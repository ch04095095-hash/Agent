from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import replace
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request as urllib_request
from urllib.error import URLError

from .config import AgentConfig
from .decision import decide_from_sensor, decide_from_sensor_llm
from .session import OellmSessionPool
from .tools import Toolset


def init_agent_log(base_dir: Path, run_id: str) -> Path:
    logs_dir = (base_dir / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"agent_history_{run_id}.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def append_agent_event(log_path: Path, event: Dict[str, Any]) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return log_path


def make_http_handler(session_pool: OellmSessionPool, tools: Toolset, agent_log_path: Path):
    sse_clients = []
    sse_lock = threading.Lock()
    event_buffer = deque(maxlen=200)

    def _broadcast_sse(event: Dict[str, Any]) -> None:
        payload = f"event: {event.get('event', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n".encode('utf-8')
        with sse_lock:
            event_buffer.append(event)
            dead = []
            for wfile in list(sse_clients):
                try:
                    wfile.write(payload)
                    wfile.flush()
                except Exception:
                    dead.append(wfile)
            for wfile in dead:
                try:
                    sse_clients.remove(wfile)
                except ValueError:
                    pass

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                return

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/sensor/decision/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    with sse_lock:
                        sse_clients.append(self.wfile)
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                    while True:
                        time.sleep(15)
                        try:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
                finally:
                    with sse_lock:
                        try:
                            sse_clients.remove(self.wfile)
                        except ValueError:
                            pass
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/sensor/decision":
                self._send_json(404, {"error": "not found"})
                return
            t0 = time.time()
            req_id = self.headers.get("X-Request-Id", "").strip() or uuid.uuid4().hex[:8]
            raw_text = ""
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                raw_text = raw.decode("utf-8", errors="replace")
                payload = json.loads(raw_text)
                if not isinstance(payload, dict):
                    raise ValueError("payload must be JSON object")
                request_ts = time.time()
                llm_error = ""
                llm_raw_response_preview = ""
                session = session_pool.acquire()
                try:
                    decision = decide_from_sensor_llm(_session=session, _tools=tools, sensor_payload=payload, llm_only=getattr(tools.cfg, "llm_only", False))
                except Exception as e:
                    llm_error = str(e) if str(e) else repr(e)
                    llm_raw_response_preview = str(getattr(e, "llm_raw_response", "") or "")[:1000]
                    if getattr(tools.cfg, "llm_only", False):
                        raise
                    decision = decide_from_sensor(_session=None, _tools=tools, sensor_payload=payload)
                    decision["decision_source"] = "rule_fallback"
                    decision["policy"] = "rule_fallback"
                    decision["fallback_reason"] = llm_error
                    decision["llm_raw_response_preview"] = llm_raw_response_preview
                response_ts = time.time()
                response = decision if isinstance(decision, dict) else {"reason": str(decision)}
                response.setdefault("request_id", req_id)
                response.setdefault("ok", True)
                public_response = {
                    "action": str(response.get("action", "FORWARD") or "FORWARD").strip().upper(),
                    "reason": str(response.get("reason", "") or "").strip(),
                    "confidence": float(response.get("confidence", 0.0) or 0.0),
                    "request_id": req_id,
                    "ok": True,
                }
                self._send_json(200, public_response)
                _broadcast_sse({
                    "event": "decision",
                    "ts": response_ts,
                    "request_id": req_id,
                    "response": response,
                })
                elapsed_ms = int((response_ts - t0) * 1000)
                evt = {
                    "event": "decision_ok",
                    "ts": response_ts,
                    "request_id": req_id,
                    "request_ts": request_ts,
                    "response_ts": response_ts,
                    "elapsed_ms": elapsed_ms,
                    "response": public_response,
                    "input_summary": {
                        "speed_mps": ((payload.get("history", {}) or {}).get("current_speed_mps", None) if isinstance(payload.get("history", {}), dict) else None),
                        "l1_action": (payload.get("current", {}) or {}).get("layer1_action", ""),
                        "l1_reasons": (payload.get("current", {}) or {}).get("layer1_reasons", []),
                    },
                }
                append_agent_event(agent_log_path, evt)
                print("AGENT_DECISION", json.dumps(evt, ensure_ascii=False), flush=True)
            except BrokenPipeError:
                return
            except Exception as e:
                import traceback
                err_text = str(e) if str(e) else repr(e)
                tb = traceback.format_exc()
                response_ts = time.time()
                elapsed_ms = int((response_ts - t0) * 1000)
                payload_preview = raw_text[:1200] if raw_text else ""
                evt = {
                    "event": "decision_error",
                    "ts": response_ts,
                    "request_id": req_id,
                    "request_ts": t0,
                    "response_ts": response_ts,
                    "elapsed_ms": elapsed_ms,
                    "error": err_text,
                    "error_type": type(e).__name__,
                    "path": self.path,
                    "content_length": self.headers.get("Content-Length", ""),
                    "content_type": self.headers.get("Content-Type", ""),
                    "payload_preview": payload_preview,
                }
                append_agent_event(agent_log_path, evt)
                print("AGENT_ERROR", json.dumps({**evt, "traceback": tb}, ensure_ascii=False), flush=True)
                self._send_json(400, {"error": err_text, "error_type": type(e).__name__, "request_id": req_id})

        def log_message(self, format, *args):
            return

    return Handler


def run_http_server(session_pool: OellmSessionPool, tools: Toolset, host: str, port: int, agent_log_path: Path) -> Any:
    handler = make_http_handler(session_pool, tools, agent_log_path)
    server = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"HTTP ready at http://{host}:{port} (POST /sensor/decision, GET /sensor/decision/stream)")
    return server


def _http_get_json(url: str, timeout: float = 3.0) -> Dict[str, Any]:
    req = urllib_request.Request(url, method="GET")
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def _spawn_local_model_service(cfg: AgentConfig) -> Optional[subprocess.Popen]:
    if not cfg.multichat_bin.exists():
        raise FileNotFoundError(f"multichat binary not found: {cfg.multichat_bin}")
    if not cfg.multichat_cfg.exists():
        raise FileNotFoundError(f"multichat config not found: {cfg.multichat_cfg}")
    env = os.environ.copy()
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    runtime_lib = str(cfg.runtime_dir / "lib")
    env["LD_LIBRARY_PATH"] = f"{runtime_lib}:{ld_library_path}" if ld_library_path else runtime_lib
    env["OELLM_AGENT_ENABLE_LLM"] = "1"
    cmd = [
        str(cfg.multichat_bin),
        "-c",
        str(cfg.multichat_cfg),
        "--mode",
        "http",
        "--port",
        str(cfg.local_model_base_port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(cfg.multichat_bin.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=env,
    )


def _wait_model_http_ready(url: str, timeout_sec: float = 120.0) -> None:
    deadline = time.time() + timeout_sec
    last_err: Optional[str] = None
    health_url = url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            data = _http_get_json(health_url, timeout=2.0)
            if not data or data.get("status") in {"ok", "ready"}:
                return
        except Exception as e:
            last_err = str(e)
        time.sleep(1.0)
    raise TimeoutError(f"timeout waiting for model HTTP service ready: {last_err or 'unknown'}")


def run_agent(cfg: AgentConfig) -> None:
    tools = Toolset(cfg.workdir, cfg)
    agent_run_id = time.strftime('%Y%m%d-%H%M%S')
    agent_log_path = init_agent_log(tools.workdir, agent_run_id)
    tools._get_vector_kb()

    model_process: Optional[subprocess.Popen] = None
    model_base_url = (cfg.model_api_url or "http://127.0.0.1:18081/infer").strip()
    model_health_base = model_base_url.rsplit('/infer', 1)[0].rstrip('/')

    # Rule-only mode: do not require/start model service.
    # decision.py currently bypasses LLM entirely.
    model_ready = True
    use_local_model = False
    session_pool = None
    append_agent_event(agent_log_path, {
        "event": "agent_started",
        "ts": time.time(),
        "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "run_id": agent_run_id,
        "host": cfg.http_host,
        "port": cfg.http_port,
        "mode": "local" if use_local_model else "http",
    })
    print(f"Local model workers: {cfg.local_model_workers}")
    print(f"Model API URL: {model_base_url}")
    print(f"Model name: {cfg.model_name}")
    print(json.dumps({"event": "model_client_config", "model_api_url": model_base_url, "model_name": cfg.model_name, "enable_local_model": cfg.enable_local_model, "local_model_workers": cfg.local_model_workers, "mode": "rule_only", "model_ready": model_ready}, ensure_ascii=False), flush=True)
    http_cfg = replace(cfg, enable_local_model=False, model_api_url=model_base_url)
    http_server = run_http_server(OellmSessionPool(http_cfg, 1), tools, cfg.http_host, cfg.http_port, agent_log_path)
    stop_event = threading.Event()
    try:
        print("Agent started. HTTP server only mode (POST /sensor/decision).")
        print("Use Ctrl+C to stop the agent.")
        while not stop_event.is_set():
            stop_event.wait(1.0)
    except KeyboardInterrupt:
        print("Agent stopping by KeyboardInterrupt...")
    finally:
        append_agent_event(agent_log_path, {
            "event": "agent_stopped",
            "ts": time.time(),
            "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "run_id": agent_run_id,
            "host": cfg.http_host,
            "port": cfg.http_port,
            "mode": "local" if use_local_model else "http",
        })
        http_server.shutdown()
        if session_pool is not None:
            session_pool.stop()
        if model_process is not None:
            try:
                model_process.terminate()
                model_process.wait(timeout=5)
            except Exception:
                try:
                    model_process.kill()
                except Exception:
                    pass
        if model_process is not None:
            try:
                model_process.terminate()
            except Exception:
                pass
