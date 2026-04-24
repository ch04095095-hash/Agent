#!/usr/bin/env python3
import argparse
import ast
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from can_decoder import CanDecoder
from online_pipeline import SlidingWindowAggregator
from risk_engine import RiskEngine
from thresholds import load_thresholds, summarize_thresholds
from kb.vector_kb import VectorKB


def load_thresholds(workdir: Path) -> Dict[str, float]:
    cfg = workdir / "config" / "thresholds.json"
    if not cfg.exists():
        raise FileNotFoundError(f"threshold config not found: {cfg}")
    obj = json.loads(cfg.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("threshold config must be a JSON object")
    return {str(k): float(v) for k, v in obj.items()}


DEFAULT_RUNTIME_DIR = Path(
    "/home/root/llm/D-Robotics_LLM_S100_1.0.0_SDK/oellm_runtime"
)


SYSTEM_PROMPT = (
    "你是一个运行在板端的实用助手。"
    "当你需要调用工具时，必须只输出一行JSON，格式为:"
    '{"action":"tool_call","tool":"read_file|write_file|run_shell|kb_search","args":{...}}。'
    "当你不需要工具时，输出:"
    '{"action":"respond","content":"..."}。'
    "不要输出除JSON之外的任何内容。"
)


DEFAULT_MODEL_API_URL = "http://127.0.0.1:18081/infer"
DEFAULT_MODEL_NAME = "oellm_multichat_worker"


@dataclass
class AgentConfig:
    runtime_dir: Path
    multichat_bin: Path
    multichat_cfg: Path
    run_bin: Path
    hbm_path: Path
    tokenizer_dir: Path
    template_path: Path
    model_type: int
    model_api_url: str
    model_name: str
    api_key: str
    enable_local_model: bool
    local_model_workers: int
    workdir: Path
    kb_db: Path
    kb_collection: str
    kb_embed_model: str
    kb_rerank_model: str
    http_host: str
    http_port: int


class OellmSession:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.out_queue: "queue.Queue[str]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._ask_lock = threading.Lock()
        self._local_mode = bool(getattr(cfg, "enable_local_model", False))
        self._http_model_base = (self.cfg.model_api_url or "").strip().rstrip("/")

    def start(self) -> None:
        if not self._local_mode:
            return
        if self.proc is not None:
            return
        if not self.cfg.multichat_bin.exists():
            raise FileNotFoundError(f"multichat binary not found: {self.cfg.multichat_bin}")
        if not self.cfg.multichat_cfg.exists():
            raise FileNotFoundError(f"multichat config not found: {self.cfg.multichat_cfg}")

        env = os.environ.copy()
        ld_library_path = env.get("LD_LIBRARY_PATH", "")
        runtime_lib = str(self.cfg.runtime_dir / "lib")
        env["LD_LIBRARY_PATH"] = f"{runtime_lib}:{ld_library_path}" if ld_library_path else runtime_lib
        cmd = [str(self.cfg.multichat_bin), "-c", str(self.cfg.multichat_cfg)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cfg.multichat_cfg.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
            env=env,
        )
        def _reader() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            while True:
                chunk = self.proc.stdout.read(1)
                if not chunk:
                    break
                try:
                    ch = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    ch = chunk.decode("utf-8", errors="replace")
                self.out_queue.put(ch)

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

        deadline = time.time() + 10
        ready_buf = []
        ready_markers = ["xlm init success", "HTTP ready", "Load dnn model success", "Init model success"]
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"local model exited early with code {self.proc.returncode}")
            try:
                ch = self.out_queue.get(timeout=0.2)
                ready_buf.append(ch)
                joined = "".join(ready_buf)
                if any(marker in joined for marker in ready_markers):
                    return
            except queue.Empty:
                continue
        preview = "".join(ready_buf)[-1000:]
        raise TimeoutError("timeout waiting for local model ready marker")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                try:
                    self.proc.stdin.write("exit\n")
                    self.proc.stdin.flush()
                except Exception:
                    pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        finally:
            self.proc = None

    def send_raw(self, text: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("local model process is not started")
        data = (text.rstrip("\n") + "\n").encode("utf-8", errors="replace")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _build_http_payload(self, prompt: str) -> Dict[str, Any]:
        try:
            user_input = json.loads(prompt)
            if not isinstance(user_input, dict):
                user_input = {"prompt": prompt}
        except Exception:
            user_input = {"prompt": prompt}
        return {
            "current": user_input.get("current", {}),
            "window": user_input.get("window", {}),
            "event_state": user_input.get("event_state", {}),
            "history": user_input.get("history", {}),
            "request_id": uuid.uuid4().hex[:8],
        }

    def _http_health(self, timeout: int = 3) -> Dict[str, Any]:
        if not self._http_model_base:
            return {"status": "disabled"}
        url = self._http_model_base + "/health"
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def ask(self, prompt: str, timeout: int = 240) -> str:
        if self._http_model_base:
            health = self._http_health(timeout=min(3, max(1, timeout)))
            if str(health.get("status", "")).lower() not in {"ok", "healthy"}:
                raise RuntimeError(f"model worker unhealthy: {health}")
            payload = self._build_http_payload(prompt)
            headers = {"Content-Type": "application/json"}
            if self.cfg.api_key:
                headers["Authorization"] = f"Bearer {self.cfg.api_key}"
            req = urllib_request.Request(
                self._http_model_base + "/infer",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib_request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
            except HTTPError as e:
                raise RuntimeError(f"model worker HTTPError {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
            except URLError as e:
                raise RuntimeError(f"model worker URLError: {e}") from e

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            if isinstance(obj, dict):
                if isinstance(obj.get("reason"), str):
                    return json.dumps(obj, ensure_ascii=False)
                if isinstance(obj.get("decision"), dict):
                    return json.dumps(obj["decision"], ensure_ascii=False)
            return raw

        if self._local_mode:
            with self._ask_lock:
                if self.proc is None:
                    self.start()
                if self.proc is not None:
                    self.send_raw(prompt)
                    raw = self._wait_for("[User] <<<", timeout=timeout)
                    assistant = self._extract_assistant(raw)
                    return assistant

        payload = {
            "model": self.cfg.model_name or "unknown",
            "messages": [
                {"role": "system", "content": "You are a strict JSON decision engine. Output only one JSON object."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        req = urllib_request.Request(self.cfg.model_api_url, data=body, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            raise RuntimeError(f"model API HTTPError {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
        except URLError as e:
            raise RuntimeError(f"model API URLError: {e}") from e

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw

        if isinstance(obj, dict):
            if "choices" in obj and isinstance(obj["choices"], list) and obj["choices"]:
                choice = obj["choices"][0] or {}
                if isinstance(choice, dict):
                    msg = choice.get("message") or {}
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    if isinstance(choice.get("text"), str):
                        return choice["text"]
            if "output" in obj and isinstance(obj.get("output"), str):
                return obj["output"]
            if "content" in obj and isinstance(obj.get("content"), str):
                return obj["content"]
        return raw

    def _wait_for(self, marker: str, timeout: int) -> str:
        acc: List[str] = []
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ch = self.out_queue.get(timeout=0.2)
                acc.append(ch)
                if marker in "".join(acc):
                    return "".join(acc)
            except queue.Empty:
                if self.proc and self.proc.poll() is not None:
                    return "".join(acc)
                continue
        raise TimeoutError(f"Timed out waiting for marker: {marker}")

    @staticmethod
    def _extract_assistant(raw: str) -> str:
        m = re.search(r"\[Assistant\]\s*>>>\s*(.*?)\s*\[User\]\s*<<<", raw, re.S)
        if m:
            return m.group(1).strip()
        return raw.strip()


class OellmSessionPool:
    def __init__(self, cfg: AgentConfig, workers: int):
        self.cfg = cfg
        self.workers = max(1, int(workers))
        self.sessions: List[OellmSession] = [OellmSession(cfg) for _ in range(self.workers)]
        self._rr_index = 0
        self._rr_lock = threading.Lock()

    def start(self) -> None:
        for s in self.sessions:
            s.start()

    def stop(self) -> None:
        for s in self.sessions:
            try:
                s.stop()
            except Exception:
                pass

    def acquire(self) -> OellmSession:
        with self._rr_lock:
            s = self.sessions[self._rr_index % len(self.sessions)]
            self._rr_index += 1
            return s


@dataclass
class WorkerHealth:
    busy: bool = False
    last_submit_ms: float = 0.0
    last_success_ms: float = 0.0
    last_elapsed_ms: float = 0.0
    consecutive_timeouts: int = 0
    last_error: str = ""
    last_request_id: str = ""


class HttpModelClient:
    def __init__(self, base_url: str, timeout_sec: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.health = WorkerHealth()
        self._lock = threading.Lock()

    def _request_json(self, path: str, payload: Optional[Dict[str, Any]] = None, method: str = "GET") -> Dict[str, Any]:
        url = self.base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(url, data=data, headers=headers, method=method)
        with urllib_request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {"raw": obj}

    def health_check(self) -> Dict[str, Any]:
        try:
            return self._request_json("/health", method="GET")
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def infer(self, payload: Dict[str, Any], timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        timeout = float(timeout_sec or self.timeout_sec)
        request_id = uuid.uuid4().hex[:8]
        body = dict(payload)
        body.setdefault("request_id", request_id)
        with self._lock:
            self.health.busy = True
            self.health.last_submit_ms = time.time() * 1000.0
            self.health.last_request_id = request_id
        t0 = time.time()
        try:
            data = self._request_json("/infer", payload=body, method="POST")
            elapsed_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self.health.busy = False
                self.health.last_elapsed_ms = elapsed_ms
                self.health.last_success_ms = time.time() * 1000.0
                self.health.consecutive_timeouts = 0
                self.health.last_error = ""
            return data
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000.0
            err = str(e)
            with self._lock:
                self.health.busy = False
                self.health.last_elapsed_ms = elapsed_ms
                self.health.consecutive_timeouts += 1
                self.health.last_error = err
            raise


class Toolset:
    def __init__(self, workdir: Path, cfg: AgentConfig):
        self.workdir = workdir.resolve()
        self.cfg = cfg
        self._vector_kb: Optional[VectorKB] = None
        self._kb_lock = threading.Lock()
        self._can_decoder = CanDecoder()
        self._window_aggregator = SlidingWindowAggregator()
        self._thresholds = load_thresholds(self.workdir)
        print("THRESHOLDS_LOADED", json.dumps(summarize_thresholds(self._thresholds), ensure_ascii=False), flush=True)
        self._risk_engine = RiskEngine(self._thresholds)
        self.model_client: Optional[HttpModelClient] = None

    def _safe_path(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (self.workdir / path).resolve()
        else:
            path = path.resolve()
        if not str(path).startswith(str(self.workdir)):
            raise ValueError(f"path not allowed: {path}")
        return path

    def read_file(self, path: str, max_chars: int = 4000) -> str:
        fp = self._safe_path(path)
        data = fp.read_text(encoding="utf-8")
        return data[:max_chars]

    def write_file(self, path: str, content: str) -> str:
        fp = self._safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"written: {fp}"

    def model_health(self) -> Dict[str, Any]:
        if self.model_client is None:
            return {"status": "disabled"}
        health = self.model_client.health_check()
        health["client_busy"] = self.model_client.health.busy
        health["client_last_submit_ms"] = self.model_client.health.last_submit_ms
        health["client_last_success_ms"] = self.model_client.health.last_success_ms
        health["client_last_elapsed_ms"] = self.model_client.health.last_elapsed_ms
        health["client_consecutive_timeouts"] = self.model_client.health.consecutive_timeouts
        health["client_last_error"] = self.model_client.health.last_error
        health["client_last_request_id"] = self.model_client.health.last_request_id
        return health

    def run_shell(self, command: str, timeout_sec: int = 10) -> str:
        allow_prefix = ["ls", "pwd", "echo", "cat", "head", "tail", "grep", "wc"]
        first = shlex.split(command)[0] if command.strip() else ""
        if first not in allow_prefix:
            raise ValueError(f"command not allowed: {first}")
        cp = subprocess.run(
            command,
            shell=True,
            cwd=str(self.workdir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (cp.stdout or "") + (cp.stderr or "")
        return out[:4000]

    def _get_vector_kb(self) -> VectorKB:
        with self._kb_lock:
            if self._vector_kb is None:
                self._vector_kb = VectorKB(
                    db_path=self.cfg.kb_db,
                    collection=self.cfg.kb_collection,
                    embed_model=self.cfg.kb_embed_model,
                    rerank_model=self.cfg.kb_rerank_model,
                )
            return self._vector_kb

    def kb_search(
        self,
        query: str,
        top_k: int = 5,
        source_type: str = "",
        dense_k: int = 30,
        bm25_k: int = 30,
        w_dense: float = 0.7,
        w_bm25: float = 0.3,
    ) -> List[Dict[str, Any]]:
        kb = self._get_vector_kb()
        return kb.search(
            query=query,
            top_k=top_k,
            source_type=source_type or None,
            dense_k=dense_k,
            bm25_k=bm25_k,
            w_dense=w_dense,
            w_bm25=w_bm25,
            rerank=True,
            rerank_top_k=max(top_k, 20),
        )

    def decode_can_frame(self, frame_id: int, data: List[int], timestamp_sec: Optional[float] = None) -> Dict[str, Any]:
        self._can_decoder.update(frame_id, data)
        decoded = self._can_decoder.state.to_dict()
        ts = timestamp_sec if timestamp_sec is not None else time.time()
        self._window_aggregator.add_sample(float(ts), decoded)
        return decoded

    def get_window_json(self) -> Dict[str, Any]:
        return self._window_aggregator.to_window_json()


def strip_json_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()



def try_parse_json(text: str) -> Optional[Dict]:
    text = strip_json_wrappers(text)

    # 1) 直接JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 2) 提取首个{...} JSON片段
    m = re.search(r"\{.*\}", text, re.S)
    candidate = m.group(0).strip() if m else ""
    if candidate:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

        # 3) 兼容常见“伪JSON”:
        # - 单引号
        # - Python True/False/None
        # - 末尾多余逗号
        normalized = candidate
        normalized = normalized.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        normalized = re.sub(r"\bTrue\b", "true", normalized)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)

        try:
            obj = json.loads(normalized)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

        # 4) 兜底：按Python字面量解析
        try:
            obj = ast.literal_eval(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    return None


def summarize_sensor(sensor_payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in [
        "vehicle_id",
        "timestamp",
        "speed_kmh",
        "engine_rpm",
        "coolant_temp_c",
        "brake_pressure_bar",
        "battery_v",
    ]:
        if k in sensor_payload:
            parts.append(f"{k}={sensor_payload[k]}")
    fault_codes = sensor_payload.get("fault_codes") or []
    if fault_codes:
        parts.append("fault_codes=" + ",".join(str(x) for x in fault_codes))
    return "; ".join(parts)


def _round_if_number(value: Any, ndigits: int = 2) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def slim_current_state(current_state: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "speed_kmh",
        "engine_rpm",
        "angle_deg",
        "slope_state",
        "gear_state",
        "emergency_stop",
        "alarm_code",
        "coolant_temp_c",
        "surface_temp_c",
        "exhaust_temp_c",
        "methane_pct",
        "co_ppm",
        "brake_pressure_bar",
        "system_pressure_bar",
        "travel_pressure_bar",
        "clamp_pressure_bar",
        "hydraulic_oil_temp_c",
        "water_tank_level_pct",
        "battery_v",
        "risk_level",
        "risk_score",
        "warning_tag",
        "layer1_action",
    ]
    slim: Dict[str, Any] = {}
    for key in keep_keys:
        if key in current_state:
            slim[key] = _round_if_number(current_state.get(key))
    return slim


def slim_risk_result(risk_result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "risk_level": risk_result.get("risk_level"),
        "warning_tag": risk_result.get("warning_tag"),
        "alarm_code": int(risk_result.get("alarm_code", 0) or 0),
        "risk_score": _round_if_number(risk_result.get("risk_score")),
        "deviation_score": _round_if_number(risk_result.get("deviation_score")),
        "suggested_actions": risk_result.get("suggested_actions", []),
    }
    event_state = risk_result.get("event_state", {})
    if isinstance(event_state, dict):
        out["event_state"] = {
            "primary_event": event_state.get("primary_event", "normal"),
            "severity": event_state.get("severity", "normal"),
            "secondary_events": event_state.get("secondary_events", []),
            "recommended_adjustments": event_state.get("recommended_adjustments", []),
            "reason": event_state.get("reason", ""),
        }
    return out


def build_model_input(decoded_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vehicle_id": decoded_state.get("vehicle_id"),
        "timestamp": decoded_state.get("timestamp"),
        "decoded_state": decoded_state,
    }


ALLOWED_ACTIONS = {"EMERGENCY_STOP", "FORWARD", "REVERSE", "ACCELERATE", "DECELERATE", "BRAKE"}
ALLOWED_RISK_LEVELS = {"normal", "warning", "danger"}
ALLOWED_OUTPUT_KEYS = {
    "action",
    "risk_level",
    "target",
    "speed_kmh",
    "reason",
    "suspected_fault",
    "recommended_adjustment",
    "monitor_next",
    "confidence",
}

ALLOWED_EVENT_SEVERITIES = {"normal", "info", "warning", "danger"}
ALLOWED_EVENT_TAGS = {
    "normal",
    "emergency_stop",
    "heartbeat_lost",
    "coolant_over_stop",
    "surface_over_stop",
    "exhaust_over_stop",
    "methane_over_stop",
    "coolant_near_alarm",
    "coolant_warning_rising",
    "water_level_low_warning",
    "brake_pressure_warning",
    "system_pressure_warning",
    "methane_near_alarm",
    "hydraulic_temp_near_alarm",
    "coolant_recovered",
    "pressure_recovered",
    "gas_recovered",
    "ready_to_move",
    "coolant_high_and_low_level",
    "coolant_high_and_rising",
    "pressure_low_and_temp_high",
    "multi_signal_warning",
    "sensor_jump",
    "sensor_stuck",
    "data_missing",
    "communication_fault",
    "timestamp_irregular",
}



def safe_fallback_decision(reason: str, evidence: List[str], suspected_fault: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "action": "HOLD",
        "risk_level": "warning",
        "target": None,
        "speed_kmh": 0,
        "reason": reason,
        "suspected_fault": suspected_fault or [],
        "recommended_adjustment": ["reduce_speed", "reduce_rpm"],
        "monitor_next": ["coolant_temp_c", "system_pressure_bar", "brake_pressure_bar"],
        "confidence": 0.2,
        "kb_evidence": evidence,
    }


def parse_can_input(sensor_payload: Dict[str, Any], tools: Toolset) -> Dict[str, Any]:
    if "frame_id" in sensor_payload and "data" in sensor_payload:
        frame_id_raw = sensor_payload.get("frame_id")
        if isinstance(frame_id_raw, str):
            frame_id = int(frame_id_raw, 16)
        else:
            frame_id = int(frame_id_raw)
        data = sensor_payload.get("data", [])
        if not isinstance(data, list):
            raise ValueError("data must be a list of 8 bytes")
        timestamp_raw = sensor_payload.get("timestamp")
        timestamp_sec: Optional[float]
        if timestamp_raw is None:
            timestamp_sec = None
        else:
            try:
                timestamp_sec = float(timestamp_raw)
            except Exception:
                timestamp_sec = None
        decoded_state = tools.decode_can_frame(frame_id, [int(x) for x in data], timestamp_sec=timestamp_sec)
        decoded_state = dict(decoded_state)
        decoded_state["frame_id"] = f"0x{frame_id:08X}"
        decoded_state["timestamp"] = sensor_payload.get("timestamp")
        decoded_state["vehicle_id"] = sensor_payload.get("vehicle_id")
        return build_model_input(decoded_state)

    if "decoded_state" in sensor_payload:
        decoded_state = dict(sensor_payload.get("decoded_state", sensor_payload))
        return build_model_input(decoded_state)

    return build_model_input(sensor_payload)


def build_event_semantics(model_input: Dict[str, Any], sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
    # 兼容旧版输入（current/window/history）与新版输入（features/history）
    current = sensor_payload.get("current", {}) if isinstance(sensor_payload.get("current", {}), dict) else {}
    window = sensor_payload.get("window", {}) if isinstance(sensor_payload.get("window", {}), dict) else {}
    signals = window.get("signals", {}) if isinstance(window.get("signals", {}), dict) else {}
    features = sensor_payload.get("features", {}) if isinstance(sensor_payload.get("features", {}), dict) else {}
    history = sensor_payload.get("history", {}) if isinstance(sensor_payload.get("history", {}), dict) else {}
    return {
        "current": current,
        "window": window,
        "signals": signals,
        "features": features,
        "history": history,
    }




ACTION_ALIASES = {
    "EMERGENCY_STOP": "EMERGENCY_STOP",
    "急停": "EMERGENCY_STOP",
    "紧急停止": "EMERGENCY_STOP",
    "FORWARD": "FORWARD",
    "MOVE": "FORWARD",
    "GO": "FORWARD",
    "继续": "FORWARD",
    "前进": "FORWARD",
    "REVERSE": "REVERSE",
    "后退": "REVERSE",
    "倒车": "REVERSE",
    "ACCELERATE": "ACCELERATE",
    "加速": "ACCELERATE",
    "DECELERATE": "DECELERATE",
    "减速": "DECELERATE",
    "BRAKE": "BRAKE",
    "STOP": "BRAKE",
    "刹车": "BRAKE",
    "制动": "BRAKE",
    "立即制动": "BRAKE",
    "HOLD": "DECELERATE",
    "WAIT": "DECELERATE",
    "保持": "DECELERATE",
    "等待": "DECELERATE",
    "暂停": "DECELERATE",
}

CONFIDENCE_ALIASES = {
    "高": 0.9,
    "较高": 0.8,
    "中": 0.6,
    "一般": 0.5,
    "较低": 0.3,
    "低": 0.2,
    "未指定": 0.5,
}

SUSPECTED_FAULT_EMPTY_MARKERS = {"未检测到", "无", "无故障", "none", "null", "未指定", "未知"}


def coerce_action(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in ACTION_ALIASES:
        return ACTION_ALIASES[upper]
    if text in ACTION_ALIASES:
        return ACTION_ALIASES[text]
    return None



def coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if text in {"未指定", "未知", "N/A", "NA", "-"}:
        return default
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        return float(m.group(0))
    return default



def coerce_confidence(value: Any, default: float = 0.5) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        conf = float(value)
        return max(0.0, min(1.0, conf))
    text = str(value).strip()
    if not text:
        return default
    if text in CONFIDENCE_ALIASES:
        return CONFIDENCE_ALIASES[text]
    if text.endswith("%"):
        number = coerce_float(text[:-1], default * 100)
        return max(0.0, min(1.0, number / 100.0))
    conf = coerce_float(text, default)
    if conf > 1:
        conf = conf / 100.0 if conf <= 100 else default
    return max(0.0, min(1.0, conf))



def coerce_suspected_fault(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(x).strip() for x in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.lower() in SUSPECTED_FAULT_EMPTY_MARKERS or text in SUSPECTED_FAULT_EMPTY_MARKERS:
            return []
        items = [text]
    cleaned = []
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        if normalized.lower() in SUSPECTED_FAULT_EMPTY_MARKERS or normalized in SUSPECTED_FAULT_EMPTY_MARKERS:
            continue
        cleaned.append(normalized)
    return cleaned



def decide_from_sensor(
    session: Any,
    tools: Toolset,
    sensor_payload: Dict[str, Any],
) -> Dict[str, Any]:
    sensor_summary = summarize_sensor(sensor_payload)
    model_input = parse_can_input(sensor_payload, tools)
    event_semantics = build_event_semantics(model_input, sensor_payload)

    # 决策阶段不注入 KB 文本，只保留结构化输入，避免语义跑偏
    evidence_ids: List[str] = []
    evidence_text = "无命中(已关闭KB注入)"

    current_state = sensor_payload.get("current", sensor_payload.get("decoded_state", model_input))
    if not isinstance(current_state, dict):
        current_state = {}

    window = sensor_payload.get("window", {})
    if not isinstance(window, dict):
        window = {}
    window_signals = window.get("signals", {}) if isinstance(window.get("signals", {}), dict) else {}
    compact_window = {}
    for k, v in window_signals.items():
        if isinstance(v, dict):
            compact_window[k] = {
                "value": _round_if_number(v.get("value")),
                "meaning": v.get("meaning", ""),
            }

    features = sensor_payload.get("features", {}) if isinstance(sensor_payload.get("features", {}), dict) else {}
    risk_features = features.get("risk", {}) if isinstance(features.get("risk", {}), dict) else {}
    ml_features = features.get("ml", {}) if isinstance(features.get("ml", {}), dict) else {}

    history = sensor_payload.get("history", {})
    if not isinstance(history, dict):
        history = {}
    history = {k: history.get(k) for k in ("last_hard_action", "stable_after_break_sec", "break_age_sec") if k in history}

    # 旧逻辑风险评估（保留）
    risk_result = tools._risk_engine.evaluate(current=current_state, window={"signals": compact_window}, history=history)
    slim_risk = slim_risk_result(risk_result)
    event_state = risk_result.get("event_state", {}) if isinstance(risk_result.get("event_state", {}), dict) else {}

    prompt_input = {
        "features": {
            "risk": {
                "level": risk_features.get("level", "normal"),
                "score": _round_if_number(risk_features.get("score", 0.0)),
                "warning_tag": risk_features.get("warning_tag", ""),
                "suggested_actions": risk_features.get("suggested_actions", []),
                "active_warnings": risk_features.get("active_warnings", []),
            },
            "ml": {
                "horizon_sec": ml_features.get("horizon_sec", 30),
                "score": _round_if_number(ml_features.get("score", 0.0)),
                "level": ml_features.get("level", "normal"),
                "model_version": ml_features.get("model_version", "none"),
                "future_warnings": ml_features.get("future_warnings", []),
            },
        },
        "history": history,
    }

    prompt = (
        "你是矿车控制决策模块，只输出一个合法 JSON。\n"
        "输入为 features.risk、features.ml、history。\n"
        "决策原则：risk.level=danger -> BRAKE；risk.level=warning 默认 DECELERATE；ml.level=warning/danger 作为前瞻风险约束。\n"
        "输出字段仅允许 action,risk_level,reason,suspected_fault,recommended_adjustment,monitor_next,confidence。\n"
        "reason 要像工程师判断：先说当前风险，再说未来风险，再说动作建议。\n"
        "只输出 JSON，不要多余文本。\n"
        f"输入: {json.dumps(prompt_input, ensure_ascii=False, separators=(',', ':'))}\n"
        "JSON:"
    )

    # 旧版提示词（保留注释，便于回滚）
    # prompt = (
    #     "你是矿车控制决策模块，只输出一个合法 JSON。\n"
    #     "输入为 current、window、history、risk_result。\n"
    #     "决策原则：risk_result.alarm_code>0 或 risk_result.risk_level=danger -> BRAKE；warning 默认 DECELERATE；若持续低速且安全边界内可 ACCELERATE；其余按工况 FORWARD/REVERSE。\n"
    #     "输出字段仅允许 action,risk_level,reason,suspected_fault,recommended_adjustment,monitor_next,confidence。\n"
    #     "reason 要像工程师判断：先说总体状态，再说关键风险，再说动作建议。\n"
    #     "只输出 JSON，不要多余文本。\n"
    #     f"输入: {json.dumps({**compact_sensor, 'risk_result': slim_risk}, ensure_ascii=False, separators=(',', ':'))}\n"
    #     "JSON:"
    # )

    model_out = session.ask(prompt, timeout=60)
    # 过滤运行时日志污染与ANSI转义
    cleaned_out = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", str(model_out))
    cleaned_out = re.sub(r"Performance\s+prefill:[^\n]*", "", cleaned_out)
    cleaned_out = re.sub(r"\[W\][^\n]*", "", cleaned_out)
    cleaned_out = strip_json_wrappers(cleaned_out)

    def _repair_decision_json(bad_text: str) -> Optional[Dict[str, Any]]:
        repair_prompt = (
            "把下面内容修复为合法 JSON 决策对象，只输出 JSON。\n"
            "输入语义仍然是 features.risk、features.ml、history。\n"
            "字段仅允许 action,risk_level,reason,suspected_fault,recommended_adjustment,monitor_next,confidence。\n"
            "action 只能是 EMERGENCY_STOP/FORWARD/REVERSE/ACCELERATE/DECELERATE/BRAKE；risk_level 只能是 normal/warning/danger。\n"
            "reason 要围绕当前风险、未来风险和历史护栏来写。\n"
            f"原始内容: {bad_text}\n"
            "只输出修复后的 JSON 对象:"
        )
        try:
            repair_out = session.ask(repair_prompt, timeout=30)
        except Exception:
            return None
        repair_cleaned = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", str(repair_out))
        repair_cleaned = re.sub(r"Performance\s+prefill:[^\n]*", "", repair_cleaned)
        repair_cleaned = re.sub(r"\[W\][^\n]*", "", repair_cleaned)
        repair_cleaned = strip_json_wrappers(repair_cleaned)
        return try_parse_json(repair_cleaned)

    def _normalize_decision_dict(d: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(d, dict):
            return None

        # 新版输出仅允许决策字段；旧的速度字段不再参与归一化。
        if set(d.keys()) - ALLOWED_OUTPUT_KEYS:
            return None

        required_keys = {"action", "risk_level", "reason", "suspected_fault", "recommended_adjustment", "monitor_next", "confidence"}
        if not required_keys.issubset(set(d.keys())):
            return None

        action = coerce_action(d.get("action"))
        risk_level = str(d.get("risk_level", "")).strip().lower()
        if action is None or risk_level not in ALLOWED_RISK_LEVELS:
            return None

        confidence = coerce_confidence(d.get("confidence", 0.5), 0.5)
        suspected_fault = coerce_suspected_fault(d.get("suspected_fault", []))
        recommended_adjustment = coerce_suspected_fault(d.get("recommended_adjustment", []))
        monitor_next = coerce_suspected_fault(d.get("monitor_next", []))

        reason = str(d.get("reason", "")).strip()
        if not reason:
            reason = "模型未提供原因"

        target = d.get("target", None)
        if target is not None:
            return None

        if not 0.0 <= confidence <= 1.0:
            return None

        # 新版不再要求 speed_kmh；保留兼容结构字段但固定为空。
        return {
            "action": action,
            "risk_level": risk_level,
            "target": None,
            "speed_kmh": None,
            "reason": reason,
            "suspected_fault": suspected_fault,
            "recommended_adjustment": recommended_adjustment,
            "monitor_next": monitor_next,
            "confidence": confidence,
        }

    # 旧逻辑（保留注释，便于回滚）
    # def _program_decision_from_event_state(evt: Dict[str, Any]) -> Dict[str, Any]:
    #     primary_event = str(evt.get("primary_event", "normal") or "normal")
    #     severity = str(evt.get("severity", "warning") or "warning").lower()
    #     secondary_events = coerce_suspected_fault(evt.get("secondary_events", []))
    #     recommended_adjustments = coerce_suspected_fault(evt.get("recommended_adjustments", []))
    #     base_reason = str(evt.get("reason", "") or "").strip() or f"事件语义: {primary_event}"
    #     speed = float(current_state.get("speed_kmh", sensor_payload.get("speed_kmh", 0)) or 0)
    #     slope_state = int(current_state.get("slope_state", 1) or 1)
    #     low_speed_persistent = speed < 5.4 and float(history.get("stable_after_break_sec", 0) or 0) > 10
    #     ...
    #     return {...}
    #
    # def _apply_control_guardrails(decision: Dict[str, Any]) -> Dict[str, Any]:
    #     out = dict(decision)
    #     speed = float(current_state.get("speed_kmh", sensor_payload.get("speed_kmh", 0.0)) or 0.0)
    #     ...
    #     return out

    def _program_decision_from_features(features: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
        risk = features.get("risk", {}) if isinstance(features.get("risk", {}), dict) else {}
        ml = features.get("ml", {}) if isinstance(features.get("ml", {}), dict) else {}

        risk_level = str(risk.get("level", "normal") or "normal").lower()
        risk_score = float(risk.get("score", 0.0) or 0.0)
        warning_tag = str(risk.get("warning_tag", "") or "").strip()
        suggested_actions = coerce_suspected_fault(risk.get("suggested_actions", []))
        active_warnings = risk.get("active_warnings", []) if isinstance(risk.get("active_warnings", []), list) else []

        ml_level = str(ml.get("level", "normal") or "normal").lower()
        ml_score = float(ml.get("score", 0.0) or 0.0)
        future_warnings = ml.get("future_warnings", []) if isinstance(ml.get("future_warnings", []), list) else []

        last_action = str(history.get("last_hard_action", history.get("last_action", "")) or "").upper()
        stable_after_break_sec = float(history.get("stable_after_break_sec", 0.0) or 0.0)
        break_age_sec = float(history.get("break_age_sec", 0.0) or 0.0)
        current_speed_kmh = float(history.get("current_speed_kmh", 0.0) or 0.0)

        reason_parts: List[str] = []
        suspected_fault: List[str] = []
        recommended_adjustment: List[str] = []
        monitor_next: List[str] = []

        if warning_tag:
            suspected_fault.append(warning_tag)
        suspected_fault.extend(suggested_actions)
        for w in active_warnings:
            tag = str(w.get("tag", "") or "").strip()
            if not tag:
                continue
            source = str(w.get("source", "") or "").strip()
            suspected_fault.append(tag)
            monitor_next.append(tag)
            if w.get("value") is None:
                continue

            # 口径区分：risk_engine 才叫“当前预警”；窗口聚合只作为“观测/趋势”
            if source == "risk_engine" or tag == warning_tag:
                reason_parts.append(f"当前预警 {tag}={w.get('value')} {w.get('unit', '')}".strip())
            else:
                reason_parts.append(f"窗口观测 {tag}={w.get('value')} {w.get('unit', '')}".strip())

        if future_warnings:
            for w in future_warnings:
                tag = str(w.get("tag", "") or "").strip()
                if tag:
                    reason_parts.append(f"未来风险 {tag}，约{w.get('eta_sec', 30)}秒内需关注")
                    suspected_fault.append(tag)
                    monitor_next.append(tag)

        low_speed_tags = {"speed_low_persistent", "speed_low_warning"}
        low_speed_signal = warning_tag in low_speed_tags or any(str(w.get("tag", "") or "").strip() in low_speed_tags for w in active_warnings)

        # 新逻辑：只看 features.risk / features.ml / history
        severe_warning_tags = {
            "travel_pressure_warning",
            "brake_pressure_bar_warning",
            "intake_pressure_kpa_warning",
            "system_pressure_bar_warning",
            "oil_pressure_kpa_warning",
            "coolant_warning",
            "hydraulic_oil_temp_warning",
            "emergency_stop",
            "communication_fault",
        }
        has_severe_warning = warning_tag in severe_warning_tags

        if low_speed_signal:
            action = "ACCELERATE"
            reason_parts.insert(0, f"低速预警需恢复速度({warning_tag or 'low_speed'})")
        elif risk_level == "danger":
            action = "BRAKE"
            reason_parts.insert(0, f"当前风险={risk_level}(score={risk_score:.3f})")
        elif risk_level == "warning":
            startup_recover_tags = {
                "travel_pressure_warning",
                "intake_pressure_kpa_warning",
                "system_pressure_bar_warning",
            }
            if current_speed_kmh < 0.8 and warning_tag in startup_recover_tags:
                action = "FORWARD"
                reason_parts.insert(0, f"起步近静止(speed={current_speed_kmh:.2f})，对{warning_tag}采取缓行恢复")
            elif current_speed_kmh < 0.6 and not has_severe_warning:
                action = "FORWARD"
                reason_parts.insert(0, f"预警但当前近静止(speed={current_speed_kmh:.2f})，先小步恢复")
            else:
                action = "DECELERATE"
                reason_parts.insert(0, f"当前风险={risk_level}(score={risk_score:.3f})")
        else:
            if ml_level in {"warning", "danger"} or ml_score >= 0.35:
                action = "DECELERATE"
                reason_parts.insert(0, f"当前风险正常，但未来风险升高(ml={ml_level}, score={ml_score:.3f})")
            elif stable_after_break_sec > 10 and last_action == "BRAKE":
                action = "ACCELERATE"
                reason_parts.insert(0, "制动后已稳定，可尝试恢复")
            elif break_age_sec > 0 and break_age_sec < 3:
                action = "DECELERATE"
                reason_parts.insert(0, "刚进入制动/减速区，先保持保守")
            else:
                action = "FORWARD"
                reason_parts.insert(0, "当前与未来风险均正常")

        # 轻量护栏
        if risk_level == "warning" and action == "ACCELERATE" and warning_tag not in {"speed_low_persistent", "speed_low_warning"}:
            action = "DECELERATE"
            reason_parts.append("预警态禁止加速")
        if risk_level == "danger" and action != "BRAKE":
            action = "BRAKE"
            reason_parts.append("危险态强制制动")
        if (
            last_action == "BRAKE"
            and stable_after_break_sec < 5
            and action in {"FORWARD", "ACCELERATE"}
            and current_speed_kmh >= 0.8
        ):
            action = "DECELERATE"
            reason_parts.append("制动后稳定不足，先减速")

        confidence = 0.95 if action == "BRAKE" else 0.92 if action == "DECELERATE" else 0.88 if action == "FORWARD" else 0.90
        if future_warnings:
            recommended_adjustment.extend([f"monitor_{str(w.get('tag','')).strip()}" for w in future_warnings if str(w.get('tag','')).strip()])
        if risk_level == "warning":
            recommended_adjustment.append("maintain_safe_speed")
        elif risk_level == "danger":
            recommended_adjustment.extend(["reduce_speed", "stop_and_inspect"])

        return {
            "action": action,
            "risk_level": "danger" if action == "BRAKE" else ("warning" if action == "DECELERATE" else "normal"),
            "target": None,
            "speed_kmh": None,
            "reason": "；".join(dict.fromkeys([p for p in reason_parts if p])),
            "suspected_fault": list(dict.fromkeys([x for x in suspected_fault if x])),
            "recommended_adjustment": list(dict.fromkeys([x for x in recommended_adjustment if x])),
            "monitor_next": list(dict.fromkeys([x for x in monitor_next if x]))[:6],
            "confidence": confidence,
        }

    def _apply_control_guardrails(decision: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
        # 新版护栏：不再依赖 current/window 数值，只基于 features/history 做最后修正。
        out = dict(decision)
        last_action = str(history.get("last_hard_action", history.get("last_action", "")) or "").upper()
        stable_after_break_sec = float(history.get("stable_after_break_sec", 0.0) or 0.0)
        if out.get("risk_level") == "danger" and out.get("action") not in {"BRAKE", "EMERGENCY_STOP"}:
            out["action"] = "BRAKE"
            out["reason"] = f"{out.get('reason', '')}；危险态强制制动".strip("；")
        if out.get("risk_level") == "warning" and out.get("action") == "ACCELERATE":
            out["action"] = "DECELERATE"
            out["reason"] = f"{out.get('reason', '')}；预警态禁止加速".strip("；")
        if out.get("risk_level") == "warning" and out.get("action") == "DECELERATE":
            current_speed_kmh = float(history.get("current_speed_kmh", 0.0) or 0.0)
            # 起步近静止时禁止“继续减速”，避免车辆锁死在0速
            if current_speed_kmh < 0.8:
                out["action"] = "FORWARD"
                out["reason"] = f"{out.get('reason', '')}；近静止时由减速切到缓行恢复".strip("；")
        if (
            last_action == "BRAKE"
            and stable_after_break_sec < 5
            and out.get("action") in {"FORWARD", "ACCELERATE"}
            and float(history.get("current_speed_kmh", 0.0) or 0.0) >= 0.8
        ):
            out["action"] = "DECELERATE"
            out["reason"] = f"{out.get('reason', '')}；制动后稳定不足，先减速".strip("；")
        return out

    program_decision = _apply_control_guardrails(_program_decision_from_features(features, history), history)
    # 旧版：如果有硬报警则强制 BRAKE（保留注释，便于回滚）
    # if risk_result.get("alarm_code", 0):
    #     program_decision["action"] = "BRAKE"
    #     program_decision["risk_level"] = "danger"
    #     program_decision["reason"] = f"风险引擎硬报警:{risk_result.get('warning_tag', '')}"
    #     program_decision["suspected_fault"] = [risk_result.get("warning_tag", "")]
    #     program_decision["recommended_adjustment"] = ["reduce_speed", "stop_and_inspect"]
    #     program_decision["monitor_next"] = ["coolant_temp_c", "system_pressure_bar", "brake_pressure_bar"]
    #     program_decision["confidence"] = max(program_decision.get("confidence", 0.0), 0.95)
    #
    # parsed = try_parse_json(cleaned_out)
    # data = _normalize_decision_dict(parsed)
    # if data is None and cleaned_out:
    #     repaired = _repair_decision_json(cleaned_out)
    #     data = _normalize_decision_dict(repaired)
    #
    # if data is None:
    #     program_decision["kb_evidence"] = evidence_ids
    #     program_decision["warning_tag"] = risk_result.get("warning_tag", "")
    #     program_decision["alarm_code"] = int(risk_result.get("alarm_code", 0) or 0)
    #     return program_decision
    #
    # data["action"] = program_decision["action"]
    # data["risk_level"] = program_decision["risk_level"]
    # data["confidence"] = max(float(data.get("confidence", 0.0) or 0.0), float(program_decision["confidence"]))
    # if not str(data.get("reason", "") or "").strip() or str(data.get("reason", "")).strip() == "模型未提供原因":
    #     data["reason"] = program_decision["reason"]
    # merged_faults = list(dict.fromkeys(program_decision["suspected_fault"] + coerce_suspected_fault(data.get("suspected_fault", []))))
    # merged_adjustments = list(dict.fromkeys(program_decision["recommended_adjustment"] + coerce_suspected_fault(data.get("recommended_adjustment", []))))
    # merged_monitor = list(dict.fromkeys(program_decision["monitor_next"] + coerce_suspected_fault(data.get("monitor_next", []))))
    # data["suspected_fault"] = merged_faults
    # data["recommended_adjustment"] = merged_adjustments
    # data["monitor_next"] = merged_monitor[:4]
    # data["kb_evidence"] = evidence_ids
    # data["warning_tag"] = risk_result.get("warning_tag", "")
    # data["alarm_code"] = int(risk_result.get("alarm_code", 0) or 0)
    # return data

    # 新版：直接以 program_decision 作为最终输出，再附上兼容字段
    program_decision["kb_evidence"] = evidence_ids
    program_decision["warning_tag"] = str(features.get("risk", {}).get("warning_tag", "") or "")
    program_decision["alarm_code"] = 110 if program_decision.get("risk_level") == "danger" else 0
    program_decision["risk_result_legacy"] = slim_risk
    return program_decision


def _init_agent_log(base_dir: Path, run_id: str) -> Path:
    logs_dir = (base_dir / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"agent_history_{run_id}.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def _append_agent_event(log_path: Path, event: Dict[str, Any]) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return log_path


def make_http_handler(session_pool: OellmSessionPool, tools: Toolset, agent_log_path: Path):
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
                session = session_pool.acquire()
                decision = decide_from_sensor(session=session, tools=tools, sensor_payload=payload)
                response_ts = time.time()
                self._send_json(200, {"decision": decision})
                elapsed_ms = int((response_ts - t0) * 1000)
                current = payload.get("current", {}) if isinstance(payload.get("current", {}), dict) else {}
                evt = {
                    "event": "decision_ok",
                    "ts": response_ts,
                    "request_id": req_id,
                    "request_ts": request_ts,
                    "response_ts": response_ts,
                    "elapsed_ms": elapsed_ms,
                    "action": decision.get("action", ""),
                    "risk_level": decision.get("risk_level", ""),
                    "confidence": decision.get("confidence", None),
                    "reason": decision.get("reason", ""),
                    "warning_tag": decision.get("warning_tag", ""),
                    "alarm_code": decision.get("alarm_code", 0),
                    "input_summary": {
                        "speed_kmh": (
                            (payload.get("history", {}) or {}).get("current_speed_kmh", None)
                            if isinstance(payload.get("history", {}), dict)
                            else None
                        ),
                        "l1_action": current.get("layer1_action", ""),
                        "l1_reasons": current.get("layer1_reasons", []),
                    },
                }
                _append_agent_event(agent_log_path, evt)
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
                _append_agent_event(agent_log_path, evt)
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
    print(f"HTTP ready at http://{host}:{port} (POST /sensor/decision)")
    return server


def run_agent(cfg: AgentConfig) -> None:
    use_local_model = bool(cfg.enable_local_model and not cfg.model_api_url)
    session_pool = OellmSessionPool(cfg, cfg.local_model_workers) if use_local_model else None
    tools = Toolset(cfg.workdir, cfg)

    agent_run_id = time.strftime('%Y%m%d-%H%M%S')
    agent_log_path = _init_agent_log(tools.workdir, agent_run_id)
    _append_agent_event(agent_log_path, {
        "event": "agent_started",
        "ts": time.time(),
        "date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        "run_id": agent_run_id,
        "host": cfg.http_host,
        "port": cfg.http_port,
        "mode": "local" if use_local_model else "http",
    })

    tools._get_vector_kb()
    if session_pool is not None:
        session_pool.start()
    print(f"Local model workers: {cfg.local_model_workers}")
    print(f"Model API URL: {cfg.model_api_url}")
    print(f"Model name: {cfg.model_name}")
    print(json.dumps({"event": "model_client_config", "model_api_url": cfg.model_api_url, "model_name": cfg.model_name, "enable_local_model": cfg.enable_local_model, "local_model_workers": cfg.local_model_workers, "mode": "local" if use_local_model else "http"}, ensure_ascii=False), flush=True)
    http_server = run_http_server(session_pool or OellmSessionPool(cfg, 1), tools, cfg.http_host, cfg.http_port, agent_log_path)
    stop_event = threading.Event()
    try:
        print("Agent started. HTTP server only mode (POST /sensor/decision).")
        print("Use Ctrl+C to stop the agent.")
        while not stop_event.is_set():
            stop_event.wait(1.0)
    except KeyboardInterrupt:
        print("Agent stopping by KeyboardInterrupt...")
    finally:
        _append_agent_event(agent_log_path, {
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


def run_once(cfg: AgentConfig, payload_text: str) -> int:
    session = OellmSession(cfg)
    tools = Toolset(cfg.workdir, cfg)
    tools._get_vector_kb()
    session.start()
    try:
        payload = try_parse_json(payload_text)
        if not isinstance(payload, dict):
            print(json.dumps({"error": "payload must be JSON object"}, ensure_ascii=False))
            return 2
        decision = decide_from_sensor(session=session, tools=tools, sensor_payload=payload)
        print(json.dumps({"decision": decision}, ensure_ascii=False))
        return 0
    finally:
        session.stop()


def build_config(args: argparse.Namespace) -> AgentConfig:
    runtime_dir = Path(args.runtime_dir).resolve()
    multichat_bin = (
        Path(args.multichat_bin).resolve()
        if args.multichat_bin
        else runtime_dir / "example/oellm_multichat/oellm_multichat"
    )
    multichat_cfg = (
        Path(args.multichat_cfg).resolve()
        if args.multichat_cfg
        else runtime_dir / "example/oellm_multichat/qwen_multichat_config.json"
    )
    run_bin = (
        Path(args.run_bin).resolve()
        if args.run_bin
        else runtime_dir / "example/oellm_run/oellm_run"
    )
    hbm_path = (
        Path(args.hbm_path).resolve()
        if args.hbm_path
        else runtime_dir / "model/Qwen2.5_1.5B_Instruct_1024.hbm"
    )
    tokenizer_dir = (
        Path(args.tokenizer_dir).resolve()
        if args.tokenizer_dir
        else runtime_dir / "config/Qwen2.5_1.5B_Instruct_config"
    )
    template_path = (
        Path(args.template_path).resolve()
        if args.template_path
        else runtime_dir / "config/Qwen2.5_1.5B_Instruct_config/Qwen2.5_1.5B_Instruct.jinja"
    )
    workdir = Path(args.workdir).resolve() if args.workdir else Path.cwd().resolve()
    kb_db = Path(args.kb_db).resolve() if args.kb_db else (workdir / "kb/api/kb/data/kb.sqlite3").resolve()
    return AgentConfig(
        runtime_dir=runtime_dir,
        multichat_bin=multichat_bin,
        multichat_cfg=multichat_cfg,
        run_bin=run_bin,
        hbm_path=hbm_path,
        tokenizer_dir=tokenizer_dir,
        template_path=template_path,
        model_type=args.model_type,
        model_api_url=args.model_api_url,
        model_name=args.model_name,
        api_key=args.api_key,
        enable_local_model=args.enable_local_model,
        local_model_workers=max(1, int(args.local_model_workers)),
        workdir=workdir,
        kb_db=kb_db,
        kb_collection=args.kb_collection,
        kb_embed_model=args.kb_embed_model,
        kb_rerank_model=args.kb_rerank_model,
        http_host="0.0.0.0",
        http_port=18080,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Board-side Python Agent based on oellm_multichat")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR), help="oellm_runtime path")
    parser.add_argument("--multichat-bin", default="", help="path to oellm_multichat binary")
    parser.add_argument("--multichat-cfg", default="", help="path to multichat config json")
    parser.add_argument("--run-bin", default="", help="path to oellm_run binary")
    parser.add_argument("--hbm-path", default="", help="path to model hbm file")
    parser.add_argument("--tokenizer-dir", default="", help="path to tokenizer dir")
    parser.add_argument("--template-path", default="", help="path to chat template jinja")
    parser.add_argument("--model-type", type=int, default=7, help="model type for oellm_run")
    parser.add_argument("--workdir", default="", help="agent allowed workspace root")

    parser.add_argument("--kb-db", default="", help="sqlite metadata db path (default: <workdir>/kb/data/kb.sqlite3)")
    parser.add_argument("--kb-collection", default="coal_truck_kb", help="faiss index name")
    parser.add_argument(
        "--kb-embed-model",
        default="/mnt/ssd/Agent/modelscope/hub/models/BAAI/bge-small-zh-v1___5",
        help="sentence-transformers embedding model local directory",
    )
    parser.add_argument(
        "--kb-rerank-model",
        default="/mnt/ssd/Agent/modelscope/hub/models/BAAI/bge-reranker-base",
        help="sentence-transformers reranker model local directory",
    )
    parser.add_argument("--model-api-url", default=DEFAULT_MODEL_API_URL, help="direct model API URL, e.g. http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="model name for API payload")
    parser.add_argument("--api-key", default="", help="optional API key")
    parser.add_argument("--enable-local-model", action="store_true", help="use local oellm_multichat process")
    parser.add_argument("--local-model-workers", type=int, default=3, help="number of local model workers (default: 3)")
    parser.add_argument("--model-timeout-sec", type=float, default=5.0, help="model request timeout (seconds)")
    parser.add_argument("--once", action="store_true", help="read one JSON payload from stdin and output one decision")
    args = parser.parse_args()
    cfg = build_config(args)
    if not args.enable_local_model and not cfg.model_api_url:
        raise ValueError("either --enable-local-model or --model-api-url is required")
    if args.once:
        import sys
        payload_text = sys.stdin.read()
        raise SystemExit(run_once(cfg, payload_text))
    run_agent(cfg)


if __name__ == "__main__":
    main()
