from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from .config import AgentConfig


class OellmSession:
    def __init__(self, cfg: AgentConfig, worker_index: int = 0):
        self.cfg = cfg
        self.worker_index = worker_index
        self.proc: Optional[subprocess.Popen] = None
        self.out_queue: "queue.Queue[str]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._ask_lock = threading.Lock()
        self._ask_count = 0
        self._started_at: Optional[float] = None
        self._max_requests_per_session = max(1, int(getattr(cfg, "max_requests_per_session", 24) or 24))
        self._max_session_age_sec = max(60.0, float(getattr(cfg, "max_session_age_sec", 300.0) or 300.0))
        model_api_url = (getattr(cfg, "model_api_url", "") or "").strip()
        self._local_mode = bool(getattr(cfg, "enable_local_model", False) and not model_api_url)
        self._local_http_port = int(getattr(cfg, "local_model_base_port", 18081)) + int(worker_index)
        self._http_model_base = "" if self._local_mode else model_api_url.rstrip("/")

    def start(self) -> None:
        if not self._local_mode:
            return
        if self.proc is not None:
            return
        if not self.cfg.run_bin.exists():
            raise FileNotFoundError(f"oellm agent runner not found: {self.cfg.run_bin}")
        if not self.cfg.hbm_path.exists():
            raise FileNotFoundError(f"model hbm not found: {self.cfg.hbm_path}")
        if not self.cfg.tokenizer_dir.exists():
            raise FileNotFoundError(f"tokenizer dir not found: {self.cfg.tokenizer_dir}")
        if not self.cfg.template_path.exists():
            raise FileNotFoundError(f"chat template not found: {self.cfg.template_path}")

        env = os.environ.copy()
        ld_library_path = env.get("LD_LIBRARY_PATH", "")
        runtime_lib = str(self.cfg.runtime_dir / "lib")
        env["LD_LIBRARY_PATH"] = f"{runtime_lib}:{ld_library_path}" if ld_library_path else runtime_lib
        cmd = [str(self.cfg.run_bin), "--hbm_path", str(self.cfg.hbm_path), "--tokenizer_dir", str(self.cfg.tokenizer_dir), "--template_path", str(self.cfg.template_path), "--model_type", str(self.cfg.model_type)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cfg.runtime_dir / "example/oellm_agent_run"),
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
        self._started_at = time.time()
        self._ask_count = 0

        deadline = time.time() + 20
        ready_buf = []
        ready_markers = ["xlm init success", "Performance prefill", "[Assistant] >>>"]
        while time.time() < deadline:
            if self.proc.poll() is not None:
                preview = "".join(ready_buf)[-2000:]
                raise RuntimeError(f"local model exited early with code {self.proc.returncode}; preview={preview}")
            try:
                ch = self.out_queue.get(timeout=0.2)
                ready_buf.append(ch)
                joined = "".join(ready_buf)
                if any(marker in joined for marker in ready_markers):
                    return
            except queue.Empty:
                continue
        preview = "".join(ready_buf)[-2000:]
        raise TimeoutError(f"timeout waiting for local model ready marker; preview={preview}")

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
            self._started_at = None
            self._ask_count = 0

    def send_raw(self, text: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("local model process is not started")
        data = (text.rstrip("\n").replace("\n", " ") + "\n").encode("utf-8", errors="replace")
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _build_http_payload(self, prompt: str) -> Dict[str, Any]:
        try:
            user_input = json.loads(prompt)
            if not isinstance(user_input, dict):
                user_input = {"prompt": prompt}
        except Exception:
            user_input = {"prompt": prompt}
        if isinstance(user_input.get("input"), dict):
            payload = dict(user_input["input"])
        else:
            payload = dict(user_input)
        for verbose_key in ("policy", "output_contract", "instruction"):
            payload.pop(verbose_key, None)
        payload["request_id"] = payload.get("request_id") or uuid.uuid4().hex[:8]
        return payload

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
        if self._local_mode:
            with self._ask_lock:
                if self._should_rotate():
                    self._rotate_session()
                while True:
                    try:
                        self.out_queue.get_nowait()
                    except queue.Empty:
                        break
                self.send_raw(prompt)
                self._ask_count += 1
                raw = self._wait_for("[User] <<<", timeout)
                return self._extract_assistant(raw)

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

    def _should_rotate(self) -> bool:
        if self.proc is None:
            return False
        if self._ask_count >= self._max_requests_per_session:
            return True
        if self._started_at is not None and (time.time() - self._started_at) >= self._max_session_age_sec:
            return True
        return False

    def _rotate_session(self) -> None:
        self.stop()
        self.start()

    @staticmethod
    def _extract_assistant(raw: str) -> str:
        m = re.search(r"\[Assistant\]\s*>>>\s*(.*?)(?:Performance prefill:|\[User\]\s*<<<)", raw, re.S)
        if m:
            return m.group(1).strip()
        return raw.strip()


class OellmSessionPool:
    def __init__(self, cfg: AgentConfig, workers: int):
        self.cfg = cfg
        self.workers = max(1, int(workers))
        self.sessions: List[OellmSession] = [OellmSession(cfg, i) for i in range(self.workers)]
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
