from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urllib_request

from oellm_agent.can.can_decoder import CanDecoder
from oellm_agent.config.thresholds.thresholds import load_thresholds, summarize_thresholds
from oellm_agent.kb.vector_kb import VectorKB
from oellm_agent.online_pipline import SlidingWindowAggregator
from oellm_agent.risk_engine import RiskEngine

from .config import AgentConfig


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
        self._thresholds = load_thresholds(Path(__file__).resolve().parents[1], model="50")
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
