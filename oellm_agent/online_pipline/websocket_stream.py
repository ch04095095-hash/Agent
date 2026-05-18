from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from oellm_agent.can.can_decoder import get_supported_ids

FrameBatch = Tuple[int, float, List[Tuple[int, List[int]]]]


class WebSocketFrameStream:
    def __init__(self, ws_url: str, realtime: bool = True, heartbeat_timeout_sec: float = 10.0):
        self.ws_url = ws_url
        self.realtime = realtime
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=4096)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._step = 0
        self._current_group: List[Tuple[int, List[int]]] = []
        self._current_group_ts: Optional[float] = None
        self._current_seen_ids: set[int] = set()
        self._supported_ids: set[int] = set(get_supported_ids('50'))
        self._last_msg_ts = time.time()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception as e:
            self._q.put({"_error": f"websocket-client not installed: {e}"})
            return

        def on_message(_ws: Any, message: str) -> None:
            try:
                data = json.loads(message)
                if isinstance(data, dict):
                    self._q.put(data)
            except Exception:
                return

        def on_error(_ws: Any, error: Any) -> None:
            self._q.put({"_error": str(error)})

        def on_close(_ws: Any, *_args: Any) -> None:
            self._q.put({"_close": True})

        def on_open(_ws: Any) -> None:
            self._q.put({"_open": True})

        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        while not self._stop.is_set():
            ws.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop.is_set():
                break
            time.sleep(1.0)

    def _coerce_frame_id(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text)
        except Exception:
            return None

    def _coerce_payload(self, value: Any) -> Optional[List[int]]:
        if not isinstance(value, list):
            return None
        try:
            payload = [int(x) & 0xFF for x in value[:8]]
            if len(payload) < 8:
                payload.extend([0] * (8 - len(payload)))
            return payload[:8]
        except Exception:
            return None

    def _drain_msg(self, msg: Dict[str, Any]) -> Optional[FrameBatch]:
        if msg.get("_error"):
            raise RuntimeError(str(msg.get("_error")))
        if msg.get("_close"):
            return None
        if not msg.get("frame_name") and not msg.get("id"):
            return None

        ts = float(msg.get("timestamp", msg.get("ts", time.time())) or time.time())
        fid = self._coerce_frame_id(msg.get("frame_name", msg.get("id")))
        payload = self._coerce_payload(msg.get("frame_content", msg.get("data")))
        if fid is None or payload is None:
            return None

        if self._current_group_ts is None:
            self._current_group_ts = ts

        # Flush current group first when a frame id repeats: this likely means next cycle starts.
        if self._current_group and fid in self._current_seen_ids:
            out = (self._step, self._current_group_ts, list(self._current_group))
            self._step += 1
            self._current_group = [(fid, payload)]
            self._current_group_ts = ts
            self._current_seen_ids = {fid}
            return out

        self._current_group.append((fid, payload))
        self._current_seen_ids.add(fid)

        # Preferred flush: all supported CAN ids are seen in this group.
        if self._supported_ids and self._supported_ids.issubset(self._current_seen_ids):
            out = (self._step, self._current_group_ts, list(self._current_group))
            self._step += 1
            self._current_group = []
            self._current_group_ts = None
            self._current_seen_ids = set()
            return out

        return None

    def __iter__(self):
        return self

    def __next__(self) -> FrameBatch:
        while not self._stop.is_set():
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._current_group and self._current_group_ts is not None:
                    if (time.time() - self._last_msg_ts) >= self.heartbeat_timeout_sec:
                        out = (self._step, self._current_group_ts, list(self._current_group))
                        self._step += 1
                        self._current_group = []
                        self._current_group_ts = None
                        self._current_seen_ids = set()
                        return out
                continue

            self._last_msg_ts = time.time()
            out = self._drain_msg(msg)
            if out is not None:
                return out

        raise StopIteration
