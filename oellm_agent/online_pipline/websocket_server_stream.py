from __future__ import annotations

import asyncio
import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import DT

FrameBatch = Tuple[int, float, List[Tuple[int, List[int]]]]


class WebSocketServerFrameStream:
    def __init__(self, host: str = '0.0.0.0', port: int = 9001, realtime: bool = True, heartbeat_timeout_sec: float = 10.0, csv_path: str | None = None):
        self.host = host
        self.port = port
        self.realtime = realtime
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.csv_path = Path(csv_path).expanduser().resolve() if csv_path else None
        self._csv_file = None
        self._csv_writer = None
        self._q: 'queue.Queue[Dict[str, Any]]' = queue.Queue(maxsize=4096)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._step = 0
        self._current_group: List[Tuple[int, List[int]]] = []
        self._current_group_ts: Optional[float] = None
        self._last_msg_ts = time.time()
        self._server = None
        self._clients: Dict[Any, asyncio.AbstractEventLoop] = {}
        self._clients_lock = threading.Lock()
        self._last_websocket: Any = None

    def start(self) -> None:
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = self.csv_path.open('w', newline='', encoding='utf-8')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=['received_ts', 'frame_name', 'frame_content_json'])
            self._csv_writer.writeheader()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait({'_close': True})
        except Exception:
            pass
        self._thread.join(timeout=2.0)
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

    def _coerce_frame_id(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text, 16) if text.lower().startswith('0x') else int(text)
        except Exception:
            return None

    def _coerce_payload(self, value: Any) -> Optional[List[int]]:
        if not isinstance(value, list):
            return None
        try:
            payload: List[int] = []
            for x in value[:8]:
                if isinstance(x, str):
                    bits = x.replace(' ', '').strip()
                    if len(bits) == 8 and set(bits) <= {'0', '1'}:
                        payload.append(int(bits, 2) & 0xFF)
                        continue
                    return None
                payload.append(int(x) & 0xFF)
            if len(payload) < 8:
                payload.extend([0] * (8 - len(payload)))
            return payload[:8]
        except Exception:
            return None

    def _append_csv_row(self, msg: Dict[str, Any]) -> None:
        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow({
                'received_ts': float(msg.get('timestamp', msg.get('ts', time.time())) or time.time()),
                'frame_name': str(msg.get('frame_name', msg.get('id', '')) or ''),
                'frame_content_json': json.dumps(msg.get('frame_content', msg.get('data', [])), ensure_ascii=False),
            })
            if self._csv_file is not None:
                self._csv_file.flush()
        except Exception:
            pass

    def _drain_msg(self, msg: Dict[str, Any]) -> Optional[FrameBatch]:
        if msg.get('_error'):
            raise RuntimeError(str(msg.get('_error')))
        if msg.get('_close'):
            return None
        fid = self._coerce_frame_id(msg.get('frame_name', msg.get('id')))
        payload = self._coerce_payload(msg.get('frame_content', msg.get('data')))
        if fid is None or payload is None:
            return None

        self._append_csv_row(msg)

        ts = float(msg.get('timestamp', msg.get('ts', time.time())) or time.time())
        if self._current_group_ts is None:
            self._current_group_ts = ts
        if ts - self._current_group_ts >= DT and self._current_group:
            out = (self._step, self._current_group_ts, list(self._current_group))
            self._step += 1
            self._current_group = []
            self._current_group_ts = ts
            self._current_group.append((fid, payload))
            return out

        self._current_group.append((fid, payload))
        return None

    def send(self, message: Dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False)
        with self._clients_lock:
            websocket = self._last_websocket
            loop = self._clients.get(websocket) if websocket is not None else None
        if websocket is None or loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(websocket.send(payload), loop)
            fut.result(timeout=1.0)
        except Exception:
            with self._clients_lock:
                self._clients.pop(websocket, None)
                if self._last_websocket is websocket:
                    self._last_websocket = None

    async def _handler(self, websocket: Any) -> None:
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            self._clients[websocket] = loop
            self._last_websocket = websocket
        try:
            async for message in websocket:
                if self._stop.is_set():
                    break
                try:
                    data = json.loads(message)
                    if isinstance(data, dict):
                        print(json.dumps({'layer': 'WEBSOCKET_SERVER', 'event': 'frame_received', 'payload': data}, ensure_ascii=False), flush=True)
                        self._q.put(data)

                        reply = json.dumps({
                            'timestamp': time.time(),
                            'frame_name': data.get('frame_name', ''),
                            'frame_content': data.get('frame_content', []),
                        }, ensure_ascii=False)
                        await websocket.send(reply)
                except Exception:
                    continue
        except Exception as e:
            self._q.put({'_error': str(e)})
        finally:
            with self._clients_lock:
                self._clients.pop(websocket, None)
                if self._last_websocket is websocket:
                    self._last_websocket = None

    def _run(self) -> None:
        try:
            import websockets  # type: ignore
        except Exception as e:
            self._q.put({'_error': f'websockets not installed: {e}'})
            return

        async def main() -> None:
            async with websockets.serve(self._handler, self.host, self.port):
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)

        try:
            asyncio.run(main())
        except Exception as e:
            if not self._stop.is_set():
                self._q.put({'_error': str(e)})

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
                        return out
                continue

            self._last_msg_ts = time.time()
            out = self._drain_msg(msg)
            if out is not None:
                return out

        raise StopIteration
