from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Optional


class ControlWebSocketSender:
    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self._q: 'queue.Queue[Dict[str, Any]]' = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait({'_stop': True})
        except Exception:
            pass
        self._thread.join(timeout=2.0)

    def send(self, payload: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(dict(payload))
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import websocket  # type: ignore
        except Exception as e:
            print(json.dumps({'layer': 'CONTROL_WS', 'event': 'sender_error', 'error': f'websocket-client not installed: {e}'}, ensure_ascii=False), flush=True)
            return

        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(self.ws_url, timeout=5)
                print(json.dumps({'layer': 'CONTROL_WS', 'event': 'connected', 'ws_url': self.ws_url}, ensure_ascii=False), flush=True)
                while not self._stop.is_set():
                    try:
                        msg = self._q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if msg.get('_stop'):
                        break
                    try:
                        ws.send(json.dumps(msg, ensure_ascii=False))
                        print(json.dumps({'layer': 'CONTROL_WS', 'event': 'sent', 'payload': msg}, ensure_ascii=False), flush=True)
                    except Exception as e:
                        print(json.dumps({'layer': 'CONTROL_WS', 'event': 'send_error', 'error': str(e)}, ensure_ascii=False), flush=True)
                        break
                try:
                    ws.close()
                except Exception:
                    pass
            except Exception as e:
                print(json.dumps({'layer': 'CONTROL_WS', 'event': 'connect_error', 'error': str(e)}, ensure_ascii=False), flush=True)
                time.sleep(1.0)
