from __future__ import annotations

import json
import threading
import time
import urllib.request
from queue import Empty, Full, Queue
from typing import Any, Dict, List


def normalize_effective_action_source(source: str) -> str:
    s = str(source or '').strip().lower()
    if s in {'l1', 'layer1'}:
        return 'l1'
    if s in {'l2', 'layer2'}:
        return 'l2'
    if s in {'rule_parking_guard', 'rule_guard', 'guardrail'}:
        return 'rule_guard'
    return 'unknown'


def call_agent_decision(agent_url: str, sensor_payload: Dict[str, float], timeout_sec: float = 12.0, request_id: str = '') -> Dict[str, object]:
    url = agent_url.rstrip('/') + '/sensor/decision'
    t0 = time.time()
    print(json.dumps({
        'layer': 'L2_CALL',
        'event': 'request_start',
        'request_id': request_id,
        'url': url,
        't_sec': sensor_payload.get('timestamp'),
        'timeout_sec': timeout_sec,
    }, ensure_ascii=False), flush=True)

    req = urllib.request.Request(
        url=url,
        data=json.dumps(sensor_payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'X-Request-Id': request_id},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
            elapsed_ms = int((time.time() - t0) * 1000)
            print(json.dumps({
                'layer': 'L2_CALL',
                'event': 'response_ok',
                'request_id': request_id,
                'status': resp.status,
                'elapsed_ms': elapsed_ms,
            }, ensure_ascii=False), flush=True)

            data = json.loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError(f'agent response invalid: {data}')
            if 'decision' in data and isinstance(data.get('decision'), dict):
                return data['decision']
            # 兼容直接返回 decision 对象的旧/简化实现
            if 'action' in data:
                return data
            raise RuntimeError(f'agent response invalid: {data}')
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace') if e.fp else ''
        elapsed_ms = int((time.time() - t0) * 1000)
        print(json.dumps({
            'layer': 'L2_CALL',
            'event': 'response_http_error',
            'request_id': request_id,
            'status': e.code,
            'elapsed_ms': elapsed_ms,
            'body': body,
        }, ensure_ascii=False), flush=True)
        raise


class L2AgentWorker:
    def __init__(self, agent_url: str, timeout_sec: float = 5, max_pending: int = 8):
        self.agent_url = agent_url
        self.timeout_sec = timeout_sec
        self._in_q: Queue[Dict[str, Any]] = Queue(maxsize=max(1, int(max_pending)))
        self._out_q: Queue[Dict[str, Any]] = Queue()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._in_q.put_nowait({'_stop': True})
        except Exception:
            pass
        self._th.join(timeout=2.0)

    def submit(self, request: Dict[str, Any]) -> bool:
        try:
            self._in_q.put_nowait(request)
            return True
        except Full:
            return False

    def poll_ready(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while True:
            try:
                out.append(self._out_q.get_nowait())
            except Empty:
                break
        return out

    def _run(self) -> None:
        print(json.dumps({'layer': 'L2_CALL', 'event': 'worker_started', 'agent_url': self.agent_url, 'timeout_sec': self.timeout_sec}, ensure_ascii=False), flush=True)
        while not self._stop.is_set():
            try:
                req = self._in_q.get(timeout=0.2)
            except Empty:
                continue
            if req.get('_stop'):
                break

            request_id = str(req.get('request_id', ''))
            payload = dict(req.get('agent_payload', {}))
            print(json.dumps({'layer': 'L2_CALL', 'event': 'worker_request_dequeued', 'request_id': request_id, 'submitted_t_sec': req.get('submitted_t_sec'), 'monitor_only': bool(req.get('monitor_only', False)), 'queue_size_hint': self._in_q.qsize()}, ensure_ascii=False), flush=True)
            try:
                decision = call_agent_decision(
                    self.agent_url,
                    payload,
                    timeout_sec=self.timeout_sec,
                    request_id=request_id,
                )
                self._out_q.put({
                    'request_id': request_id,
                    'decision': decision,
                    'error': None,
                    'submitted_t_sec': req.get('submitted_t_sec'),
                    'submitted_action': str(payload.get('submitted_action', '') or ''),
                })
                print(json.dumps({'layer': 'L2_CALL', 'event': 'worker_request_done', 'request_id': request_id}, ensure_ascii=False), flush=True)
            except Exception as e:
                print(json.dumps({'layer': 'L2_CALL', 'event': 'worker_request_error', 'request_id': request_id, 'error': str(e)}, ensure_ascii=False), flush=True)
                self._out_q.put({
                    'request_id': request_id,
                    'decision': None,
                    'error': str(e),
                    'submitted_t_sec': req.get('submitted_t_sec'),
                })
