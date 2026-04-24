#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import threading
import time
import urllib.request
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from statistics import mean, pstdev
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from can_decoder import CanDecoder
from closed_loop_sim_generator import ClosedLoopSim
from simulate_sensor_pipeline import encode_frames
from thresholds import load_thresholds, summarize_thresholds
from risk_engine import RiskEngine
from ml_risk_model import MLRiskModel


DT = 0.1


def _normalize_effective_action_source(source: str) -> str:
    s = str(source or '').strip().lower()
    return 'l1' if s == 'l1' else 'l2'


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


SHORT_WINDOW_SEC = _env_float("OELLM_WINDOW_SEC", 5.0)
SHORT_WINDOW_SIZE = max(1, int(round(SHORT_WINDOW_SEC / DT)))
SHORT_DECISION_EVERY = max(1, _env_int("OELLM_DECISION_EVERY_STEPS", int(round(2.0 / DT))))
SLOW_DIAG_EVERY_SEC = max(1, _env_int("OELLM_SLOW_DIAG_EVERY_SEC", 5))
KEEP_RAW_SAMPLES = os.getenv("OELLM_KEEP_RAW_SAMPLES", "1").strip() not in {"0", "false", "False"}


@dataclass
class SlidingWindowAggregator:
    window_sec: float = SHORT_WINDOW_SEC
    step_sec: float = 1.0
    keep_raw_samples: bool = KEEP_RAW_SAMPLES
    _samples: Deque[Dict[str, Any]] = field(default_factory=deque)

    def add_sample(self, timestamp_sec: float, state: Dict[str, Any]) -> None:
        self._samples.append({"timestamp_sec": float(timestamp_sec), "state": dict(state)})
        self._trim(timestamp_sec)

    def _trim(self, current_ts: float) -> None:
        cutoff = current_ts - self.window_sec
        while self._samples and float(self._samples[0]["timestamp_sec"]) < cutoff:
            self._samples.popleft()

    @staticmethod
    def _meaning_rank(meaning: str) -> int:
        if not meaning:
            return 0
        if meaning.startswith("异常"):
            return 4
        if meaning in {"过高", "较高", "较低", "偏高", "偏低", "水位低", "温度过高", "表面温度过高", "甲烷浓度较高", "甲烷浓度过高"}:
            return 3
        if meaning == "正常":
            return 1
        return 2

    @staticmethod
    def _compress_signal_samples(signal_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not signal_samples:
            return {"value": None, "unit": "", "meaning": ""}
        latest = signal_samples[-1]
        meaningful = [s for s in signal_samples if str(s.get("meaning", "")).strip() and str(s.get("meaning", "")) != "正常"]
        if meaningful:
            chosen = max(meaningful, key=lambda x: (SlidingWindowAggregator._meaning_rank(str(x.get("meaning", ""))), float(x.get("timestamp_sec", 0.0))))
        else:
            chosen = latest
        return {
            "value": chosen.get("value"),
            "unit": chosen.get("unit", ""),
            "meaning": chosen.get("meaning", ""),
        }

    def to_window_json(self) -> Dict[str, Any]:
        samples = list(self._samples)
        if not samples:
            return {
                "window_sec": self.window_sec,
                "step_sec": self.step_sec,
                "sample_count": 0,
                "window_start_ts": None,
                "window_end_ts": None,
                "signals": {},
                "samples": [],
            }

        window_start = float(samples[0]["timestamp_sec"])
        window_end = float(samples[-1]["timestamp_sec"])
        all_keys = set()
        for sample in samples:
            state = sample["state"]
            for key, value in state.items():
                if isinstance(value, dict) and "meaning" in value:
                    all_keys.add(key)

        signals: Dict[str, Any] = {}
        for key in sorted(all_keys):
            signal_samples: List[Dict[str, Any]] = []
            for sample in samples:
                value = sample["state"].get(key)
                if isinstance(value, dict):
                    signal_samples.append({"timestamp_sec": sample["timestamp_sec"], **value})
            if signal_samples:
                signals[key] = self._compress_signal_samples(signal_samples)

        raw_samples = []
        if self.keep_raw_samples:
            raw_samples = [{"timestamp_sec": s["timestamp_sec"], "state": s["state"]} for s in samples]

        return {
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "sample_count": len(samples),
            "window_start_ts": window_start,
            "window_end_ts": window_end,
            "signals": signals,
            "samples": raw_samples,
        }

TH = load_thresholds(Path(__file__).resolve().parent)


def hard_safety_check(s: Dict[str, float]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if int(s.get('emergency_stop', 0)) == 1:
        reasons.append('emergency_stop')
    if int(s.get('can_heartbeat_ok', 1)) == 0:
        reasons.append('can_heartbeat_lost')

    methane = float(s.get('methane_pctlel', s.get('methane_pct', 0)) or 0)
    if methane >= TH['methane_stop']:
        reasons.append('methane_over_stop')

    coolant = float(s.get('coolant_temp_c', 0) or 0)
    if coolant >= TH['coolant_stop']:
        reasons.append('coolant_over_stop')

    surface = float(s.get('surface_temp_c', s.get('surface_temp', 0)) or 0)
    if surface >= TH.get('surface_temp_high', 150.0):
        reasons.append('surface_over_stop')

    exhaust = float(s.get('exhaust_temp_c', s.get('exhaust_temp', 0)) or 0)
    if exhaust >= TH['exhaust_stop']:
        reasons.append('exhaust_over_stop')

    intake_p = s.get('intake_pressure_kpa')
    if intake_p is not None and float(intake_p) < TH['intake_pressure_min']:
        reasons.append('intake_pressure_out_of_range')

    water_level = s.get('water_tank_level_pct', s.get('water_level_pct'))
    if water_level is not None and float(water_level) < TH['water_level_alarm_min']:
        reasons.append('water_level_low')

    oil_p = s.get('oil_pressure_kpa')
    if oil_p is not None and float(oil_p) < TH['oil_pressure_min_kpa']:
        reasons.append('oil_pressure_low')

    diesel_level = s.get('diesel_level_cm', s.get('diesel_level'))
    if diesel_level is not None and float(diesel_level) < TH['diesel_level_min']:
        reasons.append('diesel_level_low')

    brake_p = s.get('brake_pressure_bar')
    if brake_p is not None and float(brake_p) < TH['brake_pressure_min']:
        reasons.append('brake_pressure_low_stop')

    walking_p = s.get('walking_pressure_bar', s.get('travel_pressure_bar', s.get('system_pressure_walk_bar')))
    if walking_p is not None and (float(walking_p) < TH['walking_pressure_min'] or float(walking_p) > TH['walking_pressure_max']):
        reasons.append('walking_pressure_over_stop')

    system_p = s.get('system_pressure_bar')
    if system_p is not None and float(system_p) < TH['system_pressure_min']:
        reasons.append('system_pressure_low_stop')

    clamp_p = s.get('clamp_pressure_bar')
    if clamp_p is not None:
        clamp_p = float(clamp_p)
        if clamp_p < TH['clamp_pressure_min'] or clamp_p > TH['clamp_pressure_max']:
            reasons.append('clamp_pressure_over_stop')

    hydraulic_temp = s.get('hydraulic_oil_temp_c')
    if hydraulic_temp is not None and float(hydraulic_temp) > TH['hydraulic_oil_temp_max']:
        reasons.append('hydraulic_oil_temp_over_stop')

    hydraulic_level = s.get('hydraulic_oil_level_pct')
    if hydraulic_level is not None and float(hydraulic_level) < TH['hydraulic_oil_level_min']:
        reasons.append('hydraulic_oil_level_low')

    make_up_oil_p = s.get('make_up_oil_pressure_bar')
    if make_up_oil_p is not None and float(make_up_oil_p) < TH['make_up_oil_pressure_min']:
        reasons.append('make_up_oil_pressure_low_stop')

    rpm = s.get('engine_rpm')
    if rpm is not None and float(rpm) > TH['rpm_max']:
        reasons.append('rpm_over_stop')

    speed = s.get('speed_kmh')
    if speed is not None and float(speed) > TH['speed_max_kmh']:
        reasons.append('speed_over_stop')

    co_ppm = s.get('co_ppm')
    if co_ppm is not None and float(co_ppm) >= TH['co_stop']:
        reasons.append('co_over_stop')

    return ('BRAKE', reasons) if reasons else ('MOVE', [])




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
            if not isinstance(data, dict) or 'decision' not in data:
                raise RuntimeError(f'agent response invalid: {data}')
            decision = data['decision']
            if not isinstance(decision, dict):
                raise RuntimeError(f'agent decision invalid: {decision}')
            return decision
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




class _L2AgentWorker:
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
        while not self._stop.is_set():
            try:
                req = self._in_q.get(timeout=0.2)
            except Empty:
                continue
            if req.get('_stop'):
                break

            request_id = str(req.get('request_id', ''))
            payload = dict(req.get('agent_payload', {}))
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
                })
            except Exception as e:
                self._out_q.put({
                    'request_id': request_id,
                    'decision': None,
                    'error': str(e),
                    'submitted_t_sec': req.get('submitted_t_sec'),
                })


def slow_diagnosis(last5_actions: Deque[str], last5_reasons: Deque[str]) -> str:
    action, cnt = Counter(last5_actions).most_common(1)[0]
    reason = Counter(last5_reasons).most_common(1)[0][0]
    if action == 'BRAKE' and cnt >= 3:
        return f'升级故障: 5秒内BRAKE占比高, 主因={reason}'
    if action == 'DECELERATE' and cnt >= 3:
        return f'维持观察: 5秒内DECELERATE占比高, 主因={reason}'
    if action in {'FORWARD', 'ACCELERATE'} and cnt >= 4:
        return '降级恢复: 状态稳定，允许继续行驶'
    return f'保持当前策略: 主动作={action}, 主因={reason}'


def stream_steps_from_csv(csv_path: Path) -> Iterator[Tuple[int, float, List[Tuple[int, List[int]]]]]:
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cur_step = None
        cur_t = 0.0
        frames: List[Tuple[int, List[int]]] = []

        for row in reader:
            step = int(row['step'])
            t_sec = float(row['t_sec'])

            fid_text = str(row['frame_id_hex']).strip()
            try:
                frame_id = int(fid_text, 16)
            except ValueError:
                # 兼容少量脏数据：提取末尾8位HEX（如“8718F183A0”）
                frame_id = int(fid_text[-8:], 16)

            payload = [int(row[f'byte{i}']) for i in range(8)]

            # 若CSV提供心跳列，则写入ID_181的byte7，供后续set_heartbeat读取
            if frame_id == 0x18F181A0 and 'can_heartbeat_ok' in row and row['can_heartbeat_ok'] != '':
                payload[7] = int(row['can_heartbeat_ok']) & 0xFF

            if cur_step is None:
                cur_step = step
                cur_t = t_sec

            if step != cur_step:
                yield cur_step, cur_t, frames
                cur_step = step
                cur_t = t_sec
                frames = []

            frames.append((frame_id, payload))

        if cur_step is not None:
            yield cur_step, cur_t, frames


def stream_steps_from_socketcan(channel: str, bitrate: int = 250000) -> Iterator[Tuple[int, float, List[Tuple[int, List[int]]]]]:
    try:
        import can  # type: ignore
    except Exception as e:
        raise RuntimeError("实时CAN模式需要安装 python-can") from e

    bus = can.interface.Bus(channel=channel, bustype='socketcan', bitrate=bitrate)
    step = 0
    start = time.time()
    bucket_end = start + DT
    frames: List[Tuple[int, List[int]]] = []

    while True:
        timeout = max(0.0, bucket_end - time.time())
        msg = bus.recv(timeout=timeout)
        now = time.time()
        if msg is not None:
            frames.append((int(msg.arbitration_id), list(msg.data[:8])))

        if now >= bucket_end:
            t_sec = step * DT
            yield step, t_sec, frames
            step += 1
            bucket_end = start + step * DT + DT
            frames = []


class ClosedLoopStepStream:
    """Closed-loop simulator stream driven by actions from run_online()."""

    def __init__(self, duration_sec: float = 120.0, default_action: str = 'FORWARD', realtime: bool = False):
        self.sim = ClosedLoopSim()
        self.total_steps = int(max(1.0, duration_sec) / DT)
        self.current_action = default_action.upper()
        self.realtime = realtime
        self.wall_start = time.time()
        self.step = 0

    def set_action(self, action: str) -> None:
        a = str(action or '').upper()
        if a in {'EMERGENCY_STOP', 'FORWARD', 'REVERSE', 'ACCELERATE', 'DECELERATE', 'BRAKE'}:
            self.current_action = a

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[int, float, List[Tuple[int, List[int]]]]:
        if self.step >= self.total_steps:
            raise StopIteration
        t_sec = self.step * DT
        if self.realtime:
            target = self.wall_start + t_sec
            now = time.time()
            if target > now:
                time.sleep(target - now)

        state = self.sim.step_once(self.current_action)
        # 调试态缓存，便于闭环链路核查 action->state 是否生效
        self.last_state = state
        self.last_speed_kmh = float(getattr(state, 'speed_kmh', 0.0) or 0.0)
        self.last_action_applied = self.current_action
        frames = encode_frames(state)
        out = (self.step, t_sec, frames)
        self.step += 1
        return out


def stream_steps_from_closed_loop(
    duration_sec: float = 120.0,
    default_action: str = 'FORWARD',
    realtime: bool = False,
) -> ClosedLoopStepStream:
    return ClosedLoopStepStream(duration_sec=duration_sec, default_action=default_action, realtime=realtime)


def run_online(
    step_stream: Iterator[Tuple[int, float, List[Tuple[int, List[int]]]]],
    agent_url: str,
    realtime: bool = False,
    l2_every_sec: float = 1.0,
    agent_timeout_sec: float = 12.0,
    run_id: Optional[str] = None,
    run_event_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    print(json.dumps({
        'layer': 'PIPELINE',
        'event': 'start',
        'realtime': realtime,
        'l2_every_sec': l2_every_sec,
        'agent_timeout_sec': agent_timeout_sec,
        'window_sec': SHORT_WINDOW_SEC,
        'window_size': SHORT_WINDOW_SIZE,
        'decision_every_steps': SHORT_DECISION_EVERY,
        'slow_diag_every_sec': SLOW_DIAG_EVERY_SEC,
        'keep_raw_samples': KEEP_RAW_SAMPLES,
        'agent_url': agent_url,
        'thresholds': summarize_thresholds(TH),
    }, ensure_ascii=False), flush=True)

    decoder = CanDecoder()
    risk_engine = RiskEngine(TH)
    ml_model = MLRiskModel()
    run_started_at = time.time()
    l2_decisions = 0
    l2_failures = 0
    l1_brake_steps = 0
    window3s: Deque[Dict[str, Any]] = deque(maxlen=SHORT_WINDOW_SIZE)
    actions_1hz: Deque[str] = deque(maxlen=SLOW_DIAG_EVERY_SEC)
    reasons_1hz: Deque[str] = deque(maxlen=SLOW_DIAG_EVERY_SEC)
    pending_feedback: Optional[Dict[str, Any]] = None
    last_feedback: Dict[str, Any] = {
        'action': '',
        'issued_t_sec': None,
        'evaluated_t_sec': None,
        'baseline_speed_kmh': None,
        'current_speed_kmh': None,
        'success': None,
        'detail': 'no_feedback_yet',
    }

    start_wall = time.time()
    l2_every_steps = max(1, int(round(l2_every_sec / DT)))
    worker = _L2AgentWorker(agent_url=agent_url, timeout_sec=agent_timeout_sec, max_pending=8)
    worker.start()
    pending_by_request: Dict[str, Dict[str, Any]] = {}
    last_effective_action = 'FORWARD'

    try:
        for step, t_sec, frames in step_stream:
            if realtime:
                target = start_wall + t_sec
                now = time.time()
                if target > now:
                    time.sleep(target - now)

            for fid, payload in frames:
                decoder.update(fid, payload)
            if frames:
                decoder.set_heartbeat(1)

            latest_state = decoder.state.to_dict()
            latest_state['t_sec'] = t_sec
            status_report = decoder.build_status_report()
            latest_state['status_report'] = status_report
            latest_state['signals'] = status_report.get('signals', {})

            risk_eval = risk_engine.evaluate(
                current=latest_state,
                window={'signals': latest_state.get('signals', {})},
                history={},
            )
            latest_state['warning_tag'] = str(risk_eval.get('warning_tag', ''))
            latest_state['risk_level'] = str(risk_eval.get('risk_level', 'normal'))
            latest_state['risk_score'] = float(risk_eval.get('risk_score', 0.0) or 0.0)
            latest_state['deviation_score'] = float(risk_eval.get('deviation_score', 0.0) or 0.0)
            latest_state['risk_suggested_actions'] = list(risk_eval.get('suggested_actions', []) or [])
            latest_state['risk_action_reason'] = str(risk_eval.get('action_reason', ''))

            ml_risk_score = float(ml_model.score(latest_state))
            latest_state['ml_risk_score'] = ml_risk_score
            latest_state['ml_warning_level'] = ml_model.level_from_score(ml_risk_score)
            latest_state['ml_model_version'] = ml_model.version

            if pending_feedback is not None:
                issued_t = float(pending_feedback.get('issued_t_sec', t_sec) or t_sec)
                if (t_sec - issued_t) >= l2_every_sec:
                    action = str(pending_feedback.get('action', '') or '').upper()
                    base_speed = float(pending_feedback.get('baseline_speed_kmh', 0.0) or 0.0)
                    cur_speed = float(latest_state.get('speed_kmh', 0.0) or 0.0)
                    success = True
                    detail = 'ok'
                    if action == 'DECELERATE':
                        if base_speed < 0.2:
                            detail = 'not_applicable_zero_speed'
                        else:
                            success = cur_speed <= (base_speed - 0.2)
                            detail = 'speed_not_down' if not success else 'speed_down_ok'
                    elif action == 'ACCELERATE':
                        if base_speed < 0.2:
                            detail = 'not_applicable_zero_speed'
                        else:
                            success = cur_speed >= (base_speed + 0.2)
                            detail = 'speed_not_up' if not success else 'speed_up_ok'
                    elif action == 'BRAKE':
                        success = cur_speed <= max(0.5, base_speed * 0.5)
                        detail = 'brake_effect_weak' if not success else 'brake_effect_ok'
                    elif action == 'EMERGENCY_STOP':
                        success = cur_speed <= 0.5
                        detail = 'not_stopped' if not success else 'stopped_ok'
                    elif action in {'FORWARD', 'REVERSE'}:
                        detail = 'command_issued'

                    last_feedback = {
                        'action': action,
                        'issued_t_sec': issued_t,
                        'evaluated_t_sec': t_sec,
                        'baseline_speed_kmh': round(base_speed, 3),
                        'current_speed_kmh': round(cur_speed, 3),
                        'success': bool(success),
                        'detail': detail,
                    }
                    pending_feedback = None

            l1_action, l1_reasons = hard_safety_check(latest_state)
            latest_state['layer1_action'] = l1_action
            latest_state['layer1_reasons'] = l1_reasons

            if l1_action != 'MOVE':
                l1_brake_steps += 1
                print(json.dumps({'layer': 'L1_10Hz', 't_sec': t_sec, 'action': l1_action, 'reasons': l1_reasons}, ensure_ascii=False), flush=True)

            if run_event_cb is not None:
                try:
                    run_event_cb({
                        'event': 'l1_tick',
                        'ts': time.time(),
                        'run_id': run_id,
                        't_sec': t_sec,
                        'action': l1_action,
                        'reasons': l1_reasons,
                        'sensor_snapshot': dict(latest_state),
                    })
                except Exception:
                    pass

            window3s.append({
                'timestamp_sec': t_sec,
                'signals': dict(latest_state.get('signals', {})),
                'layer1_action': l1_action,
                'layer1_reasons': list(l1_reasons),
            })

            if step > 0 and step % l2_every_steps == 0 and len(window3s) >= SHORT_WINDOW_SIZE:
                window_agg = SlidingWindowAggregator(window_sec=SHORT_WINDOW_SEC, step_sec=DT, keep_raw_samples=False)
                for item in window3s:
                    window_agg.add_sample(float(item['timestamp_sec']), item['signals'])
                window_json = window_agg.to_window_json()

                history_state = {
                    'last_hard_action': latest_state.get('layer1_action', 'FORWARD'),
                    'stable_after_break_sec': float(sum(1 for item in reversed(window3s) if item.get('layer1_action') == 'MOVE')) * DT if latest_state.get('layer1_action') == 'MOVE' else 0.0,
                    'break_age_sec': float(sum(1 for item in reversed(window3s) if item.get('layer1_action') == 'BRAKE')) * DT if latest_state.get('layer1_action') == 'BRAKE' else 0.0,
                    'current_speed_kmh': float(latest_state.get('speed_kmh', 0.0) or 0.0),
                    'recent_actions': list(actions_1hz)[-3:],
                    'recent_reasons': list(reasons_1hz)[-3:],
                    'last_action_feedback': dict(last_feedback),
                }

                # 旧版payload（保留注释，不删除）
                # agent_payload = {
                #     'vehicle_id': 'sim-truck-01',
                #     'timestamp': f'{t_sec:.1f}',
                #     'current': latest_state,
                #     'window': window_json,
                #     'history': history_state,
                # }

                # 新版精简payload：仅 features.risk + features.ml + history
                # 窗口信息只用于构造当前 risk，不再直接透传给 agent。
                active_warnings: List[Dict[str, Any]] = []
                for sig_key, sig_val in (window_json.get('signals', {}) or {}).items():
                    if not isinstance(sig_val, dict):
                        continue
                    meaning = str(sig_val.get('meaning', '') or '').strip()
                    if (not meaning) or meaning == '正常':
                        continue

                    # 避免把窗口观测误写成“当前预警”，尤其 speed_kmh 容易被误读为阈值越限
                    tag = f'{sig_key}_warning'
                    if sig_key == 'speed_kmh':
                        tag = 'speed_kmh_observed'

                    active_warnings.append({
                        'tag': tag,
                        'value': sig_val.get('value'),
                        'unit': sig_val.get('unit', ''),
                        'meaning': meaning,
                        'source': f'window.signals.{sig_key}',
                    })

                warning_tag = str(latest_state.get('warning_tag', '') or '').strip()
                if warning_tag and all(str(w.get('tag', '')) != warning_tag for w in active_warnings):
                    active_warnings.insert(0, {
                        'tag': warning_tag,
                        'value': latest_state.get('risk_score', 0.0),
                        'unit': 'risk_score',
                        'meaning': str(latest_state.get('risk_level', 'normal') or 'normal'),
                        'source': 'risk_engine',
                    })

                future_warnings: List[Dict[str, Any]] = []
                ml_level = str(latest_state.get('ml_warning_level', 'normal') or 'normal')
                ml_score = float(latest_state.get('ml_risk_score', 0.0) or 0.0)

                # 占位版前瞻：还没有训练好的30秒预测模型时，先用规则把“未来可能出现的风险”显式透出。
                # 逻辑：当前已经在 warning/danger 的信号，默认视为未来30秒仍需关注；
                # 同时基于当前风险分/ML分给出概率式占位，不做真实预测值。
                future_tags = []
                for w in active_warnings:
                    tag = str(w.get('tag', '') or '').strip()
                    if tag and tag not in future_tags:
                        future_tags.append(tag)

                if not future_tags:
                    if ml_level in {'warning', 'danger'}:
                        future_tags = [warning_tag or 'future_risk_placeholder']
                    elif ml_score >= 0.35:
                        future_tags = ['future_risk_placeholder']

                for tag in future_tags[:6]:
                    future_warnings.append({
                        'tag': tag,
                        'value': None,
                        'unit': '',
                        'eta_sec': 30,
                        'source': 'ml_placeholder',
                        'confidence': round(min(0.95, 0.15 + ml_score * 0.8 + (0.15 if ml_level in {'warning', 'danger'} else 0.0)), 3),
                        'meaning': '占位版前瞻：待训练模型输出真实未来值',
                    })

                agent_payload = {
                    'vehicle_id': 'sim-truck-01',
                    'timestamp': f'{t_sec:.1f}',
                    'features': {
                        'risk': {
                            'level': str(latest_state.get('risk_level', 'normal') or 'normal'),
                            'score': float(latest_state.get('risk_score', 0.0) or 0.0),
                            'warning_tag': warning_tag,
                            'suggested_actions': list(latest_state.get('risk_suggested_actions', []) or []),
                            'active_warnings': active_warnings,
                        },
                        'ml': {
                            'horizon_sec': 30,
                            'score': ml_score,
                            'level': ml_level,
                            'model_version': str(latest_state.get('ml_model_version', 'none') or 'none'),
                            'future_warnings': future_warnings,
                        },
                    },
                    'history': history_state,
                }

                request_id = uuid.uuid4().hex[:8]
                enqueued = worker.submit({
                    'request_id': request_id,
                    'agent_payload': agent_payload,
                    'submitted_t_sec': t_sec,
                })
                if enqueued:
                    pending_by_request[request_id] = {
                        't_sec': t_sec,
                        'latest_state': dict(latest_state),
                        'l1_action': l1_action,
                        'window_json': window_json,
                    }
                else:
                    l2_failures += 1
                    print(json.dumps({
                        'layer': 'L2_CALL',
                        'event': 'agent_worker_queue_full',
                        'request_id': request_id,
                        'fallback_action': 'DECELERATE',
                    }, ensure_ascii=False), flush=True)
                    if hasattr(step_stream, 'set_action'):
                        try:
                            step_stream.set_action('DECELERATE')
                            last_effective_action = 'DECELERATE'
                        except Exception:
                            pass

            for res in worker.poll_ready():
                request_id = str(res.get('request_id', ''))
                pending = pending_by_request.pop(request_id, None)
                if pending is None:
                    continue

                snap_state = dict(pending.get('latest_state', {}))
                l1_action_at_submit = str(pending.get('l1_action', 'MOVE'))
                window_json = pending.get('window_json', {})

                if res.get('error'):
                    l2_failures += 1
                    err_text = str(res.get('error', ''))
                    print(json.dumps({'layer': 'L2_CALL', 'event': 'agent_call_failed', 'request_id': request_id, 'error': err_text, 'fallback_action': 'DECELERATE'}, ensure_ascii=False), flush=True)
                    fallback_reason = 'agent超时回退减速' if ('timed out' in err_text.lower() or 'timeout' in err_text.lower()) else 'agent调用失败回退减速'
                    decision = {'action': 'DECELERATE', 'reason': fallback_reason, 'confidence': 0.3}
                else:
                    decision = dict(res.get('decision', {}) or {})

                l2_action = str(decision.get('action', 'DECELERATE')).upper()
                effective_action = l2_action
                effective_action_source = 'l2'
                if l1_action_at_submit == 'BRAKE' and effective_action != 'BRAKE':
                    effective_action = 'BRAKE'
                    effective_action_source = 'l1'
                effective_action_source = _normalize_effective_action_source(effective_action_source)

                if hasattr(step_stream, 'set_action'):
                    try:
                        step_stream.set_action(effective_action)
                        last_effective_action = effective_action
                    except Exception:
                        pass

                if hasattr(step_stream, 'last_speed_kmh'):
                    print(json.dumps({
                        'layer': 'CLOSED_LOOP_TRACE',
                        'event': 'action_feedback',
                        'request_id': request_id,
                        'l2_action': l2_action,
                        'effective_action': effective_action,
                        'effective_action_source': effective_action_source,
                        'stream_action_applied': getattr(step_stream, 'last_action_applied', ''),
                        'stream_last_speed_kmh': round(float(getattr(step_stream, 'last_speed_kmh', 0.0) or 0.0), 3),
                        'decoded_speed_kmh': round(float(snap_state.get('speed_kmh', 0.0) or 0.0), 3),
                        'warning_tag': snap_state.get('warning_tag', ''),
                        'risk_level': snap_state.get('risk_level', 'normal'),
                    }, ensure_ascii=False), flush=True)

                l2_reason = str(decision.get('reason', ''))
                l2_conf = float(decision.get('confidence', 0.5))
                print(json.dumps({
                    'layer': 'L2_CALL',
                    'event': 'decision_parsed',
                    'request_id': request_id,
                    'action': l2_action,
                    'effective_action': effective_action,
                    'effective_action_source': effective_action_source,
                    'risk_level': snap_state.get('risk_level', 'normal'),
                    'warning_tag': snap_state.get('warning_tag', ''),
                    'ml_risk_score': snap_state.get('ml_risk_score', 0.0),
                    'ml_warning_level': snap_state.get('ml_warning_level', 'normal'),
                    'ml_model_version': snap_state.get('ml_model_version', 'none'),
                    'reason': l2_reason,
                    'confidence': round(l2_conf, 3),
                }, ensure_ascii=False), flush=True)

                l2_decisions += 1
                out_l2 = {
                    'layer': 'L2_1Hz',
                    't_sec': pending.get('t_sec', snap_state.get('t_sec', 0.0)),
                    'action': l2_action,
                    'effective_action': effective_action,
                    'effective_action_source': effective_action_source,
                    'risk_level': snap_state.get('risk_level', 'normal'),
                    'warning_tag': snap_state.get('warning_tag', ''),
                    'risk_score': snap_state.get('risk_score', 0.0),
                    'ml_risk_score': snap_state.get('ml_risk_score', 0.0),
                    'ml_warning_level': snap_state.get('ml_warning_level', 'normal'),
                    'ml_model_version': snap_state.get('ml_model_version', 'none'),
                    'reason': l2_reason,
                    'confidence': round(l2_conf, 3),
                    'window': window_json,
                }
                print(json.dumps(out_l2, ensure_ascii=False), flush=True)

                actions_1hz.append(effective_action)
                reasons_1hz.append(l2_reason)
                pending_feedback = {
                    'action': effective_action,
                    'issued_t_sec': float(pending.get('t_sec', 0.0) or 0.0),
                    'baseline_speed_kmh': float(snap_state.get('speed_kmh', 0.0) or 0.0),
                }
                print(json.dumps({'layer': 'L2_FEEDBACK', 'event': 'feedback_state', 'last_feedback': last_feedback, 'pending_feedback': pending_feedback}, ensure_ascii=False), flush=True)

                if run_event_cb is not None:
                    try:
                        run_event_cb({
                            'event': 'l2_tick',
                            'ts': time.time(),
                            'run_id': run_id,
                            'request_id': request_id,
                            't_sec': pending.get('t_sec', snap_state.get('t_sec', 0.0)),
                            'l2_action': l2_action,
                            'effective_action': effective_action,
                            'effective_action_source': effective_action_source,
                            'risk_level': snap_state.get('risk_level', 'normal'),
                            'warning_tag': snap_state.get('warning_tag', ''),
                            'risk_score': snap_state.get('risk_score', 0.0),
                            'risk_action_reason': snap_state.get('risk_action_reason', ''),
                            'ml_risk_score': snap_state.get('ml_risk_score', 0.0),
                            'ml_warning_level': snap_state.get('ml_warning_level', 'normal'),
                            'ml_model_version': snap_state.get('ml_model_version', 'none'),
                            'l2_reason': l2_reason,
                            'l2_confidence': round(l2_conf, 3),
                            'l1_action': l1_action_at_submit,
                            'l1_reasons': snap_state.get('layer1_reasons', []),
                            'feedback': dict(last_feedback),
                        })
                    except Exception:
                        pass

        if hasattr(step_stream, 'set_action'):
            try:
                step_stream.set_action(last_effective_action)
            except Exception:
                pass
    finally:
        worker.stop()

    elapsed_sec = round(time.time() - run_started_at, 3)
    summary = {
        'run_id': run_id or f'run-{int(time.time())}',
        'elapsed_sec': elapsed_sec,
        'l2_decisions': l2_decisions,
        'l2_failures': l2_failures,
        'l1_brake_steps': l1_brake_steps,
        'l1_brake_ratio': round(l1_brake_steps / max(1, step + 1), 4) if 'step' in locals() else 0.0,
        'last_feedback': last_feedback,
    }
    print(json.dumps({'layer': 'RUN_SUMMARY', **summary}, ensure_ascii=False), flush=True)
    return summary


def _init_run_log(base_dir: Path, run_id: str) -> Path:
    logs_dir = (base_dir / 'logs').resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f'run_history_{run_id}.jsonl'
    path.write_text('', encoding='utf-8')
    return path


def _append_run_event(log_path: Path, event: Dict[str, Any]) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    return log_path


def _save_run_record(log_path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> Path:
    rec = {
        'event': 'run_completed',
        'ts': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'source': args.source,
        'agent_url': args.agent_url,
        'l2_every_sec': args.l2_every_sec,
        'window_sec': SHORT_WINDOW_SEC,
        'run': summary,
    }
    return _append_run_event(log_path, rec)


def main() -> None:
    parser = argparse.ArgumentParser(description='Online 3-layer pipeline (single chain for csv/socketcan/closed_loop).')
    # 默认使用闭环仿真，避免误读旧CSV导致“起步即高速/大量报警”
    parser.add_argument('--source', choices=['csv', 'socketcan', 'closed_loop'], default='closed_loop')
    parser.add_argument('--source-csv', default='/mnt/ssd/Agent/oellm_agent/sim_data/sim_can_frames_10min_10hz.csv')
    parser.add_argument('--can-channel', default='can0')
    parser.add_argument('--can-bitrate', type=int, default=250000)
    # 默认时长改短：便于快速观察前30~120秒起步与报警变化
    parser.add_argument('--sim-duration-sec', type=float, default=300, help='closed_loop source duration (sec)')
    parser.add_argument('--sim-default-action', default='FORWARD', help='closed_loop default action when agent call fails')
    parser.add_argument('--realtime', action='store_true', help='CSV/closed_loop回放按0.1秒真实节奏跑')
    parser.add_argument('--agent-url', default='http://127.0.0.1:18080', help='agent HTTP base url')
    # 默认1秒调用一次L2，便于观察起步阶段动作切换
    parser.add_argument('--l2-every-sec', type=float, default=2.0, help='L2调用agent周期(秒)，默认1.0，建议1.0~2.0')
    parser.add_argument('--agent-timeout-sec', type=float, default=5.0, help='agent HTTP调用超时(秒)')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{args.source}"
    log_path = _init_run_log(base_dir, run_id)

    # 启动即落一条，确保中断也有记录
    _append_run_event(log_path, {
        'event': 'run_started',
        'ts': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        'run_id': run_id,
        'source': args.source,
        'agent_url': args.agent_url,
        'l2_every_sec': args.l2_every_sec,
        'window_sec': SHORT_WINDOW_SEC,
    })

    def _run_event_cb(evt: Dict[str, Any]) -> None:
        _append_run_event(log_path, evt)

    summary: Dict[str, Any]
    try:
        if args.source == 'csv':
            stream = stream_steps_from_csv(Path(args.source_csv))
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                realtime=args.realtime,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=_run_event_cb,
            )
        elif args.source == 'socketcan':
            stream = stream_steps_from_socketcan(args.can_channel, args.can_bitrate)
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                realtime=False,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=_run_event_cb,
            )
        else:
            stream = stream_steps_from_closed_loop(
                duration_sec=args.sim_duration_sec,
                default_action=args.sim_default_action,
                realtime=args.realtime,
            )
            summary = run_online(
                stream,
                agent_url=args.agent_url,
                realtime=args.realtime,
                l2_every_sec=args.l2_every_sec,
                agent_timeout_sec=args.agent_timeout_sec,
                run_id=run_id,
                run_event_cb=_run_event_cb,
            )

        rec_path = _save_run_record(log_path, summary, args)
        print(json.dumps({'layer': 'RUN_RECORD', 'path': str(rec_path), 'run_id': summary.get('run_id')}, ensure_ascii=False), flush=True)
    except Exception as e:
        _append_run_event(log_path, {
            'event': 'run_crashed',
            'ts': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'run_id': run_id,
            'error': str(e),
        })
        raise


if __name__ == '__main__':
    main()
