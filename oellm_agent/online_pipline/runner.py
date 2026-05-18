from __future__ import annotations

import csv
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from oellm_agent.can.can_decoder import CanDecoder
from oellm_agent.can.control_frame_encoder import decision_to_control_frame
from oellm_agent.can.signal_processor import SignalProcessor
from oellm_agent.config.thresholds.thresholds import load_thresholds, normalize_model, summarize_thresholds
from oellm_agent.risk_engine import RiskEngine

from .agent_client import L2AgentWorker, normalize_effective_action_source
from .config import KEEP_RAW_SAMPLES, SHORT_WINDOW_SEC, SHORT_WINDOW_SIZE, SLOW_DIAG_EVERY_SEC
from .safety import hard_safety_check
from .windowing import SlidingWindowAggregator


def run_online(
    step_stream: Iterator[Tuple[int, float, List[Tuple[int, List[int]]]]],
    agent_url: str,
    model: str = '50',
    realtime: bool = True,
    l2_every_sec: float = 2.0,
    agent_timeout_sec: float = 5.0,
    run_id: Optional[str] = None,
    run_event_cb: Optional[Any] = None,
    control_sender: Optional[Any] = None,
) -> Dict[str, Any]:
    model = normalize_model(model)
    th = load_thresholds(Path(__file__).resolve().parents[1], model=model)
    window_group_count = SHORT_WINDOW_SIZE
    # adaptive timing: decision is time-driven rather than step-driven
    l2_every_steps = 1

    print(json.dumps({
        'layer': 'PIPELINE',
        'event': 'start',
        'model': model,
        'realtime': realtime,
        'l2_every_sec': l2_every_sec,
        'agent_timeout_sec': agent_timeout_sec,
        'group_period_sec': 'dynamic',
        'window_sec': SHORT_WINDOW_SEC,
        'window_groups': window_group_count,
        'decision_every_groups': l2_every_steps,
        'slow_diag_every_sec': SLOW_DIAG_EVERY_SEC,
        'keep_raw_samples': KEEP_RAW_SAMPLES,
        'agent_url': agent_url,
        'thresholds': summarize_thresholds(th),
    }, ensure_ascii=False), flush=True)

    decoder = CanDecoder(model=model)
    signal_processor = SignalProcessor(model=model, thresholds=th)
    risk_engine = RiskEngine(th)
    run_started_at = time.time()
    l2_decisions = 0
    l2_failures = 0
    l1_brake_steps = 0
    action_rewrite_counter: Dict[str, int] = {}
    action_source_counter: Dict[str, int] = {}
    window3s: Deque[Dict[str, Any]] = deque(maxlen=window_group_count)
    actions_1hz: Deque[str] = deque(maxlen=SLOW_DIAG_EVERY_SEC)
    reasons_1hz: Deque[str] = deque(maxlen=SLOW_DIAG_EVERY_SEC)
    action_events: Deque[Dict[str, Any]] = deque(maxlen=20)
    current_action_started_t_sec = 0.0
    low_speed_started_t_sec: Optional[float] = None
    warning_started_at: Dict[str, float] = {}
    drive_context_active_started_t_sec: Optional[float] = None
    agent_control_enabled = False
    agent_control_enabled_since_t_sec: Optional[float] = None
    drive_gear_grace_sec = float(th.get('decision_drive_gear_grace_sec', 15.0))
    drive_effective_hold_sec = float(th.get('decision_drive_effective_hold_sec', 3.0))
    drive_starting_rpm_max = float(th.get('decision_drive_starting_rpm_max', 1300.0))
    drive_effective_rpm_min = float(th.get('decision_drive_effective_rpm_min', 1400.0))
    drive_idle_context = False
    l1_latched = False
    l1_latched_reasons: List[str] = []
    l1_latched_since_t_sec: Optional[float] = None
    l1_recovery_clear_started_t_sec: Optional[float] = None
    pending_feedback: Optional[Dict[str, Any]] = None
    last_feedback: Dict[str, Any] = {
        'action': '',
        'issued_t_sec': None,
        'evaluated_t_sec': None,
        'baseline_speed_mps': None,
        'current_speed_mps': None,
        'success': None,
        'detail': 'no_feedback_yet',
    }

    start_wall = time.time()
    decisions_csv_path = Path(__file__).resolve().parents[1] / 'logs' / 'agent_decisions.csv'
    decisions_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fieldnames = [
        'schema_version', 'run_id', 'request_id', 'step', 't_sec', 'wall_ts',
        'speed_mps', 'engine_rpm', 'gear_state', 'vehicle_context',
        'travel_pressure_bar', 'brake_pressure_bar', 'system_pressure_bar',
        'overall_risk_level', 'overall_risk_score', 'warning_tags_json',
        'speed_delta_5s_mps', 'speed_trend_5s', 'engine_rpm_delta_5s', 'engine_rpm_trend_5s',
        'speed_low_duration_sec', 'same_warning_duration_sec',
        'l1_action', 'l1_reasons_json', 'l2_action', 'effective_action', 'effective_action_source', 'rewritten',
        'feedback_success', 'feedback_detail', 'baseline_speed_mps', 'current_speed_mps', 'speed_delta_mps',
    ]
    if not decisions_csv_path.exists() or decisions_csv_path.stat().st_size == 0:
        with decisions_csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

    worker = L2AgentWorker(agent_url=agent_url, timeout_sec=agent_timeout_sec, max_pending=8)
    worker.start()
    pending_by_request: Dict[str, Dict[str, Any]] = {}
    last_final_action = 'HOLD'
    current_action_started_t_sec = 0.0
    low_speed_started_t_sec: Optional[float] = None
    warning_started_at: Dict[str, float] = {}
    last_decision_wall_ts: float = 0.0
    last_effective_action = 'HOLD'

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

            raw_state = decoder.state.to_dict()
            latest_state, status_report = signal_processor.process(raw_state)
            latest_state['t_sec'] = t_sec
            latest_state['status_report'] = status_report
            latest_state['signals'] = status_report.get('signals', {})

            gear_state = int(latest_state.get('gear_state', 0) or 0)
            gear_state_byte1 = int(latest_state.get('gear_state_byte1', gear_state) or gear_state)
            latest_state['gear_state_byte1'] = gear_state_byte1
            drive_mode_active = gear_state in {3, 4}
            drive_gear_age_sec = (t_sec - agent_control_enabled_since_t_sec) if agent_control_enabled_since_t_sec is not None else 0.0
            engine_rpm_for_context = float(latest_state.get('engine_rpm', 0.0) or 0.0)
            speed_for_context = float(latest_state.get('speed_mps', 0.0) or 0.0)
            in_drive_gear_grace = drive_mode_active and drive_gear_age_sec < drive_gear_grace_sec
            drive_starting_context = drive_mode_active and speed_for_context < 0.3 and engine_rpm_for_context < 1000.0
            drive_effective_context = drive_mode_active and (not (drive_gear_age_sec < drive_gear_grace_sec or drive_starting_context))
            if drive_effective_context:
                if drive_context_active_started_t_sec is None:
                    drive_context_active_started_t_sec = t_sec
            else:
                drive_context_active_started_t_sec = None
            drive_effective_duration_sec = (t_sec - drive_context_active_started_t_sec) if drive_context_active_started_t_sec is not None else 0.0
            active_motion_context = drive_effective_context and drive_effective_duration_sec >= drive_effective_hold_sec
            drive_idle_context = drive_mode_active and not in_drive_gear_grace and not drive_starting_context and not active_motion_context
            control_state_enabled = gear_state_byte1 in {3, 4}
            control_entered_by_byte2 = control_state_enabled
            l1_alarm = bool(latest_state.get('alarm_code', 0) or latest_state.get('emergency_stop', 0))
            if gear_state in {3, 4} and not agent_control_enabled:
                agent_control_enabled = True
                agent_control_enabled_since_t_sec = t_sec
                print(json.dumps({
                    'layer': 'PIPELINE',
                    'event': 'agent_control_enabled_by_gear',
                    't_sec': t_sec,
                    'gear_state': gear_state,
                }, ensure_ascii=False), flush=True)
            elif gear_state not in {3, 4} and agent_control_enabled:
                agent_control_enabled = False
                agent_control_enabled_since_t_sec = None
                drive_context_active_started_t_sec = None
                print(json.dumps({
                    'layer': 'PIPELINE',
                    'event': 'agent_control_disabled_by_gear',
                    't_sec': t_sec,
                    'gear_state': gear_state,
                }, ensure_ascii=False), flush=True)
                if hasattr(step_stream, 'set_action'):
                    try:
                        step_stream.set_action('HOLD')
                    except Exception:
                        pass
                last_final_action = 'HOLD'
            elif gear_state not in {3, 4} and not agent_control_enabled:
                print(json.dumps({
                    'layer': 'PIPELINE',
                    'event': 'agent_control_not_enabled_by_gear',
                    't_sec': t_sec,
                    'gear_state': gear_state,
                    'gear_state_byte1': gear_state_byte1,
                    'control_state_enabled': bool(control_state_enabled),
                    'vehicle_context': 'parking' if not drive_mode_active else 'starting' if (in_drive_gear_grace or drive_idle_context or drive_starting_context) else 'control',
                }, ensure_ascii=False), flush=True)

            latest_state['drive_gear_age_sec'] = drive_gear_age_sec
            latest_state['control_state'] = {
                'enabled': bool(control_state_enabled),
                'entered_by_byte2': bool(control_entered_by_byte2),
                'l1_alarm': bool(l1_alarm),
            }
            latest_state['drive_gear_age_sec'] = drive_gear_age_sec
            engine_rpm_for_context = float(latest_state.get('engine_rpm', 0.0) or 0.0)
            speed_for_context = float(latest_state.get('speed_mps', 0.0) or 0.0)
            in_drive_gear_grace = drive_mode_active and drive_gear_age_sec < drive_gear_grace_sec
            drive_starting_context = drive_mode_active and speed_for_context < 0.3 and engine_rpm_for_context < 1000.0
            drive_effective_context = drive_mode_active and (not (drive_gear_age_sec < drive_gear_grace_sec or drive_starting_context))
            if drive_effective_context:
                if drive_context_active_started_t_sec is None:
                    drive_context_active_started_t_sec = t_sec
            else:
                drive_context_active_started_t_sec = None
            drive_effective_duration_sec = (t_sec - drive_context_active_started_t_sec) if drive_context_active_started_t_sec is not None else 0.0
            active_motion_context = drive_effective_context and drive_effective_duration_sec >= drive_effective_hold_sec
            drive_idle_context = drive_mode_active and not in_drive_gear_grace and not drive_starting_context and not active_motion_context
            latest_state['vehicle_context'] = 'parking' if not drive_mode_active else 'starting' if (in_drive_gear_grace or drive_idle_context or drive_starting_context) else 'control'
            latest_state['drive_effective_duration_sec'] = drive_effective_duration_sec
            latest_state['control_state'] = {
                'enabled': bool(control_state_enabled),
                'entered_by_byte2': bool(control_entered_by_byte2),
                'l1_alarm': bool(l1_alarm),
            }
            non_drive_normal_low_signals = {'speed_mps', 'brake_pressure_bar', 'travel_pressure_bar', 'walking_pressure_bar', 'system_pressure_bar', 'clamp_pressure_bar'}
            sensor_abnormal_reasons: List[str] = []
            for sig_key, sig_val in (latest_state.get('signals', {}) or {}).items():
                if not isinstance(sig_val, dict) or not sig_val.get('valid', True):
                    continue
                meaning = str(sig_val.get('meaning', '') or '').strip()
                if not meaning or meaning == '正常':
                    continue
                if ((not drive_mode_active) or in_drive_gear_grace or drive_idle_context or drive_starting_context) and sig_key in non_drive_normal_low_signals:
                    continue
                sensor_abnormal_reasons.append(f'{sig_key}:{meaning}')

            l1_action_raw, l1_reasons_raw = hard_safety_check(latest_state, th)
            if l1_alarm:
                l1_action_raw = 'EMERGENCY_STOP'
                if 'l1_alarm_force_brake' not in l1_reasons_raw:
                    l1_reasons_raw = list(l1_reasons_raw) + ['l1_alarm_force_brake']
            if (not drive_mode_active) or in_drive_gear_grace or drive_idle_context or drive_starting_context:
                l1_reasons_raw = [r for r in l1_reasons_raw if r not in {'brake_pressure_low_stop', 'walking_pressure_over_stop', 'travel_pressure_low_stop', 'speed_over_stop'}]
                l1_action_raw = 'EMERGENCY_STOP' if l1_reasons_raw else 'HOLD'
            if l1_action_raw != 'HOLD':
                if not l1_latched:
                    l1_latched_since_t_sec = t_sec
                l1_latched = True
                l1_latched_reasons = list(l1_reasons_raw)
                l1_recovery_clear_started_t_sec = None
            elif l1_latched:
                if not sensor_abnormal_reasons:
                    if l1_recovery_clear_started_t_sec is None:
                        l1_recovery_clear_started_t_sec = t_sec
                    if (t_sec - l1_recovery_clear_started_t_sec) >= 60.0:
                        l1_latched = False
                        l1_latched_reasons = []
                        l1_latched_since_t_sec = None
                        l1_recovery_clear_started_t_sec = None
                else:
                    l1_recovery_clear_started_t_sec = None

            l1_action = 'EMERGENCY_STOP' if l1_latched else 'HOLD'
            l1_reasons = list(l1_latched_reasons if l1_latched else [])
            latest_state['layer1_action'] = l1_action
            latest_state['layer1_reasons'] = l1_reasons
            latest_state['layer1_raw_action'] = l1_action_raw
            latest_state['layer1_raw_reasons'] = list(l1_reasons_raw)
            latest_state['layer1_latched'] = bool(l1_latched)
            latest_state['layer1_latched_since_t_sec'] = l1_latched_since_t_sec
            latest_state['sensor_abnormal_reasons'] = list(sensor_abnormal_reasons)
            latest_state['sensors_all_normal'] = not sensor_abnormal_reasons
            latest_state['layer1_recovery_clear_sec'] = (t_sec - l1_recovery_clear_started_t_sec) if l1_recovery_clear_started_t_sec is not None else 0.0

            if l1_action == 'HOLD':
                risk_eval = risk_engine.evaluate(
                    current=latest_state,
                    window={'signals': latest_state.get('signals', {})},
                    history={},
                )
                latest_state['warnings'] = list(risk_eval.get('warnings', []) or [])
                latest_state['overall_risk_level'] = str(risk_eval.get('overall_level', 'normal') or 'normal')
                latest_state['overall_risk_score'] = float(risk_eval.get('overall_score', 0.0) or 0.0)
                latest_state['risk_suggested_actions'] = list(risk_eval.get('suggested_actions', []) or [])
            else:
                latest_state['warnings'] = []
                latest_state['overall_risk_level'] = 'normal'
                latest_state['overall_risk_score'] = 0.0
                latest_state['risk_suggested_actions'] = []

            if pending_feedback is not None:
                issued_t = float(pending_feedback.get('issued_t_sec', t_sec) or t_sec)
                if (t_sec - issued_t) >= l2_every_sec:
                    action = str(pending_feedback.get('action', '') or '').upper()
                    base_speed = float(pending_feedback.get('baseline_speed_mps', 0.0) or 0.0)
                    cur_speed = float(latest_state.get('speed_mps', 0.0) or 0.0)
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
                            success = cur_speed >= (base_speed + 0.08)
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
                        'baseline_speed_mps': round(base_speed, 3),
                        'current_speed_mps': round(cur_speed, 3),
                        'success': bool(success),
                        'detail': detail,
                    }
                    pending_feedback = None

            if l1_action != 'MOVE':
                l1_brake_steps += 1
                if hasattr(step_stream, 'set_action'):
                    try:
                        step_stream.set_action('EMERGENCY_STOP')
                        if last_final_action != 'EMERGENCY_STOP':
                            current_action_started_t_sec = t_sec
                        last_final_action = 'EMERGENCY_STOP'
                    except Exception:
                        pass
                print(json.dumps({'layer': 'L1_GROUP', 't_sec': t_sec, 'action': l1_action, 'reasons': l1_reasons}, ensure_ascii=False), flush=True)

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
                'warnings': list(latest_state.get('warnings', []) or []),
                'layer1_action': l1_action,
                'layer1_reasons': list(l1_reasons),
            })
            # frame-count window: keep last 8 groups
            while len(window3s) > window_group_count:
                window3s.popleft()

            # dual gate: window-full + min decision interval
            now_wall = time.time()
            window_ready = len(window3s) >= window_group_count
            interval_ready = (last_decision_wall_ts <= 0.0) or ((now_wall - last_decision_wall_ts) >= float(l2_every_sec))
            should_decide = window_ready and interval_ready
            if should_decide:
                last_decision_wall_ts = now_wall
                window_agg = SlidingWindowAggregator(window_sec=SHORT_WINDOW_SEC, step_sec=0.0, keep_raw_samples=False)
                for item in window3s:
                    window_agg.add_sample(float(item['timestamp_sec']), item['signals'])
                window_json = window_agg.to_window_json()

                current_speed = float(latest_state.get('speed_mps', 0.0) or 0.0)
                current_engine_rpm = float(latest_state.get('engine_rpm', 0.0) or 0.0)
                current_travel_pressure = float(latest_state.get('travel_pressure_bar', 0.0) or 0.0)
                current_angle_deg = float(latest_state.get('angle_deg', 0.0) or 0.0)
                current_payload_tons = float(latest_state.get('payload_tons', 0.0) or 0.0)
                speed_samples = [float(((item.get('signals', {}) or {}).get('speed_mps', {}) or {}).get('value', 0.0) or 0.0) for item in window3s]
                rpm_samples = [float(((item.get('signals', {}) or {}).get('engine_rpm', {}) or {}).get('value', 0.0) or 0.0) for item in window3s]
                speed_delta = round(speed_samples[-1] - speed_samples[0], 3) if len(speed_samples) >= 2 else 0.0
                rpm_delta = round(rpm_samples[-1] - rpm_samples[0], 1) if len(rpm_samples) >= 2 else 0.0
                speed_trend = 'rising' if speed_delta > 0.08 else 'falling' if speed_delta < -0.08 else 'stable'
                rpm_trend = 'rising' if rpm_delta > 40.0 else 'falling' if rpm_delta < -40.0 else 'stable'
                rpm_idle = 800.0
                rpm_step = 80.0
                if current_speed < th['speed_low_warning_mps']:
                    if low_speed_started_t_sec is None:
                        low_speed_started_t_sec = t_sec
                else:
                    low_speed_started_t_sec = None
                low_speed_duration_sec = (t_sec - low_speed_started_t_sec) if low_speed_started_t_sec is not None else 0.0
                warning_tags_now = [str(w.get('tag', '') or '').strip() for w in latest_state.get('warnings', []) if isinstance(w, dict) and str(w.get('tag', '') or '').strip()]
                active_warning_tags = set(warning_tags_now)
                for tag in active_warning_tags:
                    warning_started_at.setdefault(tag, t_sec)
                for tag in list(warning_started_at.keys()):
                    if tag not in active_warning_tags:
                        warning_started_at.pop(tag, None)
                primary_warning = warning_tags_now[0] if warning_tags_now else ''
                same_warning_duration_sec = (t_sec - warning_started_at.get(primary_warning, t_sec)) if primary_warning else 0.0
                last_action_event = action_events[-1] if action_events else {}
                last_action_age_sec = round(t_sec - float(last_action_event.get('t_sec', t_sec) or t_sec), 3) if last_action_event else None
                same_action_duration_sec = round(t_sec - current_action_started_t_sec, 3)
                action_response = dict(last_feedback)
                if pending_feedback is not None:
                    action_response = {
                        'action': str(pending_feedback.get('action', '') or ''),
                        'issued_t_sec': pending_feedback.get('issued_t_sec'),
                        'evaluated_t_sec': None,
                        'baseline_speed_mps': pending_feedback.get('baseline_speed_mps'),
                        'current_speed_mps': round(current_speed, 3),
                        'speed_delta_mps': round(current_speed - float(pending_feedback.get('baseline_speed_mps', current_speed) or current_speed), 3),
                        'success': None,
                        'detail': 'pending',
                    }
                elif action_response.get('baseline_speed_mps') is not None and action_response.get('current_speed_mps') is not None:
                    action_response['speed_delta_mps'] = round(float(action_response.get('current_speed_mps') or 0.0) - float(action_response.get('baseline_speed_mps') or 0.0), 3)

                history_state = {
                    'layer1_action': l1_action,
                    'layer1_reasons': list(l1_reasons),
                    'agent_control_enabled': bool(agent_control_enabled),
                    'agent_control_enabled_since_t_sec': agent_control_enabled_since_t_sec,
                    'gear_state': gear_state,
                    'drive_direction': 'REVERSE' if gear_state == 4 else 'FORWARD',
                    'layer1_latched': bool(l1_latched),
                    'layer1_latched_since_t_sec': l1_latched_since_t_sec,
                    'layer1_recovery_clear_sec': latest_state.get('layer1_recovery_clear_sec', 0.0),
                    'sensors_all_normal': latest_state.get('sensors_all_normal', False),
                    'sensor_abnormal_reasons': latest_state.get('sensor_abnormal_reasons', []),
                    'last_hard_action': latest_state.get('layer1_action', 'FORWARD'),
                    'last_agent_action': str(last_action_event.get('agent_action', '') or ''),
                    'last_effective_action': last_effective_action,
                    'last_action_age_sec': last_action_age_sec,
                    'same_action_duration_sec': same_action_duration_sec,
                    'stable_after_brake_sec': float(t_sec - l1_latched_since_t_sec) if (latest_state.get('layer1_action') == 'HOLD' and l1_latched_since_t_sec is not None) else 0.0,
                    'brake_age_sec': float(t_sec - l1_latched_since_t_sec) if (latest_state.get('layer1_action') == 'EMERGENCY_STOP' and l1_latched_since_t_sec is not None) else 0.0,
                    'current_speed_mps': current_speed,
                    'current_engine_rpm': current_engine_rpm,
                    'vehicle_context': latest_state.get('vehicle_context', ''),
                    'drive_gear_age_sec': drive_gear_age_sec,
                    'control_state': latest_state.get('control_state', {}),
                    'drive_effective_duration_sec': latest_state.get('drive_effective_duration_sec', 0.0),
                    'decision_drive_gear_grace_sec': drive_gear_grace_sec,
                    'decision_drive_effective_hold_sec': drive_effective_hold_sec,
                    'engine_rpm_high_warning': float(th.get('engine_rpm_high_warning', 2200.0)),
                    'engine_rpm_high_stop': float(th.get('engine_rpm_high_stop', 2300.0)),
                    'engine_idle_rpm': rpm_idle,
                    'engine_rpm_step': rpm_step,
                    'engine_rpm_trend_5s': rpm_trend,
                    'engine_rpm_delta_5s': rpm_delta,
                    'travel_pressure_bar': current_travel_pressure,
                    'angle_deg': current_angle_deg,
                    'payload_tons': current_payload_tons,
                    'speed_low_duration_sec': round(low_speed_duration_sec, 3),
                    'speed_trend_5s': speed_trend,
                    'speed_delta_5s_mps': speed_delta,
                    'same_warning_tag': primary_warning,
                    'same_warning_duration_sec': round(same_warning_duration_sec, 3),
                    'recent_actions': list(actions_1hz)[-5:],
                    'recent_reasons': list(reasons_1hz)[-5:],
                    'last_action_feedback': dict(last_feedback),
                    'action_response': action_response,
                }

                risk_warnings = [w for w in list(latest_state.get('warnings', []) or []) if isinstance(w, dict)]
                tag_to_sensor = {
                    'coolant_warning': 'coolant_temp_c',
                    'surface_temp_warning': 'surface_temp_c',
                    'exhaust_warning': 'exhaust_temp_c',
                    'hydraulic_oil_temp_warning': 'hydraulic_oil_temp_c',
                    'travel_pressure_warning': 'travel_pressure_bar',
                    'clamp_pressure_low': 'clamp_pressure_bar',
                    'clamp_pressure_high': 'clamp_pressure_bar',
                    'brake_pressure_low': 'brake_pressure_bar',
                    'brake_pressure_high': 'brake_pressure_bar',
                    'make_up_oil_pressure_low': 'make_up_oil_pressure_bar',
                    'intake_pressure_low': 'intake_pressure_kpa',
                    'oil_pressure_low': 'oil_pressure_kpa',
                    'system_pressure_low': 'system_pressure_bar',
                    'hydraulic_oil_level_low': 'hydraulic_oil_level_pct',
                    'diesel_level_low': 'diesel_level_cm',
                    'water_tank_low': 'water_tank_level_pct',
                    'rpm_high': 'engine_rpm',
                    'speed_high': 'speed_mps',
                    'speed_low_persistent': 'speed_mps',
                }

                def normalize_agent_warning(item: Dict[str, Any]) -> Dict[str, Any]:
                    tag = str(item.get('tag', '') or '').strip()
                    sensor_key = str(item.get('sensor_key', '') or '').strip()
                    if not sensor_key:
                        sensor_key = tag_to_sensor.get(tag, '')
                    level = str(item.get('level', '') or 'warning').strip() or 'warning'
                    source = str(item.get('source', '') or 'risk_engine').strip() or 'risk_engine'
                    unit = str(item.get('unit', '') or '')
                    meaning = str(item.get('meaning', '') or '').strip()
                    reason = str(item.get('reason', '') or '').strip()
                    return {
                        'level': level,
                        'tag': tag,
                        'sensor_key': sensor_key,
                        'value': item.get('value'),
                        'unit': unit,
                        'meaning': meaning,
                        'reason': reason,
                        'score': float(item.get('score', 0.0) or 0.0),
                        'alarm_code': int(item.get('alarm_code', 0) or 0),
                        'suggested_actions': [str(x) for x in (item.get('suggested_actions', []) or []) if str(x)],
                        'monitor_next': [str(x) for x in (item.get('monitor_next', []) or []) if str(x)],
                        'source': source,
                    }

                agent_warnings: List[Dict[str, Any]] = []
                covered_sensor_keys = set()
                covered_tags = set()
                for risk_warning in risk_warnings:
                    normalized = normalize_agent_warning(dict(risk_warning))
                    tag = normalized['tag']
                    sensor_key = normalized['sensor_key']
                    if tag:
                        covered_tags.add(tag)
                    if sensor_key:
                        covered_sensor_keys.add(sensor_key)
                        sig_val = (window_json.get('signals', {}) or {}).get(sensor_key, {})
                        meaning = str(sig_val.get('meaning', '') or '').strip() if isinstance(sig_val, dict) else ''
                        if meaning and meaning != '正常':
                            normalized['meaning'] = meaning
                    agent_warnings.append(normalized)

                for sig_key, sig_val in (window_json.get('signals', {}) or {}).items():
                    if not isinstance(sig_val, dict):
                        continue
                    meaning = str(sig_val.get('meaning', '') or '').strip()
                    if (not meaning) or meaning == '正常' or sig_key in covered_sensor_keys:
                        continue
                    if sig_key in {'speed_mps', 'brake_pressure_bar', 'travel_pressure_bar', 'walking_pressure_bar'} and ((not drive_mode_active) or in_drive_gear_grace or drive_idle_context or drive_starting_context):
                        continue
                    if t_sec < 30.0 and sig_key in {'speed_mps', 'brake_pressure_bar', 'travel_pressure_bar', 'walking_pressure_bar'}:
                        continue

                    if sig_key == 'speed_mps':
                        if str(meaning) != '速度偏低':
                            continue
                        tag = 'speed_mps_observed'
                    else:
                        tag = f'{sig_key}_warning'
                    if tag in covered_tags:
                        continue

                    agent_warnings.append(normalize_agent_warning({
                        'level': 'warning',
                        'tag': tag,
                        'sensor_key': sig_key,
                        'value': sig_val.get('value'),
                        'unit': sig_val.get('unit', ''),
                        'meaning': meaning,
                        'reason': '',
                        'score': 0.5,
                        'suggested_actions': [],
                        'monitor_next': [sig_key],
                        'alarm_code': 0,
                        'source': 'decoder_semantics',
                    }))

                payload_overall_level = str(latest_state.get('overall_risk_level', 'normal') or 'normal')
                payload_overall_score = float(latest_state.get('overall_risk_score', 0.0) or 0.0)
                if payload_overall_level == 'normal' and agent_warnings:
                    payload_overall_level = 'warning'
                    payload_overall_score = max(payload_overall_score, max(float(w.get('score', 0.0) or 0.0) for w in agent_warnings))

                agent_payload = {
                    'schema_version': 'agent_payload_v1',
                    'vehicle_id': 'sim-truck-01',
                    'timestamp': f'{t_sec:.1f}',
                    'current': {
                        'speed_mps': current_speed,
                        'engine_rpm': current_engine_rpm,
                        'gear_state': gear_state,
                        'emergency_stop': int(latest_state.get('emergency_stop', 0) or 0),
                        'travel_pressure_bar': current_travel_pressure,
                        'brake_pressure_bar': float(latest_state.get('brake_pressure_bar', 0.0) or 0.0),
                    },
                    'context': {
                        'vehicle_context': latest_state.get('vehicle_context', ''),
                        'drive_mode_active': drive_mode_active,
                        'drive_gear_age_sec': drive_gear_age_sec,
                        'drive_effective_duration_sec': latest_state.get('drive_effective_duration_sec', 0.0),
                        'drive_grace_sec': drive_gear_grace_sec,
                        'drive_effective_hold_sec': drive_effective_hold_sec,
                    },
                    'features': {
                        'risk': {
                            'overall_level': payload_overall_level,
                            'overall_score': payload_overall_score,
                            'suggested_actions': list(latest_state.get('risk_suggested_actions', []) or []),
                            'warnings': agent_warnings,
                        },
                    },
                    'history': history_state,
                }

                request_id = uuid.uuid4().hex[:8]
                requested_action = ''
                print(json.dumps({'layer': 'L2_CALL', 'event': 'prepare_agent_payload', 'request_id': request_id, 't_sec': t_sec, 'agent_control_enabled': bool(agent_control_enabled), 'l1_latched': bool(l1_latched), 'gear_state': gear_state, 'agent_url': agent_url}, ensure_ascii=False), flush=True)
                if not agent_control_enabled:
                    print(json.dumps({
                        'layer': 'MONITOR',
                        'event': 'agent_not_in_control_sensor_monitoring',
                        'request_id': request_id,
                        't_sec': t_sec,
                        'gear_state': gear_state,
                        'drive_direction': 'REVERSE' if gear_state == 4 else 'FORWARD',
                        'overall_level': payload_overall_level,
                        'overall_score': round(payload_overall_score, 3),
                        'warnings': agent_warnings,
                        'sensor_snapshot': {k: latest_state.get(k) for k in ['speed_mps', 'coolant_temp_c', 'surface_temp_c', 'hydraulic_oil_level_pct', 'diesel_level_cm', 'water_tank_level_pct', 'brake_pressure_bar', 'travel_pressure_bar', 'system_pressure_bar']},
                        'decision': None,
                        'control_action': None,
                        'note': 'agent未接管车辆',
                    }, ensure_ascii=False), flush=True)
                    enqueued = False
                elif l1_latched:
                    enqueued = worker.submit({
                        'request_id': request_id,
                        'agent_payload': agent_payload,
                        'submitted_t_sec': t_sec,
                        'monitor_only': True,
                    })
                else:
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
                        'requested_action': requested_action,
                    }
                    print(json.dumps({'layer': 'L2_CALL', 'event': 'worker_submit_ok', 'request_id': request_id, 'pending_count': len(pending_by_request)}, ensure_ascii=False), flush=True)
                elif agent_control_enabled and not l1_latched:
                    print(json.dumps({
                        'layer': 'L2_CALL',
                        'event': 'agent_worker_busy_skip',
                        'request_id': request_id,
                        'hold_action': last_final_action,
                    }, ensure_ascii=False), flush=True)

            for res in worker.poll_ready():
                request_id = str(res.get('request_id', ''))
                pending = pending_by_request.pop(request_id, None)
                if pending is None:
                    continue

                snap_state = dict(pending.get('latest_state', {}))
                l1_action_at_submit = str(pending.get('l1_action', 'HOLD'))
                window_json = pending.get('window_json', {})

                requested_action = str(pending.get('requested_action', '') or '')
                submitted_action = requested_action
                if res.get('error'):
                    l2_failures += 1
                    err_text = str(res.get('error', ''))
                    print(json.dumps({'layer': 'L2_CALL', 'event': 'agent_call_failed', 'request_id': request_id, 'error': err_text, 'fallback_action': 'DECELERATE'}, ensure_ascii=False), flush=True)
                    fallback_reason = 'agent超时回退减速' if ('timed out' in err_text.lower() or 'timeout' in err_text.lower()) else 'agent调用失败回退减速'
                    decision = {'action': 'DECELERATE', 'reason': fallback_reason, 'confidence': 0.3}
                else:
                    decision = dict(res.get('decision', {}) or {})

                l2_action = str(decision.get('action', 'DECELERATE')).upper()
                if l2_action == 'FORWARD' and float(snap_state.get('speed_mps', 0.0) or 0.0) < 0.2 and gear_state in {3, 4}:
                    l2_action = 'ACCELERATE'
                    decision['action'] = 'ACCELERATE'
                    decision['reason'] = (str(decision.get('reason', '') or '') + '；' if decision.get('reason') else '') + '起步静止改用加速'
                    decision['confidence'] = max(float(decision.get('confidence', 0.3) or 0.3), 0.5)

                if l2_action == 'START' and float(snap_state.get('speed_mps', 0.0) or 0.0) < 0.2 and gear_state in {3, 4}:
                    l2_action = 'ACCELERATE'
                    decision['action'] = 'ACCELERATE'
                    decision['reason'] = (str(decision.get('reason', '') or '') + '；' if decision.get('reason') else '') + '起步静止改用加速'
                    decision['confidence'] = max(float(decision.get('confidence', 0.3) or 0.3), 0.5)

                requested_action = requested_action or l2_action
                effective_action = l2_action
                effective_action_source = 'l2'
                if l1_action_at_submit == 'EMERGENCY_STOP' and effective_action != 'EMERGENCY_STOP':
                    effective_action = 'EMERGENCY_STOP'
                    effective_action_source = 'l1'
                if str(snap_state.get('vehicle_context', '')) == 'parking' and effective_action != 'HOLD':
                    effective_action = 'HOLD'
                    effective_action_source = 'rule_parking_guard'
                    decision['reason'] = (str(decision.get('reason', '') or '') + '；' if decision.get('reason') else '') + 'parking态禁止运动控制，动作收敛为HOLD'
                effective_action_source = normalize_effective_action_source(effective_action_source)

                if hasattr(step_stream, 'last_speed_mps'):
                    print(json.dumps({
                        'layer': 'CLOSED_LOOP_TRACE',
                        'event': 'action_feedback',
                        'request_id': request_id,
                        'requested_action': requested_action,
                        'l2_action': l2_action,
                        'final_action': effective_action,
                        'action_source': effective_action_source,
                        'stream_action_applied': getattr(step_stream, 'last_action_applied', ''),
                        'stream_last_speed_mps': round(float(getattr(step_stream, 'last_speed_mps', 0.0) or 0.0), 3),
                        'decoded_speed_mps': round(float(snap_state.get('speed_mps', 0.0) or 0.0), 3),
                        'overall_risk_level': snap_state.get('overall_risk_level', 'normal'),
                        'warning_tags': [str(w.get('tag', '')) for w in snap_state.get('warnings', []) if isinstance(w, dict) and str(w.get('tag', ''))],
                    }, ensure_ascii=False), flush=True)

                l2_reason = str(decision.get('reason', ''))
                l2_conf = float(decision.get('confidence', 0.5))
                print(json.dumps({
                    'layer': 'L2_CALL',
                    'event': 'decision_parsed',
                    'request_id': request_id,
                    'submitted_action': requested_action,
                    'action': l2_action,
                    'effective_action': effective_action,
                    'effective_action_source': effective_action_source,
                    'overall_risk_level': snap_state.get('overall_risk_level', 'normal'),
                    'warning_tags': [str(w.get('tag', '')) for w in snap_state.get('warnings', []) if isinstance(w, dict) and str(w.get('tag', ''))],
                    'reason': l2_reason,
                    'confidence': round(l2_conf, 3),
                }, ensure_ascii=False), flush=True)

                l2_decisions += 1
                out_l2 = {
                    'layer': 'L2_1Hz',
                    't_sec': pending.get('t_sec', snap_state.get('t_sec', 0.0)),
                    'submitted_action': requested_action,
                    'action': l2_action,
                    'effective_action': effective_action,
                    'effective_action_source': effective_action_source,
                    'stream_action_applied': getattr(step_stream, 'last_action_applied', ''),
                    'overall_risk_level': snap_state.get('overall_risk_level', 'normal'),
                    'warning_tags': [str(w.get('tag', '')) for w in snap_state.get('warnings', []) if isinstance(w, dict) and str(w.get('tag', ''))],
                    'overall_risk_score': snap_state.get('overall_risk_score', 0.0),
                    'reason': l2_reason,
                    'confidence': round(l2_conf, 3),
                    'window': window_json,
                }
                print(json.dumps(out_l2, ensure_ascii=False), flush=True)

                rewrite_key = f'{requested_action}->{effective_action}'
                action_rewrite_counter[rewrite_key] = action_rewrite_counter.get(rewrite_key, 0) + 1
                action_source_counter[effective_action_source] = action_source_counter.get(effective_action_source, 0) + 1
                if effective_action != last_effective_action:
                    current_action_started_t_sec = float(pending.get('t_sec', snap_state.get('t_sec', 0.0)) or 0.0)
                last_effective_action = effective_action
                actions_1hz.append(effective_action)
                reasons_1hz.append(l2_reason)
                action_events.append({
                    't_sec': float(pending.get('t_sec', snap_state.get('t_sec', 0.0)) or 0.0),
                    'agent_action': l2_action,
                    'effective_action': effective_action,
                    'effective_action_source': effective_action_source,
                    'reason': l2_reason,
                    'warning_tags': [str(w.get('tag', '')) for w in snap_state.get('warnings', []) if isinstance(w, dict) and str(w.get('tag', ''))],
                })
                control_frame = decision_to_control_frame({'action': effective_action})
                if control_frame is not None:
                    control_frame_id, control_payload = control_frame
                    control_message = {
                        'timestamp': time.time(),
                        'frame_name': hex(control_frame_id),
                        'frame_content': control_payload,
                    }
                    if control_sender is not None:
                        try:
                            control_sender.send(control_message)
                        except Exception:
                            pass
                    print(json.dumps({'layer': 'CONTROL_FRAME', 'event': 'encoded', 'request_id': request_id, 'decision_action': effective_action, 'frame_name': hex(control_frame_id), 'frame_content': control_payload}, ensure_ascii=False), flush=True)
                pending_feedback = {
                    'action': effective_action,
                    'issued_t_sec': float(pending.get('t_sec', 0.0) or 0.0),
                    'baseline_speed_mps': float(snap_state.get('speed_mps', 0.0) or 0.0),
                }
                print(json.dumps({'layer': 'L2_FEEDBACK', 'event': 'feedback_state', 'last_feedback': last_feedback, 'pending_feedback': pending_feedback}, ensure_ascii=False), flush=True)

                warning_tags_list = [str(w.get('tag', '')) for w in snap_state.get('warnings', []) if isinstance(w, dict) and str(w.get('tag', ''))]
                try:
                    feedback_speed_delta = None
                    if last_feedback.get('baseline_speed_mps') is not None and last_feedback.get('current_speed_mps') is not None:
                        feedback_speed_delta = round(float(last_feedback.get('current_speed_mps') or 0.0) - float(last_feedback.get('baseline_speed_mps') or 0.0), 3)
                    row = {
                        'schema_version': 'agent_decision_v1',
                        'run_id': run_id or '',
                        'request_id': request_id,
                        'step': step,
                        't_sec': pending.get('t_sec', snap_state.get('t_sec', 0.0)),
                        'wall_ts': time.time(),
                        'speed_mps': float(snap_state.get('speed_mps', 0.0) or 0.0),
                        'engine_rpm': float(snap_state.get('engine_rpm', 0.0) or 0.0),
                        'gear_state': int(snap_state.get('gear_state', 0) or 0),
                        'vehicle_context': str(snap_state.get('vehicle_context', '') or ''),
                        'travel_pressure_bar': float(snap_state.get('travel_pressure_bar', 0.0) or 0.0),
                        'brake_pressure_bar': float(snap_state.get('brake_pressure_bar', 0.0) or 0.0),
                        'system_pressure_bar': float(snap_state.get('system_pressure_bar', 0.0) or 0.0),
                        'overall_risk_level': str(snap_state.get('overall_risk_level', 'normal') or 'normal'),
                        'overall_risk_score': float(snap_state.get('overall_risk_score', 0.0) or 0.0),
                        'warning_tags_json': json.dumps(warning_tags_list, ensure_ascii=False),
                        'speed_delta_5s_mps': float(history_state.get('speed_delta_5s_mps', 0.0) or 0.0),
                        'speed_trend_5s': str(history_state.get('speed_trend_5s', '') or ''),
                        'engine_rpm_delta_5s': float(history_state.get('engine_rpm_delta_5s', 0.0) or 0.0),
                        'engine_rpm_trend_5s': str(history_state.get('engine_rpm_trend_5s', '') or ''),
                        'speed_low_duration_sec': float(history_state.get('speed_low_duration_sec', 0.0) or 0.0),
                        'same_warning_duration_sec': float(history_state.get('same_warning_duration_sec', 0.0) or 0.0),
                        'l1_action': l1_action_at_submit,
                        'l1_reasons_json': json.dumps(list(snap_state.get('layer1_reasons', []) or []), ensure_ascii=False),
                        'l2_action': l2_action,
                        'effective_action': effective_action,
                        'effective_action_source': effective_action_source,
                        'rewritten': int(l2_action != effective_action),
                        'feedback_success': '' if last_feedback.get('success') is None else int(bool(last_feedback.get('success'))),
                        'feedback_detail': str(last_feedback.get('detail', '') or ''),
                        'baseline_speed_mps': '' if last_feedback.get('baseline_speed_mps') is None else float(last_feedback.get('baseline_speed_mps') or 0.0),
                        'current_speed_mps': '' if last_feedback.get('current_speed_mps') is None else float(last_feedback.get('current_speed_mps') or 0.0),
                        'speed_delta_mps': '' if feedback_speed_delta is None else feedback_speed_delta,
                    }
                    with decisions_csv_path.open('a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                        writer.writerow(row)
                except Exception:
                    pass

                if run_event_cb is not None:
                    try:
                        run_event_cb({
                            'event': 'l2_tick',
                            'ts': time.time(),
                            'run_id': run_id,
                            'request_id': request_id,
                            't_sec': pending.get('t_sec', snap_state.get('t_sec', 0.0)),
                            'submitted_action': requested_action,
                            'l2_action': l2_action,
                            'effective_action': effective_action,
                            'effective_action_source': effective_action_source,
                            'stream_action_applied': getattr(step_stream, 'last_action_applied', ''),
                            'overall_risk_level': snap_state.get('overall_risk_level', 'normal'),
                            'warning_tags': warning_tags_list,
                            'overall_risk_score': snap_state.get('overall_risk_score', 0.0),
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
                if last_effective_action != 'HOLD':
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
        'action_rewrite_counter': action_rewrite_counter,
        'action_source_counter': action_source_counter,
    }
    print(json.dumps({'layer': 'RUN_SUMMARY', **summary}, ensure_ascii=False), flush=True)
    return summary
