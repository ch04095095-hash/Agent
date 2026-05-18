#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

try:
    from oellm_agent.risk_engine import RiskEngine
except ModuleNotFoundError:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from oellm_agent.risk_engine import RiskEngine


ACTION_SET = {"FORWARD", "ACCELERATE", "DECELERATE", "BRAKE"}
RISK_LEVEL_SET = {"normal", "warning", "high_warning", "danger"}
CRUISE_TARGET_SPEED_MPS = 1.0
SPEED_LOW_WARNING_MPS = 0.15



def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass
class Thresholds:
    speed_high_stop_mps: float = 2.5
    speed_low_warning_mps: float = 0.15
    engine_rpm_high_warning: float = 2200.0
    engine_rpm_high_stop: float = 2300.0
    coolant_temp_high_warning: float = 93.0
    coolant_temp_high_stop: float = 95.0
    surface_temp_high_warning: float = 147.0
    surface_temp_high_stop: float = 150.0
    exhaust_temp_high_warning: float = 65.0
    exhaust_temp_high_stop: float = 69.0
    brake_pressure_low_warning: float = 150.0
    brake_pressure_low_stop: float = 60.0
    travel_pressure_low_warning: float = 26.0
    travel_pressure_low_stop: float = 25.0
    travel_pressure_high_warning: float = 250.0
    travel_pressure_high_stop: float = 380.0
    system_pressure_low_warning: float = 150.0
    system_pressure_low_stop: float = 80.0
    methane_high_stop: float = 0.5
    co_high_stop_ppm: float = 24.0
    decision_drive_gear_grace_sec: float = 15.0
    decision_drive_effective_hold_sec: float = 3.0
    decision_speed_low_effective_hold_sec: float = 60.0



def _load_thresholds(path: Path) -> Thresholds:
    data = json.loads(path.read_text(encoding="utf-8"))
    t = Thresholds()
    for k, v in data.items():
        if hasattr(t, k):
            try:
                setattr(t, k, float(v))
            except Exception:
                pass
    return t



def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default



def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default



def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))



def _parse_ts(ts: str) -> Optional[float]:
    try:
        # Format like 2026-05-05 12:32:59.944.304
        # Keep only first two microsecond groups.
        if not ts:
            return None
        parts = ts.split(".")
        if len(parts) >= 3:
            ts = parts[0] + "." + parts[1][:6].ljust(6, "0")
        import datetime as _dt

        dt = _dt.datetime.fromisoformat(ts)
        return dt.timestamp()
    except Exception:
        return None



def _trend(values: List[float]) -> Tuple[str, float]:
    if len(values) < 2:
        return "stable", 0.0
    delta = values[-1] - values[0]
    if delta > 0.15:
        return "rising", round(delta, 3)
    if delta < -0.15:
        return "falling", round(delta, 3)
    return "stable", round(delta, 3)



def _window_delta(window: List[float]) -> float:
    if len(window) < 2:
        return 0.0
    return round(window[-1] - window[0], 3)



def _overall_risk(row: Dict[str, Any], t: Thresholds) -> Tuple[str, List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []

    speed = _safe_float(row.get("speed_mps"))
    engine_rpm = _safe_float(row.get("engine_rpm"))
    brake_pressure = _safe_float(row.get("brake_pressure_bar"))
    travel_pressure = _safe_float(row.get("travel_pressure_bar"))
    system_pressure = _safe_float(row.get("system_pressure_bar"))
    coolant = _safe_float(row.get("coolant_temp_c"))
    surface = _safe_float(row.get("surface_temp_c"))
    exhaust = _safe_float(row.get("exhaust_temp_c"))
    methane = _safe_float(row.get("methane_pct"))
    co = _safe_float(row.get("co_ppm"))
    emergency_stop = _safe_int(row.get("emergency_stop"))

    def add(tag: str, value: Any, unit: str = "", reason: str = "", score: float = 0.5):
        warnings.append({
            "level": "warning",
            "tag": tag,
            "value": value,
            "unit": unit,
            "reason": reason,
            "score": score,
            "suggested_actions": [],
            "monitor_next": [],
            "alarm_code": 0,
            "source": "rule",
        })

    if emergency_stop:
        add("emergency_stop", 1, "", "急停触发", 1.0)
        return "danger", warnings
    if methane >= t.methane_high_stop:
        add("methane_high_stop", methane, "%", "甲烷超限", 1.0)
    if co >= t.co_high_stop_ppm:
        add("co_high_stop_ppm", co, "ppm", "一氧化碳超限", 1.0)
    if engine_rpm >= t.engine_rpm_high_stop:
        add("rpm_high_stop", engine_rpm, "rpm", "发动机转速超限", 1.0)
    elif engine_rpm >= t.engine_rpm_high_warning:
        add("rpm_high", engine_rpm, "rpm", "发动机转速高预警", 0.7)
    if speed >= t.speed_high_stop_mps:
        add("speed_high", speed, "m/s", "车速超限", 1.0)
    elif speed <= t.speed_low_warning_mps:
        add("speed_low_persistent", speed, "m/s", "车速偏低", 0.45)
    if brake_pressure <= t.brake_pressure_low_stop:
        add("brake_pressure_low", brake_pressure, "bar", "制动压力过低", 1.0)
    elif brake_pressure <= t.brake_pressure_low_warning:
        add("brake_pressure_low", brake_pressure, "bar", "制动压力低预警", 0.7)
    if travel_pressure <= t.travel_pressure_low_stop:
        add("travel_pressure_warning", travel_pressure, "bar", "行走压力过低", 1.0)
    elif travel_pressure <= t.travel_pressure_low_warning:
        add("travel_pressure_warning", travel_pressure, "bar", "行走压力低预警", 0.7)
    if travel_pressure >= t.travel_pressure_high_stop:
        add("travel_pressure_warning", travel_pressure, "bar", "行走压力超限", 1.0)
    elif travel_pressure >= t.travel_pressure_high_warning:
        add("travel_pressure_warning", travel_pressure, "bar", "行走压力高预警", 0.7)
    if system_pressure <= t.system_pressure_low_stop:
        add("system_pressure_low", system_pressure, "bar", "系统压力过低", 1.0)
    elif system_pressure <= t.system_pressure_low_warning:
        add("system_pressure_low", system_pressure, "bar", "系统压力低预警", 0.7)
    if coolant >= t.coolant_temp_high_stop:
        add("coolant_warning", coolant, "C", "冷却液温度高", 1.0)
    elif coolant >= t.coolant_temp_high_warning:
        add("coolant_warning", coolant, "C", "冷却液温度高预警", 0.7)
    if surface >= t.surface_temp_high_stop:
        add("surface_temp_warning", surface, "C", "表面温度高", 1.0)
    elif surface >= t.surface_temp_high_warning:
        add("surface_temp_warning", surface, "C", "表面温度高预警", 0.7)
    if exhaust >= t.exhaust_temp_high_stop:
        add("exhaust_warning", exhaust, "C", "排气温度高", 1.0)
    elif exhaust >= t.exhaust_temp_high_warning:
        add("exhaust_warning", exhaust, "C", "排气温度高预警", 0.7)

    if any(w["score"] >= 1.0 for w in warnings):
        return "danger", warnings
    if any(w["score"] >= 0.7 for w in warnings):
        return "warning", warnings
    return "normal", warnings



def _vehicle_context(row: Dict[str, Any], t: Thresholds, low_speed_sec: float) -> str:
    gear = _safe_int(row.get("gear_state"))
    speed = _safe_float(row.get("speed_mps"))
    engine_rpm = _safe_float(row.get("engine_rpm"))
    if gear <= 1:
        return "not_in_drive"
    if low_speed_sec < t.decision_drive_gear_grace_sec:
        if speed < 0.2:
            return "drive_idle"
        if engine_rpm < 800:
            return "drive_grace"
        return "drive_starting"
    if speed < 0.5 and engine_rpm < 1300:
        return "drive_transition"
    return "drive_effective"



def _action_response(last_action: str, speed_delta: float, current_speed: float) -> Dict[str, Any]:
    if not last_action:
        return {"detail": "no_feedback_yet"}
    if last_action == "ACCELERATE":
        if current_speed < 0.05:
            return {"detail": "not_applicable_zero_speed"}
        if speed_delta > 0.05:
            return {"detail": "speed_up_ok"}
        if speed_delta < -0.05:
            return {"detail": "speed_not_up"}
        return {"detail": "command_issued"}
    if last_action == "DECELERATE":
        if speed_delta < -0.05:
            return {"detail": "speed_down_ok"}
        return {"detail": "command_issued"}
    if last_action == "BRAKE":
        if current_speed < 0.05:
            return {"detail": "stopped_ok"}
        return {"detail": "command_issued"}
    return {"detail": "command_issued"}



def _rule_decision(current: Dict[str, Any], history: Dict[str, Any], risk_level: str, warnings: List[Dict[str, Any]], t: Thresholds) -> Dict[str, Any]:
    speed = _safe_float(current.get("speed_mps"))
    engine_rpm = _safe_float(current.get("engine_rpm"))
    last_action = str(history.get("last_effective_action", "") or "").upper()
    last_action_age_sec = _safe_float(history.get("last_action_age_sec"), 0.0)
    same_action_duration_sec = _safe_float(history.get("same_action_duration_sec"), 0.0)
    speed_trend_5s = str(history.get("speed_trend_5s", "") or "").lower()
    speed_delta_5s_mps = _safe_float(history.get("speed_delta_5s_mps"), 0.0)
    engine_rpm_delta_5s = _safe_float(history.get("engine_rpm_delta_5s"), 0.0)
    action_response = str(_as_dict(history.get("action_response")).get("detail", "") or "")
    layer1_action = str(history.get("layer1_action", "") or "").upper()
    vehicle_context = str(history.get("vehicle_context", "") or "")

    warning_tags = [str(w.get("tag", "") or "").strip() for w in warnings if str(w.get("tag", "") or "").strip()]
    has_speed_high = any(tag == "speed_high" for tag in warning_tags)
    has_non_speed_warning = any(tag not in {"speed_low_persistent", "speed_high"} for tag in warning_tags)
    low_speed = speed < CRUISE_TARGET_SPEED_MPS

    if layer1_action == "BRAKE":
        action = "BRAKE"
        reason = "L1硬保护触发BRAKE，必须优先停车检查。"
        risk_out = "danger"
        confidence = 0.99
    elif risk_level == "danger":
        action = "BRAKE" if any(tag in {"emergency_stop", "rpm_high_stop", "speed_high"} for tag in warning_tags) else "DECELERATE"
        reason = "当前存在严重风险，优先执行安全减速。"
        risk_out = "danger"
        confidence = 0.98
    elif risk_level == "warning":
        if vehicle_context in {"drive_grace", "drive_idle", "drive_starting", "drive_transition"} and speed < 0.6 and not has_non_speed_warning:
            action = "FORWARD"
            reason = "起步或驱动建立阶段存在轻微预警，先保持FORWARD观察。"
        elif speed > 2.0 and has_non_speed_warning:
            action = "DECELERATE"
            reason = "当前速度较高且存在异常，优先减速。"
        elif has_speed_high:
            action = "DECELERATE"
            reason = "速度高预警，优先减速。"
        else:
            action = "DECELERATE" if has_non_speed_warning else "FORWARD"
            reason = "当前为预警态，结合速度和异常类型选择保守动作。"
        risk_out = "warning"
        confidence = 0.86
    else:
        if low_speed and not has_non_speed_warning and engine_rpm < t.engine_rpm_high_warning and not (last_action == "ACCELERATE" and last_action_age_sec < 4.0):
            action = "ACCELERATE"
            reason = "当前风险正常且速度低于目标，执行一次补速。"
        elif last_action == "ACCELERATE" and last_action_age_sec < 4.0:
            action = "FORWARD"
            reason = "刚执行加速且仍在冷却窗口内，本轮保持FORWARD观察。"
        elif action_response == "pending":
            action = "FORWARD"
            reason = "上一动作仍在响应窗口内，保持FORWARD观察。"
        elif speed < 0.15 and speed_trend_5s in {"stable", "falling"} and same_action_duration_sec > 8.0 and last_action == "ACCELERATE":
            action = "FORWARD"
            reason = "连续加速后速度仍无明显改善，先转FORWARD观察并避免重复按键。"
        else:
            action = "FORWARD"
            reason = "当前风险正常，先保持FORWARD观察。"
        risk_out = "normal"
        confidence = 0.9 if action == "ACCELERATE" else 0.8

    if action == "ACCELERATE" and engine_rpm >= t.engine_rpm_high_warning - 80:
        action = "FORWARD"
        reason += " 发动机转速接近上限，本轮不继续加速。"
        confidence = min(confidence, 0.82)
    if action == "ACCELERATE" and last_action == "ACCELERATE" and last_action_age_sec < 4.0:
        action = "FORWARD"
        reason += " 加速冷却未完成，改为FORWARD。"
        confidence = min(confidence, 0.8)
    if action == "DECELERATE" and speed < 0.15 and risk_out != "danger":
        action = "FORWARD"
        reason += " 当前已近静止，不继续减速，转为观察。"
        confidence = min(confidence, 0.78)

    suspected_fault = []
    monitor_next = []
    recommended_adjustment = []
    for w in warnings:
        tag = str(w.get("tag", "") or "").strip()
        if not tag:
            continue
        suspected_fault.append(tag)
        monitor_next.extend([tag])
        suggested = w.get("suggested_actions", [])
        if isinstance(suggested, list):
            recommended_adjustment.extend([str(x) for x in suggested if str(x).strip()])

    if risk_out in {"warning", "danger"}:
        recommended_adjustment.append("maintain_safe_speed")
    if action == "ACCELERATE":
        recommended_adjustment.append("check_speed_response")
    elif action == "DECELERATE":
        recommended_adjustment.append("monitor_pressure_and_temperature")
    else:
        recommended_adjustment.append("continue_observation")

    reason = reason.strip()
    if not reason:
        reason = "当前状态下选择保守动作。"

    return {
        "action": action,
        "risk_level": risk_out,
        "reason": reason,
        "confidence": round(confidence, 3),
        "suspected_fault": list(dict.fromkeys(suspected_fault)),
        "recommended_adjustment": list(dict.fromkeys(recommended_adjustment)),
        "monitor_next": list(dict.fromkeys(monitor_next))[:6],
        "warning_tags": warning_tags,
        "policy": "rule_generated",
        "decision_source": "rule_generated",
        "history_used": {
            "layer1_action": layer1_action,
            "last_effective_action": last_action,
            "last_action_age_sec": last_action_age_sec,
            "same_action_duration_sec": same_action_duration_sec,
            "speed_trend_5s": speed_trend_5s,
            "speed_delta_5s_mps": speed_delta_5s_mps,
            "engine_rpm_delta_5s": engine_rpm_delta_5s,
            "action_response_detail": action_response,
            "vehicle_context": vehicle_context,
        },
    }



def _rows_to_samples(rows: List[Dict[str, Any]], thresholds: Thresholds, window_sec: float, stride: int) -> Iterable[Dict[str, Any]]:
    speed_hist: Deque[float] = deque(maxlen=max(2, int(round(window_sec * 2))))
    rpm_hist: Deque[float] = deque(maxlen=max(2, int(round(window_sec * 2))))
    last_action = "FORWARD"
    last_action_idx = -999999
    current_action_start_idx = 0
    low_speed_start_idx: Optional[int] = None
    last_ts: Optional[float] = None

    for idx, row in enumerate(rows):
        current = {
            "speed_mps": _safe_float(row.get("speed_mps")),
            "engine_rpm": _safe_float(row.get("engine_rpm")),
            "travel_pressure_bar": _safe_float(row.get("travel_pressure_bar")),
            "brake_pressure_bar": _safe_float(row.get("brake_pressure_bar")),
            "system_pressure_bar": _safe_float(row.get("system_pressure_bar")),
            "coolant_temp_c": _safe_float(row.get("coolant_temp_c")),
            "surface_temp_c": _safe_float(row.get("surface_temp_c")),
            "exhaust_temp_c": _safe_float(row.get("exhaust_temp_c")),
            "methane_pct": _safe_float(row.get("methane_pct")),
            "co_ppm": _safe_float(row.get("co_ppm")),
            "gear_state": _safe_int(row.get("gear_state")),
            "emergency_stop": _safe_int(row.get("emergency_stop")),
        }
        speed_hist.append(current["speed_mps"])
        rpm_hist.append(current["engine_rpm"])

        ts = _parse_ts(str(row.get("timestamp", "") or ""))
        if last_ts is None:
            last_ts = ts
        dt = 0.5
        if ts is not None and last_ts is not None:
            dt = max(0.1, ts - last_ts)
            last_ts = ts

        low_speed = current["speed_mps"] < thresholds.speed_low_warning_mps
        if low_speed and low_speed_start_idx is None:
            low_speed_start_idx = idx
        if not low_speed:
            low_speed_start_idx = None

        speed_delta_5s = _window_delta(list(speed_hist))
        engine_rpm_delta_5s = _window_delta(list(rpm_hist))
        speed_trend_5s, _ = _trend(list(speed_hist))
        rpm_trend_5s, _ = _trend(list(rpm_hist))
        low_speed_duration_sec = 0.0
        if low_speed_start_idx is not None:
            low_speed_duration_sec = round((idx - low_speed_start_idx + 1) * dt, 3)

        risk_level, warnings = _overall_risk(row, thresholds)
        vehicle_context = _vehicle_context(row, thresholds, low_speed_duration_sec)
        action_response = _action_response(last_action, speed_delta_5s, current["speed_mps"])

        history = {
            "layer1_action": "FORWARD",
            "layer1_reasons": [],
            "last_effective_action": last_action,
            "last_action_age_sec": round((idx - last_action_idx) * dt, 3) if last_action_idx >= 0 else 0.0,
            "same_action_duration_sec": round((idx - current_action_start_idx + 1) * dt, 3),
            "speed_trend_5s": speed_trend_5s,
            "speed_delta_5s_mps": speed_delta_5s,
            "engine_rpm_delta_5s": engine_rpm_delta_5s,
            "speed_low_duration_sec": low_speed_duration_sec,
            "action_response": action_response,
            "vehicle_context": vehicle_context,
            "current_speed_mps": current["speed_mps"],
            "current_engine_rpm": current["engine_rpm"],
        }
        decision = _rule_decision(current, history, risk_level, warnings, thresholds)
        decision["scenario"] = vehicle_context
        decision["risk_level_raw"] = risk_level
        if decision["risk_level"] == "danger" and decision["action"] == "FORWARD":
            decision["action"] = "DECELERATE"
            decision["reason"] = "严重风险场景不应输出FORWARD，改为DECELERATE观察/减速。"
            decision["decision_source"] = "rule_generated_adjusted"
        if decision["risk_level"] == "danger" and decision["action"] == "ACCELERATE":
            decision["action"] = "DECELERATE"
            decision["reason"] = "严重风险场景不应加速，改为DECELERATE。"
            decision["decision_source"] = "rule_generated_adjusted"

        if idx % stride != 0:
            # Still update last_action state heuristically from the chosen action for temporal coherence.
            if decision["action"] != last_action:
                last_action = decision["action"]
                last_action_idx = idx
                current_action_start_idx = idx
            continue

        sample = {
            "instruction": "你是矿车控制决策器，只输出JSON，不要解释。",
            "input": {
                "current": current,
                "history": history,
                "features": {
                    "risk": {
                        "overall_level": risk_level,
                        "warnings": warnings,
                    }
                },
            },
            "output": {k: decision[k] for k in ["action", "risk_level", "reason", "confidence", "suspected_fault", "recommended_adjustment", "monitor_next", "warning_tags", "policy", "decision_source", "scenario", "risk_level_raw"] if k in decision},
        }
        yield sample

        if decision["action"] != last_action:
            last_action = decision["action"]
            last_action_idx = idx
            current_action_start_idx = idx



def generate_dataset(input_csv: Path, thresholds_path: Path, output_jsonl: Path, valid_jsonl: Optional[Path] = None,
                     window_sec: float = 5.0, stride: int = 1, valid_ratio: float = 0.1, max_rows: Optional[int] = None) -> Tuple[int, int]:
    thresholds = _load_thresholds(thresholds_path)
    rows = _load_rows(input_csv)
    risk_engine = RiskEngine(thresholds.__dict__)
    if max_rows is not None:
        rows = rows[:max_rows]
    samples = list(_rows_to_samples(rows, thresholds, window_sec=window_sec, stride=max(1, stride)))
    if not samples:
        output_jsonl.write_text("", encoding="utf-8")
        if valid_jsonl:
            valid_jsonl.write_text("", encoding="utf-8")
        return 0, 0

    split = int(len(samples) * (1.0 - valid_ratio))
    split = max(1, min(split, len(samples) - 1)) if len(samples) > 1 else len(samples)
    train_samples = samples[:split]
    valid_samples = samples[split:] if len(samples) > 1 else []

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if valid_jsonl is not None:
        valid_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with valid_jsonl.open("w", encoding="utf-8") as f:
            for sample in valid_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return len(train_samples), len(valid_samples)



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT training data from decoded CAN csv + thresholds + rule decision.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    train_n, valid_n = generate_dataset(
        input_csv=args.input_csv,
        thresholds_path=args.thresholds,
        output_jsonl=args.output,
        valid_jsonl=args.valid_output,
        window_sec=args.window_sec,
        stride=args.stride,
        valid_ratio=args.valid_ratio,
        max_rows=args.max_rows,
    )
    print(json.dumps({"train": train_n, "valid": valid_n, "output": str(args.output), "valid_output": str(args.valid_output) if args.valid_output else ""}, ensure_ascii=False))


if __name__ == "__main__":
    main()
