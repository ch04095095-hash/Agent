#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


try:
    from oellm_agent.risk_engine import RiskEngine  # type: ignore
except Exception:  # pragma: no cover
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from oellm_agent.risk_engine import RiskEngine  # type: ignore


CRUISE_TARGET_SPEED_MPS = 1.0
STARTUP_GRACE_SEC = 15.0
BOUNDARY_MARGIN = {
    "speed_mps": 0.25,
    "engine_rpm": 120.0,
    "brake_pressure_bar": 15.0,
    "travel_pressure_bar": 20.0,
    "system_pressure_bar": 20.0,
    "coolant_temp_c": 4.0,
    "surface_temp_c": 4.0,
    "exhaust_temp_c": 4.0,
}


@dataclass
class Thresholds:
    speed_low_warning_mps: float = 0.15
    speed_high_warning_mps: float = 2.2
    speed_high_stop_mps: float = 2.5
    engine_rpm_high_warning: float = 2200.0
    engine_rpm_high_stop: float = 2300.0
    brake_pressure_low_warning: float = 150.0
    brake_pressure_low_stop: float = 60.0
    travel_pressure_low_warning: float = 26.0
    travel_pressure_low_stop: float = 25.0
    travel_pressure_high_warning: float = 250.0
    travel_pressure_high_stop: float = 380.0
    system_pressure_low_warning: float = 150.0
    system_pressure_low_stop: float = 80.0
    coolant_temp_high_warning: float = 93.0
    coolant_temp_high_stop: float = 95.0
    surface_temp_high_warning: float = 147.0
    surface_temp_high_stop: float = 150.0
    exhaust_temp_high_warning: float = 65.0
    exhaust_temp_high_stop: float = 69.0
    methane_high_stop: float = 0.5
    co_high_stop_ppm: float = 24.0
    decision_drive_gear_grace_sec: float = 15.0
    decision_drive_effective_hold_sec: float = 3.0
    decision_speed_low_effective_hold_sec: float = 60.0



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



def _safe_str(v: Any, default: str = "") -> str:
    s = str(v or "").strip()
    return s if s else default



def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}



def _load_thresholds(path: Path) -> Thresholds:
    data = json.loads(path.read_text(encoding="utf-8"))
    th = Thresholds()
    for k, v in data.items():
        if hasattr(th, k):
            try:
                setattr(th, k, float(v))
            except Exception:
                pass
    return th



def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))



def _parse_ts(ts: str) -> Optional[float]:
    try:
        if not ts:
            return None
        parts = ts.split(".")
        if len(parts) >= 3:
            ts = parts[0] + "." + parts[1][:6].ljust(6, "0")
        return dt.datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None



def _window_delta(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(values[-1] - values[0], 3)



def _trend(values: List[float]) -> str:
    if len(values) < 2:
        return "stable"
    delta = values[-1] - values[0]
    if delta > 0.15:
        return "rising"
    if delta < -0.15:
        return "falling"
    return "stable"



def _build_current(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "speed_mps": _safe_float(row.get("speed_mps")),
        "engine_rpm": _safe_float(row.get("engine_rpm")),
        "gear_state": _safe_int(row.get("gear_state")),
        "emergency_stop": _safe_int(row.get("emergency_stop")),
        "brake_pressure_bar": _safe_float(row.get("brake_pressure_bar")),
        "travel_pressure_bar": _safe_float(row.get("travel_pressure_bar")),
        "system_pressure_bar": _safe_float(row.get("system_pressure_bar")),
        "clamp_pressure_bar": _safe_float(row.get("clamp_pressure_bar")),
        "coolant_temp_c": _safe_float(row.get("coolant_temp_c")),
        "surface_temp_c": _safe_float(row.get("surface_temp_c")),
        "exhaust_temp_c": _safe_float(row.get("exhaust_temp_c")),
        "intake_pressure_kpa": _safe_float(row.get("intake_pressure_kpa")),
        "hydraulic_oil_temp_c": _safe_float(row.get("hydraulic_oil_temp_c")),
        "hydraulic_oil_level_pct": _safe_float(row.get("hydraulic_oil_level_pct")),
        "oil_pressure_kpa": _safe_float(row.get("oil_pressure_kpa")),
        "diesel_level_cm": _safe_float(row.get("diesel_level_cm")),
        "water_tank_level_pct": _safe_float(row.get("water_tank_level_pct")),
        "battery_v": _safe_float(row.get("battery_v")),
        "total_mileage_km": _safe_float(row.get("total_mileage_km")),
        "runtime_min": _safe_float(row.get("runtime_min")),
        "load_state": _safe_int(row.get("load_state")),
        "methane_pct": _safe_float(row.get("methane_pct")),
        "co_ppm": _safe_float(row.get("co_ppm")),
        "intake_temp_c": _safe_float(row.get("intake_temp_c")),
    }



def _vehicle_context(speed: float, rpm: float, gear: int, low_speed_sec: float, th: Thresholds) -> str:
    if gear <= 1:
        return "not_in_drive"
    if low_speed_sec < th.decision_drive_gear_grace_sec:
        if speed < 0.2:
            return "drive_idle"
        if rpm < 800:
            return "drive_grace"
        return "drive_starting"
    if speed < 0.5 and rpm < 1300:
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



def _near_boundary(val: float, warn: float, stop: float, margin: float) -> bool:
    return abs(val - warn) <= margin or abs(val - stop) <= margin



def _is_boundary_row(current: Dict[str, Any], history: Dict[str, Any], th: Thresholds) -> Tuple[bool, str]:
    speed = current["speed_mps"]
    rpm = current["engine_rpm"]
    brake = current["brake_pressure_bar"]
    travel = current["travel_pressure_bar"]
    system_p = current["system_pressure_bar"]
    coolant = current["coolant_temp_c"]
    surface = current["surface_temp_c"]
    exhaust = current["exhaust_temp_c"]
    gear = current["gear_state"]
    low_speed_sec = history["speed_low_duration_sec"]
    vehicle_context = history["vehicle_context"]

    if gear > 1 and low_speed_sec <= th.decision_drive_gear_grace_sec and speed < 0.2 and brake <= th.brake_pressure_low_warning + 10:
        return True, "startup_brake_low_but_normal"
    if vehicle_context in {"drive_starting", "drive_idle", "drive_grace"} and brake <= th.brake_pressure_low_warning + 5:
        return True, "startup_drive_pressure_boundary"
    if _near_boundary(speed, 0.15, 0.0, BOUNDARY_MARGIN["speed_mps"]):
        return True, "speed_low_boundary"
    if _near_boundary(speed, 1.0, 0.0, 0.25):
        return True, "cruise_target_boundary"
    if _near_boundary(rpm, th.engine_rpm_high_warning, th.engine_rpm_high_stop, BOUNDARY_MARGIN["engine_rpm"]):
        return True, "rpm_boundary"
    if _near_boundary(brake, th.brake_pressure_low_warning, th.brake_pressure_low_stop, BOUNDARY_MARGIN["brake_pressure_bar"]):
        return True, "brake_boundary"
    if _near_boundary(travel, th.travel_pressure_low_warning, th.travel_pressure_low_stop, BOUNDARY_MARGIN["travel_pressure_bar"]):
        return True, "travel_pressure_low_boundary"
    if _near_boundary(travel, th.travel_pressure_high_warning, th.travel_pressure_high_stop, BOUNDARY_MARGIN["travel_pressure_bar"]):
        return True, "travel_pressure_high_boundary"
    if _near_boundary(system_p, th.system_pressure_low_warning, th.system_pressure_low_stop, BOUNDARY_MARGIN["system_pressure_bar"]):
        return True, "system_pressure_boundary"
    if _near_boundary(coolant, th.coolant_temp_high_warning, th.coolant_temp_high_stop, BOUNDARY_MARGIN["coolant_temp_c"]):
        return True, "coolant_boundary"
    if _near_boundary(surface, th.surface_temp_high_warning, th.surface_temp_high_stop, BOUNDARY_MARGIN["surface_temp_c"]):
        return True, "surface_temp_boundary"
    if _near_boundary(exhaust, th.exhaust_temp_high_warning, th.exhaust_temp_high_stop, BOUNDARY_MARGIN["exhaust_temp_c"]):
        return True, "exhaust_boundary"
    if vehicle_context == "drive_transition" and speed < 0.5 and rpm < 1300:
        return True, "transition_boundary"
    return False, ""



def _build_history(idx: int, rows: List[Dict[str, Any]], ts_cache: List[Optional[float]], th: Thresholds) -> Dict[str, Any]:
    row = rows[idx]
    current = _build_current(row)
    speed_hist: Deque[float] = deque(maxlen=11)
    rpm_hist: Deque[float] = deque(maxlen=11)
    start = max(0, idx - 10)
    for j in range(start, idx + 1):
        speed_hist.append(_safe_float(rows[j].get("speed_mps")))
        rpm_hist.append(_safe_float(rows[j].get("engine_rpm")))

    ts = ts_cache[idx]
    prev_ts = ts_cache[idx - 1] if idx > 0 else None
    dt = 0.5
    if ts is not None and prev_ts is not None:
        dt = max(0.1, ts - prev_ts)

    low_speed = current["speed_mps"] < th.speed_low_warning_mps
    low_start = idx
    while low_start > 0:
        prev_speed = _safe_float(rows[low_start - 1].get("speed_mps"))
        if prev_speed < th.speed_low_warning_mps:
            low_start -= 1
        else:
            break
    low_speed_duration_sec = round((idx - low_start + 1) * dt, 3) if low_speed else 0.0

    last_action = _safe_str(rows[idx - 1].get("last_action"), "") if idx > 0 else "FORWARD"
    if idx > 0 and _safe_str(rows[idx - 1].get("last_action"), ""):
        last_action = _safe_str(rows[idx - 1].get("last_action"), "")
    elif idx > 0:
        last_action = _safe_str(rows[idx - 1].get("decision_action"), "FORWARD")
    else:
        last_action = "FORWARD"

    speed_delta_5s = _window_delta(list(speed_hist))
    engine_rpm_delta_5s = _window_delta(list(rpm_hist))
    speed_trend_5s = _trend(list(speed_hist))
    rpm_trend_5s = _trend(list(rpm_hist))
    vehicle_context = _vehicle_context(current["speed_mps"], current["engine_rpm"], current["gear_state"], low_speed_duration_sec, th)

    history = {
        "layer1_action": _safe_str(rows[idx].get("layer1_action"), "FORWARD") if rows[idx].get("layer1_action") else "FORWARD",
        "layer1_reasons": [],
        "last_effective_action": last_action,
        "last_action_age_sec": round(dt if idx > 0 else 0.0, 3),
        "same_action_duration_sec": round((idx - low_start + 1) * dt, 3) if idx >= low_start else 0.0,
        "speed_trend_5s": speed_trend_5s,
        "speed_delta_5s_mps": speed_delta_5s,
        "engine_rpm_delta_5s": engine_rpm_delta_5s,
        "speed_low_duration_sec": low_speed_duration_sec,
        "action_response": _action_response(last_action, speed_delta_5s, current["speed_mps"]),
        "vehicle_context": vehicle_context,
        "current_speed_mps": current["speed_mps"],
        "current_engine_rpm": current["engine_rpm"],
        "engine_rpm_trend_5s": rpm_trend_5s,
    }
    return history



def _build_reason(sample: Dict[str, Any], boundary_kind: str, target_action: str) -> str:
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}
    speed = _safe_float(current.get("speed_mps"))
    rpm = _safe_float(current.get("engine_rpm"))
    brake = _safe_float(current.get("brake_pressure_bar"))
    travel = _safe_float(current.get("travel_pressure_bar"))
    system_p = _safe_float(current.get("system_pressure_bar"))
    scenario = _safe_str(sample.get("output", {}).get("scenario"), _safe_str(history.get("vehicle_context"), "unknown"))
    last_action = _safe_str(history.get("last_effective_action"), "")
    trend = _safe_str(history.get("speed_trend_5s"), "stable")
    age = _safe_float(history.get("last_action_age_sec"))

    if boundary_kind in {"startup_brake_low_but_normal", "startup_drive_pressure_boundary"}:
        return f"起步/接管建立阶段({scenario})，制动压力{brake:.1f}bar偏低但符合建立过程，优先FORWARD观察。"
    if boundary_kind == "speed_low_boundary":
        return f"车速处于边界区间{speed:.2f}m/s，结合场景{scenario}和趋势{trend}，先保守观察。"
    if boundary_kind == "cruise_target_boundary":
        return f"车速接近巡航目标{speed:.2f}m/s，结合上次动作{last_action}(已持续{age:.1f}s)避免频繁切换。"
    if boundary_kind == "rpm_boundary":
        return f"发动机转速{rpm:.0f}rpm接近阈值边界，优先保守处理以避免过度加速。"
    if boundary_kind == "brake_boundary":
        return f"制动压力{brake:.1f}bar处于边界，结合场景{scenario}判断是否需要减速而非误报。"
    if boundary_kind == "travel_pressure_low_boundary":
        return f"行走压力{travel:.1f}bar接近低压边界，需结合速度趋势{trend}决定是否仅观察或减速。"
    if boundary_kind == "travel_pressure_high_boundary":
        return f"行走压力{travel:.1f}bar接近高压边界，结合当前速度{speed:.2f}m/s避免误判。"
    if boundary_kind == "system_pressure_boundary":
        return f"系统压力{system_p:.1f}bar处于边界区间，需结合驱动阶段判断是否属于正常波动。"
    if boundary_kind == "coolant_boundary":
        return f"冷却液温度处于阈值边界，当前优先结合趋势和场景进行保守决策。"
    if boundary_kind == "surface_temp_boundary":
        return f"表面温度处于边界，当前动作优先以安全观察为主。"
    if boundary_kind == "exhaust_boundary":
        return f"排气温度处于边界，需避免过早进入高风险判定。"
    if boundary_kind == "transition_boundary":
        return f"驱动过渡阶段({scenario})，速度{speed:.2f}m/s尚低，先按过渡态观察。"
    if target_action == "ACCELERATE":
        return f"正常低速边界，速度{speed:.2f}m/s且转速{rpm:.0f}rpm允许提升，执行一次补速。"
    if target_action == "DECELERATE":
        return f"边界风险场景，结合压力/温度接近阈值，执行DECELERATE保守处理。"
    return f"边界条件下保持FORWARD观察，避免把正常波动误判为故障。"



def _sample_target_action(risk_level: str, boundary_kind: str, rng: random.Random) -> str:
    if boundary_kind in {"startup_brake_low_but_normal", "startup_drive_pressure_boundary", "speed_low_boundary", "cruise_target_boundary", "transition_boundary"}:
        return rng.choice(["FORWARD", "ACCELERATE"])
    if boundary_kind in {"brake_boundary", "system_pressure_boundary", "travel_pressure_low_boundary", "coolant_boundary", "surface_temp_boundary", "exhaust_boundary", "rpm_boundary", "travel_pressure_high_boundary"}:
        return rng.choice(["FORWARD", "DECELERATE"])
    if risk_level == "danger":
        return rng.choice(["DECELERATE", "BRAKE"])
    if risk_level == "warning":
        return rng.choice(["FORWARD", "DECELERATE"])
    return rng.choice(["FORWARD", "ACCELERATE"])



def augment_boundary_samples(
    input_csv: Path,
    thresholds_path: Path,
    output_jsonl: Path,
    valid_jsonl: Optional[Path] = None,
    max_per_kind: int = 120,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    rows = _load_rows(input_csv)
    th = _load_thresholds(thresholds_path)
    engine = RiskEngine(thresholds=json.loads(thresholds_path.read_text(encoding="utf-8")))
    ts_cache = [_parse_ts(_safe_str(r.get("timestamp"), "")) for r in rows]
    rng = random.Random(seed)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    candidates: List[Tuple[int, str]] = []

    for idx, row in enumerate(rows):
        current = _build_current(row)
        history = _build_history(idx, rows, ts_cache, th)
        boundary, boundary_kind = _is_boundary_row(current, history, th)
        if not boundary:
            continue

        # Build a risk-engine view in the same shape that runtime uses.
        risk_eval = engine.evaluate(current=current, window={}, history=history)
        scenario = _safe_str(history.get("vehicle_context"), "unknown")
        risk_level = _safe_str(risk_eval.get("overall_level"), "normal")

        # Focus on boundary-adjacent, not fully obvious cases.
        target_action = _sample_target_action(risk_level, boundary_kind, rng)
        if risk_level == "danger":
            target_action = rng.choice(["DECELERATE", "BRAKE"])
        elif risk_level == "warning" and boundary_kind in {"startup_brake_low_but_normal", "startup_drive_pressure_boundary"}:
            target_action = rng.choice(["FORWARD", "ACCELERATE"])

        sample = {
            "instruction": "你是矿车控制决策器，只输出JSON，不要解释。",
            "input": {
                "current": current,
                "history": history,
                "features": {"risk": risk_eval},
            },
            "output": {
                "action": target_action,
                "risk_level": risk_level,
                "reason": "",
                "confidence": 0.92 if risk_level == "normal" else 0.88,
                "suspected_fault": [w.get("tag") for w in risk_eval.get("warnings", []) if isinstance(w, dict) and w.get("tag")],
                "recommended_adjustment": list(dict.fromkeys(risk_eval.get("suggested_actions", []))) or ["continue_observation"],
                "monitor_next": list(dict.fromkeys(risk_eval.get("monitor_next", [])))[:6],
                "warning_tags": [w.get("tag") for w in risk_eval.get("warnings", []) if isinstance(w, dict) and w.get("tag")],
                "policy": "boundary_augmented",
                "decision_source": "boundary_augmented",
                "scenario": scenario,
                "risk_level_raw": risk_level,
                "boundary_kind": boundary_kind,
            },
        }
        sample["output"]["reason"] = _build_reason(sample, boundary_kind, target_action)
        buckets[boundary_kind].append(sample)
        candidates.append((idx, boundary_kind))

    augmented: List[Dict[str, Any]] = []
    for boundary_kind, items in buckets.items():
        if len(items) <= max_per_kind:
            augmented.extend(items)
            continue
        augmented.extend(items[:max_per_kind])

    # De-duplicate by a stable key to avoid overfitting to exact states.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in augmented:
        key = json.dumps({
            "scenario": _safe_str(s.get("output", {}).get("scenario"), ""),
            "boundary_kind": _safe_str(s.get("output", {}).get("boundary_kind"), ""),
            "risk_level": _safe_str(s.get("output", {}).get("risk_level"), ""),
            "action": _safe_str(s.get("output", {}).get("action"), ""),
            "speed_mps": round(_safe_float(s.get("input", {}).get("current", {}).get("speed_mps")), 2),
            "engine_rpm": round(_safe_float(s.get("input", {}).get("current", {}).get("engine_rpm")), 0),
            "warning_tags": tuple(sorted([str(x) for x in s.get("output", {}).get("warning_tags", []) if str(x).strip()])),
        }, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    # Split train/valid.
    split_idx = int(len(deduped) * (1.0 - valid_ratio))
    split_idx = max(1, min(split_idx, len(deduped) - 1)) if len(deduped) > 1 else len(deduped)
    train_rows = deduped[:split_idx]
    valid_rows = deduped[split_idx:] if len(deduped) > 1 else []

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if valid_jsonl is not None:
        valid_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with valid_jsonl.open("w", encoding="utf-8") as f:
            for row in valid_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_csv": str(input_csv),
        "thresholds": str(thresholds_path),
        "output": str(output_jsonl),
        "valid_output": str(valid_jsonl) if valid_jsonl else "",
        "candidates": len(candidates),
        "boundary_counts": {k: len(v) for k, v in sorted(buckets.items(), key=lambda kv: kv[0])},
        "train": len(train_rows),
        "valid": len(valid_rows),
        "total": len(deduped),
    }
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description="Augment boundary-condition samples from real decoded CSV data.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--max-per-kind", type=int, default=120)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = augment_boundary_samples(
        input_csv=args.input_csv,
        thresholds_path=args.thresholds,
        output_jsonl=args.output,
        valid_jsonl=args.valid_output,
        max_per_kind=args.max_per_kind,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
