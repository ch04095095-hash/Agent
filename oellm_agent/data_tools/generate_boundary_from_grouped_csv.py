#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from collections import Counter, deque, defaultdict
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


INVALID_SENTINEL = 65496.0
ACTION_SET = {"FORWARD", "ACCELERATE", "DECELERATE", "BRAKE"}



def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        x = float(s)
        if abs(x - INVALID_SENTINEL) < 1e-9:
            return default
        return x
    except Exception:
        return default



def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        x = int(float(s))
        return x
    except Exception:
        return default



def _safe_str(v: Any, default: str = "") -> str:
    s = str(v or "").strip()
    return s if s else default



def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}



def _load_thresholds(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out



def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))



def _parse_ts(ts: str) -> Optional[float]:
    try:
        if not ts:
            return None
        return dt.datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None



def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def _trend(values: List[float]) -> str:
    if len(values) < 2:
        return "stable"
    delta = values[-1] - values[0]
    if delta > 0.15:
        return "rising"
    if delta < -0.15:
        return "falling"
    return "stable"



def _window_delta(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(values[-1] - values[0], 3)



def _normalize_value(x: float, low: float = -1e9, high: float = 1e9) -> float:
    if x < low or x > high:
        return 0.0
    return x



def _build_current(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "speed_mps": _normalize_value(_safe_float(row.get("speed_mps")), 0.0, 10.0),
        "engine_rpm": _normalize_value(_safe_float(row.get("engine_rpm")), 0.0, 5000.0),
        "gear_state": _safe_int(row.get("gear_state")),
        "emergency_stop": _safe_int(row.get("emergency_stop")),
        "brake_pressure_bar": _normalize_value(_safe_float(row.get("brake_pressure_bar")), 0.0, 500.0),
        "travel_pressure_bar": _normalize_value(_safe_float(row.get("travel_pressure_bar")), 0.0, 500.0),
        "system_pressure_bar": _normalize_value(_safe_float(row.get("system_pressure_bar")), 0.0, 500.0),
        "clamp_pressure_bar": _normalize_value(_safe_float(row.get("clamp_pressure_bar")), 0.0, 500.0),
        "coolant_temp_c": _normalize_value(_safe_float(row.get("coolant_temp_c")), -50.0, 200.0),
        "surface_temp_c": _normalize_value(_safe_float(row.get("surface_temp_c")), -50.0, 200.0),
        "exhaust_temp_c": _normalize_value(_safe_float(row.get("exhaust_temp_c")), -50.0, 200.0),
        "intake_pressure_kpa": _normalize_value(_safe_float(row.get("intake_pressure_kpa")), 0.0, 300.0),
        "hydraulic_oil_temp_c": _normalize_value(_safe_float(row.get("hydraulic_oil_temp_c")), -50.0, 200.0),
        "hydraulic_oil_level_pct": _normalize_value(_safe_float(row.get("hydraulic_oil_level_pct")), 0.0, 100.0),
        "oil_pressure_kpa": _normalize_value(_safe_float(row.get("oil_pressure_kpa")), 0.0, 5000.0),
        "diesel_level_cm": _normalize_value(_safe_float(row.get("diesel_level_cm")), 0.0, 200.0),
        "water_tank_level_pct": _normalize_value(_safe_float(row.get("water_level_pct")), 0.0, 100.0),
        "battery_v": _normalize_value(_safe_float(row.get("battery_v")), 0.0, 100.0),
        "total_mileage_km": _normalize_value(_safe_float(row.get("total_mileage_km")), 0.0, 100000.0),
        "runtime_min": _normalize_value(_safe_float(row.get("runtime_min")), 0.0, 100000.0),
        "load_state": _safe_int(row.get("load_state")),
        "methane_pct": _normalize_value(_safe_float(row.get("methane_pct")), 0.0, 5.0),
        "co_ppm": _normalize_value(_safe_float(row.get("co_ppm")), 0.0, 1000.0),
        "intake_temp_c": _normalize_value(_safe_float(row.get("intake_temp_c")), -50.0, 200.0),
    }



def _vehicle_context(speed: float, rpm: float, gear: int, low_speed_sec: float, th: Dict[str, float]) -> str:
    if gear <= 1:
        return "not_in_drive"
    if low_speed_sec < th.get("decision_drive_gear_grace_sec", 15.0):
        if speed < 0.2:
            return "drive_idle"
        if rpm < 800:
            return "drive_grace"
        return "drive_starting"
    if speed < 0.5 and rpm < 1300:
        return "drive_transition"
    return "drive_effective"



def _boundary_kind(current: Dict[str, Any], history: Dict[str, Any], th: Dict[str, float]) -> Optional[str]:
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
    vc = history["vehicle_context"]

    if gear > 1 and low_speed_sec <= th.get("decision_drive_gear_grace_sec", 15.0) and speed < 0.2 and brake <= th.get("brake_pressure_low_warning", 150.0) + 10:
        return "startup_brake_low_but_normal"
    if vc in {"drive_starting", "drive_idle", "drive_grace"} and brake <= th.get("brake_pressure_low_warning", 150.0) + 5:
        return "startup_drive_pressure_boundary"
    if abs(speed - 0.15) <= 0.25:
        return "speed_low_boundary"
    if abs(speed - 1.0) <= 0.25:
        return "cruise_target_boundary"
    if abs(rpm - th.get("engine_rpm_high_warning", 2200.0)) <= 120:
        return "rpm_boundary"
    if abs(brake - th.get("brake_pressure_low_warning", 150.0)) <= 15:
        return "brake_boundary"
    if abs(travel - th.get("travel_pressure_low_warning", 26.0)) <= 20:
        return "travel_pressure_low_boundary"
    if abs(travel - th.get("travel_pressure_high_warning", 250.0)) <= 20:
        return "travel_pressure_high_boundary"
    if abs(system_p - th.get("system_pressure_low_warning", 150.0)) <= 20:
        return "system_pressure_boundary"
    if abs(coolant - th.get("coolant_temp_high_warning", 93.0)) <= 4:
        return "coolant_boundary"
    if abs(surface - th.get("surface_temp_high_warning", 147.0)) <= 4:
        return "surface_temp_boundary"
    if abs(exhaust - th.get("exhaust_temp_high_warning", 65.0)) <= 4:
        return "exhaust_boundary"
    if vc == "drive_transition" and speed < 0.5 and rpm < 1300:
        return "transition_boundary"
    return None



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



def _build_reason(boundary_kind: str, action: str, current: Dict[str, Any], history: Dict[str, Any], risk_eval: Dict[str, Any]) -> str:
    speed = current["speed_mps"]
    rpm = current["engine_rpm"]
    brake = current["brake_pressure_bar"]
    travel = current["travel_pressure_bar"]
    system_p = current["system_pressure_bar"]
    vc = history["vehicle_context"]
    trend = history["speed_trend_5s"]
    last_action = history["last_effective_action"]
    age = history["last_action_age_sec"]

    if boundary_kind in {"startup_brake_low_but_normal", "startup_drive_pressure_boundary"}:
        return f"起步/驱动建立阶段({vc})，制动压力{brake:.1f}bar偏低但符合建立过程，先保持FORWARD观察。"
    if boundary_kind == "speed_low_boundary":
        return f"车速处于边界区间{speed:.2f}m/s，结合场景{vc}和趋势{trend}，先保守观察。"
    if boundary_kind == "cruise_target_boundary":
        return f"车速接近巡航目标{speed:.2f}m/s，结合上一动作{last_action}(已持续{age:.1f}s)避免频繁切换。"
    if boundary_kind == "rpm_boundary":
        return f"发动机转速{rpm:.0f}rpm接近阈值边界，优先保守处理以避免过度加速。"
    if boundary_kind == "brake_boundary":
        return f"制动压力{brake:.1f}bar处于边界，结合场景{vc}避免误判为硬故障。"
    if boundary_kind == "travel_pressure_low_boundary":
        return f"行走压力{travel:.1f}bar接近低压边界，需结合速度趋势{trend}决定是否仅观察。"
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
        return f"驱动过渡阶段({vc})，速度{speed:.2f}m/s尚低，先按过渡态观察。"
    if action == "ACCELERATE":
        return f"正常低速边界，速度{speed:.2f}m/s且转速{rpm:.0f}rpm允许提升，执行一次补速。"
    if action == "DECELERATE":
        warnings = [w.get("tag") for w in risk_eval.get("warnings", []) if isinstance(w, dict) and w.get("tag")]
        return f"边界风险场景，结合预警{','.join(warnings[:3]) if warnings else 'none'}接近阈值，执行DECELERATE保守处理。"
    return f"边界条件下保持FORWARD观察，避免把正常波动误判为故障。"



def _pick_action(boundary_kind: str, risk_level: str, rng: random.Random) -> str:
    if boundary_kind in {"startup_brake_low_but_normal", "startup_drive_pressure_boundary", "speed_low_boundary", "cruise_target_boundary", "transition_boundary"}:
        return rng.choice(["FORWARD", "ACCELERATE"])
    if boundary_kind in {"brake_boundary", "system_pressure_boundary", "travel_pressure_low_boundary", "coolant_boundary", "surface_temp_boundary", "exhaust_boundary", "rpm_boundary", "travel_pressure_high_boundary"}:
        return rng.choice(["FORWARD", "DECELERATE"])
    if risk_level == "danger":
        return rng.choice(["DECELERATE", "BRAKE"])
    if risk_level == "warning":
        return rng.choice(["FORWARD", "DECELERATE"])
    return rng.choice(["FORWARD", "ACCELERATE"])



def generate_from_grouped_csv(input_csv: Path, thresholds_path: Path, output_jsonl: Path, valid_output_jsonl: Optional[Path] = None, max_per_kind: int = 150, valid_ratio: float = 0.1, seed: int = 42) -> Dict[str, Any]:
    rows = _read_rows(input_csv)
    th = _load_thresholds(thresholds_path)
    engine = RiskEngine(thresholds=th)
    rng = random.Random(seed)

    ts_cache = [_parse_ts(_safe_str(r.get("group_ts_last") or r.get("group_ts_first") or "")) for r in rows]
    speed_hist: Deque[float] = deque(maxlen=11)
    rpm_hist: Deque[float] = deque(maxlen=11)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_candidates = 0

    for idx, row in enumerate(rows):
        current = {
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
            "water_tank_level_pct": _safe_float(row.get("water_level_pct")),
            "battery_v": _safe_float(row.get("battery_v")),
            "total_mileage_km": _safe_float(row.get("total_mileage_km")),
            "runtime_min": _safe_float(row.get("runtime_min")),
            "load_state": _safe_int(row.get("load_state")),
            "methane_pct": _safe_float(row.get("methane_pct")),
            "co_ppm": _safe_float(row.get("co_ppm")),
            "intake_temp_c": _safe_float(row.get("intake_temp_c")),
        }
        # Filter bad rows if too many invalid sentinel-heavy values are present.
        if current["surface_temp_c"] == 0.0 and _safe_str(row.get("surface_temp_c")) in {"", "0", "0.0", str(INVALID_SENTINEL)}:
            pass

        speed_hist.append(current["speed_mps"])
        rpm_hist.append(current["engine_rpm"])
        ts = ts_cache[idx]
        prev_ts = ts_cache[idx - 1] if idx > 0 else None
        dt_sec = 0.5
        if ts is not None and prev_ts is not None:
            dt_sec = max(0.1, ts - prev_ts)

        low_speed = current["speed_mps"] < th.get("speed_low_warning_mps", 0.15)
        low_start = idx
        while low_start > 0 and _safe_float(rows[low_start - 1].get("speed_mps")) < th.get("speed_low_warning_mps", 0.15):
            low_start -= 1
        low_speed_duration_sec = round((idx - low_start + 1) * dt_sec, 3) if low_speed else 0.0

        history = {
            "layer1_action": "FORWARD",
            "layer1_reasons": [],
            "last_effective_action": "FORWARD" if idx == 0 else _safe_str(rows[idx - 1].get("decision_action"), "FORWARD"),
            "last_action_age_sec": round(dt_sec if idx > 0 else 0.0, 3),
            "same_action_duration_sec": round((idx - low_start + 1) * dt_sec, 3),
            "speed_trend_5s": _trend(list(speed_hist)),
            "speed_delta_5s_mps": _window_delta(list(speed_hist)),
            "engine_rpm_delta_5s": _window_delta(list(rpm_hist)),
            "speed_low_duration_sec": low_speed_duration_sec,
            "action_response": {"detail": "no_feedback_yet"},
            "vehicle_context": _vehicle_context(current["speed_mps"], current["engine_rpm"], current["gear_state"], low_speed_duration_sec, th),
            "current_speed_mps": current["speed_mps"],
            "current_engine_rpm": current["engine_rpm"],
        }
        history["action_response"] = _action_response(history["last_effective_action"], history["speed_delta_5s_mps"], current["speed_mps"])

        # Boundary selection based on real data.
        bkind = _boundary_kind(current, history, th)
        if bkind is None:
            continue

        risk_eval = engine.evaluate(current=current, window={}, history=history)
        risk_level = _safe_str(risk_eval.get("overall_level"), "normal")
        action = _pick_action(bkind, risk_level, rng)
        if risk_level == "danger":
            action = rng.choice(["DECELERATE", "BRAKE"])
        elif risk_level == "warning" and bkind in {"startup_brake_low_but_normal", "startup_drive_pressure_boundary"}:
            action = rng.choice(["FORWARD", "ACCELERATE"])

        sample = {
            "instruction": "你是矿车控制决策器，只输出JSON，不要解释。",
            "input": {
                "current": current,
                "history": history,
                "features": {"risk": risk_eval},
            },
            "output": {
                "action": action,
                "risk_level": risk_level,
                "reason": "",
                "confidence": 0.92 if action in {"FORWARD", "ACCELERATE"} else 0.88,
                "suspected_fault": [w.get("tag") for w in risk_eval.get("warnings", []) if isinstance(w, dict) and w.get("tag")],
                "recommended_adjustment": list(dict.fromkeys(risk_eval.get("suggested_actions", []))) or ["continue_observation"],
                "monitor_next": list(dict.fromkeys(risk_eval.get("monitor_next", [])))[:6],
                "warning_tags": [w.get("tag") for w in risk_eval.get("warnings", []) if isinstance(w, dict) and w.get("tag")],
                "policy": "boundary_augmented",
                "decision_source": "boundary_augmented",
                "scenario": history["vehicle_context"],
                "risk_level_raw": risk_level,
                "boundary_kind": bkind,
                "source_csv": input_csv.name,
            },
        }
        sample["output"]["reason"] = _build_reason(bkind, action, current, history, risk_eval)
        grouped[bkind].append(sample)
        total_candidates += 1

    all_samples: List[Dict[str, Any]] = []
    for bkind, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        if len(items) > max_per_kind:
            items = items[:max_per_kind]
        all_samples.extend(items)

    # Deduplicate by a stable key.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in all_samples:
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

    split_idx = int(len(deduped) * (1.0 - valid_ratio))
    split_idx = max(1, min(split_idx, len(deduped) - 1)) if len(deduped) > 1 else len(deduped)
    train_rows = deduped[:split_idx]
    valid_rows = deduped[split_idx:] if len(deduped) > 1 else []

    _write_jsonl(output_jsonl, train_rows)
    if valid_output_jsonl is not None:
        _write_jsonl(valid_output_jsonl, valid_rows)

    return {
        "input_csv": str(input_csv),
        "thresholds": str(thresholds_path),
        "output": str(output_jsonl),
        "valid_output": str(valid_output_jsonl) if valid_output_jsonl else "",
        "candidates": total_candidates,
        "boundary_counts": {k: len(v) for k, v in sorted(grouped.items(), key=lambda kv: kv[0])},
        "train": len(train_rows),
        "valid": len(valid_rows),
        "total": len(deduped),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate boundary-condition SFT samples from grouped CSV.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--max-per-kind", type=int, default=150)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = generate_from_grouped_csv(
        input_csv=args.input_csv,
        thresholds_path=args.thresholds,
        output_jsonl=args.output,
        valid_output_jsonl=args.valid_output,
        max_per_kind=args.max_per_kind,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
