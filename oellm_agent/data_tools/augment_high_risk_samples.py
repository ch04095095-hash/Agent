#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ACTIONS = {"FORWARD", "ACCELERATE", "DECELERATE", "BRAKE"}
RISK_LEVELS = {"normal", "warning", "high_warning", "danger"}
HIGH_RISK_TARGET_BUCKETS = {"danger::DECELERATE", "danger::BRAKE"}



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



def _safe_str(v: Any, default: str = "") -> str:
    s = str(v or "").strip()
    return s if s else default



def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def _bucket(sample: Dict[str, Any]) -> str:
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}
    risk = _safe_str(out.get("risk_level"), "unknown")
    action = _safe_str(out.get("action"), "unknown")
    scenario = _safe_str(out.get("scenario"), _safe_str(history.get("vehicle_context"), "unknown"))
    if scenario == "not_in_drive":
        return "not_in_drive"
    return f"{risk}::{action}"



def _extract_tags(sample: Dict[str, Any]) -> Tuple[str, ...]:
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    tags = out.get("warning_tags", [])
    if not isinstance(tags, list):
        tags = []
    return tuple(sorted(_safe_str(t) for t in tags if _safe_str(t)))



def _extract_reason_tokens(sample: Dict[str, Any]) -> List[str]:
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    reason = _safe_str(out.get("reason"), "")
    if not reason:
        return []
    return [p for p in reason.replace("；", " ").replace(",", " ").split() if p]



def _rebuild_reason(sample: Dict[str, Any], target_action: str, target_risk: str, tags: List[str]) -> str:
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}
    scenario = _safe_str(sample.get("output", {}).get("scenario"), _safe_str(history.get("vehicle_context"), "unknown"))
    speed = _safe_float(current.get("speed_mps"))
    rpm = _safe_float(current.get("engine_rpm"))
    last_action = _safe_str(history.get("last_effective_action"), "")
    age = _safe_float(history.get("last_action_age_sec"))
    trend = _safe_str(history.get("speed_trend_5s"), "stable")
    action_resp = _safe_str((history.get("action_response") or {}).get("detail", ""), "")

    top_tags = tags[:4]
    if target_action == "BRAKE":
        return f"高风险场景({target_risk})，触发{','.join(top_tags) if top_tags else '严重预警'}，立即BRAKE优先保护。"
    if target_action == "DECELERATE":
        if top_tags:
            return f"高风险场景({target_risk})，主要预警{','.join(top_tags)}，结合场景{scenario}与速度趋势{trend}，执行DECELERATE。"
        return f"高风险场景({target_risk})，结合场景{scenario}与速度趋势{trend}，执行DECELERATE。"
    if target_action == "FORWARD":
        return f"当前高风险但处于观测/冷却窗口，结合场景{scenario}、上一动作{last_action}(已持续{age:.1f}s)和响应{action_resp}，保持FORWARD。"
    return f"当前高风险场景({target_risk})，结合速度{speed:.2f}m/s和转速{rpm:.0f}rpm，执行ACCELERATE。"



def _mutate_to_high_risk(sample: Dict[str, Any], target_action: str, target_risk: str, seed: int) -> Dict[str, Any]:
    rnd = random.Random(seed)
    s = deepcopy(sample)
    inp = s.setdefault("input", {}) if isinstance(s.get("input"), dict) else {}
    out = s.setdefault("output", {}) if isinstance(s.get("output"), dict) else {}
    current = inp.setdefault("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.setdefault("history", {}) if isinstance(inp.get("history"), dict) else {}
    features = inp.setdefault("features", {}) if isinstance(inp.get("features"), dict) else {}
    risk = features.setdefault("risk", {}) if isinstance(features.get("risk"), dict) else {}

    # Force a high-risk, control-relevant context.
    speed = _safe_float(current.get("speed_mps"))
    rpm = _safe_float(current.get("engine_rpm"))
    travel_pressure = _safe_float(current.get("travel_pressure_bar"))
    brake_pressure = _safe_float(current.get("brake_pressure_bar"))
    system_pressure = _safe_float(current.get("system_pressure_bar"))
    coolant = _safe_float(current.get("coolant_temp_c"))
    surface = _safe_float(current.get("surface_temp_c"))
    exhaust = _safe_float(current.get("exhaust_temp_c"))

    # Scenario-specific coercion.
    if target_action == "BRAKE":
        current["emergency_stop"] = 1
        history["layer1_action"] = "BRAKE"
        history["last_effective_action"] = "BRAKE"
        history["action_response"] = {"detail": "command_issued"}
        history["vehicle_context"] = "drive_effective"
        current["speed_mps"] = max(speed, 0.3 + rnd.random() * 0.8)
        current["engine_rpm"] = max(rpm, 900 + rnd.random() * 600)
        risk["overall_level"] = "danger"
        risk["warnings"] = [
            {"level": "high_warning", "tag": "emergency_stop", "value": 1, "unit": "", "reason": "急停触发", "score": 1.0, "suggested_actions": ["BRAKE"], "monitor_next": ["speed_mps"], "alarm_code": 0, "source": "augmented"},
            {"level": "high_warning", "tag": "brake_pressure_low", "value": min(brake_pressure, 40.0) if brake_pressure else 40.0, "unit": "bar", "reason": "制动压力过低", "score": 0.95, "suggested_actions": ["BRAKE"], "monitor_next": ["brake_pressure_bar", "speed_mps"], "alarm_code": 0, "source": "augmented"},
        ]
    else:
        # DECELERATE-focused high-risk state.
        history["layer1_action"] = "FORWARD"
        history["last_effective_action"] = "ACCELERATE" if _safe_str(history.get("last_effective_action"), "") != "BRAKE" else "BRAKE"
        history["action_response"] = {"detail": rnd.choice(["speed_not_up", "command_issued", "pending"])}
        history["vehicle_context"] = rnd.choice(["drive_effective", "drive_transition", "drive_starting"])
        current["speed_mps"] = max(speed, rnd.choice([0.2, 0.5, 0.8, 1.2, 1.8]))
        current["engine_rpm"] = max(rpm, rnd.choice([1200, 1500, 1800, 2100]))
        # Create a more plausible serious-risk profile by stressing pressures/temps.
        if rnd.random() < 0.5:
            current["brake_pressure_bar"] = min(brake_pressure if brake_pressure else 120.0, rnd.choice([55.0, 60.0, 75.0, 90.0]))
        if rnd.random() < 0.5:
            current["travel_pressure_bar"] = min(travel_pressure if travel_pressure else 120.0, rnd.choice([20.0, 24.0, 28.0, 35.0]))
        if rnd.random() < 0.5:
            current["system_pressure_bar"] = min(system_pressure if system_pressure else 130.0, rnd.choice([70.0, 75.0, 85.0, 95.0]))
        if rnd.random() < 0.5:
            current["coolant_temp_c"] = max(coolant, rnd.choice([95.0, 98.0, 102.0, 110.0]))
        if rnd.random() < 0.3:
            current["surface_temp_c"] = max(surface, rnd.choice([150.0, 155.0, 160.0]))
        if rnd.random() < 0.3:
            current["exhaust_temp_c"] = max(exhaust, rnd.choice([69.0, 75.0, 82.0]))

        history["speed_trend_5s"] = rnd.choice(["falling", "stable"])
        history["speed_delta_5s_mps"] = round(rnd.choice([-0.8, -0.5, -0.3, 0.0]), 3)
        history["engine_rpm_delta_5s"] = round(rnd.choice([-120, -40, 0, 60]), 3)
        history["same_action_duration_sec"] = round(max(_safe_float(history.get("same_action_duration_sec"), 0.0), rnd.choice([3.0, 4.0, 5.5, 7.0])), 3)
        history["last_action_age_sec"] = round(max(_safe_float(history.get("last_action_age_sec"), 0.0), rnd.choice([0.5, 1.0, 2.0, 3.0])), 3)

        tags: List[Dict[str, Any]] = []
        if current["coolant_temp_c"] >= 95.0:
            tags.append({"level": "high_warning", "tag": "coolant_warning", "value": current["coolant_temp_c"], "unit": "C", "reason": "冷却液温度高", "score": 1.0, "suggested_actions": ["DECELERATE"], "monitor_next": ["coolant_temp_c"], "alarm_code": 0, "source": "augmented"})
        if current["brake_pressure_bar"] <= 60.0:
            tags.append({"level": "high_warning", "tag": "brake_pressure_low", "value": current["brake_pressure_bar"], "unit": "bar", "reason": "制动压力过低", "score": 1.0, "suggested_actions": ["DECELERATE"], "monitor_next": ["brake_pressure_bar"], "alarm_code": 0, "source": "augmented"})
        if current["travel_pressure_bar"] <= 25.0:
            tags.append({"level": "high_warning", "tag": "travel_pressure_warning", "value": current["travel_pressure_bar"], "unit": "bar", "reason": "行走压力过低", "score": 1.0, "suggested_actions": ["DECELERATE"], "monitor_next": ["travel_pressure_bar"], "alarm_code": 0, "source": "augmented"})
        if current["system_pressure_bar"] <= 80.0:
            tags.append({"level": "high_warning", "tag": "system_pressure_low", "value": current["system_pressure_bar"], "unit": "bar", "reason": "系统压力低", "score": 1.0, "suggested_actions": ["DECELERATE"], "monitor_next": ["system_pressure_bar"], "alarm_code": 0, "source": "augmented"})
        if current["engine_rpm"] >= 2200:
            tags.append({"level": "high_warning", "tag": "rpm_high", "value": current["engine_rpm"], "unit": "rpm", "reason": "发动机转速高", "score": 0.9, "suggested_actions": ["DECELERATE"], "monitor_next": ["engine_rpm"], "alarm_code": 0, "source": "augmented"})
        if current["speed_mps"] >= 2.2:
            tags.append({"level": "high_warning", "tag": "speed_high", "value": current["speed_mps"], "unit": "m/s", "reason": "车速高", "score": 0.9, "suggested_actions": ["DECELERATE"], "monitor_next": ["speed_mps"], "alarm_code": 0, "source": "augmented"})

        if not tags:
            tags.append({"level": "high_warning", "tag": "speed_low_persistent", "value": current["speed_mps"], "unit": "m/s", "reason": "行驶状态车速持续偏低", "score": 0.8, "suggested_actions": ["DECELERATE"], "monitor_next": ["speed_mps", "engine_rpm"], "alarm_code": 0, "source": "augmented"})

        risk["overall_level"] = "danger"
        risk["warnings"] = tags

    # Final output target.
    out["action"] = target_action
    out["risk_level"] = "danger"
    out["risk_level_raw"] = "danger"
    out["scenario"] = history.get("vehicle_context", "drive_effective")
    out["warning_tags"] = [t["tag"] for t in risk.get("warnings", []) if isinstance(t, dict) and t.get("tag")]
    out["confidence"] = 0.98 if target_action == "BRAKE" else 0.95
    out["policy"] = "augmented_high_risk"
    out["decision_source"] = "augmented_high_risk"
    out["reason"] = _rebuild_reason(s, target_action, "danger", out.get("warning_tags", []))
    out["suspected_fault"] = list(dict.fromkeys(out.get("warning_tags", [])))
    out["monitor_next"] = list(dict.fromkeys([t["monitor_next"][0] if isinstance(t.get("monitor_next"), list) and t.get("monitor_next") else t["tag"] for t in risk.get("warnings", []) if isinstance(t, dict) and t.get("tag")]))[:6]
    out["recommended_adjustment"] = ["maintain_safe_speed", "monitor_pressure_and_temperature"] if target_action == "DECELERATE" else ["immediate_stop"]

    return s



def augment_dataset(input_path: Path, output_path: Path, target_danger_decelerate: int = 600, target_danger_brake: int = 200, seed: int = 42) -> Dict[str, Any]:
    samples = _read_jsonl(input_path)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_bucket[_bucket(s)].append(s)

    rnd = random.Random(seed)
    augmented: List[Dict[str, Any]] = []
    stats = Counter()

    # Keep all existing samples.
    augmented.extend(samples)
    stats["base"] = len(samples)

    # Generate additional dangerous decelerate samples.
    base_candidates = by_bucket.get("warning::DECELERATE", []) + by_bucket.get("normal_low::FORWARD", []) + by_bucket.get("normal_low::ACCELERATE", []) + by_bucket.get("danger::DECELERATE", [])
    if not base_candidates:
        base_candidates = samples

    def make_n(target_bucket: str, count: int, action: str) -> None:
        if count <= 0:
            return
        for i in range(count):
            src = deepcopy(rnd.choice(base_candidates))
            aug = _mutate_to_high_risk(src, action, "danger", seed + i * 17 + (0 if action == "DECELERATE" else 10000))
            bucket = _bucket(aug)
            if bucket != target_bucket:
                # If mutation didn't land exactly, force the output bucket/action.
                aug["output"]["action"] = action
                aug["output"]["risk_level"] = "danger"
                aug["output"]["risk_level_raw"] = "danger"
                aug["output"]["scenario"] = _safe_str(aug["output"].get("scenario"), "drive_effective")
                if action == "BRAKE":
                    aug["output"]["reason"] = "高风险/硬保护场景，立即BRAKE优先保护。"
                else:
                    aug["output"]["reason"] = "高风险场景，执行DECELERATE降低风险。"
            augmented.append(aug)
            stats[target_bucket] += 1

    # Determine current counts to top up.
    current_counts = Counter(_bucket(s) for s in samples)
    need_decelerate = max(0, target_danger_decelerate - current_counts.get("danger::DECELERATE", 0))
    need_brake = max(0, target_danger_brake - current_counts.get("danger::BRAKE", 0))

    make_n("danger::DECELERATE", need_decelerate, "DECELERATE")
    make_n("danger::BRAKE", need_brake, "BRAKE")

    # Deduplicate after augmentation.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in augmented:
        key = json.dumps({
            "scenario": _safe_str(s.get("output", {}).get("scenario"), ""),
            "risk_level": _safe_str(s.get("output", {}).get("risk_level"), ""),
            "action": _safe_str(s.get("output", {}).get("action"), ""),
            "speed_mps": round(_safe_float(s.get("input", {}).get("current", {}).get("speed_mps")), 2),
            "engine_rpm": round(_safe_float(s.get("input", {}).get("current", {}).get("engine_rpm")), 0),
            "warning_tags": tuple(sorted(_extract := tuple(_safe_str(t) for t in (s.get("output", {}).get("warning_tags", []) if isinstance(s.get("output", {}).get("warning_tags", []), list) else [])))),
        }, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)

    _write_jsonl(output_path, deduped)

    out_counts = Counter(_bucket(s) for s in deduped)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "base": len(samples),
        "augmented_total": len(deduped),
        "target_danger_decelerate": target_danger_decelerate,
        "target_danger_brake": target_danger_brake,
        "current_counts": dict(current_counts),
        "final_counts": dict(out_counts),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Augment high-risk samples for mine-truck control SFT.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-danger-decelerate", type=int, default=600)
    parser.add_argument("--target-danger-brake", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = augment_dataset(
        input_path=args.input,
        output_path=args.output,
        target_danger_decelerate=args.target_danger_decelerate,
        target_danger_brake=args.target_danger_brake,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
