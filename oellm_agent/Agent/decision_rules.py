from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

CRUISE_TARGET_SPEED_MPS = 1.0
SPEED_LOW_WARNING_MPS = 0.15
FAULT_MODE_MIN_SPEED_MPS = SPEED_LOW_WARNING_MPS

LOW_SPEED_TAGS = {"speed_low_persistent", "speed_low_warning", "speed_mps_observed"}
SEVERE_WARNING_TAGS = {
    "travel_pressure_warning",
    "brake_pressure_bar_warning",
    "intake_pressure_kpa_warning",
    "system_pressure_bar_warning",
    "oil_pressure_kpa_warning",
    "coolant_warning",
    "hydraulic_oil_temp_warning",
    "emergency_stop",
    "communication_fault",
}
STARTUP_RECOVER_TAGS = {
    "travel_pressure_warning",
    "intake_pressure_kpa_warning",
    "system_pressure_bar_warning",
}
ALLOWED_ACTIONS = {"HOLD", "ACCELERATE", "DECELERATE", "BRAKE", "EMERGENCY_STOP"}
NON_SPEED_WARNING_TAGS = {
    "travel_pressure_warning",
    "brake_pressure_low",
    "brake_pressure_high",
    "brake_pressure_bar_warning",
    "intake_pressure_kpa_warning",
    "system_pressure_bar_warning",
    "oil_pressure_kpa_warning",
    "coolant_warning",
    "hydraulic_oil_temp_warning",
    "clamp_pressure_low",
    "clamp_pressure_high",
}
MOTION_STATE_PARKING = "parking"
MOTION_STATE_STARTING = "starting"
MOTION_STATE_CONTROL = "control"
MOTION_STATE_UNKNOWN = "unknown"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique_nonempty(items: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in items if x]))


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _normalize_risk_level(value: Any, fallback: str = "normal") -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "l0": "normal",
        "l1": "normal",
        "safe": "normal",
        "stable": "normal",
        "low": "normal",
        "normal": "normal",
        "medium": "warning",
        "warn": "warning",
        "warning": "warning",
        "high": "high_warning",
        "high_warning": "high_warning",
        "danger": "high_warning",
        "critical": "high_warning",
        "emergency": "high_warning",
    }
    return mapping.get(text, fallback if fallback in {"normal", "warning", "high_warning"} else "normal")


def _build_reason_from_warnings(
    warnings: List[Dict[str, Any]],
    active_warnings: List[Dict[str, Any]],
) -> tuple[List[str], List[str], List[str], List[str]]:
    reason_parts: List[str] = []
    suspected_fault: List[str] = []
    monitor_next: List[str] = []
    suggested_actions: List[str] = []

    for warning in warnings:
        tag = str(warning.get("tag", "") or "").strip()
        if not tag:
            continue
        suspected_fault.append(tag)
        suggested_actions.extend(_coerce_string_list(warning.get("suggested_actions", [])))
        monitor_next.extend(_coerce_string_list(warning.get("monitor_next", [])))
        value = warning.get("value", warning.get("score"))
        unit = str(warning.get("unit", "") or "")
        if value is not None:
            reason_parts.append(f"风险 {tag}={value} {unit}".strip())
        reason = str(warning.get("reason", "") or "").strip()
        if reason:
            reason_parts.append(reason)

    for warning in active_warnings:
        tag = str(warning.get("tag", "") or "").strip()
        if not tag:
            continue
        source = str(warning.get("source", "") or "").strip()
        if source != "risk_engine":
            suspected_fault.append(tag)
        monitor_next.append(tag)
        if warning.get("value") is None or source == "risk_engine":
            continue
        reason_parts.append(f"窗口观测 {tag}={warning.get('value')} {warning.get('unit', '')}".strip())

    return reason_parts, suspected_fault, monitor_next, suggested_actions


def _derive_motion_state(vehicle_context: str, current_speed_mps: float, gear_state: int) -> str:
    if gear_state in {1, 2}:
        return MOTION_STATE_PARKING
    if vehicle_context in {"drive_grace", "drive_idle", "drive_starting"}:
        return MOTION_STATE_STARTING
    if vehicle_context == "control":
        return MOTION_STATE_CONTROL
    if current_speed_mps < 0.3:
        return MOTION_STATE_STARTING if gear_state in {3, 4} else MOTION_STATE_UNKNOWN
    return MOTION_STATE_CONTROL if gear_state in {3, 4} else MOTION_STATE_UNKNOWN


def _select_action(
    overall_level: str,
    overall_score: float,
    warning_tags: List[str],
    active_warnings: List[Dict[str, Any]],
    last_action: str,
    stable_after_brake_sec: float,
    brake_age_sec: float,
    current_speed_mps: float,
    vehicle_context: str,
    reason_parts: List[str],
    control_state: bool,
    control_entered_by_byte2: bool,
    l1_alarm: bool,
    gear_state: int,
) -> str:
    if l1_alarm:
        reason_parts.insert(0, "检测到L1报警，直接急停")
        return "EMERGENCY_STOP"

    motion_state = _derive_motion_state(vehicle_context, current_speed_mps, gear_state)
    primary_warning_tag = warning_tags[0] if warning_tags else ""
    low_speed_signal = current_speed_mps < SPEED_LOW_WARNING_MPS and (
        any(tag in LOW_SPEED_TAGS for tag in warning_tags)
        or any(str(w.get("tag", "") or "").strip() in LOW_SPEED_TAGS for w in active_warnings)
    )

    if motion_state == MOTION_STATE_PARKING:
        reason_parts.insert(0, "parking态仅允许HOLD，不下发运动控制")
        return "HOLD"

    if motion_state == MOTION_STATE_STARTING:
        if low_speed_signal or current_speed_mps < 0.3:
            reason_parts.insert(0, "起步阶段，优先加速建立车速")
            return "ACCELERATE"
        if overall_level == "high_warning":
            reason_parts.insert(0, f"起步阶段高风险(score={overall_score:.3f})，执行减速")
            return "DECELERATE"
        if overall_level == "warning":
            reason_parts.insert(0, f"起步阶段预警(score={overall_score:.3f})，保持HOLD")
            return "HOLD"

    if motion_state == MOTION_STATE_CONTROL:
        if low_speed_signal:
            if primary_warning_tag in LOW_SPEED_TAGS:
                reason_parts.insert(0, f"低速持续，尝试恢复速度({primary_warning_tag or 'low_speed'})")
            else:
                reason_parts.insert(0, f"低速伴随{primary_warning_tag or 'warning'}，先缓行恢复")
            return "ACCELERATE"

        if overall_level == "high_warning":
            reason_parts.insert(0, f"当前风险={overall_level}(score={overall_score:.3f})")
            return "DECELERATE"

        if overall_level == "warning":
            startup_tag = next((tag for tag in warning_tags if tag in STARTUP_RECOVER_TAGS), "")
            if current_speed_mps < 0.8 and startup_tag:
                reason_parts.insert(0, f"低速阶段(speed={current_speed_mps:.2f})，对{startup_tag}继续观察")
                return "HOLD"
            if current_speed_mps < 0.6:
                reason_parts.insert(0, f"低速缓行阶段(speed={current_speed_mps:.2f})，保持HOLD")
                return "HOLD"
            reason_parts.insert(0, f"当前风险={overall_level}(score={overall_score:.3f})")
            return "DECELERATE"

        if stable_after_brake_sec > 10 and last_action == "BRAKE":
            reason_parts.insert(0, "制动后已稳定，可尝试恢复")
            return "ACCELERATE"
        if 0 < brake_age_sec < 10:
            reason_parts.insert(0, "刚进入制动/减速区，先保持保守")
            return "DECELERATE"

        reason_parts.insert(0, "当前风险正常，保持HOLD")
        return "HOLD"

    reason_parts.insert(0, "默认保持保守")
    return "HOLD"


def _apply_control_guardrails(
    action: str,
    overall_level: str,
    warning_tags: List[str],
    last_action: str,
    stable_after_brake_sec: float,
    current_speed_mps: float,
    vehicle_context: str,
    reason_parts: List[str],
    control_state: bool,
    control_entered_by_byte2: bool,
    l1_alarm: bool,
    last_action_age_sec: float,
) -> str:
    if l1_alarm:
        reason_parts.append("L1报警强制急停")
        return "EMERGENCY_STOP"
    if control_state and not control_entered_by_byte2:
        reason_parts.append("未满足Byte2进入条件，维持制动")
        return "BRAKE"
    if vehicle_context == "parking" and action != "HOLD":
        reason_parts.append("parking态禁止运动控制，强制HOLD")
        return "HOLD"
    # A1: accelerate cooldown (2s)
    if action == "ACCELERATE" and last_action == "ACCELERATE" and 0.0 <= float(last_action_age_sec or 0.0) < 2.0:
        reason_parts.append("加速冷却中(<2s)，转为HOLD")
        action = "HOLD"

    if overall_level == "warning" and action == "ACCELERATE" and not any(tag in LOW_SPEED_TAGS for tag in warning_tags):
        if vehicle_context == "starting" and current_speed_mps < 0.6:
            reason_parts.append("起步/驱动建立阶段预警，保持HOLD")
            action = "HOLD"
        else:
            reason_parts.append("预警态禁止加速")
            action = "DECELERATE"
    if last_action == "BRAKE" and stable_after_brake_sec < 5 and action in {"HOLD", "ACCELERATE"} and current_speed_mps >= 0.8:
        reason_parts.append("制动后稳定不足，先保持")
        action = "DECELERATE"
    # F1: opposite-action switch must pass HOLD for 1s
    if (last_action == "ACCELERATE" and action == "DECELERATE") or (last_action == "DECELERATE" and action == "ACCELERATE"):
        if 0.0 <= float(last_action_age_sec or 0.0) < 5.0:
            reason_parts.append("加减速反向切换需经HOLD(5s)")
            action = "HOLD"

    # D2: stop decelerating at very low speed
    if action == "DECELERATE" and current_speed_mps < 0.2:
        reason_parts.append("低速(<0.2m/s)停止减速，转为HOLD")
        action = "HOLD"

    if overall_level == "warning" and action == "DECELERATE" and current_speed_mps < 0.8:
        if current_speed_mps < 0.15:
            reason_parts.append("当前已近静止，停止继续减速，转为HOLD")
            action = "HOLD"
        elif vehicle_context == "starting":
            reason_parts.append("起步/驱动建立阶段不过度减速，转为HOLD")
            action = "HOLD"
        else:
            brake_pressure_only = warning_tags and all(tag in {"brake_pressure_low", "brake_pressure_high", "brake_pressure_bar_warning"} for tag in warning_tags)
            if brake_pressure_only and current_speed_mps < 0.6:
                reason_parts.append("制动压力仅轻微预警且车速已很低，停止继续减速，转为HOLD")
                action = "HOLD"
            elif any(tag in NON_SPEED_WARNING_TAGS for tag in warning_tags):
                reason_parts.append("存在非速度异常，近静止也保持减速/等待恢复")
            else:
                reason_parts.append("近静止时由减速切到HOLD")
                action = "HOLD"
    return action


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("no JSON object found in LLM response")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON response is not an object")
    return obj


def _build_llm_prompt(sensor_payload: Dict[str, Any]) -> str:
    current = _as_dict(sensor_payload.get("current"))
    history = _as_dict(sensor_payload.get("history"))
    features = _as_dict(sensor_payload.get("features"))
    risk = _as_dict(features.get("risk"))
    context = _as_dict(sensor_payload.get("context"))
    baseline_decision = _as_dict(sensor_payload.get("baseline_decision"))
    compact = {
        "schema_version": str(sensor_payload.get("schema_version", "agent_payload_v1") or "agent_payload_v1"),
        "vehicle_id": str(sensor_payload.get("vehicle_id", "") or ""),
        "timestamp": str(sensor_payload.get("timestamp", "") or ""),
        "current": {
            "speed_mps": current.get("speed_mps", history.get("current_speed_mps", 0.0)),
            "engine_rpm": current.get("engine_rpm", history.get("current_engine_rpm", 0.0)),
            "gear_state": current.get("gear_state", history.get("gear_state", 0)),
            "emergency_stop": current.get("emergency_stop", 0),
            "travel_pressure_bar": current.get("travel_pressure_bar", 0.0),
            "brake_pressure_bar": current.get("brake_pressure_bar", 0.0),
        },
        "context": {
            "vehicle_context": context.get("vehicle_context", history.get("vehicle_context", "")),
            "drive_mode_active": context.get("drive_mode_active", history.get("agent_control_enabled", False)),
            "drive_gear_age_sec": context.get("drive_gear_age_sec", history.get("drive_gear_age_sec", 0.0)),
            "drive_effective_duration_sec": context.get("drive_effective_duration_sec", history.get("drive_effective_duration_sec", 0.0)),
        },
        "history": {
            "layer1_action": history.get("layer1_action", ""),
            "layer1_reasons": history.get("layer1_reasons", []),
            "last_effective_action": history.get("last_effective_action", ""),
            "last_action_age_sec": history.get("last_action_age_sec", 0.0),
            "same_action_duration_sec": history.get("same_action_duration_sec", 0.0),
            "speed_trend_5s": history.get("speed_trend_5s", ""),
            "speed_delta_5s_mps": history.get("speed_delta_5s_mps", history.get("delta_speed_5s", 0.0)),
            "speed_low_duration_sec": history.get("speed_low_duration_sec", 0.0),
            "engine_rpm_delta_5s": history.get("engine_rpm_delta_5s", history.get("delta_engine_rpm_5s", 0.0)),
            "action_response": _as_dict(history.get("action_response")).get("detail", history.get("action_response", "")),
        },
        "risk": {
            "overall_level": risk.get("overall_level", "normal"),
            "overall_score": risk.get("overall_score", 0.0),
            "suggested_actions": risk.get("suggested_actions", []),
            "warnings": risk.get("warnings", []),
        },
        "baseline_decision": {
            "action": baseline_decision.get("action", "FORWARD"),
            "reason": baseline_decision.get("reason", ""),
            "confidence": baseline_decision.get("confidence", 0.0),
            "decision_source": baseline_decision.get("decision_source", "rule_engine"),
            "policy": baseline_decision.get("policy", "rule_first"),
            "rule_summary": baseline_decision.get("rule_summary", {}),
        },
    }
    return (
        "你是车辆控制决策器，负责在规则基线之上做有限修正。"
        "只输出一个合法JSON对象，不要输出markdown、代码块、注释或解释。"
        "必须且只能包含以下字段：action,reason,confidence。"
        "所有键名必须使用双引号，不能写成 reason:、confidence: 这种形式。"
        "action只能取HOLD/ACCELERATE/DECELERATE/BRAKE/EMERGENCY_STOP。"
        "如果基线动作为EMERGENCY_STOP，除非输入中存在明确恢复条件，否则必须保持EMERGENCY_STOP。"
        "如果基线动作为FORWARD且车辆处于gear_state 3/4并且speed_mps接近0，优先考虑ACCELERATE。"
        "当risk为normal且baseline_decision为FORWARD时，除非你能从上下文明确判断需要起步/恢复，否则不要把动作改成减速或刹车。"
        "confidence必须是0到1之间的数字。"
        "reason必须使用中文短句，简洁说明你是否接受或修正了基线动作。"
        "输入=" + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    )


def _apply_hard_constraints(decision: Dict[str, Any], sensor_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    adjusted = dict(decision or {})
    notes: List[str] = []
    history = _as_dict(sensor_payload.get("history"))
    layer1_action = str(history.get("layer1_action", "") or "").upper()
    layer1_reasons = _coerce_string_list(history.get("layer1_reasons", []))
    last_action = str(history.get("last_effective_action", "") or "").upper()
    last_action_age_sec = float(history.get("last_action_age_sec", 0.0) or 0.0)

    action = str(adjusted.get("action", "") or "").upper()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid action from LLM: {action}")

    if layer1_action == "EMERGENCY_STOP" and action != "EMERGENCY_STOP":
        action = "EMERGENCY_STOP"
        notes.append(f"L1硬保护覆盖动作({','.join(layer1_reasons) or 'unknown'})")

    if action == "ACCELERATE" and last_action == "ACCELERATE" and last_action_age_sec < 4.0:
        action = "HOLD"
        notes.append("加速冷却约束生效：4秒内最多一次ACCELERATE")

    adjusted["action"] = action
    return adjusted, notes
