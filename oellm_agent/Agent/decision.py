from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict

from .decision_rules import (
    _apply_control_guardrails,
    _build_llm_prompt,
    _build_reason_from_warnings,
    _normalize_risk_level,
    _select_action,
)
from .tools import Toolset


_DECISION_CACHE_MAX = 256
_DECISION_CACHE_TTL_SEC = 3.0
_DECISION_CACHE_LOCK = threading.Lock()
_DECISION_CACHE: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()


def _extract_json_candidate(text: str) -> str:
    s = str(text or "")
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return ""
    return s[start : end + 1]


def _repair_json_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    s = s.replace("\r", "")
    s = re.sub(r'"\s*([A-Za-z_][A-Za-z0-9_ ]*?)\s*:\s*"', lambda m: f'"{m.group(1).strip()}": "', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    if s.count("{") > s.count("}"):
        s += "}"
    return s


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _normalize_action(value: Any) -> str:
    action = str(value or "HOLD").strip().upper()
    if action not in {"HOLD", "ACCELERATE", "DECELERATE", "BRAKE", "EMERGENCY_STOP"}:
        return "HOLD"
    return action


def _normalize_decision_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj or {})
    normalized = {
        "action": _normalize_action(out.get("action", "HOLD")),
        "reason": str(out.get("reason", "") or "").strip(),
        "confidence": 0.0,
    }
    try:
        normalized["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0.0) or 0.0)))
    except Exception:
        normalized["confidence"] = 0.0
    return normalized


def _build_cache_key(sensor_payload: Dict[str, Any], llm_only: bool) -> str:
    current = sensor_payload.get("current", {}) if isinstance(sensor_payload.get("current"), dict) else {}
    context = sensor_payload.get("context", {}) if isinstance(sensor_payload.get("context"), dict) else {}
    features = sensor_payload.get("features", {}) if isinstance(sensor_payload.get("features"), dict) else {}
    risk = features.get("risk", {}) if isinstance(features.get("risk"), dict) else {}
    history = sensor_payload.get("history", {}) if isinstance(sensor_payload.get("history"), dict) else {}

    compact = {
        "llm_only": bool(llm_only),
        "vehicle_id": str(sensor_payload.get("vehicle_id", "") or ""),
        "timestamp_bucket": str(sensor_payload.get("timestamp", "") or "")[:6],
        "current": {
            "speed_mps": round(float(current.get("speed_mps", 0.0) or 0.0), 2),
            "engine_rpm": round(float(current.get("engine_rpm", 0.0) or 0.0), 0),
            "gear_state": int(current.get("gear_state", 0) or 0),
            "emergency_stop": int(current.get("emergency_stop", 0) or 0),
            "travel_pressure_bar": round(float(current.get("travel_pressure_bar", 0.0) or 0.0), 1),
            "brake_pressure_bar": round(float(current.get("brake_pressure_bar", 0.0) or 0.0), 1),
        },
        "context": {
            "vehicle_context": str(context.get("vehicle_context", "") or ""),
            "drive_mode_active": bool(context.get("drive_mode_active", False)),
            "drive_gear_age_sec": round(float(context.get("drive_gear_age_sec", 0.0) or 0.0), 1),
            "drive_effective_duration_sec": round(float(context.get("drive_effective_duration_sec", 0.0) or 0.0), 1),
        },
        "history": {
            "layer1_action": str(history.get("layer1_action", "") or ""),
            "last_effective_action": str(history.get("last_effective_action", "") or ""),
            "speed_delta_5s_mps": round(float(history.get("speed_delta_5s_mps", 0.0) or 0.0), 2),
            "engine_rpm_delta_5s": round(float(history.get("engine_rpm_delta_5s", 0.0) or 0.0), 1),
            "speed_low_duration_sec": round(float(history.get("speed_low_duration_sec", 0.0) or 0.0), 1),
        },
        "risk": {
            "overall_level": str(risk.get("overall_level", "normal") or "normal"),
            "overall_score": round(float(risk.get("overall_score", 0.0) or 0.0), 3),
            "warning_tags": [
                str(w.get("tag", "") or "")
                for w in (risk.get("warnings", []) or [])
                if isinstance(w, dict) and str(w.get("tag", "") or "")
            ][:3],
        },
    }
    raw = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _repair_and_parse_llm_output(raw: str) -> tuple[Dict[str, Any] | None, str, bool]:
    text = str(raw or "")
    candidate = _extract_json_candidate(text)
    if not candidate:
        return None, text, False
    repaired = _repair_json_text(candidate)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return _normalize_decision_obj(obj), repaired, repaired != candidate
    except Exception:
        pass
    return None, text, False


def _cache_get(cache_key: str) -> Dict[str, Any] | None:
    now = time.time()
    with _DECISION_CACHE_LOCK:
        item = _DECISION_CACHE.get(cache_key)
        if item is None:
            return None
        created_at, value = item
        if (now - created_at) > _DECISION_CACHE_TTL_SEC:
            _DECISION_CACHE.pop(cache_key, None)
            return None
        _DECISION_CACHE.move_to_end(cache_key)
        return dict(value)


def _cache_put(cache_key: str, decision: Dict[str, Any]) -> None:
    with _DECISION_CACHE_LOCK:
        _DECISION_CACHE[cache_key] = (time.time(), dict(decision))
        _DECISION_CACHE.move_to_end(cache_key)
        while len(_DECISION_CACHE) > _DECISION_CACHE_MAX:
            _DECISION_CACHE.popitem(last=False)


def _parse_rule_decision(sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
    features = sensor_payload.get("features", {}) if isinstance(sensor_payload.get("features"), dict) else {}
    risk = features.get("risk", {}) if isinstance(features.get("risk"), dict) else {}
    history = sensor_payload.get("history", {}) if isinstance(sensor_payload.get("history"), dict) else {}
    current = sensor_payload.get("current", {}) if isinstance(sensor_payload.get("current"), dict) else {}
    control = _as_dict(sensor_payload.get("control_state"))
    control_state = bool(control.get("enabled", False))
    control_entered_by_byte2 = bool(control.get("entered_by_byte2", False))
    l1_alarm = bool(control.get("l1_alarm", False) or risk.get("l1_alarm", False))

    warnings = risk.get("warnings", []) if isinstance(risk.get("warnings", []), list) else []
    active_warnings = [w for w in warnings if isinstance(w, dict)]
    warning_tags = [str(w.get("tag", "") or "").strip() for w in active_warnings if str(w.get("tag", "") or "").strip()]

    reason_parts, suspected_fault, monitor_next, suggested_actions = _build_reason_from_warnings(active_warnings, active_warnings)
    overall_level = _normalize_risk_level(risk.get("overall_level", "normal"), fallback="normal")
    try:
        overall_score = max(0.0, float(risk.get("overall_score", 0.0) or 0.0))
    except Exception:
        overall_score = 0.0
    last_action = str(history.get("last_effective_action", "") or "").upper()
    stable_after_brake_sec = float(history.get("stable_after_brake_sec", 0.0) or 0.0)
    brake_age_sec = float(history.get("brake_age_sec", 0.0) or 0.0)
    current_speed_mps = float(current.get("speed_mps", 0.0) or 0.0)
    vehicle_context = str(history.get("vehicle_context", sensor_payload.get("context", {}).get("vehicle_context", "") or "") or "")

    gear_state = int(current.get("gear_state", 0) or 0)
    action = _select_action(
        overall_level=overall_level,
        overall_score=overall_score,
        warning_tags=warning_tags,
        active_warnings=active_warnings,
        last_action=last_action,
        stable_after_brake_sec=stable_after_brake_sec,
        brake_age_sec=brake_age_sec,
        current_speed_mps=current_speed_mps,
        vehicle_context=vehicle_context,
        reason_parts=reason_parts,
        control_state=control_state,
        control_entered_by_byte2=control_entered_by_byte2,
        l1_alarm=l1_alarm,
        gear_state=gear_state,
    )
    action = _apply_control_guardrails(
        action=action,
        overall_level=overall_level,
        warning_tags=warning_tags,
        last_action=last_action,
        stable_after_brake_sec=stable_after_brake_sec,
        current_speed_mps=current_speed_mps,
        vehicle_context=vehicle_context,
        reason_parts=reason_parts,
        control_state=control_state,
        control_entered_by_byte2=control_entered_by_byte2,
        l1_alarm=l1_alarm,
        last_action_age_sec=float(history.get("last_action_age_sec", 0.0) or 0.0),
    )
    reason = "；".join([p for p in reason_parts if p])
    if not reason:
        reason = "规则决策"
    return {
        "action": action,
        "reason": reason,
        "decision_source": "rule_engine",
        "policy": "rule_baseline",
        "llm_only": False,
        "cache_hit": False,
        "llm_skipped": True,
        "rule_summary": {
            "overall_level": overall_level,
            "overall_score": round(overall_score, 3),
            "warning_tags": warning_tags,
            "suspected_fault": suspected_fault,
            "monitor_next": monitor_next,
            "suggested_actions": suggested_actions,
            "vehicle_context": vehicle_context,
            "current_speed_mps": round(current_speed_mps, 3),
        },
    }


def _refine_with_llm(
    _session: Any,
    sensor_payload: Dict[str, Any],
    baseline_decision: Dict[str, Any],
) -> Dict[str, Any]:
    if _session is None:
        raise RuntimeError("LLM session is unavailable")

    prompt = _build_llm_prompt({
        **dict(sensor_payload or {}),
        "baseline_decision": dict(baseline_decision or {}),
    })
    raw = _session.ask(prompt)
    parsed, exposed_raw, repaired = _repair_and_parse_llm_output(raw)

    if parsed is None:
        return {
            **dict(baseline_decision or {}),
            "decision_source": "llm_raw",
            "policy": "llm_refine_fallback",
            "raw_output": exposed_raw,
            "llm_refine_failed": True,
        }

    out = _normalize_decision_obj(parsed)
    refined = dict(baseline_decision or {})
    refined.update(out)
    refined["decision_source"] = "llm_refine"
    refined["policy"] = "rule_baseline_llm_refine"
    refined["raw_output"] = exposed_raw
    refined["llm_refine_failed"] = False
    if repaired:
        refined["_repair"] = {"repaired": True}

    baseline_action = str((baseline_decision or {}).get("action", "HOLD") or "HOLD").upper()
    refined_action = str(refined.get("action", baseline_action) or baseline_action).upper()
    high_risk = str(_as_dict(_as_dict(sensor_payload.get("features")).get("risk")).get("overall_level", "normal") or "normal").lower() in {"warning", "high_warning", "danger"}
    if baseline_action == "BRAKE" and refined_action != "BRAKE":
        refined["action"] = "BRAKE"
        refined["reason"] = (str(refined.get("reason", "") or "") + "；" if refined.get("reason") else "") + "基线硬保护保持制动"
    elif not high_risk and baseline_action != refined_action and refined_action in {"DECELERATE", "BRAKE"}:
        refined["action"] = baseline_action
        refined["reason"] = (str(refined.get("reason", "") or "") + "；" if refined.get("reason") else "") + "正常态保留基线动作"

    refined["confidence"] = max(float(baseline_decision.get("confidence", 0.4) or 0.4), float(refined.get("confidence", 0.0) or 0.0))
    return refined


def decide_from_sensor_llm(
    _session: Any,
    _tools: Toolset,
    sensor_payload: Dict[str, Any],
    llm_only: bool = False,
) -> Dict[str, Any]:
    # Rule-only mode: LLM path is disabled intentionally.
    baseline = _parse_rule_decision(sensor_payload)
    baseline["decision_source"] = "rule_engine"
    baseline["policy"] = "rule_only"
    baseline["llm_only"] = False
    baseline["llm_skipped"] = True
    baseline["cache_hit"] = False
    return baseline


def decide_from_sensor(_session: Any, _tools: Toolset, sensor_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible entry point (rule-only)."""
    return decide_from_sensor_llm(_session, _tools, sensor_payload, llm_only=False)
