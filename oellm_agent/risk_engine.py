from __future__ import annotations

from typing import Any, Dict, List, Tuple


ALLOWED_RISK_LEVELS = {"normal", "warning", "danger"}


class RiskEngine:
    """Rule-first risk engine for mine truck state assessment."""

    ACTION_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
        "remote": {"actions": ["reduce_speed"], "next": ["speed_kmh", "engine_rpm"]},
        "operator": {"actions": ["notify_operator", "现场处置"], "next": ["coolant_temp_c", "hydraulic_oil_temp_c", "travel_pressure_bar", "brake_pressure_bar", "system_pressure_bar", "water_tank_level_pct", "diesel_level_cm"]},
        "stop": {"actions": ["stop_and_inspect"], "next": ["emergency_stop", "can_heartbeat_ok"]},
        "recovery": {"actions": ["resume_operation"], "next": ["coolant_temp_c", "system_pressure_bar", "brake_pressure_bar"]},
        "default": {"actions": ["monitor_trend"], "next": []},
    }

    def __init__(self, thresholds: Dict[str, float]):
        self.th = thresholds

    def classify_warning(self, current: Dict[str, Any]) -> Tuple[str, str]:
        if int(current.get("emergency_stop", 0) or 0) in (1, 2, 3):
            return "danger", "emergency_stop"
        if int(current.get("can_heartbeat_ok", 1) or 1) == 0:
            return "danger", "communication_fault"

        methane = float(current.get("methane_pct", current.get("methane_pctlel", 0.0)) or 0.0)
        if methane >= self.th["methane_alarm"]:
            return "warning", "methane_warning"
        if float(current.get("co_ppm", 0.0) or 0.0) >= self.th.get("co_warn", 24.0):
            return "warning", "co_warning"
        if float(current.get("coolant_temp_c", 0.0) or 0.0) > self.th["coolant_temp_warn"]:
            return "warning", "coolant_warning"
        if float(current.get("surface_temp_c", 0.0) or 0.0) > float(self.th.get("surface_temp_warn", 147.0)):
            return "warning", "surface_temp_warning"
        if float(current.get("exhaust_temp_c", 0.0) or 0.0) > self.th["exhaust_temp_warn"]:
            return "warning", "exhaust_warning"
        if float(current.get("hydraulic_oil_temp_c", 0.0) or 0.0) > self.th["hydraulic_oil_temp_warn"]:
            return "warning", "hydraulic_oil_temp_warning"

        travel = float(current.get("travel_pressure_bar", current.get("walking_pressure_bar", 0.0)) or 0.0)
        if travel < self.th["travel_pressure_warn_low"] or travel > self.th["travel_pressure_warn_high"]:
            return "warning", "travel_pressure_warning"
        clamp = float(current.get("clamp_pressure_bar", 0.0) or 0.0)
        if clamp < self.th["clamp_pressure_warn_low"]:
            return "warning", "clamp_pressure_low"
        if clamp > self.th["clamp_pressure_warn_high"]:
            return "warning", "clamp_pressure_high"
        if float(current.get("brake_pressure_bar", 0.0) or 0.0) < self.th["brake_pressure_warn_low"]:
            return "warning", "brake_pressure_low"
        if float(current.get("make_up_oil_pressure_bar", 0.0) or 0.0) < self.th["make_up_oil_pressure_warn_low"]:
            return "warning", "make_up_oil_pressure_low"
        if float(current.get("fire_system_pressure_bar", 999.0) or 999.0) < float(self.th.get("fire_system_pressure_warn_low", 10.0)):
            return "warning", "fire_system_pressure_low"
        if float(current.get("intake_pressure_kpa", 0.0) or 0.0) < self.th["intake_pressure_warn_low"]:
            return "warning", "intake_pressure_low"
        if float(current.get("intake_temp_c", 0.0) or 0.0) > float(self.th.get("intake_temp_high", 60.0)):
            return "warning", "intake_temp_high"
        if float(current.get("oil_pressure_kpa", 0.0) or 0.0) < self.th["oil_pressure_warn_kpa"]:
            return "warning", "oil_pressure_low"
        if float(current.get("system_pressure_bar", 0.0) or 0.0) < self.th["system_pressure_warn_low"]:
            return "warning", "system_pressure_low"

        if float(current.get("hydraulic_oil_level_pct", 100.0) or 100.0) < self.th["hydraulic_oil_level_low"]:
            return "warning", "hydraulic_oil_level_low"
        if float(current.get("diesel_level_cm", 999.0) or 999.0) < self.th["diesel_level_low"]:
            return "warning", "diesel_level_low"
        if float(current.get("water_tank_level_pct", current.get("water_level_pct", 999.0)) or 999.0) < self.th["water_level_warn_low"]:
            return "warning", "water_tank_low"

        rpm = float(current.get("engine_rpm", 0.0) or 0.0)
        speed_kmh = float(current.get("speed_kmh", 0.0) or 0.0)
        if rpm > self.th["rpm_warn_high"]:
            return "warning", "rpm_high"
        if speed_kmh > self.th["speed_warn_high_kmh"]:
            return "warning", "speed_high"
        if speed_kmh < self.th["speed_warn_low_kmh"]:
            return "warning", "speed_low_persistent"

        return "normal", ""

    def derive_alarm_code(self, current: Dict[str, Any], warning_tag: str) -> int:
        if not warning_tag:
            return 0

        # 按用户指定的映射表
        if warning_tag == "hydraulic_oil_temp_warning":
            return 101 if float(current.get("hydraulic_oil_temp_c", 0.0) or 0.0) > self.th["hydraulic_oil_temp_high"] else 0
        if warning_tag == "hydraulic_oil_level_low":
            return 102
        if warning_tag == "coolant_warning":
            return 106 if float(current.get("coolant_temp_c", 0.0) or 0.0) > self.th["coolant_temp_high"] else 0
        if warning_tag == "surface_temp_warning":
            return 107 if float(current.get("surface_temp_c", 0.0) or 0.0) > float(self.th.get("surface_temp_high", 150.0)) else 0
        if warning_tag == "exhaust_warning":
            return 108 if float(current.get("exhaust_temp_c", 0.0) or 0.0) > self.th["exhaust_temp_high"] else 0
        if warning_tag == "methane_warning":
            return 109 if float(current.get("methane_pct", current.get("methane_pctlel", 0.0)) or 0.0) >= self.th["methane_stop"] else 0
        if warning_tag == "speed_high":
            return 110 if float(current.get("speed_kmh", 0.0) or 0.0) > self.th["speed_alarm_high_kmh"] else 0
        if warning_tag == "diesel_level_low":
            return 111 if float(current.get("diesel_level_cm", 999.0) or 999.0) < self.th["diesel_level_low"] else 0
        if warning_tag == "water_tank_low":
            return 112 if float(current.get("water_tank_level_pct", current.get("water_level_pct", 999.0)) or 999.0) < self.th["water_level_low"] else 0
        if warning_tag == "oil_pressure_low":
            return 113 if float(current.get("oil_pressure_kpa", 0.0) or 0.0) < self.th["oil_pressure_low_kpa"] else 0
        if warning_tag == "intake_pressure_low":
            return 114 if float(current.get("intake_pressure_kpa", 0.0) or 0.0) < float(self.th.get("intake_pressure_low", self.th.get("intake_pressure_min", 90.0))) else 0
        if warning_tag == "co_warning":
            return 130 if float(current.get("co_ppm", 0.0) or 0.0) >= self.th.get("co_alarm", 24.0) else 0
        if warning_tag == "travel_pressure_warning":
            v = float(current.get("travel_pressure_bar", current.get("walking_pressure_bar", 0.0)) or 0.0)
            return 117 if (v < self.th["travel_pressure_warn_low"] or v > self.th["travel_pressure_high"]) else 0
        if warning_tag == "brake_pressure_low":
            return 118 if float(current.get("brake_pressure_bar", 0.0) or 0.0) < self.th["brake_pressure_low"] else 0
        if warning_tag == "clamp_pressure_low":
            return 119 if float(current.get("clamp_pressure_bar", 0.0) or 0.0) < self.th["clamp_pressure_low"] else 0
        if warning_tag == "system_pressure_low":
            return 120 if float(current.get("system_pressure_bar", 0.0) or 0.0) < self.th["system_pressure_low"] else 0
        if warning_tag == "make_up_oil_pressure_low":
            return 121 if float(current.get("make_up_oil_pressure_bar", 0.0) or 0.0) < self.th["make_up_oil_pressure_low"] else 0
        if warning_tag == "fire_system_pressure_low":
            return 122 if float(current.get("fire_system_pressure_bar", 999.0) or 999.0) < float(self.th.get("fire_system_pressure_low", 8.0)) else 0
        if warning_tag == "intake_temp_high":
            return 135 if float(current.get("intake_temp_c", 0.0) or 0.0) > float(self.th.get("intake_temp_high", 60.0)) else 0

        return 0

    def _suggest_actions(self, warning_tag: str, current: Dict[str, Any], risk_level: str) -> Tuple[List[str], List[str], str]:
        if risk_level == "danger":
            return self.ACTION_TEMPLATES["stop"]["actions"], self.ACTION_TEMPLATES["stop"]["next"], "硬底线触发，立即停机检查"
        if warning_tag in {"methane_warning", "co_warning", "coolant_warning", "surface_temp_warning", "exhaust_warning", "hydraulic_oil_temp_warning", "travel_pressure_warning", "brake_pressure_low", "clamp_pressure_low", "system_pressure_low", "make_up_oil_pressure_low", "fire_system_pressure_low", "water_tank_low", "diesel_level_low", "hydraulic_oil_level_low", "intake_pressure_low", "intake_temp_high", "oil_pressure_low", "rpm_high", "speed_high", "speed_low_persistent"}:
            return self.ACTION_TEMPLATES["remote"]["actions"], self.ACTION_TEMPLATES["remote"]["next"], "预警期优先远程降速，现场整改同步跟进"
        if warning_tag in {"communication_fault", "emergency_stop"}:
            return self.ACTION_TEMPLATES["stop"]["actions"], self.ACTION_TEMPLATES["stop"]["next"], "通信或急停异常，立即停机检查"
        if warning_tag == "":
            return self.ACTION_TEMPLATES["default"]["actions"], self.ACTION_TEMPLATES["default"]["next"], "当前无预警，持续监测"
        return self.ACTION_TEMPLATES["remote"]["actions"], self.ACTION_TEMPLATES["remote"]["next"], "预警期先执行远程降速，其他措施由现场处置"

    def evaluate(self, current: Dict[str, Any], window: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
        risk_level, warning_tag = self.classify_warning(current)
        alarm_code = self.derive_alarm_code(current, warning_tag)
        suggested_actions, monitor_next, action_reason = self._suggest_actions(warning_tag, current, risk_level)

        primary_event = "normal"
        severity = "normal"
        recommended_action = "MOVE"
        recommended_adjustments: List[str] = []
        reason = "窗口状态正常"
        secondary_events: List[str] = []

        coolant_val = float(current.get("coolant_temp_c", 0.0) or 0.0)
        water_val = float(current.get("water_tank_level_pct", current.get("water_level_pct", 100.0)) or 100.0)
        methane_val = float(current.get("methane_pct", current.get("methane_pctlel", 0.0)) or 0.0)
        brake_val = float(current.get("brake_pressure_bar", 0.0) or 0.0)
        system_val = float(current.get("system_pressure_bar", 0.0) or 0.0)
        hydro_val = float(current.get("hydraulic_oil_temp_c", 0.0) or 0.0)
        speed_val = float(current.get("speed_kmh", 0.0) or 0.0)
        rpm_val = float(current.get("engine_rpm", 0.0) or 0.0)

        coolant_meaning = str(window.get("signals", {}).get("coolant_temp_c", {}).get("meaning", ""))
        water_meaning = str(window.get("signals", {}).get("water_tank_level_pct", {}).get("meaning", window.get("signals", {}).get("water_level_pct", {}).get("meaning", "")))
        methane_meaning = str(window.get("signals", {}).get("methane_pct", {}).get("meaning", ""))
        brake_meaning = str(window.get("signals", {}).get("brake_pressure_bar", {}).get("meaning", ""))
        system_meaning = str(window.get("signals", {}).get("system_pressure_bar", {}).get("meaning", ""))
        hydro_meaning = str(window.get("signals", {}).get("hydraulic_oil_temp_c", {}).get("meaning", ""))

        last_action = str(history.get("last_hard_action", history.get("last_action", "")) or "").upper()
        stable_after_break_sec = float(history.get("stable_after_break_sec", 0) or 0)
        break_age_sec = float(history.get("break_age_sec", 0) or 0)

        hard_events: List[str] = []
        if int(current.get("emergency_stop", 0) or 0) in (1, 2, 3):
            hard_events.append("emergency_stop")
        if int(current.get("can_heartbeat_ok", 1) or 1) == 0:
            hard_events.append("heartbeat_lost")
        if methane_val >= self.th["methane_stop"]:
            hard_events.append("methane_over_stop")
        if coolant_val >= self.th["coolant_temp_high"]:
            hard_events.append("coolant_over_stop")
        if float(current.get("surface_temp_c", 0) or 0) >= self.th["surface_temp_high"]:
            hard_events.append("surface_over_stop")
        if float(current.get("exhaust_temp_c", 0) or 0) >= self.th["exhaust_temp_high"]:
            hard_events.append("exhaust_over_stop")
        intake_val = float(current.get("intake_pressure_kpa", 0) or 0)
        intake_alarm_low = float(self.th.get("intake_pressure_low", self.th.get("intake_pressure_min", 90.0)))
        if intake_val < intake_alarm_low:
            hard_events.append("intake_pressure_low")
        if water_val < self.th["water_level_low"]:
            hard_events.append("water_level_low")
        if float(current.get("oil_pressure_kpa", 0) or 0) < self.th["oil_pressure_low_kpa"]:
            hard_events.append("oil_pressure_low")
        if float(current.get("diesel_level_cm", 999) or 999) < self.th["diesel_level_low"]:
            hard_events.append("diesel_level_low")
        if brake_val < self.th["brake_pressure_low"]:
            hard_events.append("brake_pressure_low")
        travel_val = float(current.get("travel_pressure_bar", current.get("walking_pressure_bar", 0)) or 0)
        if travel_val < self.th["travel_pressure_warn_low"] or travel_val > self.th["travel_pressure_high"]:
            hard_events.append("travel_pressure_out_of_range")
        if system_val < self.th["system_pressure_low"]:
            hard_events.append("system_pressure_low")
        clamp_val = float(current.get("clamp_pressure_bar", 0.0) or 0.0)
        if clamp_val < self.th["clamp_pressure_low"] or clamp_val > self.th["clamp_pressure_high"]:
            hard_events.append("clamp_pressure_out_of_range")
        make_up_val = float(current.get("make_up_oil_pressure_bar", 0.0) or 0.0)
        if make_up_val < self.th["make_up_oil_pressure_low"]:
            hard_events.append("make_up_oil_pressure_low")
        if hydro_val > self.th["hydraulic_oil_temp_high"]:
            hard_events.append("hydraulic_oil_temp_high")
        if float(current.get("hydraulic_oil_level_pct", 100.0) or 100.0) < self.th["hydraulic_oil_level_low"]:
            hard_events.append("hydraulic_oil_level_low")

        if hard_events:
            primary_event = hard_events[0]
            secondary_events = hard_events[1:]
            severity = "danger"
            recommended_action = "BRAKE"
            reason = f"硬底线触发: {','.join(hard_events)}"
        elif last_action == "BRAKE" and (stable_after_break_sec >= 5 or break_age_sec >= 5) and coolant_val < 93 and "正常" in coolant_meaning:
            primary_event = "ready_to_move"
            severity = "info"
            recommended_action = "MOVE"
            recommended_adjustments = ["resume_operation"]
            reason = "停机后状态已恢复并稳定，可放行"
        elif "异常" in coolant_meaning and ("水位低" in water_meaning or water_val < 32):
            primary_event = "coolant_high_and_low_level"
            severity = "warning"
            recommended_action = "HOLD"
            recommended_adjustments = ["reduce_speed", "add_coolant", "inspect_leak"]
            reason = "冷却水温异常且液位偏低"
        elif "异常" in coolant_meaning and coolant_val >= 95:
            primary_event = "coolant_warning_rising"
            severity = "warning"
            recommended_action = "HOLD"
            recommended_adjustments = ["reduce_speed", "monitor_coolant"]
            reason = "冷却水温接近报警并需持续关注"
        elif "较高" in methane_meaning or "过高" in methane_meaning or methane_val >= 0.5:
            primary_event = "methane_near_alarm"
            severity = "warning"
            recommended_action = "HOLD"
            recommended_adjustments = ["ventilate", "monitor_gas"]
            reason = "甲烷浓度接近或进入异常区间"
        elif "较低" in brake_meaning or brake_val < self.th["brake_pressure_warn_low"]:
            primary_event = "brake_pressure_warning"
            severity = "warning"
            recommended_action = "HOLD"
            recommended_adjustments = ["inspect_brake_system"]
            reason = "制动压力偏低"
        elif "异常" in system_meaning or system_val < self.th["system_pressure_warn_low"]:
            primary_event = "system_pressure_warning"
            severity = "warning"
            recommended_action = "HOLD"
            recommended_adjustments = ["inspect_pressure_source"]
            reason = "系统压力偏低或异常"
        elif "较高" in hydro_meaning or hydro_val >= 67:
            primary_event = "hydraulic_temp_near_alarm"
            severity = "warning"
            recommended_action = "HOLD"
            recommended_adjustments = ["check_hydraulics", "reduce_load"]
            reason = "液压油温接近报警值"
        elif last_action == "BRAKE" and stable_after_break_sec > 0:
            primary_event = "recovering"
            severity = "info"
            recommended_action = "HOLD"
            recommended_adjustments = ["continue_monitoring"]
            reason = "处于恢复观察期"

        if speed_val > 26 or rpm_val > 2600:
            secondary_events.append("high_speed_or_rpm")
            if primary_event == "normal":
                primary_event = "overload_risk"
                severity = "warning"
                recommended_action = "HOLD"
                recommended_adjustments = ["reduce_speed", "reduce_rpm"]
                reason = "车速或转速偏高"

        semantic_summary = [m for m in [coolant_meaning, water_meaning, methane_meaning, brake_meaning, system_meaning, hydro_meaning] if m and m != "正常"]

        return {
            "warning_tag": warning_tag,
            "alarm_code": alarm_code,
            "risk_level": risk_level,
            "risk_score": 0.95 if risk_level == "danger" else (0.7 if risk_level == "warning" else 0.1),
            "deviation_score": 0.95 if risk_level == "danger" else (0.6 if risk_level == "warning" else 0.1),
            "suggested_actions": suggested_actions,
            "monitor_next": monitor_next,
            "action_reason": action_reason,
            "event_state": {
                "primary_event": primary_event,
                "secondary_events": secondary_events,
                "severity": severity,
                "recommended_action": recommended_action,
                "recommended_adjustments": recommended_adjustments,
                "reason": reason,
                "confidence_hint": 0.5 if primary_event == "normal" else 0.8,
                "signals": {
                    "coolant_temp_c": {"value": coolant_val, "meaning": coolant_meaning},
                    "water_tank_level_pct": {"value": water_val, "meaning": water_meaning},
                    "methane_pct": {"value": methane_val, "meaning": methane_meaning},
                    "brake_pressure_bar": {"value": brake_val, "meaning": brake_meaning},
                    "system_pressure_bar": {"value": system_val, "meaning": system_meaning},
                    "hydraulic_oil_temp_c": {"value": hydro_val, "meaning": hydro_meaning},
                    "speed_kmh": {"value": speed_val, "meaning": str(window.get("signals", {}).get("speed_kmh", {}).get("meaning", ""))},
                    "engine_rpm": {"value": rpm_val, "meaning": str(window.get("signals", {}).get("engine_rpm", {}).get("meaning", ""))},
                },
                "semantic_summary": semantic_summary,
                "history": {
                    "last_hard_action": last_action,
                    "stable_after_break_sec": stable_after_break_sec,
                    "break_age_sec": break_age_sec,
                },
            },
        }
