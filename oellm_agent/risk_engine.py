from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from oellm_agent.vehicle_context import build_vehicle_context

ALLOWED_RISK_LEVELS = {"normal", "warning", "high_warning"}


class RiskEngine:
    """Rule-first multi-risk engine for mine truck state assessment."""

    ACTION_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
        "reduce_speed": {"actions": ["reduce_speed"], "next": ["speed_mps", "engine_rpm"]},
        "increase_speed": {"actions": ["increase_speed"], "next": ["speed_mps", "engine_rpm", "travel_pressure_bar", "system_pressure_bar"]},
        "temperature_control": {"actions": ["reduce_speed", "monitor_cooling_trend"], "next": ["coolant_temp_c", "hydraulic_oil_temp_c", "exhaust_temp_c", "surface_temp_c", "engine_rpm", "speed_mps"]},
        "onsite": {"actions": ["notify_operator", "现场处置"], "next": ["water_tank_level_pct", "diesel_level_cm", "hydraulic_oil_level_pct", "oil_pressure_kpa", "brake_pressure_bar", "system_pressure_bar", "travel_pressure_bar", "clamp_pressure_bar"]},
        "diagnose_low_speed": {"actions": ["attempt_accelerate", "monitor_response", "notify_operator_if_no_response"], "next": ["speed_mps", "engine_rpm", "travel_pressure_bar", "system_pressure_bar", "brake_pressure_bar"]},
        "default": {"actions": ["monitor_trend"], "next": []},
    }

    def __init__(self, thresholds: Dict[str, float]):
        self.th = thresholds

    @staticmethod
    def _risk_rank(level: str) -> int:
        return {"normal": 0, "warning": 1, "high_warning": 2}.get(str(level or "normal"), 0)

    @staticmethod
    def _score_from_ratio(ratio: float) -> float:
        return max(0.0, ratio)

    @classmethod
    def _score_high(cls, value: float, warn: float, stop: float) -> float:
        if stop <= warn:
            return 1.0
        ratio = (value - warn) / (stop - warn)
        return cls._score_from_ratio(ratio)

    @classmethod
    def _score_low(cls, value: float, warn: float, stop: float) -> float:
        if warn <= stop:
            return 1.0
        ratio = (warn - value) / (warn - stop)
        return cls._score_from_ratio(ratio)

    @classmethod
    def _score_range(cls, value: float, low_warn: float, low_stop: float, high_warn: float, high_stop: float) -> Optional[float]:
        if value < low_warn:
            return cls._score_low(value, low_warn, low_stop)
        if value > high_warn:
            return cls._score_high(value, high_warn, high_stop)
        return None

    @staticmethod
    def overall_level(warnings: List[Dict[str, Any]]) -> str:
        if any(str(w.get("level", "")) == "high_warning" for w in warnings):
            return "high_warning"
        if any(str(w.get("level", "")) == "warning" for w in warnings):
            return "warning"
        return "normal"

    @staticmethod
    def overall_score(warnings: List[Dict[str, Any]]) -> float:
        if not warnings:
            return 0.0
        scores = [max(0.0, float(w.get("score", 0.0) or 0.0)) for w in warnings]
        if not scores:
            return 0.0
        combined_safe_prob = 1.0
        for score in scores:
            combined_safe_prob *= max(0.0, 1.0 - score)
        return round(1.0 - combined_safe_prob, 3)

    def classify_warnings(self, current: Dict[str, Any]) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        flags = build_vehicle_context(current)

        def add(level: str, tag: str, value: Any = None, unit: str = "", reason: str = "", score: Optional[float] = None) -> None:
            if score is None:
                score = 0.0
            score = float(score)
            level = self._classify_warning_level(tag, score)
            actions, _monitor_next, action_reason = self._suggest_actions(tag, current, level)
            warnings.append({
                "level": level,
                "tag": tag,
                "value": value,
                "unit": unit,
                "score": round(score, 3),
                "suggested_actions": actions,
                "reason": reason or action_reason,
                "source": "risk_engine",
            })

        coolant = float(current.get("coolant_temp_c", 0.0) or 0.0)
        if coolant > self.th["coolant_temp_high_warning"]:
            add("warning", "coolant_warning", coolant, "℃", "冷却液温度预警", self._score_high(coolant, self.th["coolant_temp_high_warning"], self.th["coolant_temp_high_stop"]))
        surface = float(current.get("surface_temp_c", 0.0) or 0.0)
        if surface > float(self.th.get("surface_temp_high_warning", 147.0)):
            add("warning", "surface_temp_warning", surface, "℃", "表面温度预警", self._score_high(surface, float(self.th.get("surface_temp_high_warning", 147.0)), float(self.th.get("surface_temp_high_stop", 150.0))))
        exhaust = float(current.get("exhaust_temp_c", 0.0) or 0.0)
        if exhaust > self.th["exhaust_temp_high_warning"]:
            add("warning", "exhaust_warning", exhaust, "℃", "排气温度预警", self._score_high(exhaust, self.th["exhaust_temp_high_warning"], self.th["exhaust_temp_high_stop"]))
        hydraulic_temp = float(current.get("hydraulic_oil_temp_c", 0.0) or 0.0)
        if hydraulic_temp > self.th["hydraulic_oil_temp_high_warning"]:
            add("warning", "hydraulic_oil_temp_warning", hydraulic_temp, "℃", "液压油温预警", self._score_high(hydraulic_temp, self.th["hydraulic_oil_temp_high_warning"], self.th["hydraulic_oil_temp_high_stop"]))

        travel = float(current.get("travel_pressure_bar", current.get("walking_pressure_bar", 0.0)) or 0.0)
        travel_score = self._score_range(travel, self.th["travel_pressure_low_warning"], self.th["travel_pressure_low_stop"], self.th["travel_pressure_high_warning"], self.th["travel_pressure_high_stop"])
        if travel_score is not None and (travel > self.th["travel_pressure_high_warning"] or flags["active_motion"]):
            add("warning", "travel_pressure_warning", travel, "bar", "行走压力预警", travel_score)
        clamp = float(current.get("clamp_pressure_bar", 0.0) or 0.0)
        if clamp < self.th["clamp_pressure_low_warning"]:
            add("warning", "clamp_pressure_low", clamp, "bar", "夹紧压力低", self._score_low(clamp, self.th["clamp_pressure_low_warning"], self.th["clamp_pressure_low_stop"]))
        if clamp > self.th["clamp_pressure_high_warning"]:
            add("warning", "clamp_pressure_high", clamp, "bar", "夹紧压力高", self._score_high(clamp, self.th["clamp_pressure_high_warning"], self.th["clamp_pressure_high_stop"]))
        brake = float(current.get("brake_pressure_bar", 0.0) or 0.0)
        brake_low_decision_threshold = min(float(self.th["brake_pressure_low_warning"]), 145.0)
        if brake < brake_low_decision_threshold and flags["active_motion"]:
            add("warning", "brake_pressure_low", brake, "bar", "制动压力低", self._score_low(brake, brake_low_decision_threshold, self.th["brake_pressure_low_stop"]))
        if brake > self.th["brake_pressure_high_warning"]:
            add("warning", "brake_pressure_high", brake, "bar", "制动压力高", self._score_high(brake, self.th["brake_pressure_high_warning"], self.th["brake_pressure_high_stop"]))
        make_up = float(current.get("make_up_oil_pressure_bar", 0.0) or 0.0)
        if make_up < self.th["make_up_oil_pressure_low_warning"]:
            add("warning", "make_up_oil_pressure_low", make_up, "bar", "补油压力低", self._score_low(make_up, self.th["make_up_oil_pressure_low_warning"], self.th["make_up_oil_pressure_low_stop"]))
        intake_p = float(current.get("intake_pressure_kpa", 0.0) or 0.0)
        if intake_p < self.th["intake_pressure_low_warning"]:
            add("warning", "intake_pressure_low", intake_p, "kPa", "进气压力低", self._score_low(intake_p, self.th["intake_pressure_low_warning"], self.th["intake_pressure_low_stop"]))
        intake_t = float(current.get("intake_temp_c", 0.0) or 0.0)
        intake_t_warn = float(self.th.get("intake_temp_high_warning", self.th.get("intake_temp_high_normal", 60.0)))
        intake_t_stop = float(self.th.get("intake_temp_high_stop", intake_t_warn + 10.0))
        if intake_t > intake_t_warn:
            add("warning", "intake_temp_high", intake_t, "℃", "进气温度高", self._score_high(intake_t, intake_t_warn, intake_t_stop))
        oil_p = float(current.get("oil_pressure_kpa", 0.0) or 0.0)
        if oil_p < self.th["oil_pressure_low_warning_kpa"]:
            add("warning", "oil_pressure_low", oil_p, "kPa", "机油压力低", self._score_low(oil_p, self.th["oil_pressure_low_warning_kpa"], self.th["oil_pressure_low_stop_kpa"]))
        system_p = float(current.get("system_pressure_bar", 0.0) or 0.0)
        system_low_decision_threshold = min(float(self.th["system_pressure_low_warning"]), 145.0)
        if system_p < system_low_decision_threshold and flags["active_motion"]:
            add("warning", "system_pressure_low", system_p, "bar", "系统压力低", self._score_low(system_p, system_low_decision_threshold, self.th["system_pressure_low_stop"]))

        hydraulic_level = float(current.get("hydraulic_oil_level_pct", 100.0) or 100.0)
        if hydraulic_level < self.th["hydraulic_oil_level_low_warning"]:
            add("warning", "hydraulic_oil_level_low", hydraulic_level, "%", "液压油液位低", self._score_low(hydraulic_level, self.th["hydraulic_oil_level_low_warning"], self.th["hydraulic_oil_level_low_stop"]))
        diesel_level = float(current.get("diesel_level_cm", 999.0) or 999.0)
        if diesel_level < self.th["diesel_level_low_warning_cm"]:
            add("warning", "diesel_level_low", diesel_level, "cm", "柴油液位低", self._score_low(diesel_level, self.th["diesel_level_low_warning_cm"], self.th["diesel_level_low_stop_cm"]))
        water_level = float(current.get("water_tank_level_pct", current.get("water_level_pct", 999.0)) or 999.0)
        if water_level < self.th["water_tank_level_low_warning"]:
            add("warning", "water_tank_low", water_level, "%", "水箱液位低", self._score_low(water_level, self.th["water_tank_level_low_warning"], self.th["water_tank_level_low_stop"]))

        rpm = float(current.get("engine_rpm", 0.0) or 0.0)
        speed_mps = float(current.get("speed_mps", 0.0) or 0.0)
        if rpm > self.th["engine_rpm_high_warning"]:
            add("warning", "rpm_high", rpm, "rpm", "发动机转速高", self._score_high(rpm, self.th["engine_rpm_high_warning"], self.th["engine_rpm_high_stop"]))
        if speed_mps > self.th["speed_high_warning_mps"]:
            add("warning", "speed_high", speed_mps, "m/s", "车速高", self._score_high(speed_mps, self.th["speed_high_warning_mps"], self.th["speed_high_stop_mps"]))
        if flags["active_motion"] and speed_mps < self.th["speed_low_warning_mps"]:
            effective_sec = float(current.get("drive_effective_duration_sec", 0.0) or 0.0)
            if effective_sec >= float(self.th.get("speed_low_warning_duration_sec", 60.0)):
                add("warning", "speed_low_persistent", speed_mps, "m/s", "行驶状态下车速持续偏低", score=0.5)

        warnings.sort(key=lambda r: (self._risk_rank(str(r.get("level", "normal"))), float(r.get("score", 0.0) or 0.0)), reverse=True)
        return warnings

    @staticmethod
    def _speed_controllable_tags() -> set[str]:
        return {
            "coolant_warning",
            "surface_temp_warning",
            "exhaust_warning",
            "hydraulic_oil_temp_warning",
            "intake_temp_high",
            "speed_high",
            "speed_low_persistent",
            "rpm_high",
        }

    @staticmethod
    def _maybe_speed_influenced_tags() -> set[str]:
        return {
            "travel_pressure_warning",
            "system_pressure_low",
            "brake_pressure_low",
            "brake_pressure_high",
            "clamp_pressure_low",
            "clamp_pressure_high",
            "make_up_oil_pressure_low",
            "intake_pressure_low",
            "oil_pressure_low",
        }

    @staticmethod
    def _onsite_required_tags() -> set[str]:
        return {
            "water_tank_low",
            "diesel_level_low",
            "hydraulic_oil_level_low",
        }

    def _classify_warning_level(self, tag: str, score: float) -> str:
        if tag in self._speed_controllable_tags():
            return "high_warning" if score >= 0.8 else "warning"
        if tag in self._maybe_speed_influenced_tags() or tag in self._onsite_required_tags():
            return "high_warning"
        return "high_warning" if score >= 0.8 else "warning"

    def derive_alarm_code(self, current: Dict[str, Any], tag: str) -> int:
        if not tag:
            return 0
        if tag == "hydraulic_oil_temp_warning":
            return 101 if float(current.get("hydraulic_oil_temp_c", 0.0) or 0.0) > self.th["hydraulic_oil_temp_high_stop"] else 0
        if tag == "hydraulic_oil_level_low":
            return 102
        if tag == "coolant_warning":
            return 106 if float(current.get("coolant_temp_c", 0.0) or 0.0) > self.th["coolant_temp_high_stop"] else 0
        if tag == "surface_temp_warning":
            return 107 if float(current.get("surface_temp_c", 0.0) or 0.0) > float(self.th.get("surface_temp_high_stop", 150.0)) else 0
        if tag == "exhaust_warning":
            return 108 if float(current.get("exhaust_temp_c", 0.0) or 0.0) > self.th["exhaust_temp_high_stop"] else 0
        if tag == "speed_high":
            return 110 if float(current.get("speed_mps", 0.0) or 0.0) > self.th["speed_high_stop_mps"] else 0
        if tag == "diesel_level_low":
            return 111 if float(current.get("diesel_level_cm", 999.0) or 999.0) < self.th["diesel_level_low_stop_cm"] else 0
        if tag == "water_tank_low":
            return 112 if float(current.get("water_tank_level_pct", current.get("water_level_pct", 999.0)) or 999.0) < self.th["water_tank_level_low_stop"] else 0
        if tag == "oil_pressure_low":
            return 113 if float(current.get("oil_pressure_kpa", 0.0) or 0.0) < self.th["oil_pressure_low_stop_kpa"] else 0
        if tag == "intake_pressure_low":
            return 114 if float(current.get("intake_pressure_kpa", 0.0) or 0.0) < self.th["intake_pressure_low_stop"] else 0
        if tag == "travel_pressure_warning":
            v = float(current.get("travel_pressure_bar", current.get("walking_pressure_bar", 0.0)) or 0.0)
            return 117 if (v < self.th["travel_pressure_low_stop"] or v > self.th["travel_pressure_high_stop"]) else 0
        if tag == "brake_pressure_low":
            return 118 if float(current.get("brake_pressure_bar", 0.0) or 0.0) < self.th["brake_pressure_low_stop"] else 0
        if tag == "brake_pressure_high":
            return 118 if float(current.get("brake_pressure_bar", 0.0) or 0.0) > self.th["brake_pressure_high_stop"] else 0
        if tag == "clamp_pressure_low":
            return 119 if float(current.get("clamp_pressure_bar", 0.0) or 0.0) < self.th["clamp_pressure_low_stop"] else 0
        if tag == "system_pressure_low":
            return 120 if float(current.get("system_pressure_bar", 0.0) or 0.0) < self.th["system_pressure_low_stop"] else 0
        if tag == "make_up_oil_pressure_low":
            return 121 if float(current.get("make_up_oil_pressure_bar", 0.0) or 0.0) < self.th["make_up_oil_pressure_low_stop"] else 0
        if tag == "intake_temp_high":
            return 135 if float(current.get("intake_temp_c", 0.0) or 0.0) > float(self.th.get("intake_temp_high_stop", self.th.get("intake_temp_high_warning", 60.0))) else 0
        return 0

    def _suggest_actions(self, tag: str, current: Dict[str, Any], level: str) -> Tuple[List[str], List[str], str]:
        if not tag:
            return self.ACTION_TEMPLATES["default"]["actions"], self.ACTION_TEMPLATES["default"]["next"], "当前无预警，持续监测"

        onsite_tags = self._onsite_required_tags()
        maybe_speed_influenced_tags = self._maybe_speed_influenced_tags()
        temperature_tags = {
            "coolant_warning",
            "surface_temp_warning",
            "exhaust_warning",
            "hydraulic_oil_temp_warning",
            "intake_temp_high",
        }

        if tag == "speed_low_persistent":
            speed = float(current.get("speed_mps", 0.0) or 0.0)
            rpm = float(current.get("engine_rpm", 0.0) or 0.0)
            travel_p = float(current.get("travel_pressure_bar", current.get("walking_pressure_bar", 0.0)) or 0.0)
            system_p = float(current.get("system_pressure_bar", 0.0) or 0.0)
            brake_p = float(current.get("brake_pressure_bar", 999.0) or 999.0)
            pressure_or_brake_limited = (
                travel_p < self.th["travel_pressure_low_warning"]
                or system_p < self.th["system_pressure_low_warning"]
                or brake_p < self.th["brake_pressure_low_warning"]
            )
            if pressure_or_brake_limited or rpm > self.th["engine_rpm_high_warning"]:
                tpl = self.ACTION_TEMPLATES["diagnose_low_speed"]
                return tpl["actions"], tpl["next"], "当前低速可能由压力、制动或动力受限导致，先尝试提速并观察响应，无响应则现场排查"
            tpl = self.ACTION_TEMPLATES["increase_speed"]
            return tpl["actions"], tpl["next"], f"当前车速 {speed:.2f} m/s 偏低，优先尝试远程提速并观察压力/转速响应"

        if tag in {"speed_high", "rpm_high"}:
            tpl = self.ACTION_TEMPLATES["reduce_speed"]
            return tpl["actions"], tpl["next"], "车速或转速偏高，可通过远程降速缓解"

        if tag in temperature_tags:
            tpl = self.ACTION_TEMPLATES["temperature_control"]
            if level == "high_warning":
                return tpl["actions"], tpl["next"], "温度接近硬保护边界，优先远程降速并连续观察冷却趋势，无法回落再通知现场排查"
            return tpl["actions"], tpl["next"], "温度类预警，可先通过降速/降负载缓解并观察趋势"

        if tag in maybe_speed_influenced_tags:
            tpl = self.ACTION_TEMPLATES["diagnose_low_speed"]
            return tpl["actions"], tpl["next"], "压力/执行系统类预警可能受车速或负载影响，但不能只靠车速控制解决；先小幅调整并观察响应，无改善则现场排查"

        if tag in onsite_tags:
            tpl = self.ACTION_TEMPLATES["onsite"]
            return tpl["actions"], tpl["next"], "液位/燃油/消防压力类问题无法通过车速控制解决，需现场补给、检漏或检查系统状态"

        tpl = self.ACTION_TEMPLATES["default"]
        return tpl["actions"], tpl["next"], "当前预警类型未归类，先持续监测趋势"

    def evaluate(self, current: Dict[str, Any], window: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
        warnings = self.classify_warnings(current)
        overall_level = self.overall_level(warnings)
        overall_score = self.overall_score(warnings)

        suggested_actions: List[str] = []
        for warning in warnings:
            suggested_actions.extend([str(x) for x in warning.get("suggested_actions", []) or [] if str(x)])

        return {
            "overall_level": overall_level,
            "overall_score": overall_score,
            "warnings": warnings,
            "suggested_actions": list(dict.fromkeys(suggested_actions)) or self.ACTION_TEMPLATES["default"]["actions"],
            "history": {
                "last_hard_action": str(history.get("last_hard_action", history.get("last_action", "")) or "").upper(),
                "stable_after_brake_sec": float(history.get("stable_after_brake_sec", 0) or 0),
                "brake_age_sec": float(history.get("brake_age_sec", 0) or 0),
            },
        }
