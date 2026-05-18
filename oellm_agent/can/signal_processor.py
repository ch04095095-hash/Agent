from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from oellm_agent.vehicle_context import STATE_SENSITIVE_LOW_SIGNALS, build_vehicle_context, should_suppress_low_signal

MODEL_DISABLED_SIGNALS: Dict[str, set[str]] = {
    "50": {"co_ppm"},
    "105": set(),
    "190": set(),
}


class SignalProcessor:
    """Post-process decoded physical values into valid signals and display semantics.

    CanDecoder keeps model/protocol selection and raw physical decoding. This layer handles
    sensor availability, invalid values, and normal/high/low display meanings.
    """

    SIGNAL_UNITS: Dict[str, str] = {
        "coolant_temp_c": "℃",
        "surface_temp_c": "℃",
        "exhaust_temp_c": "℃",
        "oil_pressure_kpa": "kPa",
        "brake_pressure_bar": "bar",
        "travel_pressure_bar": "bar",
        "system_pressure_bar": "bar",
        "clamp_pressure_bar": "bar",
        "speed_mps": "m/s",
        "engine_rpm": "rpm",
        "angle_deg": "deg",
        "hydraulic_oil_temp_c": "℃",
        "hydraulic_oil_level_pct": "%",
        "make_up_oil_pressure_bar": "bar",
        "fire_system_pressure_bar": "bar",
        "water_tank_level_pct": "%",
        "intake_pressure_kpa": "kPa",
        "diesel_level_cm": "cm",
        "intake_temp_c": "℃",
        "co_ppm": "ppm",
        "methane_pct": "%",
    }

    SIGNAL_ORDER: Tuple[str, ...] = tuple(SIGNAL_UNITS.keys())

    def __init__(self, model: str, thresholds: Dict[str, float]):
        self.model = str(model).strip().lower() or "190"
        self.th = thresholds
        self.signal_rules = self._build_signal_rules_from_thresholds(thresholds)

    @staticmethod
    def _meaning_high(warn: float, stop: float, high_label: str) -> List[Tuple[float, str]]:
        return [
            (warn, "正常"),
            (max(warn, stop) - 1e-9, high_label),
            (float("inf"), ""),
        ]

    @staticmethod
    def _meaning_low(warn: float, stop: float, low_label: str) -> List[Tuple[float, str]]:
        return [
            (stop, ""),
            (warn, low_label),
            (float("inf"), "正常"),
        ]

    @staticmethod
    def _meaning_range(low_warn: float, low_stop: float, high_warn: float, high_stop: float, low_label: str, high_label: str) -> List[Tuple[float, str]]:
        return [
            (low_stop, ""),
            (low_warn, low_label),
            (high_warn, "正常"),
            (max(high_warn, high_stop) - 1e-9, high_label),
            (float("inf"), ""),
        ]

    @classmethod
    def _build_signal_rules_from_thresholds(cls, th: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        return {
            "coolant_temp_c": {"meaning": cls._meaning_high(th["coolant_temp_high_warning"], th["coolant_temp_high_stop"], "冷却水温偏高")},
            "surface_temp_c": {"meaning": cls._meaning_high(th["surface_temp_high_warning"], th["surface_temp_high_stop"], "表面温度偏高")},
            "exhaust_temp_c": {"meaning": cls._meaning_high(th["exhaust_temp_high_warning"], th["exhaust_temp_high_stop"], "尾气温度偏高")},
            "intake_pressure_kpa": {"meaning": cls._meaning_low(th["intake_pressure_low_warning"], th["intake_pressure_low_stop"], "进气压力偏低")},
            "water_tank_level_pct": {"meaning": cls._meaning_low(th["water_tank_level_low_warning"], th["water_tank_level_low_stop"], "水箱液位偏低")},
            "oil_pressure_kpa": {"meaning": cls._meaning_low(th["oil_pressure_low_warning_kpa"], th["oil_pressure_low_stop_kpa"], "机油压力偏低")},
            "diesel_level_cm": {"meaning": cls._meaning_low(th["diesel_level_low_warning_cm"], th["diesel_level_low_stop_cm"], "柴油液位偏低")},
            "brake_pressure_bar": {"meaning": cls._meaning_range(th["brake_pressure_low_warning"], th["brake_pressure_low_stop"], th["brake_pressure_high_warning"], th["brake_pressure_high_stop"], "制动压力偏低", "制动压力偏高")},
            "travel_pressure_bar": {"meaning": cls._meaning_range(th["travel_pressure_low_warning"], th["travel_pressure_low_stop"], th["travel_pressure_high_warning"], th["travel_pressure_high_stop"], "行走压力偏低", "行走压力偏高")},
            "system_pressure_bar": {"meaning": [(th["system_pressure_low_stop"], ""), (th["system_pressure_low_warning"], "系统压力偏低"), (180.0, "正常"), (float("inf"), "系统压力偏高")]},
            "clamp_pressure_bar": {"meaning": cls._meaning_range(th["clamp_pressure_low_warning"], th["clamp_pressure_low_stop"], th["clamp_pressure_high_warning"], th["clamp_pressure_high_stop"], "夹紧压力偏低", "夹紧压力偏高")},
            "hydraulic_oil_temp_c": {"meaning": cls._meaning_high(th["hydraulic_oil_temp_high_warning"], th["hydraulic_oil_temp_high_stop"], "液压油温偏高")},
            "hydraulic_oil_level_pct": {"meaning": cls._meaning_low(th["hydraulic_oil_level_low_warning"], th["hydraulic_oil_level_low_stop"], "液压油液位偏低")},
            "make_up_oil_pressure_bar": {"meaning": [(th["make_up_oil_pressure_low_stop"], ""), (th["make_up_oil_pressure_low_warning"], "补油压力偏低"), (th.get("make_up_oil_pressure_high_normal", 28.0), "正常"), (float("inf"), "补油压力偏高")]},
            "engine_rpm": {"meaning": cls._meaning_high(th["engine_rpm_high_warning"], th["engine_rpm_high_stop"], "转速偏高")},
            "speed_mps": {"meaning": [(th["speed_low_warning_mps"], "速度偏低"), (th["speed_high_warning_mps"], "正常"), (th["speed_high_stop_mps"] - 1e-9, "车速偏高"), (float("inf"), "")]},
        }

    def _lookup_threshold_label(self, key: str, value: Optional[float], state: Optional[Dict[str, Any]] = None) -> str:
        if value is None:
            return ""
        if state is not None:
            ctx = build_vehicle_context(state)
            if key in STATE_SENSITIVE_LOW_SIGNALS and should_suppress_low_signal(key, state):
                if key == "speed_mps" and float(value or 0.0) < float(self.th.get("speed_high_warning_mps", 2.2)):
                    return "正常"
                if key in {"brake_pressure_bar", "travel_pressure_bar", "walking_pressure_bar", "system_pressure_bar", "clamp_pressure_bar"}:
                    rules = self.signal_rules.get(key, {}).get("meaning", [])
                    if rules:
                        low_warn_limit = float(rules[1][0]) if len(rules) > 1 else float("inf")
                        if float(value or 0.0) <= low_warn_limit:
                            return "正常"
            if ctx.get("active_motion"):
                if key == "brake_pressure_bar" and float(value or 0.0) >= 145.0:
                    return "正常"
                if key == "system_pressure_bar" and float(value or 0.0) >= 145.0:
                    return "正常"
        for limit, label in self.signal_rules.get(key, {}).get("meaning", []):
            if value <= limit:
                return label
        return "正常"

    def _disabled_reason(self, key: str) -> str:
        if self.model == "50" and key == "co_ppm":
            return "model_50_no_co_sensor"
        return f"model_{self.model}_signal_disabled"

    def _build_signal_meta(self, key: str, value: Any, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        unit = self.SIGNAL_UNITS.get(key, "")
        if key in MODEL_DISABLED_SIGNALS.get(self.model, set()):
            return {
                "value": None,
                "raw_value": value,
                "unit": unit,
                "meaning": "",
                "valid": False,
                "invalid_reason": self._disabled_reason(key),
            }
        return {
            "value": value,
            "unit": unit,
            "meaning": self._lookup_threshold_label(key, value if isinstance(value, (int, float)) else None, state),
            "valid": True,
            "invalid_reason": "",
        }

    def process(self, state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        processed_state = dict(state)
        signals: Dict[str, Dict[str, Any]] = {}
        for key in self.SIGNAL_ORDER:
            signal = self._build_signal_meta(key, state.get(key), state)
            signals[key] = signal
            if not signal.get("valid", True):
                processed_state[key] = None

        return processed_state, {
            "frame_id": None,
            "decoded": processed_state,
            "signals": signals,
        }
