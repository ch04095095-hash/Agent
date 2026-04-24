from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REQUIRED_KEYS: List[str] = [
    "methane_stop",
    "co_stop",
    "coolant_stop",
    "exhaust_stop",
    "intake_pressure_min",
    "water_level_alarm_min",
    "oil_pressure_min_kpa",
    "diesel_level_min",
    "brake_pressure_min",
    "walking_pressure_min",
    "walking_pressure_max",
    "system_pressure_min",
    "clamp_pressure_min",
    "clamp_pressure_max",
    "hydraulic_oil_temp_max",
    "hydraulic_oil_level_min",
    "make_up_oil_pressure_min",
    "rpm_max",
    "speed_max_kmh",
    "coolant_temp_warn",
    "coolant_temp_high",
    "exhaust_temp_warn",
    "exhaust_temp_high",
    "oil_pressure_warn_kpa",
    "hydraulic_oil_temp_warn",
    "hydraulic_oil_temp_high",
    "hydraulic_oil_level_low",
    "make_up_oil_pressure_warn_low",
    "brake_pressure_warn_low",
    "travel_pressure_warn_low",
    "travel_pressure_warn_high",
    "travel_pressure_high",
    "system_pressure_warn_low",
    "system_pressure_low",
    "clamp_pressure_warn_low",
    "clamp_pressure_warn_high",
    "intake_pressure_warn_low",
    "water_level_warn_low",
    "water_level_low",
    "diesel_level_low",
    "methane_alarm",
    "co_warn",
    "co_alarm",
    "speed_warn_high_kmh",
    "speed_alarm_high_kmh",
    "speed_warn_low_kmh",
    "rpm_warn_high",
    "rpm_alarm_high",
]


def load_thresholds(base_dir: Path) -> Dict[str, float]:
    cfg_path = (base_dir / "config" / "thresholds.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"threshold config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("threshold config must be a JSON object")

    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"threshold config missing keys: {missing}")

    out: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[k] = float(v)
        except Exception as e:
            raise ValueError(f"threshold '{k}' must be numeric, got {v!r}") from e

    return out


def summarize_thresholds(th: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "methane_stop",
        "co_stop",
        "coolant_stop",
        "exhaust_stop",
        "brake_pressure_min",
        "walking_pressure_min",
        "walking_pressure_max",
        "system_pressure_min",
        "speed_max_kmh",
        "rpm_max",
    ]
    return {k: th[k] for k in keys if k in th}
