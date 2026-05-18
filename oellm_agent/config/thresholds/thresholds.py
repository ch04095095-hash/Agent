from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REQUIRED_KEYS: List[str] = [
    "methane_high_stop",
    "co_high_stop_ppm",
    "coolant_temp_high_warning",
    "coolant_temp_high_stop",
    "surface_temp_high_warning",
    "surface_temp_high_stop",
    "exhaust_temp_high_warning",
    "exhaust_temp_high_stop",
    "intake_pressure_low_warning",
    "intake_pressure_low_stop",
    "water_tank_level_low_warning",
    "water_tank_level_low_stop",
    "oil_pressure_low_warning_kpa",
    "oil_pressure_low_stop_kpa",
    "diesel_level_low_warning_cm",
    "diesel_level_low_stop_cm",
    "brake_pressure_low_warning",
    "brake_pressure_low_stop",
    "brake_pressure_high_warning",
    "brake_pressure_high_stop",
    "travel_pressure_low_warning",
    "travel_pressure_low_stop",
    "travel_pressure_high_warning",
    "travel_pressure_high_stop",
    "system_pressure_low_warning",
    "system_pressure_low_stop",
    "clamp_pressure_low_warning",
    "clamp_pressure_low_stop",
    "clamp_pressure_high_warning",
    "clamp_pressure_high_stop",
    "hydraulic_oil_temp_high_warning",
    "hydraulic_oil_temp_high_stop",
    "hydraulic_oil_level_low_warning",
    "hydraulic_oil_level_low_stop",
    "make_up_oil_pressure_low_warning",
    "make_up_oil_pressure_low_stop",
    "engine_rpm_high_warning",
    "engine_rpm_high_stop",
    "speed_low_warning_mps",
    "speed_high_warning_mps",
    "speed_high_stop_mps",
]

OPTIONAL_KEYS_WITH_DEFAULTS: Dict[str, float] = {
    "co_high_warning_ppm": 24.0,
    "make_up_oil_pressure_high_normal": 28.0,
    "travel_pressure_low_stop_duration_sec": 10.0,
    "speed_low_warning_duration_sec": 0.0,
    "decision_drive_gear_grace_sec": 15.0,
    "decision_drive_effective_hold_sec": 3.0,
    "decision_drive_starting_rpm_max": 1300.0,
    "decision_drive_effective_rpm_min": 1400.0,
    "decision_brake_pressure_low_normal_floor": 145.0,
    "decision_system_pressure_low_normal_floor": 145.0,
    "decision_speed_low_effective_hold_sec": 60.0,
}

MODEL_ALIASES = {
    "190": "190",
    "50": "50",
    "105": "105",
    "50/105": "105",
    "105/50": "105",
}


def normalize_model(model: str) -> str:
    key = str(model).strip().lower()
    return MODEL_ALIASES.get(key, "190")


def load_thresholds(base_dir: Path, model: str = "50") -> Dict[str, float]:
    model_key = normalize_model(model)
    cfg_path = (base_dir / "config" / "thresholds" / f"{model_key}.json").resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"threshold config not found for model '{model_key}': {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("threshold config must be a JSON object")

    missing_keys = [key for key in REQUIRED_KEYS if key not in obj]
    if missing_keys:
        raise ValueError(f"threshold config missing required keys: {missing_keys}")

    normalized = dict(obj)
    for key, default in OPTIONAL_KEYS_WITH_DEFAULTS.items():
        normalized.setdefault(key, default)

    out: Dict[str, float] = {}
    for k, v in normalized.items():
        try:
            out[k] = float(v)
        except Exception as e:
            raise ValueError(f"threshold '{k}' must be numeric, got {v!r}") from e

    return out


def summarize_thresholds(th: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "methane_high_stop",
        "co_high_stop_ppm",
        "coolant_temp_high_stop",
        "surface_temp_high_stop",
        "exhaust_temp_high_stop",
        "brake_pressure_low_stop",
        "travel_pressure_low_stop",
        "travel_pressure_high_stop",
        "system_pressure_low_stop",
        "speed_high_stop_mps",
        "engine_rpm_high_stop",
    ]
    return {k: th[k] for k in keys if k in th}
