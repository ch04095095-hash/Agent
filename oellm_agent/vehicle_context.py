from __future__ import annotations

from typing import Any, Dict


RUNNING_GEARS = {3, 4}
STATE_SENSITIVE_LOW_SIGNALS = {
    "speed_mps",
    "brake_pressure_bar",
    "travel_pressure_bar",
    "system_pressure_bar",
    "clamp_pressure_bar",
}


def build_vehicle_context(state: Dict[str, Any], drive_gear_grace_sec: float | None = None) -> Dict[str, Any]:
    """Return state-aware context used to gate pressure/speed low-value judgments.

    Low speed/pressure values are normal in idle, neutral, freshly engaged gear,
    and early startup. They only become decision-relevant once the vehicle has
    entered an effective driving state for a short period.
    """
    if drive_gear_grace_sec is None:
        drive_gear_grace_sec = float(state.get("decision_drive_gear_grace_sec", 15.0) or 15.0)
    drive_effective_hold_sec = float(state.get("decision_drive_effective_hold_sec", 3.0) or 3.0)
    drive_starting_rpm_max = float(state.get("decision_drive_starting_rpm_max", 1300.0) or 1300.0)
    drive_effective_rpm_min = float(state.get("decision_drive_effective_rpm_min", 1400.0) or 1400.0)

    gear_state = int(state.get("gear_state", 1) or 1)
    engine_rpm = float(state.get("engine_rpm", 0.0) or 0.0)
    speed_mps = float(state.get("speed_mps", 0.0) or 0.0)
    drive_gear_age_sec = float(state.get("drive_gear_age_sec", 0.0) or 0.0)
    effective_duration_sec = float(state.get("drive_effective_duration_sec", 0.0) or 0.0)

    in_drive = gear_state in RUNNING_GEARS
    in_grace = in_drive and drive_gear_age_sec < drive_gear_grace_sec
    drive_idle = in_drive and engine_rpm <= 900.0 and speed_mps < 0.15
    drive_starting = in_drive and speed_mps < 0.3 and engine_rpm < 1000.0
    drive_effective_candidate = in_drive and not (in_grace or drive_starting)
    active_motion = drive_effective_candidate and effective_duration_sec >= drive_effective_hold_sec

    if not in_drive:
        name = "parking"
    elif in_grace or drive_starting:
        name = "starting"
    else:
        name = "control"

    suppress_low_signals = (not in_drive) or in_grace or drive_idle or drive_starting
    return {
        "name": name,
        "gear_state": gear_state,
        "engine_rpm": engine_rpm,
        "speed_mps": speed_mps,
        "drive_gear_age_sec": drive_gear_age_sec,
        "drive_effective_duration_sec": effective_duration_sec,
        "in_drive": in_drive,
        "in_grace": in_grace,
        "drive_idle": drive_idle,
        "drive_starting": drive_starting,
        "drive_effective_candidate": drive_effective_candidate,
        "active_motion": active_motion,
        "suppress_low_signals": suppress_low_signals,
    }


def should_suppress_low_signal(key: str, state: Dict[str, Any], drive_gear_grace_sec: float = 15.0) -> bool:
    if key not in STATE_SENSITIVE_LOW_SIGNALS:
        return False
    return bool(build_vehicle_context(state, drive_gear_grace_sec=drive_gear_grace_sec)["suppress_low_signals"])
