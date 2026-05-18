from __future__ import annotations

from typing import Dict, List, Tuple

from oellm_agent.vehicle_context import build_vehicle_context


def hard_safety_check(s: Dict[str, float], th: Dict[str, float]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    flags = build_vehicle_context(s)
    emergency_stop = int(s.get('emergency_stop', 0) or 0)
    gear_state = int(s.get('gear_state', 1) or 1)
    control_state = s.get('control_state', {}) if isinstance(s.get('control_state'), dict) else {}
    control_enabled = bool(control_state.get('enabled', False))
    control_entered_by_byte2 = bool(control_state.get('entered_by_byte2', False))
    if emergency_stop in {1, 2, 3}:
        reasons.append('linked_vehicle_emergency_stop')
    if int(s.get('can_heartbeat_ok', 1)) == 0:
        reasons.append('can_heartbeat_lost')

    methane = float(s.get('methane_pctlel', s.get('methane_pct', 0)) or 0)
    if methane >= th['methane_high_stop']:
        reasons.append('methane_over_stop')

    coolant = float(s.get('coolant_temp_c', 0) or 0)
    if coolant >= th['coolant_temp_high_stop']:
        reasons.append('coolant_over_stop')

    surface = float(s.get('surface_temp_c', s.get('surface_temp', 0)) or 0)
    if surface >= th.get('surface_temp_high_stop', 150.0):
        reasons.append('surface_over_stop')

    exhaust = float(s.get('exhaust_temp_c', s.get('exhaust_temp', 0)) or 0)
    if exhaust >= th['exhaust_temp_high_stop']:
        reasons.append('exhaust_over_stop')

    intake_p = s.get('intake_pressure_kpa')
    if intake_p is not None and float(intake_p) < th['intake_pressure_low_stop']:
        reasons.append('intake_pressure_out_of_range')

    water_level = s.get('water_tank_level_pct', s.get('water_level_pct'))
    if water_level is not None and float(water_level) < th['water_tank_level_low_stop']:
        reasons.append('water_level_low')

    oil_p = s.get('oil_pressure_kpa')
    if oil_p is not None and float(oil_p) < th['oil_pressure_low_stop_kpa']:
        reasons.append('oil_pressure_low')

    diesel_level = s.get('diesel_level_cm', s.get('diesel_level'))
    if diesel_level is not None and float(diesel_level) < th['diesel_level_low_stop_cm']:
        reasons.append('diesel_level_low')

    motion_state = str(flags.get('name', '') or '')

    if motion_state == 'parking':
        byte1 = int(s.get('gear_state_byte1', 0) or 0)
        if byte1 != 0:
            reasons.append('parking_neutral_byte1_nonzero')

    # 四项压力：上限全时监测；下限仅在 control 态监测
    in_control = bool(motion_state == 'control' and control_enabled and control_entered_by_byte2)

    brake_p = s.get('brake_pressure_bar')
    if brake_p is not None:
        brake_p = float(brake_p)
        if brake_p > th['brake_pressure_high_stop']:
            reasons.append('brake_pressure_high_stop')
        elif in_control and flags['active_motion'] and brake_p < th['brake_pressure_low_stop']:
            reasons.append('brake_pressure_low_stop')

    walking_p = s.get('walking_pressure_bar', s.get('travel_pressure_bar', s.get('system_pressure_walk_bar')))
    if walking_p is not None:
        walking_p = float(walking_p)
        if walking_p > th['travel_pressure_high_stop']:
            reasons.append('walking_pressure_over_stop')
        elif in_control and flags['active_motion'] and walking_p < th['travel_pressure_low_stop']:
            reasons.append('walking_pressure_over_stop')

    system_p = s.get('system_pressure_bar')
    if system_p is not None:
        system_p = float(system_p)
        system_high_stop = th.get('system_pressure_high_stop')
        if system_high_stop is not None and system_p > float(system_high_stop):
            reasons.append('system_pressure_high_stop')
        elif in_control and system_p < th['system_pressure_low_stop']:
            reasons.append('system_pressure_low_stop')

    clamp_p = s.get('clamp_pressure_bar')
    if clamp_p is not None:
        clamp_p = float(clamp_p)
        if clamp_p > th['clamp_pressure_high_stop']:
            reasons.append('clamp_pressure_over_stop')
        elif in_control and clamp_p < th['clamp_pressure_low_stop']:
            reasons.append('clamp_pressure_over_stop')

    # 其余传感器全时监测（parking/starting/control 均生效）
    hydraulic_temp = s.get('hydraulic_oil_temp_c')
    if hydraulic_temp is not None and float(hydraulic_temp) > th['hydraulic_oil_temp_high_stop']:
        reasons.append('hydraulic_oil_temp_over_stop')

    hydraulic_level = s.get('hydraulic_oil_level_pct')
    if hydraulic_level is not None and float(hydraulic_level) < th['hydraulic_oil_level_low_stop']:
        reasons.append('hydraulic_oil_level_low')

    make_up_oil_p = s.get('make_up_oil_pressure_bar')
    if make_up_oil_p is not None and float(make_up_oil_p) < th['make_up_oil_pressure_low_stop']:
        reasons.append('make_up_oil_pressure_low_stop')

    rpm = s.get('engine_rpm')
    if rpm is not None and float(rpm) > th['engine_rpm_high_stop']:
        reasons.append('rpm_over_stop')

    speed = s.get('speed_mps')
    if speed is not None and float(speed) > th['speed_high_stop_mps']:
        reasons.append('speed_over_stop')

    co_ppm = s.get('co_ppm')
    if co_ppm is not None and float(co_ppm) >= th['co_high_stop_ppm']:
        reasons.append('co_over_stop')

    return ('BRAKE', reasons) if reasons else ('MOVE', [])
