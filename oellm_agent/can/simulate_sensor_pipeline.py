#!/usr/bin/env python3
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

DT = 0.5
DURATION_SEC = 600
TOTAL_STEPS = int(DURATION_SEC / DT)  # 6000
SEED = 20260408

TH = {
    "coolant_temp_warn": 95.0,
    "coolant_temp_high": 98.0,
    "surface_temp_warn": 147.0,
    "surface_temp_high": 150.0,
    "exhaust_temp_warn": 67.0,
    "exhaust_temp_high": 70.0,
    "oil_pressure_low_kpa": 70.0,
    "hydraulic_oil_temp_warn": 70.0,
    "hydraulic_oil_temp_high": 80.0,
    "hydraulic_oil_level_low": 40.0,
    "make_up_oil_pressure_low": 26.0,
    "make_up_oil_pressure_high": 28.0,
    "fire_system_pressure_low": 1.0,
    "fire_system_pressure_high": 25.0,
    "brake_pressure_high": 40.0,
    "travel_pressure_high": 60.0,
    "system_pressure_high": 150.0,
    "clamp_pressure_low": 110.0,
    "clamp_pressure_high": 140.0,
    "intake_pressure_low": 90.0,
    "intake_pressure_high": 100.0,
    "water_level_low": 30.0,
    "diesel_level_low": 30.0,
    "methane_alarm": 0.5,
    "methane_stop": 1.0,
}

# 让正常态占大头，异常更少、更短、更明确
CRITICAL_EVENT_PROB = 0.028  # 约2.8%：轻微预警/趋势问题
OVERSHOOT_EVENT_PROB = 0.004  # 约0.4%：少量硬底线
LLM_CASE_PROB = 0.010  # 趋势类片段的启动概率
START_TS = datetime(2026, 4, 8, 9, 0, 0, tzinfo=timezone.utc)


@dataclass
class SensorState:
    speed_mps: float
    engine_rpm: float
    coolant_temp_c: float
    surface_temp_c: float
    exhaust_temp_c: float
    intake_pressure_kpa: float
    radiator_level_pct: float
    oil_pressure_kpa: float
    water_tank_level_pct: float
    diesel_level_cm: float
    intake_temp_c: float
    brake_pressure_bar: float
    travel_pressure_bar: float
    system_pressure_bar: float
    clamp_pressure_bar: float
    methane_pctlel: float
    co_ppm: float
    battery_v: float
    angle_deg: float
    total_mileage_km: float
    runtime_min: float
    payload_tons: float
    cabin_id: int
    slope_state: int
    firewall_pressure_kpa: float

    # 0x18F181A0
    alarm_code: int
    risk_level: str
    warning_tag: str
    gear_state: int
    emergency_stop: int
    hydraulic_oil_temp_c: float
    hydraulic_oil_level_pct: float
    make_up_oil_pressure_bar: float
    fire_system_pressure_bar: float
    shua_qu_state: int

    # 链路健康（非181协议字段，供L1链路监测）
    can_heartbeat_ok: int


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def base_state(step: int) -> SensorState:
    t = step * DT

    # 构造更接近真实矿车的运行节奏：平路-上坡-平路-下坡循环
    cycle = int(t // 45) % 4
    if cycle == 0:
        slope_state = 1
        target_payload = 100 + 6 * math.sin(t / 45)
        base_speed = 10.8
    elif cycle == 1:
        slope_state = 2
        target_payload = 165 + 9 * math.sin(t / 40)
        base_speed = 8.8
    elif cycle == 2:
        slope_state = 1
        target_payload = 130 + 8 * math.sin(t / 35)
        base_speed = 10.0
    else:
        slope_state = 3
        target_payload = 88 + 5 * math.sin(t / 38)
        base_speed = 11.2

    angle = clamp((8 if slope_state == 2 else (-6 if slope_state == 3 else 1.0)) + 1.0 * math.sin(t / 22) + random.gauss(0, 0.14), -14, 14)
    payload = clamp(target_payload + random.gauss(0, 1.0), 40, 240)
    cabin_id = 1 if (step // int(35 / DT)) % 2 == 0 else 2

    # 起步速度与负重/轨道摩擦(坡度表征)/发动机转速建立耦合
    # 1) 先估计发动机转速（受基础工况、负重与坡度影响）
    rpm = clamp(
        780
        + 46 * max(0.0, base_speed)
        + 0.55 * max(0.0, payload - 110)
        + 18.0 * max(0.0, angle)
        + random.gauss(0, 8),
        700,
        1850,
    )

    # 2) 将“负重 + 坡度”映射成轨道阻力（等价摩擦负载）
    payload_norm = max(0.0, min(1.6, payload / 200.0))
    slope_up_norm = max(0.0, angle) / 14.0
    rolling_resistance = 0.10 + 0.30 * payload_norm + 0.34 * slope_up_norm

    # 3) 将发动机可用牵引映射成速度（起步阶段逐步放开）
    traction_norm = max(0.0, min(1.2, (rpm - 780.0) / 980.0))
    net_drive = max(0.0, traction_norm - rolling_resistance)
    startup_ramp = max(0.15, min(1.0, t / 20.0))
    target_speed = (2.0 + 9.8 * net_drive) * startup_ramp

    # 与 thresholds.json 对齐：speed_warn_high_mps=13、speed_alarm_high_mps=14
    speed = clamp(target_speed + 0.20 * math.sin(t / 16.0) + random.gauss(0, 0.06), 0.3, 12.6)

    # 运行工况耦合：把负载/时间/转速映射到热与消耗量
    load_ratio = max(0.0, min(1.4, payload / 180.0))
    rpm_ratio = max(0.0, min(1.3, rpm / 1800.0))
    speed_ratio = max(0.0, min(1.3, speed / 12.5))
    uphill_ratio = max(0.0, angle) / 14.0
    runtime_hr = t / 3600.0

    thermal_stress = 0.50 * load_ratio + 0.30 * rpm_ratio + 0.20 * uphill_ratio

    # 冷却水温：由负载/转速/坡度和累计运行时间共同抬升
    coolant = clamp(
        80.8
        + 7.8 * thermal_stress
        + 1.15 * math.log1p(runtime_hr * 9.0)
        + 0.45 * max(0.0, speed_ratio - 0.85)
        + random.gauss(0, 0.06),
        80.0,
        93.0,
    )

    # 表面温度：较冷却水更敏感，受负载和转速叠加驱动
    surface = clamp(
        95.0
        + 16.5 * thermal_stress
        + 1.9 * math.log1p(runtime_hr * 8.0)
        + 0.75 * max(0.0, speed_ratio - 0.8)
        + random.gauss(0, 0.20),
        92.0,
        142.0,
    )

    exhaust = clamp(50.5 + 1.2 * load_ratio + 2.9 * rpm_ratio + 1.0 * speed_ratio + 0.8 * uphill_ratio + random.gauss(0, 0.14), 48, 64.8)

    # 进气压力受海拔/坡度(等效)、转速需求和过滤阻塞(工时累积)影响
    filter_clog = min(0.18, runtime_hr * 0.05 + 0.02 * load_ratio)
    intake_pressure = clamp(111.5 - 7.0 * filter_clog - 1.9 * uphill_ratio - 1.1 * max(0.0, rpm_ratio - 0.95) + random.gauss(0, 0.16), 104, 116)

    # 冷却水液位：热应力、转速和工时共同消耗
    coolant_loss_index = 0.35 + 0.45 * thermal_stress + 0.20 * max(0.0, rpm_ratio - 0.85)
    radiator_level = clamp(83.0 - coolant_loss_index * runtime_hr * 10.0 + 0.42 * math.sin(t / 220) + random.gauss(0, 0.10), 52, 100)

    # 机油压力：与转速正相关，受油温和负荷拖累
    oil_temp_effect = max(0.0, coolant - 88.0)
    oil_pressure = clamp(340 + 95 * rpm_ratio - 1.8 * oil_temp_effect - 7.5 * load_ratio + random.gauss(0, 1.3), 390, 480)

    # 水箱液位：热负荷与运行时间耦合
    water_tank = clamp(69.0 - (0.30 + 0.34 * thermal_stress + 0.10 * speed_ratio) * runtime_hr * 10.0 + random.gauss(0, 0.07), 44, 100)

    # 柴油液位：负载×转速×坡度决定瞬时油耗，并随工时累计
    fuel_burn_index = 0.40 * load_ratio + 0.35 * rpm_ratio + 0.15 * speed_ratio + 0.10 * uphill_ratio
    diesel_level = clamp(71.0 - (1.40 + 1.55 * fuel_burn_index) * runtime_hr * 10.0 + random.gauss(0, 0.05), 40, 100)

    intake_temp = clamp(17.5 + 0.06 * abs(angle) + 0.9 * max(0.0, speed_ratio - 0.7) + 0.5 * max(0.0, coolant - 88.0) / 6.0 + random.gauss(0, 0.09), -20, 36)

    # 液压油温/液位、补油压力、消防压力联动
    hydraulic_oil_temp = clamp(50.5 + 8.5 * (0.45 * load_ratio + 0.35 * rpm_ratio + 0.20 * speed_ratio) + 1.2 * math.log1p(runtime_hr * 8.0) + random.gauss(0, 0.16), 20, 67.0)
    hydraulic_oil_level = clamp(84.0 - (0.18 + 0.25 * max(0.0, hydraulic_oil_temp - 58.0) / 10.0 + 0.10 * load_ratio) * runtime_hr * 10.0 + random.gauss(0, 0.16), 50, 100)
    make_up_oil = clamp(27.6 - 0.06 * max(0.0, hydraulic_oil_temp - 60.0) - 0.015 * max(0.0, speed - 9.0) + 0.010 * max(0.0, hydraulic_oil_level - 70.0) + random.gauss(0, 0.04), 26.5, 27.8)
    fire_sys = clamp(12.6 - 0.025 * runtime_hr * 10.0 + 0.18 * math.sin(t / 300) + random.gauss(0, 0.05), 4.0, 23.0)

    # 液压与制动压力联动：负载/坡度/速度共同作用
    brake = clamp(156 + 1.2 * max(0.0, -angle) + 0.55 * max(0.0, speed - 6.0) + 0.03 * max(0.0, payload - 130) + random.gauss(0, 0.65), 150, 178.0)
    travel = clamp(30 + 0.065 * payload + 0.85 * max(0.0, angle) + 0.95 * speed_ratio + random.gauss(0, 0.30), 26, 70.0)
    system = clamp(148 + 0.05 * payload + 0.55 * rpm_ratio - 0.25 * max(0.0, hydraulic_oil_temp - 55.0) + random.gauss(0, 0.30), 150, 176.0)
    clamp_p = clamp(118 + 0.18 * (system - 150) + 0.01 * max(0.0, payload - 130) + random.gauss(0, 0.16), 118, 138.0)

    # 气体与工况联动：负载/温度/下坡通风影响
    methane = clamp(0.025 + 0.00020 * max(0.0, payload - 140) + 0.010 * (1 if slope_state == 3 else 0) + 0.0007 * max(0.0, runtime_hr * 60 - 5) + random.gauss(0, 0.002), 0.0, 0.20)
    co = clamp(5.5 + 0.012 * max(0.0, payload - 120) + 0.9 * max(0.0, exhaust - 58.0) + 0.30 * rpm_ratio + random.gauss(0, 0.25), 0, 40)

    # 电气系统：高负载和高温会导致电压轻微下探
    battery = clamp(25.25 - 0.18 * max(0.0, thermal_stress - 0.9) - 0.06 * max(0.0, coolant - 90.0) / 5.0 + 0.04 * math.sin(t / 120) + random.gauss(0, 0.015), 22, 27.5)

    # 液压油温/液位、补油压力、消防压力联动
    hydraulic_oil_temp = clamp(50.5 + 8.5 * (0.45 * load_ratio + 0.35 * rpm_ratio + 0.20 * speed_ratio) + 1.2 * math.log1p(runtime_hr * 8.0) + random.gauss(0, 0.16), 20, 67.0)
    hydraulic_oil_level = clamp(84.0 - (0.18 + 0.25 * max(0.0, hydraulic_oil_temp - 58.0) / 10.0 + 0.10 * load_ratio) * runtime_hr * 10.0 + random.gauss(0, 0.16), 50, 100)
    make_up_oil = clamp(27.6 - 0.06 * max(0.0, hydraulic_oil_temp - 60.0) - 0.015 * max(0.0, speed - 9.0) + 0.010 * max(0.0, hydraulic_oil_level - 70.0) + random.gauss(0, 0.04), 26.5, 27.8)
    fire_sys = clamp(12.6 - 0.025 * runtime_hr * 10.0 + 0.18 * math.sin(t / 300) + random.gauss(0, 0.05), 4.0, 23.0)

    firewall_pressure = clamp(205 + 0.62 * payload + 1.4 * max(0.0, speed - 8.0) - 1.1 * max(0.0, hydraulic_oil_temp - 60.0) + random.gauss(0, 1.2), 180, 260)
    shua_qu_state = 2 if speed > 0.5 else 1

    miles = 12500.0 + t * max(0.0, speed) / 3600.0
    runtime = t / 60.0

    return SensorState(
        speed_mps=speed,
        engine_rpm=rpm,
        coolant_temp_c=coolant,
        surface_temp_c=surface,
        exhaust_temp_c=exhaust,
        intake_pressure_kpa=intake_pressure,
        radiator_level_pct=radiator_level,
        oil_pressure_kpa=oil_pressure,
        water_tank_level_pct=water_tank,
        diesel_level_cm=diesel_level,
        intake_temp_c=intake_temp,
        brake_pressure_bar=brake,
        travel_pressure_bar=travel,
        system_pressure_bar=system,
        clamp_pressure_bar=clamp_p,
        methane_pctlel=methane,
        co_ppm=co,
        battery_v=battery,
        angle_deg=angle,
        total_mileage_km=miles,
        runtime_min=runtime,
        payload_tons=payload,
        cabin_id=cabin_id,
        slope_state=slope_state,
        firewall_pressure_kpa=firewall_pressure,
        alarm_code=0,
        risk_level="normal",
        warning_tag="",
        gear_state=2 if speed > 0.5 else 1,
        emergency_stop=0,
        hydraulic_oil_temp_c=hydraulic_oil_temp,
        hydraulic_oil_level_pct=hydraulic_oil_level,
        make_up_oil_pressure_bar=make_up_oil,
        fire_system_pressure_bar=fire_sys,
        shua_qu_state=shua_qu_state,
        can_heartbeat_ok=1,
    )


def classify_warning(s: SensorState) -> Tuple[str, str]:
    # 先给出预警标签，报警码由后续硬阈值映射得到
    if s.emergency_stop in (1, 2, 3):
        return "danger", "emergency_stop"
    if s.can_heartbeat_ok == 0:
        return "danger", "communication_fault"
    if s.methane_pctlel >= TH["methane_alarm"]:
        return "warning", "methane_warning"
    if s.co_ppm >= 50:
        return "warning", "co_warning"
    if s.coolant_temp_c >= TH["coolant_temp_warn"]:
        return "warning", "coolant_warning"
    if s.surface_temp_c >= TH["surface_temp_warn"]:
        return "warning", "surface_warning"
    if s.exhaust_temp_c >= TH["exhaust_temp_warn"]:
        return "warning", "exhaust_warning"
    if s.hydraulic_oil_temp_c >= TH["hydraulic_oil_temp_warn"]:
        return "warning", "hydraulic_oil_temp_warning"
    if s.hydraulic_oil_level_pct <= TH["hydraulic_oil_level_low"]:
        return "warning", "hydraulic_oil_level_low"
    if s.make_up_oil_pressure_bar < TH["make_up_oil_pressure_low"]:
        return "warning", "make_up_oil_pressure_low"
    if s.fire_system_pressure_bar > TH["fire_system_pressure_high"] or s.fire_system_pressure_bar < TH["fire_system_pressure_low"]:
        return "warning", "fire_system_pressure_warning"
    if s.brake_pressure_bar > TH["brake_pressure_high"]:
        return "warning", "brake_pressure_high"
    if s.travel_pressure_bar > TH["travel_pressure_high"]:
        return "warning", "travel_pressure_high"
    if s.system_pressure_bar > TH["system_pressure_high"]:
        return "warning", "system_pressure_high"
    if s.oil_pressure_kpa <= TH["oil_pressure_low_kpa"]:
        return "warning", "oil_pressure_low"
    if s.diesel_level_cm <= TH["diesel_level_low"]:
        return "warning", "diesel_level_low"
    if s.water_tank_level_pct <= TH["water_level_low"]:
        return "warning", "water_tank_low"
    if s.payload_tons >= 300:
        return "warning", "overload_warning"
    # 稳态但存在轻微工况压力
    if s.speed_mps < 1.0 and s.payload_tons > 170:
        return "warning", "idle_heavy_load"
    return "normal", ""


def derive_alarm_code(s: SensorState, warning_tag: str) -> int:
    # 正常状态不报码
    if warning_tag == "":
        return 0

    # 硬报警：真正达到报警线时才映射
    if warning_tag == "emergency_stop":
        return 101
    if warning_tag == "hydraulic_oil_level_low":
        return 102
    if warning_tag == "communication_fault":
        return 103
    if warning_tag == "coolant_warning":
        return 106 if s.coolant_temp_c >= TH["coolant_temp_high"] else 0
    if warning_tag == "surface_warning":
        return 107 if s.surface_temp_c >= TH["surface_temp_high"] else 0
    if warning_tag == "exhaust_warning":
        return 108 if s.exhaust_temp_c >= TH["exhaust_temp_high"] else 0
    if warning_tag == "methane_warning":
        return 109 if s.methane_pctlel >= TH["methane_stop"] else 0
    if warning_tag == "co_warning":
        return 109 if s.co_ppm >= 3000 else 0
    if warning_tag == "overload_warning":
        return 110 if s.payload_tons >= 380 else 0
    if warning_tag == "diesel_level_low":
        return 111 if s.diesel_level_cm <= TH["diesel_level_low"] else 0
    if warning_tag == "water_tank_low":
        return 112 if s.water_tank_level_pct <= TH["water_level_low"] else 0
    if warning_tag == "oil_pressure_low":
        return 113 if s.oil_pressure_kpa <= TH["oil_pressure_low_kpa"] else 0
    if warning_tag == "make_up_oil_pressure_low":
        return 121 if s.make_up_oil_pressure_bar < TH["make_up_oil_pressure_low"] else 0
    if warning_tag == "fire_system_pressure_warning":
        return 122 if (s.fire_system_pressure_bar > TH["fire_system_pressure_high"] or s.fire_system_pressure_bar < TH["fire_system_pressure_low"]) else 0
    if warning_tag == "travel_pressure_high":
        return 117 if s.travel_pressure_bar > TH["travel_pressure_high"] else 0
    if warning_tag == "brake_pressure_high":
        return 118 if s.brake_pressure_bar > TH["brake_pressure_high"] else 0
    if warning_tag == "system_pressure_high":
        return 120 if s.system_pressure_bar > TH["system_pressure_high"] else 0
    if warning_tag == "hydraulic_oil_temp_warning":
        return 0
    if warning_tag == "idle_heavy_load":
        return 0
    return 0


def inject_critical_event(s: SensorState, event: str) -> None:
    if event == "coolant_overheat":
        s.coolant_temp_c = random.uniform(95.0, 99.0)
    elif event == "hydraulic_oil_overheat":
        s.hydraulic_oil_temp_c = random.uniform(70.0, 79.8)
    elif event == "hydraulic_oil_level_low":
        s.hydraulic_oil_level_pct = random.uniform(0.0, 39.0)
    elif event == "water_tank_low":
        s.water_tank_level_pct = random.uniform(0.0, 29.0)
    elif event == "make_up_oil_pressure_out":
        s.make_up_oil_pressure_bar = random.choice([random.uniform(0.0, 25.0), random.uniform(29.0, 60.0)])
    elif event == "fire_system_pressure_out":
        s.fire_system_pressure_bar = random.choice([random.uniform(0.0, 0.9), random.uniform(25.1, 30.0)])
    elif event == "brake_pressure_low":
        s.brake_pressure_bar = random.uniform(0.0, 39.0)
    elif event == "travel_pressure_low":
        s.travel_pressure_bar = random.uniform(0.0, 59.0)
    elif event == "system_pressure_low":
        s.system_pressure_bar = random.uniform(0.0, 109.0)
    elif event == "methane_high":
        s.methane_pctlel = random.uniform(0.5, 0.99)
    elif event == "emergency_stop":
        s.emergency_stop = random.choice([1, 2, 3])
    elif event == "co_high":
        s.co_ppm = random.uniform(50.0, 2999.0)
    elif event == "diesel_level_low":
        s.diesel_level_cm = random.uniform(0.0, 29.0)
    elif event == "water_tank_low":
        s.water_tank_level_pct = random.uniform(0.0, 29.0)
    elif event == "overload":
        s.payload_tons = random.uniform(300.0, 379.0)
    elif event == "steep_slope":
        s.angle_deg = random.choice([-18.0, 18.0])
        s.slope_state = 2 if s.angle_deg > 0 else 3



def inject_overshoot_event(s: SensorState, event: str) -> None:
    if event == "coolant_overheat":
        s.coolant_temp_c = random.uniform(95.5, 99.5)
    elif event == "hydraulic_oil_overheat":
        s.hydraulic_oil_temp_c = random.uniform(70.0, 79.8)
    elif event == "hydraulic_oil_level_low":
        s.hydraulic_oil_level_pct = random.uniform(0.0, 35.0)
    elif event == "make_up_oil_pressure_out":
        s.make_up_oil_pressure_bar = random.choice([random.uniform(0.0, 24.0), random.uniform(29.0, 60.0)])
    elif event == "fire_system_pressure_out":
        s.fire_system_pressure_bar = random.choice([random.uniform(0.0, 0.8), random.uniform(25.2, 30.0)])
    elif event == "brake_pressure_low":
        s.brake_pressure_bar = random.uniform(0.0, 30.0)
    elif event == "travel_pressure_low":
        s.travel_pressure_bar = random.uniform(0.0, 45.0)
    elif event == "system_pressure_low":
        s.system_pressure_bar = random.uniform(0.0, 100.0)
    elif event == "methane_high":
        s.methane_pctlel = random.uniform(0.5, 0.99)
    elif event == "emergency_stop":
        s.emergency_stop = random.choice([1, 2, 3])
    elif event == "co_high":
        s.co_ppm = random.uniform(50.0, 2999.0)
    elif event == "overload":
        s.payload_tons = random.uniform(300.0, 379.0)
    elif event == "steep_slope":
        s.angle_deg = random.choice([-20.0, 20.0])
        s.slope_state = 2 if s.angle_deg > 0 else 3


def inject_llm_case(s: SensorState, case: str, phase: float) -> None:
    # 全在硬阈值内，但趋势明显
    if case == "coolant_fast_rise":
        s.coolant_temp_c = clamp(82 + phase * 12.5 + random.gauss(0, 0.2), 80, 97.6)
    elif case == "hydraulic_oil_temp_rise":
        s.hydraulic_oil_temp_c = clamp(58 + phase * 10.0 + random.gauss(0, 0.2), 55, 79.6)
    elif case == "hydraulic_oil_level_drop":
        s.hydraulic_oil_level_pct = clamp(72 - phase * 26 + random.gauss(0, 0.5), 41.0, 100)
    elif case == "water_tank_drop":
        s.water_tank_level_pct = clamp(52 - phase * 24 + random.gauss(0, 0.5), 30.5, 100)
    elif case == "make_up_oil_pressure_drift":
        s.make_up_oil_pressure_bar = clamp(28.0 - phase * 1.8 + random.gauss(0, 0.08), 26.1, 28.0)
    elif case == "fire_system_pressure_drift":
        s.fire_system_pressure_bar = clamp(18.0 + phase * 6.0 + random.gauss(0, 0.08), 1.2, 24.8)
    elif case == "brake_pressure_softening":
        s.brake_pressure_bar = clamp(105 - phase * 32 + random.gauss(0, 1.0), 41.0, 160)
    elif case == "travel_pressure_softening":
        s.travel_pressure_bar = clamp(95 - phase * 28 + random.gauss(0, 1.0), 61.0, 180)
    elif case == "system_pressure_drift_down":
        s.system_pressure_bar = clamp(154 - phase * 42 + random.gauss(0, 1.0), 111.0, 190)
    elif case == "methane_rising_near_limit":
        s.methane_pctlel = clamp(0.55 + phase * 0.38 + random.gauss(0, 0.01), 0.2, 0.99)
    elif case == "speed_up_trend":
        s.speed_mps = clamp(12 + phase * 11 + random.gauss(0, 0.15), 0, 26)
        s.engine_rpm = clamp(1300 + phase * 700 + random.gauss(0, 15), 700, 2600)
    elif case == "angle_swing":
        s.angle_deg = clamp(-15 + phase * 30 + random.gauss(0, 0.5), -40, 40)
    elif case == "payload_rise":
        s.payload_tons = clamp(220 + phase * 140 + random.gauss(0, 1.0), 0, 400)
    elif case == "co_rise":
        s.co_ppm = clamp(150 + phase * 180 + random.gauss(0, 2.0), 0, 3000)
    elif case == "mileage_growth":
        s.total_mileage_km = clamp(s.total_mileage_km + phase * 0.02, 0, 677721)
    elif case == "runtime_growth":
        s.runtime_min = clamp(s.runtime_min + phase * 0.1, 0, 16777215)
    elif case == "cabin_shift":
        s.cabin_id = 1 if phase < 0.5 else 2
    elif case == "slope_state_switch":
        s.slope_state = 2 if phase < 0.5 else 3


def u16_bytes(v: int) -> Tuple[int, int]:
    v = max(0, min(65535, v))
    return (v & 0xFF, (v >> 8) & 0xFF)


def i16_bytes(v: int) -> Tuple[int, int]:
    v = max(-32768, min(32767, v))
    if v < 0:
        v = (1 << 16) + v
    return (v & 0xFF, (v >> 8) & 0xFF)


def u32_bytes(v: int) -> Tuple[int, int, int, int]:
    v = max(0, min(4294967295, v))
    return (v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF)


def encode_frames(s: SensorState, model: str = "50") -> List[Tuple[int, List[int]]]:
    # 严格对齐 can_decoder.py 的车型协议
    # 2.1 发送报警号、挡位状态、急停状态、液压油温度、液压油液位、补油压力、灭火器系统压力、甩驱状态
    coolant_u8 = max(0, min(255, int(round(s.coolant_temp_c + 40.0))))
    hyd_oil_temp = max(0, min(80, int(round(s.hydraulic_oil_temp_c))))
    hyd_oil_level = max(0, min(100, int(round(s.hydraulic_oil_level_pct))))
    make_up_oil_p = max(0, min(60, int(round(s.make_up_oil_pressure_bar))))
    fire_sys_p = max(0, min(25, int(round(s.fire_system_pressure_bar))))
    shua_qu_state = max(1, min(5, int(round(s.shua_qu_state))))

    # 2.2 制动压力、行走压力、系统压力、夹紧压力：4 个 u16 小端，需与 can_decoder 对齐
    brake_p = max(0, min(65535, int(round(s.brake_pressure_bar))))
    travel_p = max(0, min(65535, int(round(s.travel_pressure_bar))))
    system_p = max(0, min(65535, int(round(s.system_pressure_bar))))
    clamp_p = max(0, min(65535, int(round(s.clamp_pressure_bar))))
    bp0, bp1 = u16_bytes(brake_p)
    tp0, tp1 = u16_bytes(travel_p)
    sp0, sp1 = u16_bytes(system_p)
    cp0, cp1 = u16_bytes(clamp_p)

    # 2.3 甲烷浓度、车速、转速、角度
    methane_raw = max(0, min(800, int(round(s.methane_pctlel * 10.0 + 400.0))))
    speed_raw = max(0, min(251, int(round(s.speed_mps * 10.0))))
    rpm_raw = max(0, min(3000, int(round(s.engine_rpm))))
    angle_raw = max(0, min(1800, int(round(s.angle_deg * 10.0 + 900.0))))
    m0, m1 = u16_bytes(methane_raw)
    s0, s1 = u16_bytes(speed_raw)
    r0, r1 = u16_bytes(rpm_raw)
    a0, a1 = u16_bytes(angle_raw)

    # 2.4 冷却水温、表面温度、尾气温度、进气压力、水箱水位、机油压力、柴油液位、进气温度
    surface_u8 = max(0, min(255, int(round(s.surface_temp_c))))
    exhaust_u8 = max(0, min(255, int(round(s.exhaust_temp_c))))
    intake_p_u8 = max(0, min(250, int(round(s.intake_pressure_kpa / 2.0))))
    water_tank_u8 = max(0, min(100, int(round(s.water_tank_level_pct))))
    oil_u8 = max(0, min(255, int(round(s.oil_pressure_kpa / 4.0))))
    diesel_level_u8 = max(0, min(100, int(round(s.diesel_level_cm))))
    intake_temp_u8 = max(0, min(140, int(round(s.intake_temp_c + 40.0))))

    # 2.5 总里程、运行时间、防冻液温度
    total_mileage_raw = max(0, min(677721, int(round(s.total_mileage_km * 10.0))))
    runtime_raw = max(0, min(16777215, int(round(s.runtime_min))))
    antifreeze_raw = max(0, min(65535, int(round((58.0 + 0.2 * math.sin(s.runtime_min / 90.0)) + 273.0) / 0.03125)))
    tm0, tm1 = u16_bytes(total_mileage_raw)
    rt0, rt1 = u16_bytes(runtime_raw)
    af0, af1 = u16_bytes(antifreeze_raw)

    # 2.9 称重、一氧化碳
    payload_raw = max(0, min(400, int(round(s.payload_tons))))
    co_raw = max(0, min(3000, int(round(s.co_ppm))))
    pw0, pw1 = u16_bytes(payload_raw)
    co0, co1 = u16_bytes(co_raw)

    # 3.0 角度，驾驶室号，上下坡状态，壳体压力
    angle3_raw = max(0, min(1800, int(round(s.angle_deg * 10.0 + 900.0))))
    angle3_0, angle3_1 = u16_bytes(angle3_raw)
    cabin = max(1, min(4, int(round(s.cabin_id))))
    slope_state = max(1, min(3, int(round(s.slope_state))))
    firewall_raw = max(0, min(65535, int(round(s.firewall_pressure_kpa))))
    fw0, fw1 = u16_bytes(firewall_raw)

    alarm_code = max(0, min(122, int(round(s.alarm_code))))
    gear_state = max(1, min(4, int(round(s.gear_state))))
    estop_state = max(0, min(3, int(round(s.emergency_stop))))

    if str(model).strip().lower() in {"50", "105"}:
        batt_raw = max(0, min(65535, int(round(s.battery_v * 100.0))))
        oil_raw = max(0, min(65535, int(round(s.oil_pressure_kpa))))
        bv0, bv1 = u16_bytes(batt_raw)
        op0, op1 = u16_bytes(oil_raw)
        intake_raw = max(0, min(65535, int(round(s.intake_pressure_kpa))))
        surface_raw = max(0, min(65535, int(round(s.surface_temp_c))))
        exhaust_raw = max(0, min(65535, int(round(s.exhaust_temp_c))))
        coolant_raw = max(0, min(65535, int(round(s.coolant_temp_c))))
        ip0, ip1 = u16_bytes(intake_raw)
        st0, st1 = u16_bytes(surface_raw)
        et0, et1 = u16_bytes(exhaust_raw)
        ct0, ct1 = u16_bytes(coolant_raw)
        mileage_raw = max(0, min(4294967295, int(round(s.total_mileage_km * 1000.0))))
        runtime_s_raw = max(0, min(4294967295, int(round(s.runtime_min * 60.0))))
        ml0, ml1, ml2, ml3 = u32_bytes(mileage_raw)
        rs0, rs1, rs2, rs3 = u32_bytes(runtime_s_raw)
        co_50_raw = max(0, min(65535, int(round(s.co_ppm))))
        intake_temp_raw = max(0, min(65535, int(round(s.intake_temp_c))))
        co50_0, co50_1 = u16_bytes(co_50_raw)
        it0, it1 = u16_bytes(intake_temp_raw)
        load_state = max(0, min(255, int(round(1 if s.payload_tons > 1 else 0))))
        methane_u8 = max(0, min(255, int(round(s.methane_pctlel * 10.0))))
        diesel_u8 = max(0, min(255, int(round(s.diesel_level_cm))))
        water_u8 = max(0, min(100, int(round(s.water_tank_level_pct))))
        return [
            (0x18F181A0, [alarm_code, gear_state, estop_state, hyd_oil_temp, hyd_oil_level, make_up_oil_p, fire_sys_p, shua_qu_state]),
            (0x18F182A0, [bp0, bp1, tp0, tp1, sp0, sp1, cp0, cp1]),
            (0x18F183A0, [bv0, bv1, s0, s1, r0, r1, op0, op1]),
            (0x18F184A0, [ip0, ip1, st0, st1, et0, et1, ct0, ct1]),
            (0x18F185A0, [ml0, ml1, ml2, ml3, rs0, rs1, rs2, rs3]),
            (0x18F186A0, [0, 0, 0, 0, load_state, methane_u8, diesel_u8, water_u8]),
            (0x18F189A0, [0, 0, 0, 0, co50_0, co50_1, it0, it1]),
        ]

    return [
        (0x18F181A0, [alarm_code, gear_state, estop_state, hyd_oil_temp, hyd_oil_level, make_up_oil_p, fire_sys_p, shua_qu_state]),
        (0x18F182A0, [bp0, bp1, tp0, tp1, sp0, sp1, cp0, cp1]),
        (0x18F183A0, [m0, m1, s0, s1, r0, r1, a0, a1]),
        (0x18F184A0, [coolant_u8, surface_u8, exhaust_u8, intake_p_u8, water_tank_u8, oil_u8, diesel_level_u8, intake_temp_u8]),
        (0x18F185A0, [tm0, tm1, rt0, rt1, af0, af1, 0, 0]),
        (0x18F189A0, [pw0, pw1, co0, co1, 0, 0, 0, 0]),
        (0x18F190A0, [angle3_0, angle3_1, cabin, slope_state, fw0, fw1, 0, 0]),
    ]


def main() -> None:
    random.seed(SEED)
    out_path = Path("/home/sunrise/oellm_agent/sim_data/sim_can_frames_10min_10hz.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    l1_events = [
        "coolant_overheat",
        "hydraulic_oil_overheat",
        "hydraulic_oil_level_low",
        "water_tank_low",
        "diesel_level_low",
        "make_up_oil_pressure_out",
        "fire_system_pressure_out",
        "brake_pressure_low",
        "travel_pressure_low",
        "system_pressure_low",
        "methane_high",
        "emergency_stop",
        "co_high",
        "overload",
        "steep_slope",
    ]
    llm_cases = [
        "coolant_fast_rise",
        "hydraulic_oil_temp_rise",
        "hydraulic_oil_level_drop",
        "water_tank_drop",
        "make_up_oil_pressure_drift",
        "fire_system_pressure_drift",
        "brake_pressure_softening",
        "travel_pressure_softening",
        "system_pressure_drift_down",
        "methane_rising_near_limit",
        "speed_up_trend",
        "angle_swing",
        "payload_rise",
        "co_rise",
        "mileage_growth",
        "runtime_growth",
        "cabin_shift",
        "slope_state_switch",
    ]

    active_llm_case = ""
    case_total_len = 0
    case_elapsed = 0
    l1_count = 0
    critical_count = 0

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "t_sec", "timestamp", "frame_id_hex",
            "byte0", "byte1", "byte2", "byte3", "byte4", "byte5", "byte6", "byte7",
            "can_heartbeat_ok", "llm_case",
        ])

        for step in range(TOTAL_STEPS):
            t_sec = round(step * DT, 1)
            ts = (START_TS + timedelta(seconds=t_sec)).isoformat()
            s = base_state(step)

            # 趋势类问题段：稀疏出现，避免长期偏离正常分布
            if case_elapsed >= case_total_len:
                if random.random() < LLM_CASE_PROB:
                    active_llm_case = random.choice(llm_cases)
                    case_total_len = random.randint(16, 36)  # 1.6-3.6s
                    case_elapsed = 0
                else:
                    active_llm_case = ""
                    case_total_len = 0
                    case_elapsed = 0

            if active_llm_case:
                phase = case_elapsed / max(1, case_total_len - 1)
                inject_llm_case(s, active_llm_case, phase)
                case_elapsed += 1

            # 临界状态与硬事件注入
            roll = random.random()
            if roll < OVERSHOOT_EVENT_PROB:
                critical_count += 1
                inject_overshoot_event(s, random.choice(l1_events))
            elif roll < OVERSHOOT_EVENT_PROB + CRITICAL_EVENT_PROB:
                critical_count += 1
                inject_llm_case(s, random.choice(llm_cases), random.random())
            elif random.random() < 0.0001:
                critical_count += 1
                inject_critical_event(s, random.choice(l1_events))

            s.risk_level, s.warning_tag = classify_warning(s)
            s.alarm_code = derive_alarm_code(s, s.warning_tag)
            if s.risk_level != "normal":
                critical_count += 1
            if s.alarm_code != 0:
                l1_count += 1

            frames = encode_frames(s)
            for fid, payload in frames:
                w.writerow([
                    step,
                    f"{t_sec:.1f}",
                    ts,
                    f"0x{fid:08X}",
                    payload[0], payload[1], payload[2], payload[3],
                    payload[4], payload[5], payload[6], payload[7],
                    s.can_heartbeat_ok,
                    active_llm_case,
                ])

    print(f"Generated: {out_path}")
    print(f"steps={TOTAL_STEPS}, rows={TOTAL_STEPS * 4}, dt={DT}s, duration={DURATION_SEC}s")
    print(f"critical_event_ratio≈{(critical_count / TOTAL_STEPS):.4%} (target≈6.0%)")
    print(f"alarm_ratio≈{(l1_count / TOTAL_STEPS):.4%} (target≈1.0% hard alarm)")


if __name__ == "__main__":
    main()
