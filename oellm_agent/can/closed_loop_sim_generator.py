#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
 
from oellm_agent.can.simulate_sensor_pipeline import (
    DT,
    SensorState,
    base_state,
    classify_warning,
    derive_alarm_code,
    encode_frames,
)

CONTROL_ACTIONS = {
    "EMERGENCY_STOP",
    "FORWARD",
    "REVERSE",
    "ACCELERATE",
    "DECELERATE",
    "BRAKE",
}

# 闭环仿真内部判级阈值（与当前运行策略保持一致，避免沿用旧协议阈值导致误报）
SIM_TH = {
    "coolant_warn": 95.0,
    "coolant_high": 98.0,
    "intake_warn_low": 104.0,
    "intake_low": 90.0,
    "water_warn_low": 20.0,
    "water_low": 10.0,
    "make_up_warn_low": 23.0,
    "make_up_low": 16.0,
    "travel_warn_low": 25.0,
    "travel_warn_high": 250.0,
    "travel_high": 380.0,
    "system_warn_low": 150.0,
    "system_low": 80.0,
    "brake_warn_low": 150.0,
    "brake_low": 60.0,
    "speed_warn_high": 7.92,
    "speed_alarm_high": 9.0,
}


class ClosedLoopSim:
    """Closed-loop mine truck process model.

    关键点：
    - 载重/液位初值在启动时随机一次，运行中连续变化，不每步重置。
    - 用纵向动力学近似：F=ma，考虑坡阻、滚阻、制动与牵引。
    - 温度/压力用一阶惯性响应，贴近真实“慢变量”特性。
    """

    def __init__(
        self,
        seed: int = 20260422,
        profile: str = "balanced",
        cooling_fault_prob: Optional[float] = None,
        warning_fault_prob: Optional[float] = None,
    ):
        random.seed(seed)
        self.step = 0
        self.last_action = "FORWARD"
        self.profile = (profile or "balanced").strip().lower()

        profile_cfg = {
            "easy": {
                "cooling_fault_prob": 0.0005,
                "warning_fault_prob": 0.0008,
                "phase_scale": 1.15,
            },
            "balanced": {
                "cooling_fault_prob": 0.0008,
                "warning_fault_prob": 0.0012,
                "phase_scale": 1.0,
            },
            "stress": {
                "cooling_fault_prob": 0.0016,
                "warning_fault_prob": 0.0022,
                "phase_scale": 0.85,
            },
        }
        if self.profile not in profile_cfg:
            self.profile = "balanced"
        cfg = profile_cfg[self.profile]
        self.cooling_fault_prob = float(cooling_fault_prob) if cooling_fault_prob is not None else float(cfg["cooling_fault_prob"])
        self.warning_fault_prob = float(warning_fault_prob) if warning_fault_prob is not None else float(cfg["warning_fault_prob"])
        self.phase_scale = float(cfg["phase_scale"])

        # -------- 固定初始工况（启动后连续演化）--------
        self.base_mass_tons = random.uniform(190.0, 220.0)  # 空车质量
        self.payload_tons = random.uniform(8.0, 12.0)
        self.water_tank_level_pct = random.uniform(72.0, 96.0)
        self.hydraulic_oil_level_pct = random.uniform(76.0, 94.0)
        self.diesel_level_cm = random.uniform(58.0, 88.0)

        self.direction = 1  # 1 forward, -1 reverse
        self.control_gate_switch_sec = random.uniform(10.0, 20.0)
        self.neutral_gate_duration_sec = random.uniform(2.0, 4.0)
        # 只保留前进档工况，避免模拟器在闭环里自行切到 REVERSE
        self.control_gate_drive_gear = 3
        # 常识约束：矿车起步速度应接近 0，不应初始即高速
        self.speed_mps = random.uniform(0.0, 0.04)  # m/s
        self.low_speed_sec = 0.0

        # 遥控器控制模型：制动状态下才能选前进/后退；加速/减速是按键脉冲，每次调整80rpm
        self.remote_idle_rpm = 800.0
        self.remote_rpm_step = 80.0
        self.remote_max_rpm = SIM_TH.get("engine_rpm_warn", 2200.0) - self.remote_rpm_step
        self.remote_target_rpm = self.remote_idle_rpm
        self.engine_rpm_actual = self.remote_idle_rpm
        self.remote_brake_latched = True
        self.remote_direction_selected = False
        self.remote_last_button = "BRAKE"
        self.remote_button_elapsed_sec = 999.0
        self.remote_button_min_interval_sec = 1.5

        # 控制输入平滑：避免油门/制动突变导致速度衰减“台阶感”
        self.throttle_cmd = 0.0
        self.brake_cmd = 0.45
        self.decel_limit = -0.55
        self.action_blend = "BRAKE"
        self.action_hold_remaining_sec = 0.0

        # 班次工况状态机：装载 -> 运输 -> 卸载 -> 返程
        self.phase = "load"
        self.phase_elapsed_sec = 0.0
        self.phase_target_sec = random.uniform(22.0, 38.0) * self.phase_scale
        self.low_speed_accel_stall_sec = 0.0

        # 慢变量状态
        self.coolant_temp_c = random.uniform(48.0, 64.0)
        self.hydraulic_oil_temp_c = random.uniform(35.0, 43.0)
        self.exhaust_temp_c = random.uniform(46.0, 52.0)
        self.surface_temp_c = random.uniform(43.0, 46.0)

        # 环境/散热异常场景：用于验证“高温->降速->回落”闭环效果
        # 注：不再按固定时长自动消失，改为“满足恢复判据后退出”。
        self.cooling_fault_active = False
        self.cooling_fault_elapsed_sec = 0.0
        self.cooling_fault_cooldown_sec = 60.0
        self.cooling_fault_cooldown_remaining_sec = 0.0
        self.cooling_fault_min_hold_sec = 12.0
        self.cooling_fault_recover_confirm_sec = 5.0
        self.cooling_fault_recover_elapsed_sec = 0.0

        # warning级场景注入（非速度类）：观察策略是否能“修复”并回落到normal
        # 注：不再按固定时长自动消失，改为“满足恢复判据后退出”。
        self.warning_fault = ''  # coolant_warning / intake_pressure_low / water_tank_low
        self.warning_fault_elapsed_sec = 0.0
        self.warning_fault_cooldown_sec = 45.0
        self.warning_fault_cooldown_remaining_sec = 0.0
        self.warning_fault_min_hold_sec = 10.0
        self.warning_fault_recover_confirm_sec = 4.0 
        self.warning_fault_recover_elapsed_sec = 0.0

        # 约 100s 后允许出现高温 warning，用于验证“高温->减速->回落”闭环效果
        self.warmup_sec = 90.0
        self.heat_stress_active = False
        self.heat_stress_elapsed_sec = -1.0
        self.heat_stress_recover_elapsed_sec = 0.0
        self.heat_stress_min_hold_sec = 24.0
        self.level_fault = ""
        self.level_fault_elapsed_sec = 0.0
        self.level_fault_cooldown_sec = 70.0
        self.level_fault_cooldown_remaining_sec = 35.0
        self.level_fault_min_hold_sec = 14.0
        self.level_fault_recover_elapsed_sec = 0.0

        # 基线环境控制：避免把无关环境风险（甲烷/CO）掺进控制策略评估
        self.limit_env_gas = True

    @property
    def mass_kg(self) -> float:
        return (self.base_mass_tons + self.payload_tons) * 1000.0

    def _action_to_throttle_brake(self, action: str) -> Tuple[float, float]:
        throttle, brake = 0.0, 0.0
        if action == "ACCELERATE":
            throttle = 0.36
        elif action == "DECELERATE":
            throttle, brake = 0.0, 0.10
        elif action == "BRAKE":
            throttle, brake = 0.04, 0.34
        elif action == "EMERGENCY_STOP":
            throttle, brake = 0.0, 1.0
        elif action in {"FORWARD", "REVERSE"}:
            throttle, brake = 0.0, 0.0
        return throttle, brake

    def _smooth_action(self, action: str) -> str:
        action = str(action or "FORWARD").upper()
        if action == "EMERGENCY_STOP":
            self.action_blend = action
            self.action_hold_remaining_sec = 0.0
            return action
        if self.action_hold_remaining_sec > 0.0 and action != self.action_blend and action not in {"BRAKE", "EMERGENCY_STOP", "FORWARD", "REVERSE"}:
            self.action_hold_remaining_sec = max(0.0, self.action_hold_remaining_sec - DT)
            return self.action_blend
        if action != self.action_blend:
            self.action_blend = action
            self.action_hold_remaining_sec = 0.8
        else:
            self.action_hold_remaining_sec = max(0.0, self.action_hold_remaining_sec - DT)
        return self.action_blend

    def _apply_remote_action(self, action: str) -> Tuple[float, float]:
        action = str(action or "FORWARD").upper()
        self.remote_button_elapsed_sec += DT
        if action in {"BRAKE", "EMERGENCY_STOP"}:
            self.remote_brake_latched = True
            self.remote_target_rpm = self.remote_idle_rpm
            self.engine_rpm_actual += min(1.0, DT / 3.0) * (self.remote_idle_rpm - self.engine_rpm_actual)
            self.remote_last_button = action
            self.remote_button_elapsed_sec = 0.0
            return 0.0, 0.42 if action == "BRAKE" else 1.0
        if action in {"FORWARD", "REVERSE"}:
            # 闭环模拟器不再允许通过动作切换行驶方向，只允许保持前进方向
            if self.remote_brake_latched or not self.remote_direction_selected:
                self.direction = 1
                self.remote_direction_selected = True
                self.remote_brake_latched = False
                self.remote_last_button = "FORWARD"
                self.remote_button_elapsed_sec = 0.0
            return 0.0, 0.0
        if action == "ACCELERATE" and (not self.remote_brake_latched) and self.remote_button_elapsed_sec >= self.remote_button_min_interval_sec:
            self.remote_target_rpm = min(self.remote_max_rpm, self.remote_target_rpm + self.remote_rpm_step)
            self.remote_last_button = action
            self.remote_button_elapsed_sec = 0.0
        elif action == "DECELERATE" and (not self.remote_brake_latched) and self.remote_button_elapsed_sec >= self.remote_button_min_interval_sec:
            self.remote_target_rpm = max(self.remote_idle_rpm, self.remote_target_rpm - self.remote_rpm_step)
            self.remote_last_button = action
            self.remote_button_elapsed_sec = 0.0
        rpm_response_tau_sec = 4.0
        rpm_alpha = min(1.0, DT / rpm_response_tau_sec)
        self.engine_rpm_actual += rpm_alpha * (self.remote_target_rpm - self.engine_rpm_actual)
        rpm_ratio = (self.engine_rpm_actual - self.remote_idle_rpm) / max(1.0, self.remote_max_rpm - self.remote_idle_rpm)
        throttle = 0.04 + 0.30 * max(0.0, min(1.0, rpm_ratio)) if self.remote_direction_selected and not self.remote_brake_latched else 0.0
        return throttle, 0.0

    def _update_direction(self, action: str) -> None:
        # 闭环里固定前进方向，不再允许模拟器自行换向
        if action in {"FORWARD", "REVERSE"}:
            self.direction = 1

    def _update_speed(self, action: str, angle_deg: float) -> None:
        action = self._smooth_action(action)
        self._update_direction(action)
        throttle_target, brake_target = self._apply_remote_action(action)

        # 一阶平滑控制输入（避免动作切换导致突变减速度）
        alpha_throttle = 0.18 if action == "ACCELERATE" and self.speed_mps < 1.5 else 0.08
        alpha_brake = 0.10
        self.throttle_cmd += alpha_throttle * (throttle_target - self.throttle_cmd)
        self.brake_cmd += alpha_brake * (brake_target - self.brake_cmd)
        throttle = max(0.0, min(1.0, self.throttle_cmd))
        brake = max(0.0, min(1.0, self.brake_cmd))

        g = 9.81
        v = max(0.0, self.speed_mps)
        theta = math.radians(angle_deg)

        # 重载/上坡自适应牵引：在FORWARD/ACCELERATE且车速偏低时，自动提高等效油门
        payload_ratio = max(0.0, min(1.0, (self.payload_tons - 80.0) / 120.0))
        uphill_ratio = max(0.0, min(1.0, angle_deg / 10.0))
        speed_ratio = max(0.0, min(1.0, v / 1.8))  # 约 6.5 km/h 对应 1.0
        f_trac_max = 0.20 * self.mass_kg * g
        f_grade = self.mass_kg * g * math.sin(theta)
        f_roll = 0.025 * self.mass_kg * g * math.cos(theta)
        f_drag = 0.9 * v * v
        low_speed_boost = 0.0
        if action == "ACCELERATE":
            torque_boost = 0.05 * payload_ratio + 0.06 * uphill_ratio
            # 低速时放大补偿，高速后自动减弱
            throttle += torque_boost * (1.0 - 0.65 * speed_ratio)
            if v < 1.5:
                low_speed_boost = (1.5 - v) / 1.5
                throttle += 0.08 * low_speed_boost
            if v < 1.5 and self.low_speed_accel_stall_sec > 4.0:
                throttle += min(0.18, 0.03 * (self.low_speed_accel_stall_sec - 4.0))
            throttle = max(0.0, min(1.0, throttle))

        f_brake = brake * 0.30 * self.mass_kg * g
        # 牵引力、坡阻、滚阻、空气阻力（简化）
        f_trac = throttle * f_trac_max
        resist = f_roll + f_drag + max(0.0, f_grade)
        net_force = f_trac - resist - f_brake
        acc = net_force / max(1.0, self.mass_kg)

        # 数值稳定与极限速度
        # 非急停工况限制最大减速度，并做缓变，避免衰减系数台阶变化
        target_decel_limit = -3.5 if action == "EMERGENCY_STOP" else -0.45
        if self.speed_mps < 0.8 and action != "EMERGENCY_STOP":
            target_decel_limit = max(target_decel_limit, -0.25)
        self.decel_limit += 0.06 * (target_decel_limit - self.decel_limit)
        min_acc = self.decel_limit

        # data_3 常态低速作业：起步助力应温和，避免模拟器远高于真实 0.3~0.7m/s 常态
        if action == "ACCELERATE" and self.speed_mps < 1.0:
            min_positive_acc = 0.04 + 0.18 * max(0.0, min(1.0, (1.0 - self.speed_mps) / 1.0))
            if self.low_speed_accel_stall_sec > 6.0:
                min_positive_acc += min(0.10, 0.015 * (self.low_speed_accel_stall_sec - 6.0))
            acc = max(acc, min_positive_acc)

        if action in {"FORWARD", "ACCELERATE"} and getattr(self, "travel_pressure_bar", 0.0) > 140.0 and self.speed_mps < 0.8:
            acc = max(acc, 0.03)

        max_acc = 0.22 if (action == "ACCELERATE" and self.speed_mps < 1.2) else 0.16
        acc = max(min_acc, min(max_acc, acc))

        # 速度积分 + 低速黏滞抑制（避免噪声导致反复贴零）
        next_speed = self.speed_mps + acc * DT + random.gauss(0.0, 0.003)
        if action in {"FORWARD", "ACCELERATE"} and 0.0 < next_speed < 0.08:
            next_speed = 0.08
        prev_speed = self.speed_mps
        self.speed_mps = max(0.0, next_speed)
        self.speed_mps = min(self.speed_mps, 1.35 if self.direction > 0 else 1.25)  # 前进/后退上限，贴近data_3常态速度
        if action == "ACCELERATE" and self.speed_mps < 1.5 and (self.speed_mps - prev_speed) < 0.015:
            self.low_speed_accel_stall_sec += DT
        else:
            self.low_speed_accel_stall_sec = max(0.0, self.low_speed_accel_stall_sec - 2.0 * DT)

    def _update_consumables(self, throttle: float, brake: float) -> None:
        # 物理一致性：负载/动作与温度共同影响介质消耗
        load_factor = self.payload_tons / 180.0

        # 柴油：牵引为主，制动和高负载带来附加工况损耗
        diesel_burn = 0.0006 + 0.0038 * throttle + 0.0006 * load_factor + 0.0003 * brake
        self.diesel_level_cm = max(0.0, self.diesel_level_cm - diesel_burn * DT)

        # 冷却水：高温时蒸发/泄放风险增大，温度越高下降越快
        coolant_temp_excess = max(0.0, self.coolant_temp_c - 88.0)
        water_drop = 0.00018 + 0.0005 * throttle + 0.0012 * (coolant_temp_excess / 10.0) + 0.0004 * load_factor
        self.water_tank_level_pct = max(0.0, self.water_tank_level_pct - water_drop * DT)

        # 液压油液位：温度高+频繁动作时损耗更快
        hyd_temp_excess = max(0.0, self.hydraulic_oil_temp_c - 60.0)
        hyd_drop = 0.00012 + 0.00045 * throttle + 0.00035 * brake + 0.0010 * (hyd_temp_excess / 10.0)
        self.hydraulic_oil_level_pct = max(0.0, self.hydraulic_oil_level_pct - hyd_drop * DT)

    def _update_cycle_phase(self) -> None:
        self.phase_elapsed_sec += DT

        # 阶段切换
        if self.phase_elapsed_sec >= self.phase_target_sec:
            if self.phase == "load":
                self.phase = "haul"
                self.phase_target_sec = random.uniform(70.0, 120.0) * self.phase_scale
            elif self.phase == "haul":
                self.phase = "dump"
                self.phase_target_sec = random.uniform(16.0, 28.0) * self.phase_scale
            elif self.phase == "dump":
                self.phase = "return"
                self.phase_target_sec = random.uniform(60.0, 105.0) * self.phase_scale
            else:
                self.phase = "load"
                self.phase_target_sec = random.uniform(22.0, 38.0) * self.phase_scale
            self.phase_elapsed_sec = 0.0

        # 分阶段载重变化（连续）
        if self.phase == "load":
            # 装载阶段，载重缓慢增加，避免一开始因快速增载造成低速平台
            self.payload_tons = min(180.0, self.payload_tons + random.uniform(0.04, 0.14))
        elif self.phase == "haul":
            # 运输阶段，载重基本稳定
            self.payload_tons = max(85.0, min(180.0, self.payload_tons + random.gauss(0.0, 0.08)))
        elif self.phase == "dump":
            # 卸载阶段，载重快速下降
            self.payload_tons = max(20.0, self.payload_tons - random.uniform(0.35, 0.95))
        elif self.phase == "return":
            # 返程阶段，空载小波动
            self.payload_tons = max(18.0, min(45.0, self.payload_tons + random.gauss(0.0, 0.06)))

    def _update_temperatures(self, throttle: float, brake: float) -> None:
        # 低速计时仅用于工况统计，不作为温升源
        if self.speed_mps < 1.5:
            self.low_speed_sec += DT
        else:
            self.low_speed_sec = max(0.0, self.low_speed_sec - DT)

        sim_time_sec = self.step * DT
        if self.heat_stress_active:
            self.heat_stress_elapsed_sec += DT

        # 以较低概率注入“散热能力下降”场景。
        # 仅在预热完成后启用；退出由恢复判据决定，不再按固定时长自动恢复。
        if self.cooling_fault_cooldown_remaining_sec > 0.0:
            self.cooling_fault_cooldown_remaining_sec = max(0.0, self.cooling_fault_cooldown_remaining_sec - DT)

        if (
            sim_time_sec >= self.warmup_sec
            and (not self.cooling_fault_active)
            and self.cooling_fault_cooldown_remaining_sec <= 0.0
            and random.random() < self.cooling_fault_prob
        ):
            self.cooling_fault_active = True
            self.cooling_fault_elapsed_sec = 0.0
            self.cooling_fault_recover_elapsed_sec = 0.0

        if self.cooling_fault_active:
            self.cooling_fault_elapsed_sec += DT

        # 非速度 warning 场景（更常见一些，便于验证“修复效果”）
        # 仅在预热完成后注入 + 冷却期，防止同类告警长期锁死
        if self.warning_fault_cooldown_remaining_sec > 0.0:
            self.warning_fault_cooldown_remaining_sec = max(0.0, self.warning_fault_cooldown_remaining_sec - DT)

        if (
            sim_time_sec >= self.warmup_sec
            and self.warning_fault == ''
            and self.warning_fault_cooldown_remaining_sec <= 0.0
            and random.random() < self.warning_fault_prob
        ):
            self.warning_fault = random.choice(['coolant_warning', 'intake_pressure_low', 'water_tank_low'])
            self.warning_fault_elapsed_sec = 0.0
            self.warning_fault_recover_elapsed_sec = 0.0

        if self.warning_fault:
            self.warning_fault_elapsed_sec += DT

        if self.level_fault_cooldown_remaining_sec > 0.0:
            self.level_fault_cooldown_remaining_sec = max(0.0, self.level_fault_cooldown_remaining_sec - DT)
        if sim_time_sec >= 50.0 and self.level_fault == '' and self.level_fault_cooldown_remaining_sec <= 0.0 and random.random() < 0.018:
            self.level_fault = random.choice(['hydraulic_oil_level_low', 'diesel_level_low', 'water_tank_low'])
            self.level_fault_elapsed_sec = 0.0
            self.level_fault_recover_elapsed_sec = 0.0
        if self.level_fault:
            self.level_fault_elapsed_sec += DT

        load_factor = self.payload_tons / 180.0

        # 速度带来的冷却风量：车速越高散热越好；主动减速/制动时额外降负载散热
        cooling_air = min(1.8, 0.24 * self.speed_mps)
        decel_cooling = 6.0 * brake + (4.0 if brake > 0.05 else 0.0)

        # 散热故障/热负荷场景时冷却目标提高，观察 agent 是否触发降速及回落
        cooling_fault_bias = 6.0 if self.cooling_fault_active else 0.0
        heat_stress_bias = 0.0
        if self.heat_stress_active:
            progress = min(1.0, self.heat_stress_elapsed_sec / 18.0)
            recovery_scale = 0.45 if brake > 0.05 else 1.0
            heat_stress_bias = 12.0 * progress * recovery_scale

        run_heat_bias = min(8.0, 0.045 * max(0.0, sim_time_sec - 20.0))
        surface_run_bias = min(18.0, 0.13 * max(0.0, sim_time_sec - 20.0))

        coolant_target = 83.5 + run_heat_bias + 6.0 * throttle + 2.8 * load_factor + 0.20 * brake + cooling_fault_bias + heat_stress_bias - cooling_air - decel_cooling
        hyd_target = 52.0 + 0.55 * run_heat_bias + 5.2 * throttle + 2.0 * load_factor + 0.20 * brake - 0.6 * cooling_air - 0.7 * decel_cooling
        exh_target = 54.0 + 0.7 * surface_run_bias + 7.5 * throttle + 1.8 * load_factor - 1.2 * decel_cooling
        surface_target = 108.0 + surface_run_bias + 5.0 * throttle + 2.5 * load_factor - 1.4 * decel_cooling

        # 一阶惯性（平滑），温度随运行时间逐步累积，减速后目标下降并回落
        self.coolant_temp_c += 0.055 * (coolant_target - self.coolant_temp_c) + random.gauss(0.0, 0.03)
        self.hydraulic_oil_temp_c += 0.055 * (hyd_target - self.hydraulic_oil_temp_c) + random.gauss(0.0, 0.03)
        self.exhaust_temp_c += 0.065 * (exh_target - self.exhaust_temp_c) + random.gauss(0.0, 0.03)
        self.surface_temp_c += 0.045 * (surface_target - self.surface_temp_c) + random.gauss(0.0, 0.04)

        self.coolant_temp_c = min(115.0, max(75.0, self.coolant_temp_c))
        self.hydraulic_oil_temp_c = min(95.0, max(35.0, self.hydraulic_oil_temp_c))
        self.exhaust_temp_c = min(98.0, max(40.0, self.exhaust_temp_c))
        self.surface_temp_c = min(165.0, max(80.0, self.surface_temp_c))

        if self.heat_stress_active:
            recovered_heat = self.coolant_temp_c < (SIM_TH["coolant_warn"] - 0.6) and brake > 0.05 and self.heat_stress_elapsed_sec >= self.heat_stress_min_hold_sec
            if recovered_heat:
                self.heat_stress_recover_elapsed_sec += DT
            else:
                self.heat_stress_recover_elapsed_sec = 0.0
            if self.heat_stress_recover_elapsed_sec >= 8.0:
                self.heat_stress_active = False

        # cooling故障退出规则：由工况恢复判据驱动，不按固定时长自动消失
        if self.cooling_fault_active:
            recovered_now = (
                self.coolant_temp_c < (SIM_TH["coolant_warn"] - 1.2)
                and self.speed_mps < 1.8
                and self.throttle_cmd < 0.22
            )
            if self.cooling_fault_elapsed_sec < self.cooling_fault_min_hold_sec:
                recovered_now = False

            if recovered_now:
                self.cooling_fault_recover_elapsed_sec += DT
            else:
                self.cooling_fault_recover_elapsed_sec = 0.0

            if self.cooling_fault_recover_elapsed_sec >= self.cooling_fault_recover_confirm_sec:
                self.cooling_fault_active = False
                self.cooling_fault_elapsed_sec = 0.0
                self.cooling_fault_recover_elapsed_sec = 0.0
                self.cooling_fault_cooldown_remaining_sec = self.cooling_fault_cooldown_sec

    def _classify_warning_local(self, s: SensorState) -> Tuple[str, str]:
        if int(s.emergency_stop or 0) in (1, 2, 3):
            return "danger", "emergency_stop"
        if int(s.can_heartbeat_ok or 1) == 0:
            return "danger", "communication_fault"

        # 硬底线
        if float(s.coolant_temp_c or 0.0) >= SIM_TH["coolant_high"]:
            return "danger", "coolant_warning"
        if float(s.speed_mps or 0.0) >= SIM_TH["speed_alarm_high"]:
            return "danger", "speed_high"

        # warning 区
        if float(s.coolant_temp_c or 0.0) >= SIM_TH["coolant_warn"]:
            return "warning", "coolant_warning"
        if float(s.intake_pressure_kpa or 0.0) < SIM_TH["intake_warn_low"]:
            return "warning", "intake_pressure_low"
        if float(s.water_tank_level_pct or 100.0) < SIM_TH["water_warn_low"]:
            return "warning", "water_tank_low"
        if float(s.make_up_oil_pressure_bar or 999.0) < SIM_TH["make_up_warn_low"]:
            return "warning", "make_up_oil_pressure_low"

        travel = float(s.travel_pressure_bar or 0.0)
        if travel < SIM_TH["travel_warn_low"] or travel > SIM_TH["travel_warn_high"]:
            return "warning", "travel_pressure_warning"
        if float(s.system_pressure_bar or 0.0) < SIM_TH["system_warn_low"]:
            return "warning", "system_pressure_low"
        if float(s.brake_pressure_bar or 0.0) < SIM_TH["brake_warn_low"]:
            return "warning", "brake_pressure_low"
        if float(s.speed_mps or 0.0) > SIM_TH["speed_warn_high"]:
            return "warning", "speed_high"

        return "normal", ""

    def _derive_alarm_code_local(self, s: SensorState, warning_tag: str) -> int:
        if not warning_tag:
            return 0
        if warning_tag == "emergency_stop":
            return 101
        if warning_tag == "communication_fault":
            return 103
        if warning_tag == "coolant_warning":
            return 106 if float(s.coolant_temp_c or 0.0) >= SIM_TH["coolant_high"] else 0
        if warning_tag == "water_tank_low":
            return 112 if float(s.water_tank_level_pct or 100.0) < SIM_TH["water_low"] else 0
        if warning_tag == "make_up_oil_pressure_low":
            return 121 if float(s.make_up_oil_pressure_bar or 999.0) < SIM_TH["make_up_low"] else 0
        if warning_tag == "travel_pressure_warning":
            v = float(s.travel_pressure_bar or 0.0)
            return 117 if (v < SIM_TH["travel_warn_low"] or v > SIM_TH["travel_high"]) else 0
        if warning_tag == "brake_pressure_low":
            return 118 if float(s.brake_pressure_bar or 0.0) < SIM_TH["brake_low"] else 0
        if warning_tag == "system_pressure_low":
            return 120 if float(s.system_pressure_bar or 0.0) < SIM_TH["system_low"] else 0
        if warning_tag == "speed_high":
            return 115 if float(s.speed_mps or 0.0) > SIM_TH["speed_alarm_high"] else 0
        return 0

    def step_once(self, action: str) -> SensorState:
        s = base_state(self.step)

        # 启动阶段速度包络：避免刚起步就出现高车速
        sim_time_sec = self.step * DT
        if sim_time_sec >= 95.0 and not self.heat_stress_active and self.heat_stress_elapsed_sec < 0.0:
            self.heat_stress_active = True
            self.heat_stress_elapsed_sec = 0.001
            self.heat_stress_recover_elapsed_sec = 0.0

        # 班次工况先演化，再叠加控制动作
        self._update_cycle_phase()

        # 分阶段坡度偏置：运输多上坡，返程多下坡；装载/卸载阶段默认近似平地
        if self.phase == "haul":
            s.slope_state = 2
            s.angle_deg = max(-16.0, min(16.0, s.angle_deg + random.uniform(1.0, 4.5)))
        elif self.phase == "return":
            s.slope_state = 3
            s.angle_deg = max(-16.0, min(16.0, s.angle_deg - random.uniform(1.0, 4.5)))
        else:
            s.slope_state = 1
            s.angle_deg = max(-2.0, min(2.0, random.gauss(0.0, 0.5)))

        if sim_time_sec < self.control_gate_switch_sec:
            action = "BRAKE"
        elif sim_time_sec < self.control_gate_switch_sec + self.neutral_gate_duration_sec:
            action = "BRAKE"
        elif self.remote_brake_latched and not self.remote_direction_selected:
            action = "REVERSE" if self.control_gate_drive_gear == 4 else "FORWARD"

        throttle, brake = self._action_to_throttle_brake(action)
        self._update_speed(action, s.angle_deg)
        self._update_temperatures(throttle, brake)
        self._update_consumables(throttle, brake)

        # 启动前 30s 做速度包络上限，单位保持 m/s
        startup_cap_mps = 1.0 + 0.01 * min(30.0, sim_time_sec)  # 1.0 -> 1.3 m/s
        if sim_time_sec < 30.0:
            self.speed_mps = min(self.speed_mps, startup_cap_mps)

        speed_mps = self.speed_mps
        s.speed_mps = speed_mps
        if sim_time_sec < self.control_gate_switch_sec + self.neutral_gate_duration_sec:
            s.gear_state = 1 if sim_time_sec < self.control_gate_switch_sec else 2
            self.remote_brake_latched = True
            self.remote_direction_selected = False
            self.remote_target_rpm = self.remote_idle_rpm
            if sim_time_sec < self.control_gate_switch_sec:
                self.engine_rpm_actual += min(1.0, DT / 3.0) * (self.remote_idle_rpm - self.engine_rpm_actual)
            else:
                self.engine_rpm_actual += min(1.0, DT / 5.0) * (920.0 - self.engine_rpm_actual)
            self.speed_mps = min(self.speed_mps, 0.04)
            s.speed_mps = self.speed_mps
        elif self.remote_brake_latched and not self.remote_direction_selected:
            self.direction = 1
            s.gear_state = self.control_gate_drive_gear
            self.remote_direction_selected = True
            self.remote_brake_latched = False
            self.action_blend = "FORWARD"
        else:
            s.gear_state = 3
        s.emergency_stop = 1 if action == "EMERGENCY_STOP" else 0

        # 关键“固定初值+连续演化”变量
        s.payload_tons = self.payload_tons
        s.water_tank_level_pct = self.water_tank_level_pct
        s.hydraulic_oil_level_pct = self.hydraulic_oil_level_pct
        s.diesel_level_cm = self.diesel_level_cm

        # 动力关联量（按 data_3 常态区间生成）
        s.engine_rpm = max(650.0, min(float(SIM_TH.get("engine_rpm_stop", 2300.0)), self.engine_rpm_actual + 12.0 * throttle - 30.0 * brake + random.gauss(0, 5.0)))
        # 行走压力：gear1 约 26bar；gear3/4 moving p50≈58, p90≈141
        travel_base = 26.0 if s.gear_state == 1 else 27.0 if s.gear_state == 2 else 32.0
        s.travel_pressure_bar = max(25.0, min(191.0, travel_base + 1.2 * max(0.0, self.engine_rpm_actual - self.remote_idle_rpm) / 80.0 + 118 * throttle + 1.4 * max(0.0, s.angle_deg) - 8 * brake + random.gauss(0, 2.0)))
        # 制动压力：动作触发上升；高温会削弱有效制动力（热衰减）
        brake_fade = max(0.0, self.coolant_temp_c - 96.0) * 0.8 + max(0.0, self.hydraulic_oil_temp_c - 72.0) * 0.5
        if s.gear_state == 1:
            s.brake_pressure_bar = max(1.0, min(168.0, 3.0 + random.gauss(0, 0.8)))
        elif s.gear_state == 2:
            s.brake_pressure_bar = max(120.0, min(168.0, 140.0 + 8.0 * min(1.0, max(0.0, sim_time_sec - self.control_gate_switch_sec) / max(0.5, self.neutral_gate_duration_sec)) + random.gauss(0, 4.0)))
        else:
            s.brake_pressure_bar = max(120.0, min(168.0, 153 + 7 * brake - brake_fade + random.gauss(0, 3.0)))

        # 系统压力：负载/牵引升高而升，连续制动和高温会拖低
        s.system_pressure_bar = max(
            80.0,
            min(190.0, 154 + 0.05 * self.payload_tons + 1.2 * throttle - 0.9 * brake - 0.3 * max(0.0, self.hydraulic_oil_temp_c - 70.0) + random.gauss(0, 1.0)),
        )

        # 夹紧压力：受系统压力与液压油温共同影响
        s.clamp_pressure_bar = max(
            70.0,
            min(145.0, 114 + 0.12 * (s.system_pressure_bar - 120.0) - 0.18 * max(0.0, self.hydraulic_oil_temp_c - 70.0) + random.gauss(0, 0.8)),
        )

        # 温度-压力耦合：油温升高通常伴随压力能力下滑
        hyd_temp_excess = max(0.0, self.hydraulic_oil_temp_c - 58.0)
        s.make_up_oil_pressure_bar = max(
            16.0,
            min(30.0, 26.9 - 0.035 * hyd_temp_excess + 0.6 * throttle - 0.25 * brake + random.gauss(0, 0.05)),
        )

        # 进气压力：节气开度正相关，制动与高负载会拉低有效进气
        s.intake_pressure_kpa = max(
            80.0,
            min(130.0, 109.0 + 4.0 * throttle - 1.8 * brake - 0.5 * max(0.0, self.payload_tons - 150.0) / 30.0 + random.gauss(0, 0.8)),
        )

        # 机油压力：转速正相关，高温负相关
        s.oil_pressure_kpa = max(
            70.0,
            min(520.0, 390.0 + 0.06 * s.engine_rpm - 1.4 * max(0.0, self.coolant_temp_c - 92.0) - 0.9 * hyd_temp_excess + random.gauss(0, 4.0)),
        )

        # 灭火系统压力：慢变量，小幅漂移
        s.fire_system_pressure_bar = max(1.0, min(24.0, 12.0 + 0.18 * math.sin(self.step * DT / 220.0) + random.gauss(0, 0.06)))

        s.coolant_temp_c = self.coolant_temp_c
        s.hydraulic_oil_temp_c = self.hydraulic_oil_temp_c
        s.exhaust_temp_c = self.exhaust_temp_c
        s.surface_temp_c = self.surface_temp_c

        # 电气与气体关联：高负载/高温时电压略降；燃烧负荷高时CO略升
        temp_load_stress = max(0.0, self.coolant_temp_c - 92.0) * 0.03 + max(0.0, self.payload_tons - 150.0) * 0.002
        s.battery_v = max(22.0, min(27.5, 25.1 - temp_load_stress + 0.12 * math.sin(self.step * DT / 180.0) + random.gauss(0, 0.03)))

        co_base = 6.0 + 0.012 * max(0.0, self.payload_tons - 120.0) + 0.20 * throttle + 0.08 * max(0.0, self.exhaust_temp_c - 60.0)
        methane_base = 0.03 + 0.00025 * max(0.0, self.payload_tons - 140.0) + 0.003 * (1 if s.slope_state == 3 else 0)
        s.co_ppm = max(0.0, min(80.0, co_base + random.gauss(0, 0.35)))
        s.methane_pctlel = max(0.0, min(0.45, methane_base + random.gauss(0, 0.004)))

        # 散热异常时同步降低水位，制造更贴近现场的高温伴生信号
        if self.cooling_fault_active:
            self.water_tank_level_pct = max(0.0, self.water_tank_level_pct - (0.0035 + 0.0015 * throttle) * DT)

        if self.level_fault == 'hydraulic_oil_level_low':
            if action in {"DECELERATE", "BRAKE", "EMERGENCY_STOP"} and self.level_fault_elapsed_sec >= self.level_fault_min_hold_sec:
                self.hydraulic_oil_level_pct = min(44.0, self.hydraulic_oil_level_pct + 0.22 * DT)
            else:
                self.hydraulic_oil_level_pct = max(40.5, min(self.hydraulic_oil_level_pct, 43.5) - 0.030 * DT)
        elif self.level_fault == 'diesel_level_low':
            if action in {"DECELERATE", "BRAKE", "EMERGENCY_STOP"} and self.level_fault_elapsed_sec >= self.level_fault_min_hold_sec:
                self.diesel_level_cm = min(33.0, self.diesel_level_cm + 0.18 * DT)
            else:
                self.diesel_level_cm = max(29.6, min(self.diesel_level_cm, 32.0) - 0.025 * DT)
        elif self.level_fault == 'water_tank_low':
            if action in {"DECELERATE", "BRAKE", "EMERGENCY_STOP"} and self.level_fault_elapsed_sec >= self.level_fault_min_hold_sec:
                self.water_tank_level_pct = min(25.0, self.water_tank_level_pct + 0.28 * DT)
            else:
                self.water_tank_level_pct = max(11.5, min(self.water_tank_level_pct, 18.5) - 0.025 * DT)

        # warning场景注入：只到warning区，不直接打到stop区
        if self.warning_fault == 'coolant_warning':
            # 渐进升温并可被控制动作“拉回”（不再按固定时长自动清除）
            progress = min(1.0, self.warning_fault_elapsed_sec / 18.0)
            target_coolant = 94.2 + 2.8 * progress  # 94.2 -> 97.0
            # 降速/制动时给予额外降温修正，避免“越降速越升温”的反直觉现象
            action_cooling = -0.35 if action in {"DECELERATE", "BRAKE", "EMERGENCY_STOP"} else 0.0
            s.coolant_temp_c = min(97.2, max(93.8, s.coolant_temp_c + 0.16 * (target_coolant - s.coolant_temp_c) + action_cooling + random.gauss(0.0, 0.05)))
        elif self.warning_fault == 'intake_pressure_low':
            # 低于 warning 线但高于 stop 线；纠偏动作会提升压力，便于闭环恢复
            action_relief = 10.5 if action in {"DECELERATE", "BRAKE", "EMERGENCY_STOP"} else 0.0
            s.intake_pressure_kpa = max(91.5, min(106.0, 93.3 + action_relief + random.uniform(-0.8, 0.7)))
        elif self.warning_fault == 'water_tank_low':
            # 低于 warning(30) 的边缘区；纠偏动作视为触发补水，支持恢复退出
            if action in {"DECELERATE", "BRAKE", "EMERGENCY_STOP"}:
                refill = 0.040 + 0.015 * brake
                self.water_tank_level_pct = min(35.0, self.water_tank_level_pct + refill * DT)
            else:
                self.water_tank_level_pct = max(27.5, self.water_tank_level_pct - (0.007 + 0.0025 * throttle) * DT)

        s.water_tank_level_pct = self.water_tank_level_pct

        # warning故障退出规则：由“恢复判据”驱动，不按固定时长自动消失
        if self.warning_fault:
            recovered_now = False
            if self.warning_fault == 'coolant_warning':
                recovered_now = float(s.coolant_temp_c or 0.0) < (SIM_TH["coolant_warn"] - 0.8)
            elif self.warning_fault == 'intake_pressure_low':
                recovered_now = float(s.intake_pressure_kpa or 0.0) >= (SIM_TH["intake_warn_low"] + 1.0)
            elif self.warning_fault == 'water_tank_low':
                recovered_now = float(self.water_tank_level_pct or 0.0) >= (SIM_TH["water_warn_low"] + 1.0)

            if self.warning_fault_elapsed_sec < self.warning_fault_min_hold_sec:
                recovered_now = False

            if recovered_now:
                self.warning_fault_recover_elapsed_sec += DT
            else:
                self.warning_fault_recover_elapsed_sec = 0.0

            if self.warning_fault_recover_elapsed_sec >= self.warning_fault_recover_confirm_sec:
                self.warning_fault = ''
                self.warning_fault_elapsed_sec = 0.0
                self.warning_fault_recover_elapsed_sec = 0.0
                self.warning_fault_cooldown_remaining_sec = self.warning_fault_cooldown_sec

        if self.level_fault:
            recovered_level = False
            if self.level_fault == 'hydraulic_oil_level_low':
                recovered_level = self.hydraulic_oil_level_pct >= 43.0
            elif self.level_fault == 'diesel_level_low':
                recovered_level = self.diesel_level_cm >= 32.0
            elif self.level_fault == 'water_tank_low':
                recovered_level = self.water_tank_level_pct >= 23.0
            if self.level_fault_elapsed_sec < self.level_fault_min_hold_sec:
                recovered_level = False
            if recovered_level:
                self.level_fault_recover_elapsed_sec += DT
            else:
                self.level_fault_recover_elapsed_sec = 0.0
            if self.level_fault_recover_elapsed_sec >= 6.0:
                self.level_fault = ''
                self.level_fault_elapsed_sec = 0.0
                self.level_fault_recover_elapsed_sec = 0.0
                self.level_fault_cooldown_remaining_sec = self.level_fault_cooldown_sec

        # 环境气体相关变量做保守钳制：尽量不因 CO / 甲烷触发预警或报警
        # 说明：仅用于闭环压测场景，保持在 warning 阈值下方，避免环境噪声干扰控制策略评估。
        if self.limit_env_gas:
            s.methane_pctlel = min(float(s.methane_pctlel), 0.35)
            s.co_ppm = min(float(s.co_ppm), 18.0)

        # 使用闭环仿真内部阈值做判级，避免旧阈值（如 brake_pressure_high=40）造成系统性误报
        s.risk_level, s.warning_tag = self._classify_warning_local(s)
        s.alarm_code = self._derive_alarm_code_local(s, s.warning_tag)

        self.last_action = action
        self.step += 1
        return s


def call_agent_decision(agent_url: str, payload: Dict[str, Any], timeout_sec: float = 8.0) -> Dict[str, Any]:
    url = agent_url.rstrip('/') + '/sensor/decision'
    req_body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url=url,
        data=req_body,
        headers={
            'Content-Type': 'application/json',
            'Connection': 'close',
            'X-Request-Id': f"sim-{int(time.time() * 1000)}-{random.randint(1000, 9999)}",
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode('utf-8', errors='replace')
        except Exception:
            err_body = "<failed_to_read_error_body>"
        print(
            json.dumps(
                {
                    "event": "agent_http_error_detail",
                    "status": int(getattr(e, "code", 0) or 0),
                    "url": url,
                    "payload_keys": sorted(list(payload.keys())),
                    "payload_size": len(req_body),
                    "response_body": err_body[:1200],
                },
                ensure_ascii=False,
            )
        )
        raise
    except Exception as e:
        print(
            json.dumps(
                {
                    "event": "agent_transport_error_detail",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "url": url,
                    "payload_keys": sorted(list(payload.keys())),
                    "payload_size": len(req_body),
                },
                ensure_ascii=False,
            )
        )
        raise

    data = json.loads(raw)
    if not isinstance(data, dict) or 'decision' not in data or not isinstance(data['decision'], dict):
        print(
            json.dumps(
                {
                    "event": "agent_invalid_response_detail",
                    "url": url,
                    "response_preview": raw[:1200],
                },
                ensure_ascii=False,
            )
        )
        raise RuntimeError(f'agent decision invalid: {data}')
    return data['decision']


def write_rows(
    writer: csv.writer,
    step: int,
    state: SensorState,
    frames: List[Tuple[int, List[int]]],
    action: str,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    for fid, payload in frames:
        writer.writerow([
            step,
            round(step * DT, 1),
            ts,
            f"0x{fid:08X}",
            *payload,
            state.can_heartbeat_ok,
            state.warning_tag,
            action,
            state.risk_level,
        ])


# 注意：
# 闭环执行入口已统一到 online_pipeline.py（--source closed_loop）。
# 本文件主要提供过程模型与数据生成能力。


def generate_closed_loop_csv(
    output_csv: Path,
    duration_sec: float,
    seed: int = 20260422,
    control_every_sec: float = 2.0,
    profile: str = "balanced",
    cooling_fault_prob: Optional[float] = None,
    warning_fault_prob: Optional[float] = None,
) -> Dict[str, Any]:
    """Generate synthetic closed-loop CAN frames for ML training.

    说明：
    - 该生成器用于扩充训练样本（尤其是 warning/danger 转换片段）。
    - 控制策略是轻量规则策略，不依赖外部 agent 服务。
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sim = ClosedLoopSim(
        seed=seed,
        profile=profile,
        cooling_fault_prob=cooling_fault_prob,
        warning_fault_prob=warning_fault_prob,
    )
    total_steps = max(1, int(duration_sec / DT))
    control_every_steps = max(1, int(control_every_sec / DT))

    action = "FORWARD"
    action_counter: Dict[str, int] = {k: 0 for k in CONTROL_ACTIONS}
    warning_counter: Dict[str, int] = {}

    # 控制平滑参数
    min_hold_steps = max(1, int(5.0 / DT))  # 动作最短保持约5秒
    action_hold_steps = 0

    # FSM 控制状态：cruise / caution / recover / emergency
    fsm_mode = "cruise"
    mode_hold_steps = 0
    warning_streak = 0
    normal_streak = 0
    danger_streak = 0
    recover_steps = 0

    last_risk_level = "normal"
    last_warning_tag = ""

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "t_sec",
            "timestamp_utc",
            "frame_id_hex",
            "byte0",
            "byte1",
            "byte2",
            "byte3",
            "byte4",
            "byte5",
            "byte6",
            "byte7",
            "can_heartbeat_ok",
            "warning_tag",
            "control_action",
            "risk_level",
        ])

        last_state: Optional[SensorState] = None
        for step in range(total_steps):
            if step % control_every_steps == 0 and last_state is not None:
                # FSM 闭环控制：emergency -> caution -> recover -> cruise
                rl = str(last_state.risk_level or "normal")
                warning_tag = str(last_state.warning_tag or "")
                last_risk_level = rl
                last_warning_tag = warning_tag

                speed_mps = float(last_state.speed_mps or 0.0)
                coolant = float(last_state.coolant_temp_c or 0.0)
                travel_p = float(last_state.travel_pressure_bar or 0.0)
                system_p = float(last_state.system_pressure_bar or 0.0)
                brake_p = float(last_state.brake_pressure_bar or 0.0)
                water_pct = float(last_state.water_tank_level_pct or 100.0)

                if rl == "danger":
                    danger_streak += 1
                    warning_streak = 0
                    normal_streak = 0
                elif rl == "warning":
                    warning_streak += 1
                    danger_streak = 0
                    normal_streak = 0
                else:
                    normal_streak += 1
                    warning_streak = 0
                    danger_streak = 0

                severe_stop = {
                    "coolant_warning",
                    "hydraulic_oil_temp_warning",
                    "brake_pressure_low",
                    "travel_pressure_warning",
                    "system_pressure_low",
                    "make_up_oil_pressure_low",
                    "communication_fault",
                    "emergency_stop",
                }

                if rl == "danger" or warning_tag in severe_stop or brake_p < 110.0 or speed_mps > 8.4:
                    target_mode = "emergency"
                elif rl == "warning" or speed_mps > 6.3 or travel_p > 220.0 or system_p < 138.0 or coolant > 94.0 or water_pct < 27.5:
                    target_mode = "caution"
                elif normal_streak >= 2:
                    target_mode = "recover"
                else:
                    target_mode = "cruise"

                # 状态切换滞回：避免模式频繁抖动
                if target_mode != fsm_mode:
                    allow_switch = (
                        target_mode == "emergency"
                        or mode_hold_steps >= max(1, int(4.0 / DT))
                        or (fsm_mode == "recover" and target_mode == "cruise" and recover_steps >= max(1, int(6.0 / DT)))
                    )
                    if allow_switch:
                        fsm_mode = target_mode
                        mode_hold_steps = 0
                        if fsm_mode == "recover":
                            recover_steps = 0
                else:
                    mode_hold_steps += control_every_steps

                # 每个模式下的动作策略
                if fsm_mode == "emergency":
                    if speed_mps > 1.2:
                        candidate = "BRAKE"
                    else:
                        candidate = "DECELERATE"
                elif fsm_mode == "caution":
                    if speed_mps > 7.0:
                        candidate = "DECELERATE"
                    elif speed_mps < 1.0 and warning_streak >= 3:
                        candidate = "FORWARD"
                    else:
                        candidate = "DECELERATE"
                elif fsm_mode == "recover":
                    recover_steps += control_every_steps
                    if speed_mps > 2.0:
                        candidate = "DECELERATE"
                    elif speed_mps < 0.8:
                        candidate = "FORWARD"
                    else:
                        candidate = "FORWARD"
                else:  # cruise
                    if speed_mps > 2.2:
                        candidate = "DECELERATE"
                    else:
                        r = random.random()
                        if speed_mps < 0.7 and r < 0.12:
                            candidate = "ACCELERATE"
                        elif r < 0.04:
                            candidate = "ACCELERATE"
                        elif r < 0.08:
                            candidate = "DECELERATE"
                        else:
                            candidate = "FORWARD"

                # 全局防护：高速度时严禁加速
                if speed_mps > 2.4:
                    candidate = "DECELERATE"
                elif speed_mps > 2.0 and candidate == "ACCELERATE":
                    candidate = "FORWARD"

                # 大跳变保护：加速不直接跳刹车
                if action == "ACCELERATE" and candidate == "BRAKE":
                    candidate = "DECELERATE"

                # 动作最短保持（emergency 态可快速介入）
                if action != candidate:
                    if fsm_mode == "emergency" or action_hold_steps >= min_hold_steps:
                        action = candidate
                        action_hold_steps = 0
                else:
                    action_hold_steps += control_every_steps
            else:
                action_hold_steps += 1

            if action == "ACCELERATE" and getattr(sim, "remote_brake_latched", True):
                action = "FORWARD"

            state = sim.step_once(action)
            state.last_action_applied = action
            frames = encode_frames(state)
            write_rows(writer, step, state, frames, action)

            action_counter[action] = action_counter.get(action, 0) + 1
            if state.warning_tag:
                warning_counter[state.warning_tag] = warning_counter.get(state.warning_tag, 0) + 1
            last_state = state

    return {
        "event": "closed_loop_data_generated",
        "output_csv": str(output_csv),
        "duration_sec": float(duration_sec),
        "dt": DT,
        "steps": total_steps,
        "rows_estimated": total_steps * 8,
        "seed": seed,
        "profile": sim.profile,
        "cooling_fault_prob": sim.cooling_fault_prob,
        "warning_fault_prob": sim.warning_fault_prob,
        "control_every_sec": float(control_every_sec),
        "action_counter": action_counter,
        "warning_counter": warning_counter,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate more closed-loop CAN data for ML training")
    parser.add_argument(
        "--output-csv",
        default="",
        help="output CSV path; if empty, auto-generate a timestamped new file",
    )
    parser.add_argument("--duration-sec", type=float, default=3600.0, help="simulation duration in seconds")
    parser.add_argument("--seed", type=int, default=20260422, help="random seed")
    parser.add_argument("--control-every-sec", type=float, default=2.0, help="control update period")
    parser.add_argument("--profile", choices=["easy", "balanced", "stress"], default="balanced", help="scenario profile for long-run diversity")
    parser.add_argument("--cooling-fault-prob", type=float, default=None, help="override cooling fault trigger probability")
    parser.add_argument("--warning-fault-prob", type=float, default=None, help="override warning fault trigger probability")
    args = parser.parse_args()

    if str(args.output_csv).strip():
        output_csv = Path(args.output_csv)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_csv = Path(__file__).resolve().parent / "sim_data" / f"closed_loop_can_frames_{ts}.csv"

    summary = generate_closed_loop_csv(
        output_csv=output_csv,
        duration_sec=float(args.duration_sec),
        seed=int(args.seed),
        control_every_sec=float(args.control_every_sec),
        profile=str(args.profile),
        cooling_fault_prob=args.cooling_fault_prob,
        warning_fault_prob=args.warning_fault_prob,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
