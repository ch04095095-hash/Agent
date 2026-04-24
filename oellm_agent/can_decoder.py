from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from thresholds import load_thresholds

# LuTech protocol frame IDs
ID_181 = 0x18F181A0
ID_182 = 0x18F182A0
ID_183 = 0x18F183A0
ID_184 = 0x18F184A0
ID_185 = 0x18F185A0
ID_186 = 0x18F186A0
ID_189 = 0x18F189A0
ID_190 = 0x18F190A0

SUPPORTED_IDS = {ID_181, ID_182, ID_183, ID_184, ID_185, ID_186, ID_189, ID_190}


@dataclass
class DecodedState:
    speed_kmh: float = 0.0
    engine_rpm: float = 0.0
    angle_deg: float = 0.0
    total_mileage_km: float = 0.0
    runtime_min: float = 0.0
    payload_tons: float = 0.0
    cabin_id: int = 0
    slope_state: int = 0
    warning_tag: str = ""
    risk_level: str = "normal"
    risk_score: float = 0.0
    deviation_score: float = 0.0
    methane_pct: float = 0.0
    co_ppm: float = 0.0
    battery_v: float = 24.0

    coolant_temp_c: float = 0.0
    surface_temp_c: float = 0.0
    exhaust_temp_c: float = 0.0
    intake_pressure_kpa: float = 0.0
    water_tank_level_pct: float = 0.0
    oil_pressure_kpa: float = 0.0
    oil_pressure_mpa: float = 0.0
    diesel_level_cm: float = 0.0
    intake_temp_c: float = 0.0

    brake_pressure_bar: float = 0.0
    travel_pressure_bar: float = 0.0
    system_pressure_bar: float = 0.0
    clamp_pressure_bar: float = 0.0

    alarm_code: int = 0
    gear_state: int = 1
    emergency_stop: int = 0
    hydraulic_oil_temp_c: float = 0.0
    hydraulic_oil_level_pct: float = 0.0
    make_up_oil_pressure_bar: float = 0.0
    fire_system_pressure_bar: float = 0.0
    shua_qu_state: int = 1

    can_heartbeat_ok: int = 1
    fault_code_byte: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed_kmh": self.speed_kmh,
            "engine_rpm": self.engine_rpm,
            "angle_deg": self.angle_deg,
            "total_mileage_km": self.total_mileage_km,
            "runtime_min": self.runtime_min,
            "payload_tons": self.payload_tons,
            "cabin_id": self.cabin_id,
            "slope_state": self.slope_state,
            "warning_tag": self.warning_tag,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "deviation_score": self.deviation_score,
            "methane_pct": self.methane_pct,
            "methane_pctlel": self.methane_pct,
            "co_ppm": self.co_ppm,
            "battery_v": self.battery_v,
            "coolant_temp_c": self.coolant_temp_c,
            "surface_temp_c": self.surface_temp_c,
            "exhaust_temp_c": self.exhaust_temp_c,
            "intake_pressure_kpa": self.intake_pressure_kpa,
            "water_tank_level_pct": self.water_tank_level_pct,
            "water_level_pct": self.water_tank_level_pct,
            "oil_pressure_kpa": self.oil_pressure_kpa,
            "oil_pressure_mpa": self.oil_pressure_mpa,
            "diesel_level_cm": self.diesel_level_cm,
            "intake_temp_c": self.intake_temp_c,
            "brake_pressure_bar": self.brake_pressure_bar,
            "travel_pressure_bar": self.travel_pressure_bar,
            "walking_pressure_bar": self.travel_pressure_bar,
            "system_pressure_bar": self.system_pressure_bar,
            "clamp_pressure_bar": self.clamp_pressure_bar,
            "alarm_code": self.alarm_code,
            "gear_state": self.gear_state,
            "emergency_stop": self.emergency_stop,
            "hydraulic_oil_temp_c": self.hydraulic_oil_temp_c,
            "hydraulic_oil_level_pct": self.hydraulic_oil_level_pct,
            "make_up_oil_pressure_bar": self.make_up_oil_pressure_bar,
            "fire_system_pressure_bar": self.fire_system_pressure_bar,
            "shua_qu_state": self.shua_qu_state,
            "can_heartbeat_ok": self.can_heartbeat_ok,
            "fault_code_byte": self.fault_code_byte,
        }


class CanDecoder:
    """Decode LuTech CAN frames into unified state."""

    SIGNAL_RULES: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _build_signal_rules_from_thresholds(th: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        return {
            "methane_pct": {
                "meaning": [
                    (th["methane_alarm"], "正常"),
                    (float("inf"), "异常：甲烷浓度超限"),
                ],
            },
            "coolant_temp_c": {
                "meaning": [(th["coolant_temp_warn"], "正常"), (float("inf"), "异常：冷却水温过高")],
            },
            "surface_temp_c": {
                "meaning": [
                    (th["surface_temp_warn"], "正常"),
                    (th["surface_temp_high"], "表面温度偏高"),
                    (float("inf"), "异常：表面温度过高"),
                ],
            },
            "exhaust_temp_c": {
                "meaning": [
                    (th["exhaust_temp_warn"], "正常"),
                    (th["exhaust_temp_high"], "尾气温度偏高"),
                    (float("inf"), "异常：尾气温度过高"),
                ],
            },
            "intake_pressure_kpa": {
                "meaning": [
                    (th["intake_pressure_min"], "异常：进气压力过低"),
                    (th["intake_pressure_warn_low"], "进气压力偏低"),
                    (float("inf"), "正常"),
                ],
            },
            "water_tank_level_pct": {
                "meaning": [
                    (th["water_level_low"], "异常：水箱液位过低"),
                    (th["water_level_warn_low"], "水箱液位偏低"),
                    (float("inf"), "正常"),
                ],
            },
            "oil_pressure_kpa": {
                "meaning": [
                    (th["oil_pressure_low_kpa"], "异常：机油压力过低"),
                    (th["oil_pressure_warn_kpa"], "机油压力偏低"),
                    (float("inf"), "正常"),
                ],
            },
            "diesel_level_cm": {
                "meaning": [(th["diesel_level_low"], "柴油液位偏低"), (float("inf"), "正常")],
            },
            "brake_pressure_bar": {
                "meaning": [
                    (th["brake_pressure_low"], "异常：制动压力过低"),
                    (th["brake_pressure_warn_low"], "制动压力偏低"),
                    (180.0, "正常"),
                    (float("inf"), "制动压力偏高"),
                ],
            },
            "travel_pressure_bar": {
                "meaning": [
                    (th["travel_pressure_warn_low"], "异常：行走压力过低"),
                    (th["travel_pressure_warn_high"], "正常"),
                    (th["travel_pressure_high"], "行走压力偏高"),
                    (float("inf"), "异常：行走压力过高"),
                ],
            },
            "system_pressure_bar": {
                "meaning": [
                    (th["system_pressure_low"], "异常：系统压力过低"),
                    (th["system_pressure_warn_low"], "系统压力偏低"),
                    (180.0, "正常"),
                    (float("inf"), "系统压力偏高"),
                ],
            },
            "clamp_pressure_bar": {
                "meaning": [
                    (th["clamp_pressure_low"], "异常：夹紧压力过低"),
                    (th["clamp_pressure_warn_low"], "夹紧压力偏低"),
                    (th["clamp_pressure_warn_high"], "正常"),
                    (th["clamp_pressure_high"], "夹紧压力偏高"),
                    (float("inf"), "异常：夹紧压力过高"),
                ],
            },
            "hydraulic_oil_temp_c": {
                "meaning": [
                    (th["hydraulic_oil_temp_warn"], "正常"),
                    (th["hydraulic_oil_temp_high"], "液压油温偏高"),
                    (float("inf"), "异常：液压油温过高"),
                ],
            },
            "hydraulic_oil_level_pct": {
                "meaning": [(th["hydraulic_oil_level_low"], "液压油液位偏低"), (float("inf"), "正常")],
            },
            "make_up_oil_pressure_bar": {
                "meaning": [
                    (th["make_up_oil_pressure_low"], "异常：补油压力过低"),
                    (th["make_up_oil_pressure_warn_low"], "补油压力偏低"),
                    (th.get("make_up_oil_pressure_high", 28.0), "正常"),
                    (float("inf"), "补油压力偏高"),
                ],
            },
            "engine_rpm": {
                "meaning": [
                    (th["rpm_warn_high"], "正常"),
                    (th["rpm_alarm_high"], "转速偏高"),
                    (float("inf"), "异常：转速过高"),
                ],
            },
            "speed_kmh": {
                "meaning": [
                    (th["speed_warn_low_kmh"], "速度偏低"),
                    (th["speed_warn_high_kmh"], "正常"),
                    (th["speed_alarm_high_kmh"], "车速偏高"),
                    (float("inf"), "异常：车速过高"),
                ],
            },
            "co_ppm": {
                "meaning": [(th["co_warn"], "正常"), (float("inf"), "异常：一氧化碳超限")],
            },
        }

    def __init__(self):
        self.state = DecodedState()
        self._last_values: Dict[str, float] = {}
        self._last_trends: Dict[str, str] = {}
        self._last_update_ts: Optional[float] = None
        self._stable_counts: Dict[str, int] = {}
        try:
            th = load_thresholds(Path(__file__).resolve().parent)
            self.SIGNAL_RULES = self._build_signal_rules_from_thresholds(th)
        except Exception:
            # 兜底：阈值文件异常时保持最小可用
            self.SIGNAL_RULES = {}

    @staticmethod
    def _u16_le(data: List[int], idx: int) -> int:
        return int(data[idx]) | (int(data[idx + 1]) << 8)

    @staticmethod
    def _apply_scale_offset(raw: int, scale: float, offset: float) -> float:
        return raw * scale + offset

    @staticmethod
    def _update_signal_memory(self, key: str, value: float) -> float:
        last = self._last_values.get(key, value)
        delta = value - last
        self._last_values[key] = value
        return delta

    def _lookup_threshold_label(self, key: str, value: Optional[float]) -> str:
        if value is None:
            return ""
        for limit, label in self.SIGNAL_RULES.get(key, {}).get("meaning", []):
            if value <= limit:
                return label
        return "正常"

    def _build_signal_meta(self, key: str, value: Optional[float], unit: str = "") -> Dict[str, Any]:
        return {
            "value": value,
            "unit": unit,
            "meaning": self._lookup_threshold_label(key, value),
        }

    def update(self, frame_id: int, data: List[int]) -> None:
        if len(data) != 8:
            raise ValueError("CAN payload must be 8 bytes")

        b = [int(x) & 0xFF for x in data]
        if frame_id == ID_181:
            # 2.1 报警号、挡位状态、急停状态、液压油温度、液压油液位、补油压力、灭火器系统压力、甩驱状态
            self.state.alarm_code = b[0]
            self.state.fault_code_byte = b[0]
            self.state.gear_state = b[1]
            self.state.emergency_stop = b[2]
            self.state.hydraulic_oil_temp_c = float(b[3])
            self.state.hydraulic_oil_level_pct = float(b[4])
            self.state.make_up_oil_pressure_bar = float(b[5])
            self.state.fire_system_pressure_bar = float(b[6])
            self.state.shua_qu_state = b[7]
        elif frame_id == ID_182:
            # 2.2 制动压力、行走压力、系统压力、夹紧压力
            self.state.brake_pressure_bar = float(b[0])
            self.state.travel_pressure_bar = float(b[1])
            self.state.system_pressure_bar = float(b[2])
            self.state.clamp_pressure_bar = float(b[3])
        elif frame_id == ID_183:
            # 2.3 甲烷浓度、车速、转速、角度
            raw_ch4 = self._u16_le(b, 0)
            raw_spd = self._u16_le(b, 2)
            raw_rpm = self._u16_le(b, 4)
            raw_ang = self._u16_le(b, 6)
            self.state.methane_pct = self._apply_scale_offset(raw_ch4, 0.1, -40.0)
            self.state.speed_kmh = self._apply_scale_offset(raw_spd, 0.1, 0.0)
            self.state.engine_rpm = float(raw_rpm)
            self.state.angle_deg = self._apply_scale_offset(raw_ang, 0.1, -90.0)
        elif frame_id == ID_184:
            # 2.4 冷却水温、表面温度、尾气温度、进气压力、水箱水位、机油压力、柴油液位、进气温度
            self.state.coolant_temp_c = self._apply_scale_offset(b[0], 1.0, -40.0)
            self.state.surface_temp_c = self._apply_scale_offset(b[1], 1.0, 0.0)
            self.state.exhaust_temp_c = self._apply_scale_offset(b[2], 1.0, 0.0)
            self.state.intake_pressure_kpa = self._apply_scale_offset(b[3], 2.0, 0.0)
            self.state.water_tank_level_pct = self._apply_scale_offset(b[4], 1.0, 0.0)
            self.state.oil_pressure_kpa = self._apply_scale_offset(b[5], 4.0, 0.0)
            self.state.oil_pressure_mpa = self.state.oil_pressure_kpa / 1000.0
            self.state.diesel_level_cm = self._apply_scale_offset(b[6], 1.0, 0.0)
            self.state.intake_temp_c = self._apply_scale_offset(b[7], 1.0, -40.0)
        elif frame_id == ID_186:
            bat_raw = self._u16_le(b, 5)
            self.state.battery_v = bat_raw * 0.01
        elif frame_id == ID_185:
            # 2.5 总里程、运行时间、防冻液温度
            total_mileage_raw = self._u16_le(b, 0)
            runtime_raw = self._u16_le(b, 2)
            antifreeze_raw = self._u16_le(b, 4)
            self.state.total_mileage_km = total_mileage_raw * 0.1
            self.state.runtime_min = float(runtime_raw)
            # 防冻液温度不单独入 state，可在 signals 中扩展
        elif frame_id == ID_189:
            # 2.9 称重、一氧化碳
            payload_raw = self._u16_le(b, 0)
            co_raw = self._u16_le(b, 2)
            self.state.payload_tons = float(payload_raw)
            self.state.co_ppm = float(co_raw)
        elif frame_id == ID_190:
            # 3.0 角度、驾驶室号、上下坡状态、壳体压力
            angle_raw = self._u16_le(b, 0)
            cabin_id = int(b[2])
            slope_state = int(b[3])
            firewall_raw = self._u16_le(b, 4)
            self.state.angle_deg = (angle_raw - 900) * 0.1
            self.state.cabin_id = cabin_id
            self.state.slope_state = slope_state

    def set_heartbeat(self, ok: int) -> None:
        self.state.can_heartbeat_ok = 1 if ok else 0

    def to_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def build_status_report(self) -> Dict[str, Any]:
        state = self.to_dict()
        signals: Dict[str, Dict[str, Any]] = {
            "coolant_temp_c": self._build_signal_meta("coolant_temp_c", state["coolant_temp_c"], unit="℃"),
            "surface_temp_c": self._build_signal_meta("surface_temp_c", state["surface_temp_c"], unit="℃"),
            "exhaust_temp_c": self._build_signal_meta("exhaust_temp_c", state["exhaust_temp_c"], unit="℃"),
            "oil_pressure_kpa": self._build_signal_meta("oil_pressure_kpa", state["oil_pressure_kpa"], unit="kPa"),
            "brake_pressure_bar": self._build_signal_meta("brake_pressure_bar", state["brake_pressure_bar"], unit="bar"),
            "travel_pressure_bar": self._build_signal_meta("travel_pressure_bar", state["travel_pressure_bar"], unit="bar"),
            "system_pressure_bar": self._build_signal_meta("system_pressure_bar", state["system_pressure_bar"], unit="bar"),
            "clamp_pressure_bar": self._build_signal_meta("clamp_pressure_bar", state["clamp_pressure_bar"], unit="bar"),
            "methane_pct": self._build_signal_meta("methane_pct", state["methane_pct"], unit="%"),
            "speed_kmh": self._build_signal_meta("speed_kmh", state["speed_kmh"], unit="km/h"),
            "engine_rpm": self._build_signal_meta("engine_rpm", state["engine_rpm"], unit="rpm"),
            "angle_deg": self._build_signal_meta("angle_deg", state["angle_deg"], unit="deg"),
            "hydraulic_oil_temp_c": self._build_signal_meta("hydraulic_oil_temp_c", state["hydraulic_oil_temp_c"], unit="℃"),
            "hydraulic_oil_level_pct": self._build_signal_meta("hydraulic_oil_level_pct", state["hydraulic_oil_level_pct"], unit="%"),
            "make_up_oil_pressure_bar": self._build_signal_meta("make_up_oil_pressure_bar", state["make_up_oil_pressure_bar"], unit="bar"),
            "fire_system_pressure_bar": self._build_signal_meta("fire_system_pressure_bar", state["fire_system_pressure_bar"], unit="bar"),
            "water_tank_level_pct": self._build_signal_meta("water_tank_level_pct", state["water_tank_level_pct"], unit="%"),
            "intake_pressure_kpa": self._build_signal_meta("intake_pressure_kpa", state["intake_pressure_kpa"], unit="kPa"),
            "diesel_level_cm": self._build_signal_meta("diesel_level_cm", state["diesel_level_cm"], unit="cm"),
            "intake_temp_c": self._build_signal_meta("intake_temp_c", state["intake_temp_c"], unit="℃"),
        }

        return {
            "frame_id": None,
            "decoded": state,
            "signals": signals,
        }


# Encoding helpers (for simulator/replay generation)
def to_u8(v: int) -> int:
    return max(0, min(255, v))


def to_u16_le(v: int) -> Tuple[int, int]:
    v = max(0, min(65535, v))
    return v & 0xFF, (v >> 8) & 0xFF
