from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

from oellm_agent.config.thresholds.thresholds import normalize_model

# LuTech 190 型号协议帧 IDs
ID_181 = 0x18F181A0
ID_182 = 0x18F182A0
ID_183 = 0x18F183A0
ID_184 = 0x18F184A0
ID_185 = 0x18F185A0
ID_186 = 0x18F186A0
ID_189 = 0x18F189A0
ID_190 = 0x18F190A0

# 190 型号协议帧（你已确认）
SUPPORTED_IDS_190 = {ID_181, ID_182, ID_183, ID_184, ID_185, ID_189, ID_190}

# 50 型号协议帧
SUPPORTED_IDS_50 = {ID_181, ID_182, ID_183, ID_184, ID_185, ID_186, ID_189}

# 105 型号协议帧
SUPPORTED_IDS_105 = {ID_181, ID_182, ID_183, ID_184, ID_185, ID_186, ID_189}

# 兼容旧调用：默认给 50/105 协议
SUPPORTED_IDS = SUPPORTED_IDS_50


def get_supported_ids(model: str = "50") -> set[int]:
    model_key = str(model).strip().lower()
    if model_key not in CanDecoder.PROTOCOL_REGISTRY:
        model_key = "50"
    return set(CanDecoder.PROTOCOL_REGISTRY[model_key]["supported_ids"])

REAL_CSV_HEADERS = {
    "timestamp": ("时间标识", "timestamp", "time", "ts"),
    "frame_name": ("帧ID", "帧名", "frame_id", "frame_name"),
    "data": ("数据", "data", "payload"),
}


@dataclass
class DecodedState:
    speed_mps: float = 0.0
    engine_rpm: float = 0.0
    angle_deg: float = 0.0
    total_mileage_km: float = 0.0
    runtime_min: float = 0.0
    payload_tons: float = 0.0
    cabin_id: int = 0
    slope_state: int = 0
    load_state: int = 0
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
            "speed_mps": self.speed_mps,
            "engine_rpm": self.engine_rpm,
            "angle_deg": self.angle_deg,
            "total_mileage_km": self.total_mileage_km,
            "runtime_min": self.runtime_min,
            "payload_tons": self.payload_tons,
            "cabin_id": self.cabin_id,
            "slope_state": self.slope_state,
            "load_state": self.load_state,
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
    """Decode LuTech CAN frames into unified state.

    支持车型：
    - 190（原有协议）
    - 50/105（同一协议分支）
    """

    PROTOCOL_REGISTRY: Dict[str, Dict[str, Any]] = {
        "190": {
            "supported_ids": SUPPORTED_IDS_190,
            "decoder_method": "_update_190",
        },
        "50": {
            "supported_ids": SUPPORTED_IDS_50,
            "decoder_method": "_update_50_105",
        },
        "105": {
            "supported_ids": SUPPORTED_IDS_105,
            "decoder_method": "_update_50_105",
        },
    }

    def __init__(self, model: str = "50"):
        self.state = DecodedState()
        model_key = normalize_model(model)
        if model_key not in self.PROTOCOL_REGISTRY:
            model_key = "50"
        self.model = model_key
        self._last_values: Dict[str, float] = {}
        self._last_trends: Dict[str, str] = {}
        self._stable_counts: Dict[str, int] = {}

    @staticmethod
    def _u16_le(data: List[int], idx: int) -> int:
        return int(data[idx]) | (int(data[idx + 1]) << 8)

    @staticmethod
    def _u24_le(data: List[int], idx: int) -> int:
        return int(data[idx]) | (int(data[idx + 1]) << 8) | (int(data[idx + 2]) << 16)

    @staticmethod
    def _u32_le(data: List[int], idx: int) -> int:
        return (
            int(data[idx])
            | (int(data[idx + 1]) << 8)
            | (int(data[idx + 2]) << 16)
            | (int(data[idx + 3]) << 24)
        )

    @staticmethod
    def _apply_scale_offset(raw: int, scale: float, offset: float) -> float:
        return raw * scale + offset

    def _update_signal_memory(self, key: str, value: float) -> float:
        last = self._last_values.get(key, value)
        delta = value - last
        self._last_values[key] = value
        return delta

    @staticmethod
    def parse_frame_name(frame_name: str) -> int:
        """将真实数据里的帧名解析成 frame_id 整数。

        支持示例：
        - "18f181a0"（真实CSV常见格式）
        - "0x18F181A0"
        - "18F181A0"
        """
        text = str(frame_name).strip()
        if not text:
            raise ValueError("empty frame name")
        text = text.lower()
        if text.startswith("0x"):
            text = text[2:]
        return int(text, 16)

    @staticmethod
    def parse_payload_hex(payload_text: str) -> List[int]:
        """将空格分隔的8字节16进制文本解析为整数数组。"""
        parts = [p for p in str(payload_text).strip().split() if p]
        if len(parts) != 8:
            raise ValueError(f"payload bytes must be 8, got {len(parts)}: {payload_text}")
        data = [int(p, 16) & 0xFF for p in parts]
        return data

    @staticmethod
    def parse_real_timestamp(ts_text: str) -> datetime:
        """解析真实数据时间戳：2026-04-24 07:45:56.233.239。

        该格式是“秒.毫秒.微秒(3位)”；将其合并为标准微秒后解析。
        """
        text = str(ts_text).strip()
        if not text:
            raise ValueError("empty timestamp")

        # 兼容标准 datetime（无双小数段）
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass

        # 兼容真实CSV格式：YYYY-mm-dd HH:MM:SS.mmm.uuu
        parts = text.rsplit(".", 2)
        if len(parts) != 3:
            raise ValueError(f"unsupported timestamp format: {ts_text}")
        base, ms, us = parts
        frac6 = (ms + us).ljust(6, "0")[:6]
        return datetime.strptime(f"{base}.{frac6}", "%Y-%m-%d %H:%M:%S.%f")

    @classmethod
    def frame_from_real_csv_row(cls, row: Dict[str, Any]) -> Tuple[datetime, int, List[int]]:
        """把真实CSV一行转换为 (timestamp, frame_id, payload_bytes)。"""
        ts_key = next((k for k in REAL_CSV_HEADERS["timestamp"] if k in row), None)
        fid_key = next((k for k in REAL_CSV_HEADERS["frame_name"] if k in row), None)
        data_key = next((k for k in REAL_CSV_HEADERS["data"] if k in row), None)

        if ts_key is None or fid_key is None or data_key is None:
            raise KeyError("row missing required columns: 时间标识/帧ID/数据")

        ts = cls.parse_real_timestamp(str(row[ts_key]))
        frame_id = cls.parse_frame_name(str(row[fid_key]))
        payload = cls.parse_payload_hex(str(row[data_key]))
        return ts, frame_id, payload

    def update(self, frame_id: int, data: List[int]) -> None:
        if len(data) != 8:
            raise ValueError("CAN payload must be 8 bytes")

        b = [int(x) & 0xFF for x in data]
        protocol = self.PROTOCOL_REGISTRY[self.model]

        supported_ids = protocol["supported_ids"]
        if frame_id not in supported_ids:
            return

        decoder_method_name = protocol["decoder_method"]
        decoder_method = getattr(self, decoder_method_name)
        decoder_method(frame_id, b)

    def _update_190(self, frame_id: int, b: List[int]) -> None:
        if frame_id == ID_181:
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
            self.state.brake_pressure_bar = float(self._u16_le(b, 0))
            self.state.travel_pressure_bar = float(self._u16_le(b, 2))
            self.state.system_pressure_bar = float(self._u16_le(b, 4))
            self.state.clamp_pressure_bar = float(self._u16_le(b, 6))
        elif frame_id == ID_183:
            raw_ch4 = self._u16_le(b, 0)
            raw_spd = self._u16_le(b, 2)
            raw_rpm = self._u16_le(b, 4)
            raw_ang = self._u16_le(b, 6)
            self.state.methane_pct = self._apply_scale_offset(raw_ch4, 0.1, -400.0)
            self.state.speed_mps = self._apply_scale_offset(raw_spd, 0.1, 0.0)
            self.state.engine_rpm = float(raw_rpm)
            self.state.angle_deg = self._apply_scale_offset(raw_ang, 0.1, -900.0)
        elif frame_id == ID_184:
            self.state.coolant_temp_c = self._apply_scale_offset(b[0], 1.0, -40.0)
            self.state.surface_temp_c = self._apply_scale_offset(b[1], 1.0, 0.0)
            self.state.exhaust_temp_c = self._apply_scale_offset(b[2], 1.0, 0.0)
            self.state.intake_pressure_kpa = self._apply_scale_offset(b[3], 2.0, 0.0)
            self.state.water_tank_level_pct = self._apply_scale_offset(b[4], 1.0, 0.0)
            self.state.oil_pressure_kpa = self._apply_scale_offset(b[5], 4.0, 0.0)
            self.state.diesel_level_cm = self._apply_scale_offset(b[6], 1.0, 0.0)
            self.state.intake_temp_c = self._apply_scale_offset(b[7], 1.0, -40.0)
        elif frame_id == ID_185:
            total_mileage_raw = self._u24_le(b, 0)
            runtime_raw = self._u24_le(b, 3)
            self.state.total_mileage_km = total_mileage_raw * 0.1
            self.state.runtime_min = float(runtime_raw)
        elif frame_id == ID_189:
            payload_raw = self._u16_le(b, 0)
            co_raw = self._u16_le(b, 2)
            self.state.payload_tons = float(payload_raw)
            self.state.co_ppm = float(co_raw)
        elif frame_id == ID_190:
            angle_raw = self._u16_le(b, 0)
            self.state.angle_deg = (angle_raw - 900) * 0.1
            self.state.cabin_id = int(b[2])
            self.state.slope_state = int(b[3])

    def _update_50_105(self, frame_id: int, b: List[int]) -> None:
        if frame_id == ID_181:
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
            self.state.brake_pressure_bar = float(self._u16_le(b, 0))
            self.state.travel_pressure_bar = float(self._u16_le(b, 2))
            self.state.system_pressure_bar = float(self._u16_le(b, 4))
            self.state.clamp_pressure_bar = float(self._u16_le(b, 6))
        elif frame_id == ID_183:
            raw_batt = self._u16_le(b, 0)
            raw_spd = self._u16_le(b, 2)
            raw_rpm = self._u16_le(b, 4)
            raw_oil = self._u16_le(b, 6)
            self.state.battery_v = raw_batt * 0.01
            self.state.speed_mps = raw_spd * 0.1
            self.state.engine_rpm = float(raw_rpm)
            self.state.oil_pressure_kpa = float(raw_oil)
        elif frame_id == ID_184:
            self.state.intake_pressure_kpa = float(self._u16_le(b, 0))
            self.state.surface_temp_c = float(self._u16_le(b, 2))
            self.state.exhaust_temp_c = float(self._u16_le(b, 4))
            self.state.coolant_temp_c = float(self._u16_le(b, 6))
        elif frame_id == ID_185:
            total_m_raw = self._u32_le(b, 0)
            runtime_s_raw = self._u32_le(b, 4)
            self.state.total_mileage_km = total_m_raw / 1000.0
            self.state.runtime_min = runtime_s_raw / 60.0
        elif frame_id == ID_186:
            self.state.load_state = int(b[4])
            self.state.methane_pct = float(b[5]) * 0.1
            self.state.diesel_level_cm = float(b[6])
            self.state.water_tank_level_pct = float(b[7])
        elif frame_id == ID_189:
            # 2.9 取后四字节：Byte5-6 一氧化碳，Byte7-8 进气温度
            self.state.co_ppm = float(self._u16_le(b, 4))
            self.state.intake_temp_c = float(self._u16_le(b, 6))
        elif frame_id == ID_189:
            # 2.9 取后四字节：Byte5-6 一氧化碳，Byte7-8 进气温度
            self.state.co_ppm = float(self._u16_le(b, 4))
            self.state.intake_temp_c = float(self._u16_le(b, 6))


    def set_heartbeat(self, ok: int) -> None:
        self.state.can_heartbeat_ok = 1 if ok else 0

    def to_dict(self) -> Dict[str, Any]:
        return self.state.to_dict()


# Encoding helpers (for simulator/replay generation)
def to_u8(v: int) -> int:
    return max(0, min(255, v))


def to_u16_le(v: int) -> Tuple[int, int]:
    v = max(0, min(65535, v))
    return v & 0xFF, (v >> 8) & 0xFF
