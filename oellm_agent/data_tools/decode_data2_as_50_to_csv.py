#!/usr/bin/env python3
"""将 data_2.csv 按 50 协议分组解码为物理量 CSV（181 开组，收齐 181/182/183/184/185/186/189 出一行）。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
for p in (PROJECT_DIR, WORKSPACE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from oellm_agent.can.can_decoder import CanDecoder, ID_181, ID_182, ID_183, ID_184, ID_185, ID_186, ID_189

INPUT_CSV = Path("/mnt/ssd/Agent/oellm_agent/data/data_2.csv")
OUTPUT_CSV = Path("/mnt/ssd/Agent/oellm_agent/data/data_2_decoded_50_grouped.csv")

GROUP_REQUIRED_IDS = (ID_181, ID_182, ID_183, ID_184, ID_185, ID_186, ID_189)
GROUP_ID_SET = set(GROUP_REQUIRED_IDS)
GROUP_START_ID = ID_181


def _finalize_group(group_idx: int, rows: List[Tuple[str, int, List[int]]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    decoder = CanDecoder(model="50")
    seen_ids = set()

    for ts, frame_id, payload in rows:
        seen_ids.add(frame_id)
        decoder.update(frame_id, payload)

    d = decoder.to_dict()
    ts_first = rows[0][0]
    ts_last = rows[-1][0]

    out: Dict[str, Any] = {
        "group_ts_first": ts_first,
        "group_ts_last": ts_last,
        "battery_v": d.get("battery_v"),
        "speed_mps": d.get("speed_mps"),
        "engine_rpm": d.get("engine_rpm"),
        "oil_pressure_kpa": d.get("oil_pressure_kpa"),
        "intake_pressure_kpa": d.get("intake_pressure_kpa"),
        "surface_temp_c": d.get("surface_temp_c"),
        "exhaust_temp_c": d.get("exhaust_temp_c"),
        "coolant_temp_c": d.get("coolant_temp_c"),
        "brake_pressure_bar": d.get("brake_pressure_bar"),
        "travel_pressure_bar": d.get("travel_pressure_bar"),
        "system_pressure_bar": d.get("system_pressure_bar"),
        "clamp_pressure_bar": d.get("clamp_pressure_bar"),
        "hydraulic_oil_temp_c": d.get("hydraulic_oil_temp_c"),
        "hydraulic_oil_level_pct": d.get("hydraulic_oil_level_pct"),
        "make_up_oil_pressure_bar": d.get("make_up_oil_pressure_bar"),
        "total_mileage_km": d.get("total_mileage_km"),
        "runtime_min": d.get("runtime_min"),
        "load_state": d.get("load_state"),
        "intake_temp_c": d.get("intake_temp_c"),
        "diesel_level_cm": d.get("diesel_level_cm"),
        "water_level_pct": d.get("water_tank_level_pct"),
        "alarm_code": d.get("alarm_code"),
        "gear_state": d.get("gear_state"),
        "emergency_stop": d.get("emergency_stop"),
        "shua_qu_state": d.get("shua_qu_state"),
    }
    return out


def main() -> None:
    grouped_rows: List[Dict[str, Any]] = []
    current_group: List[Tuple[str, int, List[int]]] = []
    current_seen: set[int] = set()
    group_idx = 0

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts, frame_id, payload = CanDecoder.frame_from_real_csv_row(row)
            except Exception:
                continue

            if frame_id not in GROUP_ID_SET:
                continue

            # 181 开新组；若上一组未收齐也先结算
            if frame_id == GROUP_START_ID and current_group:
                one = _finalize_group(group_idx, current_group)
                if one is not None:
                    grouped_rows.append(one)
                group_idx += 1
                current_group = []
                current_seen = set()

            current_group.append((ts.isoformat(timespec="microseconds"), frame_id, payload))
            current_seen.add(frame_id)

            # 收齐 181/182/183/184/185/186/189 立即结算出一行
            if all(fid in current_seen for fid in GROUP_REQUIRED_IDS):
                one = _finalize_group(group_idx, current_group)
                if one is not None:
                    grouped_rows.append(one)
                group_idx += 1
                current_group = []
                current_seen = set()

    # 收尾：若最后一组不为空，也输出（可能不完整）
    one = _finalize_group(group_idx, current_group)
    if one is not None:
        grouped_rows.append(one)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if grouped_rows:
        with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(grouped_rows[0].keys()))
            writer.writeheader()
            writer.writerows(grouped_rows)

    print(str(OUTPUT_CSV))
    print(f"grouped_rows={len(grouped_rows)}")


if __name__ == "__main__":
    main()
