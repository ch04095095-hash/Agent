#!/usr/bin/env python3
"""将 data_1.csv 按 105 协议解码，导出明细 JSON 与统计 JSON。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from oellm_agent.can.can_decoder import CanDecoder, SUPPORTED_IDS_105

INPUT_CSV = Path("/mnt/ssd/Agent/oellm_agent/data/data_1.csv")
OUTPUT_JSON = Path("/mnt/ssd/Agent/oellm_agent/data/data_1_decoded_105.json")
OUTPUT_SUMMARY = Path("/mnt/ssd/Agent/oellm_agent/data/data_1_decoded_105_summary.json")


def main() -> None:
    decoder = CanDecoder(model="105")

    decoded_rows: List[Dict[str, Any]] = []
    rows_seen = 0
    rows_parsed = 0
    rows_supported = 0

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_seen += 1
            try:
                ts, frame_id, payload = CanDecoder.frame_from_real_csv_row(row)
                rows_parsed += 1
            except Exception:
                continue

            if frame_id not in SUPPORTED_IDS_105:
                continue
            rows_supported += 1

            decoder.update(frame_id, payload)
            state = decoder.to_dict()

            decoded_rows.append(
                {
                    "timestamp": ts.isoformat(timespec="microseconds"),
                    "frame_id": f"0x{frame_id:08X}",
                    "payload_hex": " ".join(f"{x:02X}" for x in payload),
                    "battery_v": state.get("battery_v"),
                    "speed_mps": state.get("speed_mps"),
                    "engine_rpm": state.get("engine_rpm"),
                    "oil_pressure_kpa": state.get("oil_pressure_kpa"),
                    "intake_pressure_kpa": state.get("intake_pressure_kpa"),
                    "surface_temp_c": state.get("surface_temp_c"),
                    "exhaust_temp_c": state.get("exhaust_temp_c"),
                    "coolant_temp_c": state.get("coolant_temp_c"),
                    "total_mileage_km": state.get("total_mileage_km"),
                    "runtime_min": state.get("runtime_min"),
                    "load_state": state.get("load_state"),
                    "methane_pct": state.get("methane_pct"),
                    "diesel_level_cm": state.get("diesel_level_cm"),
                    "water_tank_level_pct": state.get("water_tank_level_pct"),
                }
            )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "input": str(INPUT_CSV),
            "output_json": str(OUTPUT_JSON),
            "rows_seen": rows_seen,
            "rows_parsed": rows_parsed,
            "rows_supported_50_105": rows_supported,
            "rows_decoded": len(decoded_rows),
        },
        "rows": decoded_rows,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = payload["summary"]
    OUTPUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(str(OUTPUT_JSON))
    print(str(OUTPUT_SUMMARY))


if __name__ == "__main__":
    main()
