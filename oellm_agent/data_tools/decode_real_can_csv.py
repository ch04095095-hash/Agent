#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from oellm_agent.can.can_decoder import CanDecoder


def _state_row(ts_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "speed_mps",
        "engine_rpm",
        "gear_state",
        "emergency_stop",
        "brake_pressure_bar",
        "travel_pressure_bar",
        "system_pressure_bar",
        "clamp_pressure_bar",
        "coolant_temp_c",
        "surface_temp_c",
        "exhaust_temp_c",
        "intake_pressure_kpa",
        "hydraulic_oil_temp_c",
        "hydraulic_oil_level_pct",
        "oil_pressure_kpa",
        "diesel_level_cm",
        "water_tank_level_pct",
        "battery_v",
        "total_mileage_km",
        "runtime_min",
        "load_state",
        "methane_pct",
        "co_ppm",
        "intake_temp_c",
    ]
    out: Dict[str, Any] = {"timestamp": ts_text}
    for key in keys:
        out[key] = state.get(key, "")
    return out


def decode_csv(input_path: Path, output_path: Path, model: str = "50") -> Tuple[int, int]:
    decoder = CanDecoder(model=model)
    rows_out: List[Dict[str, Any]] = []
    frames = 0
    groups = 0
    current_group_seen: set[int] = set()
    latest_ts = ""

    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts, frame_id, payload = CanDecoder.frame_from_real_csv_row(row)
            except Exception:
                continue
            ts_text = str(row.get("时间标识", "") or ts.isoformat())
            if frame_id in current_group_seen and current_group_seen:
                rows_out.append(_state_row(latest_ts, decoder.state.to_dict()))
                groups += 1
                current_group_seen = set()
            decoder.update(frame_id, payload)
            current_group_seen.add(frame_id)
            latest_ts = ts_text
            frames += 1

    if current_group_seen:
        rows_out.append(_state_row(latest_ts, decoder.state.to_dict()))
        groups += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys()) if rows_out else ["timestamp"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return frames, groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode real CAN CSV into physical time-series rows.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--model", default="50", choices=["50", "105", "190"])
    args = parser.parse_args()
    output = args.output or args.input.with_name(args.input.stem + "_decoded.csv")
    frames, groups = decode_csv(args.input, output, model=args.model)
    print(f"frames={frames}")
    print(f"decoded_rows={groups}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
