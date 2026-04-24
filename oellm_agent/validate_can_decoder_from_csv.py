import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from can_decoder import CanDecoder

CSV_PATH = Path("/mnt/ssd/Agent/oellm_agent/sim_data/sim_can_frames_10min_10hz.csv")
OUTPUT_PATH = Path("/mnt/ssd/Agent/oellm_agent/sim_data/can_decoder_first_10s.json")
MAX_T_SEC = 20.0


def parse_frame_id(text: str) -> int:
    text = str(text).strip()
    try:
        return int(text, 16)
    except ValueError:
        return int(text[-8:], 16)


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_step_snapshots(rows: List[Dict[str, Any]], max_t_sec: float = MAX_T_SEC) -> List[Dict[str, Any]]:
    decoder = CanDecoder()
    snapshots: List[Dict[str, Any]] = []
    current_step = None

    for row in rows:
        t_sec = float(row["t_sec"])
        if t_sec > max_t_sec:
            break

        step = int(row["step"])
        frame_id = parse_frame_id(row["frame_id_hex"])
        payload = [int(row[f"byte{i}"]) for i in range(8)]
        heartbeat_ok = int(row.get("can_heartbeat_ok") or 1)

        if current_step is None:
            current_step = step
        if step != current_step:
            report = decoder.build_status_report()
            snapshots.append(
                {
                    "step": current_step,
                    "t_sec": float(prev_row["t_sec"]),
                    "timestamp": prev_row.get("timestamp", ""),
                    "decoded": report["decoded"],
                    "signals": report["signals"],
                    "llm_case": prev_row.get("llm_case", ""),
                }
            )
            current_step = step

        decoder.update(frame_id, payload)
        decoder.set_heartbeat(heartbeat_ok)
        prev_row = row

    if rows and current_step is not None:
        report = decoder.build_status_report()
        snapshots.append(
            {
                "step": current_step,
                "t_sec": float(prev_row["t_sec"]),
                "timestamp": prev_row.get("timestamp", ""),
                "decoded": report["decoded"],
                "signals": report["signals"],
                "llm_case": prev_row.get("llm_case", ""),
            }
        )

    return snapshots


def build_summary(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    decoded_rows = [x["decoded"] for x in snapshots]
    coolant_values = [float(x.get("coolant_temp_c", 0.0) or 0.0) for x in decoded_rows]
    water_values = [float(x.get("water_tank_level_pct", 0.0) or 0.0) for x in decoded_rows]
    methane_values = [float(x.get("methane_pct", 0.0) or 0.0) for x in decoded_rows]
    brake_values = [float(x.get("brake_pressure_bar", 0.0) or 0.0) for x in decoded_rows]
    system_values = [float(x.get("system_pressure_bar", 0.0) or 0.0) for x in decoded_rows]

    return {
        "csv_path": str(CSV_PATH),
        "steps": len(snapshots),
        "coolant_temp_c": {
            "min": min(coolant_values) if coolant_values else None,
            "max": max(coolant_values) if coolant_values else None,
        },
        "water_tank_level_pct": {
            "min": min(water_values) if water_values else None,
            "max": max(water_values) if water_values else None,
        },
        "methane_pct": {
            "min": min(methane_values) if methane_values else None,
            "max": max(methane_values) if methane_values else None,
        },
        "brake_pressure_bar": {
            "min": min(brake_values) if brake_values else None,
            "max": max(brake_values) if brake_values else None,
        },
        "system_pressure_bar": {
            "min": min(system_values) if system_values else None,
            "max": max(system_values) if system_values else None,
        },
    }


def main() -> None:
    rows = load_rows(CSV_PATH)
    snapshots = build_step_snapshots(rows, max_t_sec=MAX_T_SEC)
    payload = {
        "summary": build_summary(snapshots),
        "full_output_first_10s": snapshots,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
