#!/usr/bin/env python3
"""Quickly verify CAN decoding against simulated CSV rows.

This script reads the simulator output CSV, reconstructs CAN frames,
decodes them with ``CanDecoder``, and prints compact JSON reports for
selected rows so you can inspect the decoded structure and rule hits.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from oellm_agent.can.can_decoder import CanDecoder


def _parse_frame_id(value: str) -> int:
    value = value.strip()
    if value.lower().startswith("0x"):
        return int(value, 16)
    return int(value)


def _read_rows(csv_path: Path, limit: int) -> Iterable[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            yield row


def _row_to_frame(row: Dict[str, str]) -> Tuple[int, List[int]]:
    frame_id = _parse_frame_id(row["frame_id_hex"])
    payload = [int(row[f"byte{i}"]) & 0xFF for i in range(8)]
    return frame_id, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify simulated CAN decoding")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="/mnt/ssd/Agent/oellm_agent/sim_data/sim_can_frames_10min_10hz.csv",
        help="Path to the simulated CAN CSV",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of CSV rows to inspect")
    parser.add_argument(
        "--group-by-step",
        action="store_true",
        help="Group frames by step and print one decoded report per step",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    decoder = CanDecoder()

    if args.group_by_step:
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in _read_rows(csv_path, args.limit * 4):
            grouped[row["step"]].append(row)

        for step in sorted(grouped.keys(), key=lambda x: float(x))[: args.limit]:
            rows = grouped[step]
            latest_frame_id = None
            latest_payload = None
            for row in rows:
                frame_id, payload = _row_to_frame(row)
                decoder.update(frame_id, payload)
                latest_frame_id = frame_id
                latest_payload = payload

            report = decoder.build_status_report()
            print(
                json.dumps(
                    {
                        "step": float(step),
                        "last_frame_id": f"0x{latest_frame_id:08X}" if latest_frame_id is not None else None,
                        "last_payload": latest_payload,
                        "report": report,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print()
        return

    for idx, row in enumerate(_read_rows(csv_path, args.limit)):
        frame_id, payload = _row_to_frame(row)
        decoder.update(frame_id, payload)
        report = decoder.build_status_report()
        print(
            json.dumps(
                {
                    "row_index": idx,
                    "step": float(row["step"]),
                    "timestamp": row["timestamp"],
                    "frame_id": row["frame_id_hex"],
                    "payload": payload,
                    "report": report,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print()


if __name__ == "__main__":
    main()
