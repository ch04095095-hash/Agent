import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Optional, Tuple

from oellm_agent.can.can_decoder import CanDecoder, ID_181, ID_182, ID_183, ID_184, ID_185, ID_189, ID_190, get_supported_ids
from oellm_agent.config.thresholds.thresholds import normalize_model

CSV_PATH = Path("/mnt/ssd/Agent/oellm_agent/data/data_1.csv")

# 为避免大文件 OOM，流式处理
MAX_ROWS: Optional[int] = None
MAX_GROUPS_TO_SAVE = 3000

MANDATORY_GROUP_IDS = (ID_181, ID_182, ID_183, ID_184, ID_185)

# 自适应周期参数（基于 ID_181 帧间隔）
DEFAULT_CYCLE_SEC = 0.5
CYCLE_SAMPLE_WINDOW = 50
MIN_CYCLE_SEC = 0.1
MAX_CYCLE_SEC = 2.0
GROUP_TIMEOUT_FACTOR = 1.8  # 当前组起始到当前帧超过 cycle*factor 判超时断组


class MinMax:
    def __init__(self) -> None:
        self.min_v: Optional[float] = None
        self.max_v: Optional[float] = None

    def update(self, v: float) -> None:
        if self.min_v is None or v < self.min_v:
            self.min_v = v
        if self.max_v is None or v > self.max_v:
            self.max_v = v

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"min": self.min_v, "max": self.max_v}


class AdaptiveCycleEstimator:
    def __init__(self) -> None:
        self.last_181_ts: Optional[datetime] = None
        self.intervals: list[float] = []
        self.cycle_sec: float = DEFAULT_CYCLE_SEC

    def update_with_181(self, ts: datetime) -> None:
        if self.last_181_ts is not None:
            dt = (ts - self.last_181_ts).total_seconds()
            if MIN_CYCLE_SEC <= dt <= MAX_CYCLE_SEC:
                self.intervals.append(dt)
                if len(self.intervals) > CYCLE_SAMPLE_WINDOW:
                    self.intervals.pop(0)
                self.cycle_sec = median(self.intervals) if self.intervals else DEFAULT_CYCLE_SEC
        self.last_181_ts = ts


def _safe_seconds_delta(ts: datetime, ts0: datetime) -> float:
    return (ts - ts0).total_seconds()


def _finalize_group(
    group_frames: Dict[int, Tuple[datetime, list[int], int]],
    ts0: Optional[datetime],
    saved_groups: list[Dict[str, Any]],
    max_groups_to_save: int,
    stats: Dict[str, Any],
    model: str,
    decode_order: Tuple[int, ...],
) -> Optional[datetime]:
    if not group_frames:
        return ts0

    if not all(fid in group_frames for fid in MANDATORY_GROUP_IDS):
        stats["groups_incomplete"] += 1
        return ts0

    decoder = CanDecoder(model=model)
    for fid in decode_order:
        frame = group_frames.get(fid)
        if frame is None:
            continue
        _, payload, _ = frame
        decoder.update(fid, payload)

    group_ts = min(item[0] for item in group_frames.values())
    if ts0 is None:
        ts0 = group_ts

    report = decoder.build_status_report()
    d = report["decoded"]
    stats["groups_decoded"] += 1

    stats["mm_coolant"].update(float(d.get("coolant_temp_c", 0.0) or 0.0))
    stats["mm_water"].update(float(d.get("water_tank_level_pct", 0.0) or 0.0))
    stats["mm_methane"].update(float(d.get("methane_pct", 0.0) or 0.0))
    stats["mm_brake"].update(float(d.get("brake_pressure_bar", 0.0) or 0.0))
    stats["mm_system"].update(float(d.get("system_pressure_bar", 0.0) or 0.0))

    if len(saved_groups) < max_groups_to_save:
        frame_dump = {}
        for fid in decode_order:
            if fid in group_frames:
                ts, payload, row_index = group_frames[fid]
                frame_dump[f"0x{fid:08X}"] = {
                    "row_index": row_index,
                    "timestamp": ts.isoformat(timespec="microseconds"),
                    "payload_hex": " ".join(f"{b:02X}" for b in payload),
                }

        saved_groups.append(
            {
                "group_index": stats["groups_decoded"],
                "timestamp": group_ts.isoformat(timespec="microseconds"),
                "t_sec": _safe_seconds_delta(group_ts, ts0),
                "frames": frame_dump,
                "decoded": d,
                "signals": report["signals"],
            }
        )

    stats["last_timestamp"] = group_ts.isoformat(timespec="microseconds")
    if stats["first_timestamp"] is None:
        stats["first_timestamp"] = stats["last_timestamp"]

    return ts0


def build_output_grouped(
    csv_path: Path,
    model: str,
    max_rows: Optional[int] = MAX_ROWS,
    max_groups_to_save: int = MAX_GROUPS_TO_SAVE,
) -> Dict[str, Any]:
    rows_seen = 0
    rows_parsed = 0

    model_key = normalize_model(model)
    supported_ids = get_supported_ids(model_key)
    optional_group_ids: Tuple[int, ...] = (ID_189, ID_190) if model_key == "190" else tuple()
    decode_order = MANDATORY_GROUP_IDS + optional_group_ids

    saved_groups: list[Dict[str, Any]] = []
    current_group_frames: Dict[int, Tuple[datetime, list[int], int]] = {}
    current_group_start_ts: Optional[datetime] = None
    ts0: Optional[datetime] = None

    cycle = AdaptiveCycleEstimator()

    stats: Dict[str, Any] = {
        "groups_decoded": 0,
        "groups_incomplete": 0,
        "groups_timeout_split": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "mm_coolant": MinMax(),
        "mm_water": MinMax(),
        "mm_methane": MinMax(),
        "mm_brake": MinMax(),
        "mm_system": MinMax(),
    }

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            rows_seen += 1
            if max_rows is not None and rows_seen > max_rows:
                break

            try:
                ts, frame_id, payload = CanDecoder.frame_from_real_csv_row(row)
                rows_parsed += 1
            except Exception:
                continue

            if frame_id not in supported_ids:
                continue

            if frame_id == ID_181:
                cycle.update_with_181(ts)

            if current_group_frames and current_group_start_ts is not None:
                timeout_sec = cycle.cycle_sec * GROUP_TIMEOUT_FACTOR
                if (ts - current_group_start_ts).total_seconds() > timeout_sec:
                    ts0 = _finalize_group(current_group_frames, ts0, saved_groups, max_groups_to_save, stats, model_key, decode_order)
                    current_group_frames = {}
                    current_group_start_ts = None
                    stats["groups_timeout_split"] += 1

            if frame_id == ID_181 and current_group_frames:
                ts0 = _finalize_group(current_group_frames, ts0, saved_groups, max_groups_to_save, stats, model_key, decode_order)
                current_group_frames = {}
                current_group_start_ts = None

            if not current_group_frames:
                current_group_start_ts = ts
            current_group_frames[frame_id] = (ts, payload, idx)

    ts0 = _finalize_group(current_group_frames, ts0, saved_groups, max_groups_to_save, stats, model_key, decode_order)

    summary = {
        "model": model_key,
        "csv_path": str(csv_path),
        "max_rows": max_rows,
        "rows_seen": rows_seen,
        "rows_parsed": rows_parsed,
        "groups_decoded": stats["groups_decoded"],
        "groups_incomplete": stats["groups_incomplete"],
        "groups_timeout_split": stats["groups_timeout_split"],
        "groups_saved": len(saved_groups),
        "groups_save_limit": max_groups_to_save,
        "mandatory_group_ids": [f"0x{x:08X}" for x in MANDATORY_GROUP_IDS],
        "optional_group_ids": [f"0x{x:08X}" for x in optional_group_ids],
        "estimated_cycle_sec": cycle.cycle_sec,
        "timeout_factor": GROUP_TIMEOUT_FACTOR,
        "estimated_group_timeout_sec": cycle.cycle_sec * GROUP_TIMEOUT_FACTOR,
        "first_timestamp": stats["first_timestamp"],
        "last_timestamp": stats["last_timestamp"],
        "coolant_temp_c": stats["mm_coolant"].to_dict(),
        "water_tank_level_pct": stats["mm_water"].to_dict(),
        "methane_pct": stats["mm_methane"].to_dict(),
        "brake_pressure_bar": stats["mm_brake"].to_dict(),
        "system_pressure_bar": stats["mm_system"].to_dict(),
    }

    return {"summary": summary, "groups": saved_groups}


def main() -> None:
    parser = argparse.ArgumentParser(description="按车型协议验证 CAN 解码结果")
    parser.add_argument("--model", default="190", help="车型: 190/105/50")
    parser.add_argument("--csv", default=str(CSV_PATH), help="输入 CSV 路径")
    parser.add_argument("--output", default="", help="输出 JSON 路径，默认自动按车型命名")
    args = parser.parse_args()

    model_key = normalize_model(args.model)
    csv_path = Path(args.csv)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_path.with_name(f"can_decoder_from_{csv_path.stem}_{model_key}.json")

    payload = build_output_grouped(csv_path, model=model_key, max_rows=MAX_ROWS, max_groups_to_save=MAX_GROUPS_TO_SAVE)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
