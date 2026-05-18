#!/usr/bin/env python3
"""
将 data 目录下 CSV 按文件名编号拼接，然后先删列，再按帧ID过滤。

用法：
python oellm_agent/merge_and_drop_columns.py \
  --input-dir oellm_agent/data \
  --output oellm_agent/data/merged_filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Tuple


DEFAULT_KEEP_FRAMES = {
    "18f181a0",
    "18f182a0",
    "18f183a0",
    "18f184a0",
    "18f185a0",
    "18f186a0",
    "18f189a0",
}


def extract_sort_key(path: Path) -> Tuple[int, int, str]:
    stem = path.stem
    nums = re.findall(r"\d+", stem)
    if nums:
        return (0, int(nums[-1]), stem)
    return (1, 10**18, stem)


def read_csv_with_fallback(path: Path) -> tuple[list[str], list[list[str]]]:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return [], []
            return rows[0], rows[1:]
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"读取失败: {path}，错误: {last_err}")


def excel_col_letter_to_index(letter: str) -> int:
    letter = letter.strip().upper()
    idx = 0
    for ch in letter:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"非法列字母: {letter}")
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def drop_indices_for_header(header: list[str], letters: list[str]) -> list[int]:
    indices = sorted(
        {excel_col_letter_to_index(l) for l in letters if l.strip()},
        reverse=True,
    )
    return [i for i in indices if 0 <= i < len(header)]


def drop_by_indices(row: list[str], drop_indices: list[int]) -> list[str]:
    out = list(row)
    for idx in drop_indices:
        if 0 <= idx < len(out):
            out.pop(idx)
    return out


def find_frame_id_col(header: list[str]) -> int:
    candidates = {"帧id", "帧名", "frame_id", "frame_name"}
    for i, col in enumerate(header):
        if str(col).strip().lower() in candidates:
            return i
    raise ValueError("未找到帧ID列（支持：帧ID/帧名/frame_id/frame_name）")


def main() -> None:
    parser = argparse.ArgumentParser(description="拼接CSV后先删列，再按帧ID过滤")
    parser.add_argument("--input-dir", default="oellm_agent/data", help="输入目录")
    parser.add_argument("--output", default="oellm_agent/data/merged_filtered.csv", help="输出文件")
    parser.add_argument("--drop-letters", default="ABCEGHIJK", help="先删除的Excel列字母")
    parser.add_argument(
        "--keep-frames",
        default=",".join(sorted(DEFAULT_KEEP_FRAMES)),
        help="保留的帧ID（逗号分隔，不区分大小写）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    csv_files = [
        p
        for p in input_dir.glob("*.csv")
        if p.is_file() and p.resolve() != output_path.resolve()
    ]
    if not csv_files:
        raise FileNotFoundError(f"目录下未找到 CSV 文件: {input_dir}")
    csv_files.sort(key=extract_sort_key)

    letters = [ch for ch in args.drop_letters.upper() if ch.isalpha()]
    keep_frames = {x.strip().lower() for x in args.keep_frames.split(",") if x.strip()}

    merged_rows_count = 0
    kept_rows_count = 0
    base_header: list[str] | None = None
    drop_indices: list[int] | None = None
    frame_col_idx: int | None = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as out_f:
        writer = csv.writer(out_f)

        for i, file_path in enumerate(csv_files):
            header, rows = read_csv_with_fallback(file_path)
            if not header:
                continue

            if i == 0 or base_header is None:
                base_header = header
                drop_indices = drop_indices_for_header(base_header, letters)
                dropped_header = drop_by_indices(base_header, drop_indices)
                frame_col_idx = find_frame_id_col(dropped_header)
                writer.writerow(dropped_header)

            for row in rows:
                merged_rows_count += 1
                target_len = len(base_header or row)
                if len(row) < target_len:
                    row = row + [""] * (target_len - len(row))
                elif len(row) > target_len:
                    row = row[:target_len]

                dropped_row = drop_by_indices(row, drop_indices or [])
                fid = dropped_row[frame_col_idx].strip().lower() if frame_col_idx is not None and frame_col_idx < len(dropped_row) else ""
                if fid in keep_frames:
                    writer.writerow(dropped_row)
                    kept_rows_count += 1

    print(f"已拼接文件数: {len(csv_files)}")
    print(f"拼接后总行数: {merged_rows_count}")
    print(f"删除列字母: {''.join(letters)}")
    print(f"保留帧ID数: {len(keep_frames)}")
    print(f"过滤后输出行数: {kept_rows_count}")
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
