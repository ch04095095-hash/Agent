#!/usr/bin/env python3
"""给组级解码后的运行数据追加预测性风险评分。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
for p in (PROJECT_DIR, WORKSPACE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from oellm_agent.predictive_risk_engine import PredictiveRiskEngine

DEFAULT_INPUT = Path("/mnt/ssd/Agent/oellm_agent/data/data_2_decoded_50_grouped.csv")
DEFAULT_OUTPUT = Path("/mnt/ssd/Agent/oellm_agent/data/data_2_predictive_risk_50.csv")
DEFAULT_THRESHOLDS = Path("/mnt/ssd/Agent/oellm_agent/config/thresholds/50.json")


def _load_thresholds(path: Path) -> Dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in obj.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="组级运行数据预测性风险评分")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="组级解码CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出CSV")
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS), help="阈值JSON")
    parser.add_argument("--window-size", type=int, default=40, help="历史窗口组数，默认40组约20秒")
    parser.add_argument("--horizon-sec", type=float, default=30.0, help="预测未来秒数")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    thresholds = _load_thresholds(Path(args.thresholds))
    engine = PredictiveRiskEngine(thresholds, window_size=args.window_size, horizon_sec=args.horizon_sec)

    output_rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row_for_engine = dict(row)
            row_for_engine["t_sec"] = idx * 0.5
            result = engine.update(row_for_engine).to_dict()
            output_rows.append({**row, **result})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_rows:
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
            writer.writeheader()
            writer.writerows(output_rows)

    level_counts: Dict[str, int] = {}
    for row in output_rows:
        level = str(row.get("risk_level", "normal"))
        level_counts[level] = level_counts.get(level, 0) + 1

    print(str(output_path))
    print(f"rows={len(output_rows)}")
    print(f"level_counts={level_counts}")


if __name__ == "__main__":
    main()
