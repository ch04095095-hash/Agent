#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default



def _safe_str(v: Any, default: str = "") -> str:
    s = str(v or "").strip()
    return s if s else default



def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def _extract_key(sample: Dict[str, Any]) -> str:
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}
    tags = out.get("warning_tags", []) if isinstance(out.get("warning_tags"), list) else []
    tags = tuple(sorted(_safe_str(t) for t in tags if _safe_str(t)))
    return json.dumps(
        {
            "scenario": _safe_str(out.get("scenario"), _safe_str(history.get("vehicle_context"), "")),
            "boundary_kind": _safe_str(out.get("boundary_kind"), ""),
            "risk_level": _safe_str(out.get("risk_level"), ""),
            "action": _safe_str(out.get("action"), ""),
            "speed_mps": round(_safe_float(current.get("speed_mps")), 2),
            "engine_rpm": round(_safe_float(current.get("engine_rpm")), 0),
            "warning_tags": tags,
            "reason": _safe_str(out.get("reason"), "")[:120],
        },
        ensure_ascii=False,
        sort_keys=True,
    )



def merge_datasets(inputs: List[Path], output: Path, valid_output: Path | None = None, valid_ratio: float = 0.1, shuffle_seed: int = 42) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for path in inputs:
        rows = _read_jsonl(path)
        source_counts[str(path)] = len(rows)
        all_rows.extend(rows)

    # Deduplicate while preserving first occurrence.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    dup_count = 0
    for row in all_rows:
        key = _extract_key(row)
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(row)

    # Deterministic pseudo-shuffle by key ordering; avoids random module dependency in output.
    deduped.sort(key=_extract_key)

    split_idx = int(len(deduped) * (1.0 - valid_ratio))
    split_idx = max(1, min(split_idx, len(deduped) - 1)) if len(deduped) > 1 else len(deduped)
    train_rows = deduped[:split_idx]
    valid_rows = deduped[split_idx:] if len(deduped) > 1 else []

    _write_jsonl(output, train_rows)
    if valid_output is not None:
        _write_jsonl(valid_output, valid_rows)

    bucket_counts = Counter()
    for row in deduped:
        out = row.get("output", {}) if isinstance(row.get("output"), dict) else {}
        risk = _safe_str(out.get("risk_level"), "unknown")
        action = _safe_str(out.get("action"), "unknown")
        bucket_counts[f"{risk}::{action}"] += 1

    return {
        "inputs": [str(p) for p in inputs],
        "source_counts": source_counts,
        "output": str(output),
        "valid_output": str(valid_output) if valid_output else "",
        "merged_total": len(all_rows),
        "deduped_total": len(deduped),
        "duplicates_removed": dup_count,
        "train": len(train_rows),
        "valid": len(valid_rows),
        "buckets": dict(sorted(bucket_counts.items(), key=lambda kv: kv[0])),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple SFT JSONL datasets into a deduplicated train/valid split.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Input JSONL datasets to merge.")
    parser.add_argument("--output", type=Path, required=True, help="Merged train output JSONL.")
    parser.add_argument("--valid-output", type=Path, default=None, help="Merged valid output JSONL.")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    summary = merge_datasets(
        inputs=args.inputs,
        output=args.output,
        valid_output=args.valid_output,
        valid_ratio=args.valid_ratio,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
