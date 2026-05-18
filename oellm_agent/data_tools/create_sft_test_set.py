#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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



def _bucket(sample: Dict[str, Any]) -> str:
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}
    risk = _safe_str(out.get("risk_level"), "unknown")
    action = _safe_str(out.get("action"), "unknown")
    scenario = _safe_str(out.get("scenario"), _safe_str(history.get("vehicle_context"), "unknown"))
    boundary_kind = _safe_str(out.get("boundary_kind"), "")
    if boundary_kind:
        return f"boundary::{boundary_kind}"
    if scenario == "not_in_drive":
        return "not_in_drive"
    return f"{risk}::{action}"



def _normalize(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only model-facing fields to make the test set clean and comparable.
    if not isinstance(sample.get("input"), dict) or not isinstance(sample.get("output"), dict):
        return sample
    out = sample["output"]
    keep_out_keys = {
        "action",
        "risk_level",
        "reason",
        "confidence",
        "suspected_fault",
        "recommended_adjustment",
        "monitor_next",
        "warning_tags",
        "policy",
        "decision_source",
        "scenario",
        "risk_level_raw",
        "boundary_kind",
    }
    sample = dict(sample)
    sample["output"] = {k: out[k] for k in keep_out_keys if k in out}
    return sample



def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = json.dumps(
            {
                "scenario": _safe_str(row.get("output", {}).get("scenario"), ""),
                "boundary_kind": _safe_str(row.get("output", {}).get("boundary_kind"), ""),
                "risk_level": _safe_str(row.get("output", {}).get("risk_level"), ""),
                "action": _safe_str(row.get("output", {}).get("action"), ""),
                "speed_mps": round(_safe_float(row.get("input", {}).get("current", {}).get("speed_mps")), 2),
                "engine_rpm": round(_safe_float(row.get("input", {}).get("current", {}).get("engine_rpm")), 0),
                "warning_tags": tuple(sorted([str(x) for x in (row.get("output", {}).get("warning_tags", []) if isinstance(row.get("output", {}).get("warning_tags", []), list) else []) if str(x).strip()])),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out



def create_test_set(
    input_path: Path,
    output_path: Path,
    size: int = 400,
    seed: int = 42,
    per_bucket_cap: int = 80,
    valid_output_path: Path | None = None,
    valid_ratio: float = 0.5,
) -> Dict[str, Any]:
    rows = [_normalize(r) for r in _read_jsonl(input_path)]
    rows = _dedupe(rows)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_bucket(row)].append(row)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    bucket_counts: Dict[str, int] = {}

    # Ensure coverage: take from every bucket first.
    for bucket in sorted(buckets.keys()):
        items = buckets[bucket]
        rng.shuffle(items)
        take = min(len(items), per_bucket_cap)
        chosen = items[:take]
        selected.extend(chosen)
        bucket_counts[bucket] = len(chosen)

    # If we have too many, trim balanced by bucket proportionally.
    if len(selected) > size:
        rng.shuffle(selected)
        selected = selected[:size]
    elif len(selected) < size:
        remaining = [r for r in rows if r not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, size - len(selected))])

    selected = _dedupe(selected)
    rng.shuffle(selected)

    split_idx = int(len(selected) * (1.0 - valid_ratio))
    split_idx = max(1, min(split_idx, len(selected) - 1)) if len(selected) > 1 else len(selected)
    train_rows = selected[:split_idx]
    valid_rows = selected[split_idx:] if len(selected) > 1 else []

    _write_jsonl(output_path, train_rows)
    if valid_output_path is not None:
        _write_jsonl(valid_output_path, valid_rows)

    final_bucket_counts = defaultdict(int)
    for row in selected:
        final_bucket_counts[_bucket(row)] += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "valid_output": str(valid_output_path) if valid_output_path else "",
        "total_available": len(rows),
        "selected": len(selected),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "target_size": size,
        "bucket_counts": dict(sorted(final_bucket_counts.items(), key=lambda kv: kv[0])),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced SFT test set from final training data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--per-bucket-cap", type=int, default=80)
    parser.add_argument("--valid-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = create_test_set(
        input_path=args.input,
        output_path=args.output,
        size=args.size,
        seed=args.seed,
        per_bucket_cap=args.per_bucket_cap,
        valid_output_path=args.valid_output,
        valid_ratio=args.valid_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
