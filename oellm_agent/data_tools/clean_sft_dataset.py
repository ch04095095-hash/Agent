#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class SampleKey:
    scenario: str
    risk_level_raw: str
    action: str
    speed_mps: float
    engine_rpm: float
    warning_tags: Tuple[str, ...]



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



def _extract_key(sample: Dict[str, Any]) -> SampleKey:
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}

    warning_tags = out.get("warning_tags", [])
    if not isinstance(warning_tags, list):
        warning_tags = []
    warning_tags = tuple(sorted(_safe_str(x) for x in warning_tags if _safe_str(x)))

    return SampleKey(
        scenario=_safe_str(out.get("scenario"), _safe_str(inp.get("history", {}).get("vehicle_context", ""), "unknown")),
        risk_level_raw=_safe_str(out.get("risk_level_raw"), _safe_str(out.get("risk_level"), "unknown")),
        action=_safe_str(out.get("action"), "unknown"),
        speed_mps=round(_safe_float(current.get("speed_mps")), 2),
        engine_rpm=round(_safe_float(current.get("engine_rpm")), 0),
        warning_tags=warning_tags,
    )



def _bucket(sample: Dict[str, Any]) -> str:
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}

    risk = _safe_str(out.get("risk_level"), "unknown")
    action = _safe_str(out.get("action"), "unknown")
    scenario = _safe_str(out.get("scenario"), _safe_str(history.get("vehicle_context"), "unknown"))
    speed = _safe_float(current.get("speed_mps"))
    last_action = _safe_str(history.get("last_effective_action"), "")
    age = _safe_float(history.get("last_action_age_sec"))

    if scenario == "not_in_drive":
        return "not_in_drive"
    if risk == "danger":
        return f"danger::{action}"
    if risk == "warning":
        return f"warning::{action}"
    if speed < 0.15:
        return f"normal_low::{action}"
    if speed < 1.0:
        if last_action == "ACCELERATE" and age < 4.0:
            return "normal_cooldown"
        return f"normal_low::{action}"
    return f"normal_cruise::{action}"



def _quality_filter(sample: Dict[str, Any]) -> Tuple[bool, str]:
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}

    action = _safe_str(out.get("action"), "")
    risk = _safe_str(out.get("risk_level"), "")
    scenario = _safe_str(out.get("scenario"), "")
    reason = _safe_str(out.get("reason"), "")
    speed = _safe_float(current.get("speed_mps"))
    engine_rpm = _safe_float(current.get("engine_rpm"))

    if action not in {"FORWARD", "ACCELERATE", "DECELERATE", "BRAKE"}:
        return False, "invalid_action"
    if risk not in {"normal", "warning", "high_warning", "danger"}:
        return False, "invalid_risk"
    if not reason or "�" in reason:
        return False, "bad_reason"

    # Avoid obviously contradictory labels.
    if risk == "danger" and action == "FORWARD":
        return False, "danger_forward"
    if risk == "danger" and speed >= 0.0 and scenario != "not_in_drive" and action == "ACCELERATE":
        return False, "danger_accelerate"
    if risk == "warning" and action == "BRAKE" and speed < 0.15:
        return False, "warning_brake_near_stop"
    if scenario == "not_in_drive" and action == "ACCELERATE" and speed < 0.05 and engine_rpm < 100:
        return False, "not_in_drive_accelerate"

    # Require minimal history fields.
    if not isinstance(history, dict) or not history:
        return False, "missing_history"

    return True, "ok"



def _maybe_overwrite_reason(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Keep current reason if it is descriptive enough; otherwise rebuild a concise one.
    inp = sample.get("input", {}) if isinstance(sample.get("input"), dict) else {}
    out = sample.get("output", {}) if isinstance(sample.get("output"), dict) else {}
    current = inp.get("current", {}) if isinstance(inp.get("current"), dict) else {}
    history = inp.get("history", {}) if isinstance(inp.get("history"), dict) else {}
    risk = inp.get("features", {}).get("risk", {}) if isinstance(inp.get("features"), dict) else {}

    action = _safe_str(out.get("action"), "")
    scenario = _safe_str(out.get("scenario"), _safe_str(history.get("vehicle_context"), "unknown"))
    risk_level = _safe_str(out.get("risk_level"), "unknown")
    tags = out.get("warning_tags", []) if isinstance(out.get("warning_tags"), list) else []
    tags = [str(x) for x in tags if str(x).strip()]

    if len(_safe_str(out.get("reason"))) >= 8 and "�" not in _safe_str(out.get("reason")):
        return sample

    speed = _safe_float(current.get("speed_mps"))
    rpm = _safe_float(current.get("engine_rpm"))
    last_action = _safe_str(history.get("last_effective_action"), "")
    age = _safe_float(history.get("last_action_age_sec"))
    speed_trend = _safe_str(history.get("speed_trend_5s"), "stable")

    if action == "ACCELERATE":
        reason = f"当前风险{risk_level}，速度{speed:.2f}m/s偏低且发动机转速{rpm:.0f}rpm允许提升，执行一次补速。"
    elif action == "FORWARD":
        reason = f"当前风险{risk_level}，结合场景{scenario}与历史动作{last_action}(已持续{age:.1f}s)、速度趋势{speed_trend}，本轮保持观察。"
    elif action == "DECELERATE":
        reason = f"当前风险{risk_level}，触发预警{','.join(tags[:3]) if tags else 'none'}，本轮减速以降低风险。"
    else:
        reason = f"当前风险{risk_level}且存在硬保护/高风险预警，立即制动。"

    sample = dict(sample)
    sample["output"] = dict(out)
    sample["output"]["reason"] = reason
    return sample



def clean_dataset(input_path: Path, output_path: Path, valid_output_path: Path | None = None, min_per_bucket: int = 150, max_per_bucket: int = 800, max_total: int | None = None) -> Dict[str, Any]:
    samples = _read_jsonl(input_path)

    cleaned: List[Dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen = set()
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for s in samples:
        ok, reason = _quality_filter(s)
        if not ok:
            rejected[reason] += 1
            continue
        s = _maybe_overwrite_reason(s)
        key = _extract_key(s)
        key_s = json.dumps(key.__dict__, ensure_ascii=False, sort_keys=True)
        if key_s in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key_s)
        bucket = _bucket(s)
        buckets[bucket].append(s)

    # Balanced sampling per bucket.
    for bucket, items in sorted(buckets.items(), key=lambda kv: kv[0]):
        if len(items) < min_per_bucket:
            # keep all if bucket is small; don't over-prune minority cases
            chosen = items
        else:
            chosen = items[:max_per_bucket]
        cleaned.extend(chosen)

    if max_total is not None and len(cleaned) > max_total:
        cleaned = cleaned[:max_total]

    # deterministic split
    split_idx = int(len(cleaned) * 0.9)
    split_idx = max(1, min(split_idx, len(cleaned) - 1)) if len(cleaned) > 1 else len(cleaned)
    train_rows = cleaned[:split_idx]
    valid_rows = cleaned[split_idx:] if len(cleaned) > 1 else []

    _write_jsonl(output_path, train_rows)
    if valid_output_path is not None:
        _write_jsonl(valid_output_path, valid_rows)

    summary = {
        "input": str(input_path),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "total_cleaned": len(cleaned),
        "buckets": {k: len(v) for k, v in sorted(buckets.items(), key=lambda kv: kv[0])},
        "rejected": dict(rejected),
        "output": str(output_path),
        "valid_output": str(valid_output_path) if valid_output_path else "",
    }
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and rebalance SFT dataset for mine-truck control.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--min-per-bucket", type=int, default=150)
    parser.add_argument("--max-per-bucket", type=int, default=800)
    parser.add_argument("--max-total", type=int, default=None)
    args = parser.parse_args()

    summary = clean_dataset(
        input_path=args.input,
        output_path=args.output,
        valid_output_path=args.valid_output,
        min_per_bucket=args.min_per_bucket,
        max_per_bucket=args.max_per_bucket,
        max_total=args.max_total,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
