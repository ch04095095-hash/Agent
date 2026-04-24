#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ml_risk_model import MLRiskModel
from train_risk_model import _read_l1_ticks


def _is_danger_tick(item: Dict[str, Any]) -> bool:
    return bool(item.get("reasons") or [])


def _build_eval_set(ticks: List[Dict[str, Any]], horizon_sec: float) -> Tuple[List[float], List[int], List[float]]:
    n = len(ticks)
    y = [0] * n
    danger_idx = [i for i in range(n) if _is_danger_tick(ticks[i])]
    j = 0
    for i in range(n):
        ti = ticks[i]["t_sec"]
        while j < len(danger_idx) and ticks[danger_idx[j]]["t_sec"] <= ti:
            j += 1
        if j < len(danger_idx):
            tj = ticks[danger_idx[j]]["t_sec"]
            if 0.0 < (tj - ti) <= horizon_sec:
                y[i] = 1

    t = [float(x["t_sec"]) for x in ticks]
    return t, y, [0.0] * n


def _roc_auc(y_true: List[int], y_score: List[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += i
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _precision_recall(y_true: List[int], y_score: List[float], th: float) -> Tuple[float, float, int, int, int]:
    tp = fp = fn = 0
    for y, s in zip(y_true, y_score):
        pred = 1 if s >= th else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall), tp, fp, fn


def _avg_early_warning_sec(t_sec: List[float], y_true: List[int], y_score: List[float], th: float) -> float:
    danger_times = [t_sec[i] for i, y in enumerate(y_true) if y == 1]
    if not danger_times:
        return 0.0

    # collapse near-duplicate positives into episodes
    episodes: List[float] = []
    for ts in danger_times:
        if not episodes or ts - episodes[-1] > 1.0:
            episodes.append(ts)

    early_list: List[float] = []
    for d in episodes:
        cands = [t for t, s in zip(t_sec, y_score) if (t < d and s >= th)]
        if not cands:
            continue
        early = d - max(cands)
        if early >= 0:
            early_list.append(early)

    if not early_list:
        return 0.0
    return float(sum(early_list) / len(early_list))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained ML risk model")
    ap.add_argument("--run-history", required=True)
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent / "models" / "risk_logreg.json"))
    ap.add_argument("--horizon-sec", type=float, default=10.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--scan-thresholds", action="store_true", help="scan thresholds from 0.30 to 0.90")
    args = ap.parse_args()

    ticks = _read_l1_ticks(Path(args.run_history))
    t_sec, y_true, _ = _build_eval_set(ticks, horizon_sec=float(args.horizon_sec))

    model = MLRiskModel(Path(args.model))
    y_score: List[float] = []
    for item in ticks:
        y_score.append(float(model.score(item.get("sensor", {}) or {})))

    auc = _roc_auc(y_true, y_score)

    if args.scan_thresholds:
        rows: List[Dict[str, Any]] = []
        best = None
        for i in range(30, 91, 5):
            th = i / 100.0
            precision, recall, tp, fp, fn = _precision_recall(y_true, y_score, th=th)
            avg_early = _avg_early_warning_sec(t_sec, y_true, y_score, th=th)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            row = {
                "threshold": round(th, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "avg_early_warning_sec": round(avg_early, 4),
            }
            rows.append(row)
            if best is None or row["f1"] > best["f1"]:
                best = row

        out = {
            "event": "risk_model_eval_scan",
            "model_loaded": bool(model.loaded),
            "model_version": model.version,
            "samples": len(y_true),
            "positives": int(sum(y_true)),
            "horizon_sec": float(args.horizon_sec),
            "auc": round(auc, 4),
            "best_by_f1": best,
            "threshold_rows": rows,
        }
        print(json.dumps(out, ensure_ascii=False))
        return

    precision, recall, tp, fp, fn = _precision_recall(y_true, y_score, th=float(args.threshold))
    avg_early = _avg_early_warning_sec(t_sec, y_true, y_score, th=float(args.threshold))

    out = {
        "event": "risk_model_eval",
        "model_loaded": bool(model.loaded),
        "model_version": model.version,
        "samples": len(y_true),
        "positives": int(sum(y_true)),
        "horizon_sec": float(args.horizon_sec),
        "threshold": float(args.threshold),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "avg_early_warning_sec": round(avg_early, 4),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
