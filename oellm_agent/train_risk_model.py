#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


FEATURES = [
    "speed_kmh",
    "engine_rpm",
    "coolant_temp_c",
    "exhaust_temp_c",
    "hydraulic_oil_temp_c",
    "methane_pct",
    "co_ppm",
    "brake_pressure_bar",
    "system_pressure_bar",
    "travel_pressure_bar",
    "intake_pressure_kpa",
    "oil_pressure_kpa",
    "water_tank_level_pct",
    "diesel_level_cm",
    "make_up_oil_pressure_bar",
]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _read_l1_ticks(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("event") != "l1_tick":
            continue
        ss = obj.get("sensor_snapshot", {}) or {}
        out.append({
            "t_sec": float(obj.get("t_sec", 0.0) or 0.0),
            "reasons": list(obj.get("reasons", []) or []),
            "sensor": ss,
        })
    out.sort(key=lambda x: x["t_sec"])
    return out


def _is_danger_tick(item: Dict[str, Any]) -> bool:
    reasons = set(str(x) for x in (item.get("reasons") or []))
    return bool(reasons)


def _build_dataset(ticks: List[Dict[str, Any]], horizon_sec: float) -> Tuple[List[List[float]], List[int], Dict[str, float], Dict[str, float]]:
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

    # collect raw X
    Xraw: List[List[float]] = []
    for i in range(n):
        s = ticks[i]["sensor"]
        row = [float(s.get(f, 0.0) or 0.0) for f in FEATURES]
        Xraw.append(row)

    # standardize
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for k, f in enumerate(FEATURES):
        vals = [r[k] for r in Xraw]
        mu = sum(vals) / max(1, len(vals))
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals))
        sigma = max(1e-6, var ** 0.5)
        mean[f] = mu
        std[f] = sigma

    X = [[(r[k] - mean[FEATURES[k]]) / std[FEATURES[k]] for k in range(len(FEATURES))] for r in Xraw]
    return X, y, mean, std


def _train_logreg(X: List[List[float]], y: List[int], lr: float = 0.05, epochs: int = 400) -> Tuple[List[float], float]:
    if not X:
        return [0.0] * len(FEATURES), 0.0
    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    # class balance weight
    pos = sum(y)
    neg = max(1, len(y) - pos)
    w_pos = neg / max(1, pos) if pos > 0 else 1.0

    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(w[j] * xi[j] for j in range(d))
            p = _sigmoid(z)
            err = (p - yi) * (w_pos if yi == 1 else 1.0)
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        n = float(len(X))
        for j in range(d):
            w[j] -= lr * (gw[j] / n)
        b -= lr * (gb / n)

    return w, b


def main() -> None:
    ap = argparse.ArgumentParser(description="Train simple logistic risk model from run_history.jsonl")
    ap.add_argument("--run-history", required=True)
    ap.add_argument("--horizon-sec", type=float, default=10.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "models" / "risk_logreg.json"))
    args = ap.parse_args()

    rh = Path(args.run_history)
    ticks = _read_l1_ticks(rh)
    X, y, mean, std = _build_dataset(ticks, horizon_sec=float(args.horizon_sec))
    w, b = _train_logreg(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "type": "logreg",
        "version": f"logreg_h{args.horizon_sec}s",
        "features": FEATURES,
        "weights": w,
        "bias": b,
        "mean": mean,
        "std": std,
        "meta": {
            "samples": len(X),
            "positives": int(sum(y)),
            "horizon_sec": float(args.horizon_sec),
            "source": str(rh),
        },
    }
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "event": "model_trained",
        "out": str(out),
        "samples": len(X),
        "positives": int(sum(y)),
        "horizon_sec": float(args.horizon_sec),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
