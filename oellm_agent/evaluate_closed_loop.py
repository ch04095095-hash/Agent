#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


TARGET_ANOMALY_REASONS = {
    "coolant_over_stop",
    "speed_over_stop",
    "intake_pressure_low",
    "water_tank_low",
}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_by_step(rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    by_step: Dict[int, Dict[str, str]] = {}
    for r in rows:
        step = int(r["step"])
        cur = by_step.get(step)
        if cur is None:
            by_step[step] = dict(r)
        else:
            # 用0x18F181A0优先补齐risk/warning/action（所有帧应一致）
            if r.get("frame_id_hex", "") == "0x18F181A0":
                by_step[step] = dict(r)
    return by_step


def compute_warning_to_normal_recovery(step_rows: Dict[int, Dict[str, str]], dt: float = 0.1) -> Tuple[int, float]:
    steps = sorted(step_rows)
    recoveries: List[float] = []
    i = 0
    while i < len(steps):
        s = steps[i]
        lvl = step_rows[s].get("risk_level", "")
        if lvl != "warning":
            i += 1
            continue

        start = s
        j = i
        while j < len(steps) and step_rows[steps[j]].get("risk_level", "") == "warning":
            j += 1

        # 从warning段结束后找第一个normal
        k = j
        while k < len(steps):
            if step_rows[steps[k]].get("risk_level", "") == "normal":
                recoveries.append((steps[k] - start) * dt)
                break
            if step_rows[steps[k]].get("risk_level", "") == "danger":
                break
            k += 1

        i = j

    return len(recoveries), (mean(recoveries) if recoveries else 0.0)


def read_l1_ticks_from_run_history(path: Path) -> List[Dict]:
    ticks: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("event") == "l1_tick":
                ticks.append(obj)
    return ticks


def summarize_recovery_episodes(l1_ticks: List[Dict], dt: float = 0.1) -> Dict:
    episodes: List[Dict] = []
    i = 0

    def reasons_of(tick: Dict) -> List[str]:
        reasons = tick.get("reasons", [])
        if not isinstance(reasons, list):
            return []
        return [str(r) for r in reasons]

    while i < len(l1_ticks):
        rs = set(reasons_of(l1_ticks[i]))
        targets = sorted(rs.intersection(TARGET_ANOMALY_REASONS))
        if not targets:
            i += 1
            continue

        start = i
        active_reason = targets[0]
        j = i
        while j < len(l1_ticks):
            cur = set(reasons_of(l1_ticks[j]))
            if active_reason not in cur:
                break
            j += 1

        end = j - 1
        start_t = float(l1_ticks[start].get("t_sec", start * dt))
        end_t = float(l1_ticks[end].get("t_sec", end * dt))

        start_snapshot = l1_ticks[start].get("sensor_snapshot", {}) or {}
        end_snapshot = l1_ticks[end].get("sensor_snapshot", {}) or {}

        recovered = j < len(l1_ticks)
        next_t = float(l1_ticks[j].get("t_sec", j * dt)) if recovered else None
        next_reasons = reasons_of(l1_ticks[j]) if recovered else []

        episodes.append(
            {
                "reason": active_reason,
                "start_t_sec": round(start_t, 3),
                "end_t_sec": round(end_t, 3),
                "duration_sec": round(end_t - start_t, 3),
                "recovered": recovered,
                "next_t_sec": round(next_t, 3) if next_t is not None else None,
                "next_reasons": next_reasons,
                "start_coolant_temp_c": start_snapshot.get("coolant_temp_c"),
                "end_coolant_temp_c": end_snapshot.get("coolant_temp_c"),
            }
        )

        i = j

    by_reason = Counter(ep["reason"] for ep in episodes)
    recovered_count = sum(1 for ep in episodes if ep["recovered"])

    return {
        "episode_count": len(episodes),
        "recovered_count": recovered_count,
        "recovery_ratio": round(recovered_count / len(episodes), 4) if episodes else 0.0,
        "episodes_by_reason": dict(by_reason),
        "episodes": episodes,
    }


def build_csv_summary(path: Path, dt: float) -> Dict:
    rows = read_rows(path)
    if not rows:
        return {"error": "empty_csv", "path": str(path)}

    step_rows = aggregate_by_step(rows)
    steps = sorted(step_rows)

    action_counter = Counter()
    risk_counter = Counter()
    warning_counter = Counter()
    alarm_steps = 0

    for s in steps:
        r = step_rows[s]
        action_counter[str(r.get("control_action", ""))] += 1
        risk_counter[str(r.get("risk_level", ""))] += 1
        warning_counter[str(r.get("warning_tag", ""))] += 1

        if r.get("frame_id_hex", "") == "0x18F181A0":
            try:
                alarm_code = int(r.get("byte0", "0") or 0)
            except Exception:
                alarm_code = 0
            if alarm_code != 0:
                alarm_steps += 1

    total_steps = len(steps)
    warning_to_normal_count, warning_to_normal_avg_sec = compute_warning_to_normal_recovery(step_rows, dt)

    # 简单抖动统计：相邻step动作切换次数
    switches = 0
    prev_action = None
    for s in steps:
        a = step_rows[s].get("control_action", "")
        if prev_action is not None and a != prev_action:
            switches += 1
        prev_action = a

    return {
        "path": str(path),
        "rows": len(rows),
        "steps": total_steps,
        "action_distribution": dict(action_counter),
        "risk_distribution": dict(risk_counter),
        "top_warning_tags": dict(warning_counter.most_common(10)),
        "alarm_steps": alarm_steps,
        "alarm_ratio": round(alarm_steps / total_steps, 4) if total_steps else 0.0,
        "warning_to_normal_recovery_count": warning_to_normal_count,
        "warning_to_normal_avg_sec": round(warning_to_normal_avg_sec, 3),
        "action_switch_count": switches,
        "action_switch_ratio": round(switches / max(1, total_steps - 1), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate closed-loop simulation artifacts")
    parser.add_argument(
        "--csv",
        default="/home/sunrise/oellm_agent/sim_data/closed_loop_can_frames.csv",
        help="Path to closed loop csv",
    )
    parser.add_argument(
        "--run-history",
        default="",
        help="Path to run_history.jsonl for recovery episode summary",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    output: Dict[str, object] = {}

    csv_path = Path(args.csv)
    if csv_path.exists():
        output["csv_summary"] = build_csv_summary(csv_path, args.dt)
    else:
        output["csv_summary"] = {"error": "csv_not_found", "path": str(csv_path)}

    run_history_path: Optional[Path] = Path(args.run_history) if args.run_history else None
    if run_history_path is not None:
        if run_history_path.exists():
            l1_ticks = read_l1_ticks_from_run_history(run_history_path)
            output["run_history_recovery_summary"] = {
                "path": str(run_history_path),
                "l1_tick_count": len(l1_ticks),
                **summarize_recovery_episodes(l1_ticks, args.dt),
            }
        else:
            output["run_history_recovery_summary"] = {
                "error": "run_history_not_found",
                "path": str(run_history_path),
            }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
