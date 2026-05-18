from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

from .config import KEEP_RAW_SAMPLES, SHORT_WINDOW_SEC
from .semantic_rules import DEFAULT_WINDOW_SEMANTIC_CONFIG, WindowSemanticConfig, is_abnormal_meaning, rank_candidates


@dataclass
class SlidingWindowAggregator:
    window_sec: float = SHORT_WINDOW_SEC
    step_sec: float = 1.0
    keep_raw_samples: bool = KEEP_RAW_SAMPLES
    semantic_config: WindowSemanticConfig = DEFAULT_WINDOW_SEMANTIC_CONFIG
    _samples: Deque[Dict[str, Any]] = field(default_factory=deque)

    def add_sample(self, timestamp_sec: float, state: Dict[str, Any]) -> None:
        self._samples.append({"timestamp_sec": float(timestamp_sec), "state": dict(state)})
        self._trim(timestamp_sec)

    def _trim(self, current_ts: float) -> None:
        cutoff = current_ts - self.window_sec
        while self._samples and float(self._samples[0]["timestamp_sec"]) < cutoff:
            self._samples.popleft()

    def _compress_signal_samples(self, signal_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not signal_samples:
            return {"value": None, "unit": "", "meaning": ""}
        latest = signal_samples[-1]
        meaningful = [s for s in signal_samples if is_abnormal_meaning(str(s.get("meaning", "")), self.semantic_config)]
        if meaningful:
            chosen = rank_candidates(meaningful, self.semantic_config)[-1]
        else:
            chosen = latest
        return {
            "value": chosen.get("value"),
            "unit": chosen.get("unit", ""),
            "meaning": chosen.get("meaning", ""),
        }

    def to_window_json(self) -> Dict[str, Any]:
        samples = list(self._samples)
        if not samples:
            return {
                "window_sec": self.window_sec,
                "step_sec": self.step_sec,
                "sample_count": 0,
                "window_start_ts": None,
                "window_end_ts": None,
                "signals": {},
                "samples": [],
            }

        window_start = float(samples[0]["timestamp_sec"])
        window_end = float(samples[-1]["timestamp_sec"])
        all_keys = set()
        for sample in samples:
            state = sample["state"]
            for key, value in state.items():
                if isinstance(value, dict) and "meaning" in value:
                    all_keys.add(key)

        signals: Dict[str, Any] = {}
        for key in sorted(all_keys):
            signal_samples: List[Dict[str, Any]] = []
            for sample in samples:
                value = sample["state"].get(key)
                if isinstance(value, dict):
                    signal_samples.append({"timestamp_sec": sample["timestamp_sec"], **value})
            if signal_samples:
                signals[key] = self._compress_signal_samples(signal_samples)

        raw_samples = []
        if self.keep_raw_samples:
            raw_samples = [{"timestamp_sec": s["timestamp_sec"], "state": s["state"]} for s in samples]

        return {
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "sample_count": len(samples),
            "window_start_ts": window_start,
            "window_end_ts": window_end,
            "signals": signals,
            "samples": raw_samples,
        }
