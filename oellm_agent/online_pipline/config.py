from __future__ import annotations

import os

DT = 0.5
DEFAULT_GROUP_PERIOD_SEC = 1.0
DEFAULT_WINDOW_GROUPS = 8
DEFAULT_DECISION_STEP_SEC = 2.0


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


SHORT_WINDOW_SIZE = max(1, env_int("OELLM_WINDOW_GROUPS", DEFAULT_WINDOW_GROUPS))
SHORT_WINDOW_SEC = env_float("OELLM_WINDOW_SEC", SHORT_WINDOW_SIZE * DT)
SHORT_DECISION_EVERY = max(1, env_int("OELLM_DECISION_EVERY_GROUPS", int(round(DEFAULT_DECISION_STEP_SEC / DT))))
SLOW_DIAG_EVERY_SEC = max(1, env_int("OELLM_SLOW_DIAG_EVERY_SEC", 5))
KEEP_RAW_SAMPLES = os.getenv("OELLM_KEEP_RAW_SAMPLES", "1").strip() not in {"0", "false", "False"}
