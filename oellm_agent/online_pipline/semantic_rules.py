from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

DEFAULT_MEANING_RANKS: Dict[str, int] = {
    "正常": 1,
    "过高": 3,
    "较高": 3,
    "较低": 3,
    "偏高": 3,
    "偏低": 3,
    "水位低": 3,
    "温度过高": 3,
    "表面温度过高": 3,
    "甲烷浓度较高": 3,
    "甲烷浓度过高": 3,
}

DEFAULT_ABNORMAL_PREFIXES: List[str] = ["异常"]
DEFAULT_NORMAL_MEANINGS = {"正常"}


@dataclass(frozen=True)
class WindowSemanticConfig:
    meaning_ranks: Dict[str, int]
    abnormal_prefixes: List[str]
    normal_meanings: set[str]


DEFAULT_WINDOW_SEMANTIC_CONFIG = WindowSemanticConfig(
    meaning_ranks=dict(DEFAULT_MEANING_RANKS),
    abnormal_prefixes=list(DEFAULT_ABNORMAL_PREFIXES),
    normal_meanings=set(DEFAULT_NORMAL_MEANINGS),
)


def meaning_rank(meaning: str, config: WindowSemanticConfig = DEFAULT_WINDOW_SEMANTIC_CONFIG) -> int:
    text = str(meaning or "").strip()
    if not text:
        return 0
    if any(text.startswith(prefix) for prefix in config.abnormal_prefixes):
        return 4
    if text in config.meaning_ranks:
        return config.meaning_ranks[text]
    return 2


def is_normal_meaning(meaning: str, config: WindowSemanticConfig = DEFAULT_WINDOW_SEMANTIC_CONFIG) -> bool:
    return str(meaning or "").strip() in config.normal_meanings


def is_abnormal_meaning(meaning: str, config: WindowSemanticConfig = DEFAULT_WINDOW_SEMANTIC_CONFIG) -> bool:
    text = str(meaning or "").strip()
    return bool(text) and not is_normal_meaning(text, config)


def rank_candidates(items: Iterable[dict], config: WindowSemanticConfig = DEFAULT_WINDOW_SEMANTIC_CONFIG) -> list[dict]:
    return sorted(
        items,
        key=lambda x: (
            meaning_rank(str(x.get("meaning", "")), config),
            float(x.get("timestamp_sec", 0.0)),
        ),
    )
