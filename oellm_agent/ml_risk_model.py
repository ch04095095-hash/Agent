from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "risk_logreg.json"


class MLRiskModel:
    """Lightweight logistic-regression risk scorer.

    Model file schema:
    {
      "type": "logreg",
      "features": ["speed_kmh", ...],
      "weights": [0.1, ...],
      "bias": -1.2,
      "mean": {"speed_kmh": 6.8, ...},
      "std": {"speed_kmh": 1.3, ...}
    }
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.loaded = False
        self.features: List[str] = []
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.version: str = "none"

        self._try_load()

    def _try_load(self) -> None:
        if not self.model_path.exists():
            self.loaded = False
            return
        obj = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.features = [str(x) for x in obj.get("features", [])]
        self.weights = [float(x) for x in obj.get("weights", [])]
        self.bias = float(obj.get("bias", 0.0) or 0.0)
        self.mean = {str(k): float(v) for k, v in (obj.get("mean", {}) or {}).items()}
        self.std = {str(k): max(1e-6, float(v)) for k, v in (obj.get("std", {}) or {}).items()}
        self.version = str(obj.get("version", self.model_path.name))
        self.loaded = bool(self.features and len(self.features) == len(self.weights))

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def score(self, current: Dict[str, Any]) -> float:
        if not self.loaded:
            return 0.0
        z = self.bias
        for f, w in zip(self.features, self.weights):
            v = float(current.get(f, 0.0) or 0.0)
            mu = self.mean.get(f, 0.0)
            sigma = self.std.get(f, 1.0)
            x = (v - mu) / sigma
            z += w * x
        return float(self._sigmoid(z))

    @staticmethod
    def level_from_score(score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "medium"
        if score >= 0.3:
            return "low"
        return "normal"
