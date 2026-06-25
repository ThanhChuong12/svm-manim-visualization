"""Reusable mathematical functions for SVM / biometric score visualisations."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.score_norm import gaussian, min_func
except ImportError:
    import numpy as np

    def gaussian(x: float, mu: float, sigma: float) -> float:
        return float(
            np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            / (sigma * np.sqrt(2 * np.pi))
        )

    def min_func(f, g):
        return lambda x: min(f(x), g(x))
