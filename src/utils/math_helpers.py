"""Reusable mathematical functions for SVM / biometric score visualisations.

NOTE: Core math logic has been moved to `core.score_norm`.
This module re-exports for backward compatibility with part1.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.score_norm import gaussian, min_func
except ImportError:
    # Fallback if core is not on path
    import numpy as np

    def gaussian(x: float, mu: float, sigma: float) -> float:
        return float(
            np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
            / (sigma * np.sqrt(2 * np.pi))
        )

    def min_func(f, g):
        return lambda x: min(f(x), g(x))
