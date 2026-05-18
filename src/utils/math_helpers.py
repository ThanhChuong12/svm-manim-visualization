"""Reusable mathematical functions for SVM / biometric score visualisations."""

import numpy as np


def gaussian(x: float, mu: float, sigma: float) -> float:
    """Normalised Gaussian probability density function (PDF)."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))


def min_func(f, g):
    """Return a callable that evaluates min(f(x), g(x)) point-wise.

    Useful for computing the visual overlap area between two distributions.
    """
    return lambda x: min(f(x), g(x))
