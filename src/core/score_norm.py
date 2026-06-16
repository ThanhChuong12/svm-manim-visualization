"""Score normalisation and distribution utilities.

Provides mathematical functions for modelling biometric score
distributions (Gaussian PDF) and computing distribution overlaps.
Used primarily by part1 (unibiometrics threshold scene).
"""

import numpy as np


def gaussian(x: float, mu: float, sigma: float) -> float:
    """Normalised Gaussian probability density function (PDF).

    Models the score distribution of genuine or impostor populations.
    """
    return float(
        np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        / (sigma * np.sqrt(2 * np.pi))
    )


def min_func(f, g):
    """Return a callable that evaluates min(f(x), g(x)) point-wise.

    Useful for computing the visual overlap area between
    genuine and impostor score distributions.
    """
    return lambda x: min(f(x), g(x))


def min_max_normalize(
    values: np.ndarray,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> np.ndarray:
    """Min-max normalisation of an array to [new_min, new_max].

    Common score normalisation technique before fusion.
    """
    v_min, v_max = values.min(), values.max()
    if v_max - v_min < 1e-12:
        return np.full_like(values, (new_min + new_max) / 2)
    return (values - v_min) / (v_max - v_min) * (new_max - new_min) + new_min


def z_score_normalize(values: np.ndarray) -> np.ndarray:
    """Z-score normalisation (mean=0, std=1).

    Another standard score normalisation technique.
    """
    mu = values.mean()
    sigma = values.std()
    if sigma < 1e-12:
        return np.zeros_like(values)
    return (values - mu) / sigma
