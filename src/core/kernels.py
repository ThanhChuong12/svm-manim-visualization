"""Kernel functions for SVM visualisations.

Provides mathematical kernel computations used across scenes,
especially the RBF (Radial Basis Function) kernel mapping for part3.
"""

import numpy as np


def rbf_z(x: float, y: float, gamma: float = 2.0) -> float:
    """RBF kernel mapping: z = exp(-γ(x² + y²)).

    Maps a 2D point to a scalar z-height representing the kernel
    feature space. Inner points (small r) get high z values;
    outer points (large r) get z ≈ 0.
    """
    return float(np.exp(-gamma * (x ** 2 + y ** 2)))


def rbf_kernel(x: np.ndarray, l: np.ndarray, gamma: float = 2.0) -> float:
    """Full RBF kernel: K(x, l) = exp(-γ ||x - l||²).

    General-purpose kernel computation between two arbitrary vectors.
    """
    diff = np.asarray(x) - np.asarray(l)
    return float(np.exp(-gamma * np.dot(diff, diff)))


def polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
    coef0: float = 1.0,
) -> float:
    """Polynomial kernel: K(x, y) = (x · y + coef0)^degree."""
    return float((np.dot(x, y) + coef0) ** degree)


def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """Linear kernel: K(x, y) = x · y."""
    return float(np.dot(x, y))
