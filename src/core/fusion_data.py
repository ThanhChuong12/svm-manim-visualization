"""Shared data generation functions for biometric score visualisations.

Centralises all synthetic dataset creation so that scene files stay focused
on animation logic rather than data wrangling.
"""

import numpy as np
from sklearn.datasets import make_circles


def generate_cluster(
    center,
    n: int = 14,
    sigma: float = 0.15,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate a 2D Gaussian cluster embedded in 3D space (z=0).

    Used by part0 (XOR trap) for generating blue/red point clusters.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=np.asarray(center)[:2], scale=sigma, size=(n, 2))
    return [np.array([x, y, 0.0]) for x, y in pts]


def scatter_2d(
    center: tuple[float, float],
    n: int,
    sigma: float,
    seed: int,
    x_clip: tuple[float, float] = (0.05, 0.95),
    y_clip: tuple[float, float] = (0.05, 0.95),
) -> list[tuple[float, float]]:
    """Return *n* (x, y) score-space points clustered around *center*.

    Used by part2 (score fusion) for genuine/impostor 2D scatter clouds.
    Points are clipped to the given axis range.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center, scale=sigma, size=(n, 2))
    pts[:, 0] = np.clip(pts[:, 0], *x_clip)
    pts[:, 1] = np.clip(pts[:, 1], *y_clip)
    return [(float(x), float(y)) for x, y in pts]


def generate_circle_data(
    n_samples: int = 80,
    noise: float = 0.08,
    factor: float = 0.4,
    scale: float = 1.6,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate concentric circle data via sklearn.datasets.make_circles.

    Returns (inner_pts, outer_pts) as lists of [x, y] numpy arrays.
    Inner = genuine (class 1), Outer = impostor (class 0).

    Used by part3 (kernel trick) for the non-linearly separable dataset.
    """
    X, y = make_circles(
        n_samples=n_samples, noise=noise,
        factor=factor, random_state=seed,
    )
    X = X * scale

    inner_pts = [X[i] for i in range(len(y)) if y[i] == 1]
    outer_pts = [X[i] for i in range(len(y)) if y[i] == 0]
    return inner_pts, outer_pts
