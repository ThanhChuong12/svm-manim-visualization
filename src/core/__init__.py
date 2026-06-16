"""Core package for shared SVM / biometric visualisation logic.

Modules:
    fusion_data  — Synthetic dataset generation (clusters, circles, scatter)
    kernels      — Kernel functions (RBF, polynomial, linear)
    score_norm   — Score normalisation and distribution utilities
"""

from core.fusion_data import generate_cluster, scatter_2d, generate_circle_data
from core.kernels import rbf_z, rbf_kernel, polynomial_kernel, linear_kernel
from core.score_norm import gaussian, min_func, min_max_normalize, z_score_normalize
