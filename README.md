# Multibiometric Fusion and SVM Visualization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Manim](https://img.shields.io/badge/Manim-0.18.x-2980b9?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.8%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

> **Báo cáo Đồ án môn học — Nhận dạng (Pattern Recognition - CSC14006)**
>
> *Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM*

---

## Table of Contents

- [1. About The Project](#1-about-the-project)
- [2. Theoretical Coverage](#2-theoretical-coverage)
  - [Part 1: Storyboard & Planning](#part-1-storyboard--planning)
  - [Part 2: Visual & Mathematical Concepts](#part-2-visual--mathematical-concepts)
- [3. Visualization Scenes & Modules](#3-visualization-scenes--modules)
- [4. Repository Structure](#4-repository-structure)
- [5. Getting Started](#5-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Rendering the Scenes](#rendering-the-scenes)
- [6. Contributors](#6-contributors)
- [7. License & Acknowledgments](#7-license--acknowledgments)

---

## 1. About The Project

This project is the **Course Project** for the *Pattern Recognition (Nhận dạng)* course at VNU-HCM University of Science. It provides a cinematic, highly-visual explanation of **Multibiometrics and Support Vector Machine (SVM) Score-Level Fusion** using the **Manim** animation engine.

The core narrative is inspired by **Chapter 14: "Introduction to Multibiometrics"** from the book *Handbook of Biometrics* (edited by Anil K. Jain, Patrick Flynn, and Arun A. Ross), written by Arun Ross, Karthik Nandakumar, and Anil K. Jain.

The project is organized into three interconnected components:

| **Part 1: Biometric & Normalization Basics** | **Part 2: SVM & Kernel Trick Fusion** | **Part 3: Limitations & Conclusion** |
| :--- | :--- | :--- |
| Unimodal biometric limitations, score matching pipeline, and score normalization techniques (Min-Max, Z-Score). | Multi-modal score vectorization, linear SVM learning (support vectors, margins), and the RBF Kernel trick to bypass the XOR trap. | Real-world trade-offs of deploying multibiometric systems, focusing on costs, processing latency, and security/privacy risks. |
| Animates the FAR/FRR trade-off using overlapping Gaussian distributions, and visualizes the score normalization workflows. | Visualizes SVM margin maximization, 3D kernel lifting of non-linear XOR data, 3D hyperplane (laser shield) slicing, and circular projection. | Animates hardware sensor costs with a balance scale, pipeline latency with stopwatches, and identity theft with security leak icons. |
| Explains why unimodal systems fail under noise/spoofing and why score-level fusion is the most popular modality-combining method. | Demonstrates how the kernel trick mathematically lifts data into higher dimensions to resolve non-separability. | Concludes by summarizing the security benefits of multibiometrics versus its operational costs. |

> **Core Philosophy:** All animations are dynamically generated from math equations and real algorithms. The SVM boundaries are computed live using `scikit-learn` in the background before rendering, ensuring mathematical correctness and visual accuracy.

---

## 2. Theoretical Coverage

### Part 1: Storyboard & Planning

The visualization project consists of 6 core scenes, taking the viewer on a pedagogical journey from unimodal basics to complex non-linear fusion and real-world system limitations:

| Scene File | Chapter / Phase | Content |
| :--- | :--- | :--- |
| [part0_intro.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part0_intro.py) | **Part 0 — Introduction** | Biometric grid scan (Face, Fingerprint, Iris, Voice), the XOR Trap (failure of a straight line in 2D), and a teaser of the 3D Kernel Lift. |
| [part1_unibiometric.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part1_unibiometric.py) | **Part 1 — Unimodal Biometrics** | Ideal vs. real unimodal score distributions (overlapping Gaussians), FAR & FRR trade-offs under threshold adjustments, and rotation-based transition to 2D. |
| [part1_5_pipeline.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part1_5_pipeline.py) | **Part 1.5 — Score Normalization Pipeline** | Comparison table of fusion levels, matcher-to-score pipeline, Min-Max normalization, and Z-score normalization workflows. |
| [part2_score_fusion.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part2_score_fusion.py) | **Part 2 — Score Fusion & Linear SVM** | Score vectorization, "Math Class Detour" (iterative SVM learning, margins, support vectors, hard vs. soft margin), and the XOR Trap (how spoofing attacks break linear classifiers). |
| [part3_kernel_lift.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part3_kernel_lift.py) | **Part 3 — The Kernel Trick & RBF SVM** | Non-linear concentric circle distributions, 3D Kernel Lift using RBF kernel, 3D hyperplane (Laser Shield) separation, 2D circular boundary projection, and Gamma parameter analysis. |
| [part4_conclusion.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part4_conclusion.py) | **Part 4 — System Limitations & Outro** | Balance-scale hardware cost animation, stopwatch-based latency comparison, privacy risks (identity theft), and the final callback to Part 0. |

---

### Part 2: Visual & Mathematical Concepts

The mathematical and visual foundations represented in the scenes cover several key pattern recognition concepts:

- **Biometric Score Normalization** — To combine scores from different modalities (e.g. Fingerprint $[0, 100]$ and Face $[0, 1]$), they must be mapped to a common scale:
  - **Min-Max Normalization**:
    $$s' = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$
  - **Z-Score Normalization**:
    $$s' = \frac{s - \mu}{\sigma}$$
- **Linear Support Vector Classifier** — Finds the optimal separating hyperplane that maximizes the margin:
  - Decision boundary is defined by $\mathbf{w}^T \mathbf{x} + b = 0$.
  - Optimization problem:
    $$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Radial Basis Function (RBF) Kernel & 3D Lift** — Lifts non-linearly separable data into high-dimensional space:
  - The kernel function: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$.
  - In `part3_kernel_lift.py`, the 3D lift is mapped relative to the origin as:
    $$z = \exp(-\gamma (x^2 + y^2))$$
  - At the decision threshold $z_{thresh} = 0.45$, the projected boundary forms a circle in 2D with radius:
    $$r = \sqrt{\frac{-\ln(z_{thresh})}{\gamma}}$$
- **Symmetric Trade-offs (FAR vs. FRR)** — Shows how threshold adjustments shift the system's operational security:
  - **False Accept Rate (FAR)**: $\text{FAR} = \int_{\eta}^{\infty} p(s | \text{Impostor}) ds$
  - **False Reject Rate (FRR)**: $\text{FRR} = \int_{-\infty}^{\eta} p(s | \text{Genuine}) ds$

---

## 3. Visualization Scenes & Modules

This repository contains modularized components separating mathematical algorithms, visual layouts, and rendering scenes:

| File / Component | Purpose | Key Classes & Functions |
| :--- | :--- | :--- |
| **[part0_intro.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part0_intro.py)** | Introductory Hook & Teaser | [IntroScene](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part0_intro.py#L120) |
| **[part1_unibiometric.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part1_unibiometric.py)** | Unimodal Thresholding & FAR/FRR | [UnibiometricsScene](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part1_unibiometric.py#L131) |
| **[part1_5_pipeline.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part1_5_pipeline.py)** | Biometric Pipeline & Normalization | [PipelineScene](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part1_5_pipeline.py#L51) |
| **[part2_score_fusion.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part2_score_fusion.py)** | 2D Vectorization, Linear SVM Detour | [ScoreCombinationScene](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part2_score_fusion.py#L81) |
| **[part3_kernel_lift.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part3_kernel_lift.py)** | RBF Kernel, 3D Lift & Gamma Slider | [KernelTrickScene](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part3_kernel_lift.py#L83) |
| **[part4_conclusion.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part4_conclusion.py)** | Hardware, Latency, Privacy, Callback | [SystemLimitationsScene](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/scenes/part4_conclusion.py#L61) |
| **[fusion_data.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/fusion_data.py)** | Core synthetic data generators | [generate_cluster](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/fusion_data.py#L12), [scatter_2d](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/fusion_data.py#L22), [generate_circle_data](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/fusion_data.py#L42) |
| **[kernels.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/kernels.py)** | Mathematical RBF logic | [rbf_z](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/kernels.py#L7) |
| **[score_norm.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/score_norm.py)** | Min-Max and Z-Score calculators | [min_max_scale](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/score_norm.py#L7), [z_score_scale](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/core/score_norm.py#L20) |
| **[math_helpers.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/math_helpers.py)** | Standard Gaussian & Min Functions | [gaussian](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/math_helpers.py#L8), [min_func](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/math_helpers.py#L14) |
| **[visual_helpers.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/visual_helpers.py)** | Reusable vector and biometric icons | [make_genuine_icon](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/visual_helpers.py#L20), [make_impostor_icon](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/visual_helpers.py#L38), [make_fingerprint_icon](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/utils/visual_helpers.py#L86) |
| **[constants.py](file:///d:/3rdY_HCMUS/manimations/svm-manim-visualization/src/constants.py)** | Design tokens, palette, and typography | Global constants (colors, main font) |

---

## 4. Repository Structure

```text
svm-manim-visualization/
│
├── assets/                           # Font files and static assets
│   ├── audio/                        # Voiceover and soundtrack tracks (empty)
│   └── images/                       # Logo or overlay images (empty)
│
├── docs/                             # Project planning and storyboards
│   ├── plan.md                       # Storyboard structure and script plans
│   ├── references.md                 # Academic references
│   └── script.md                     # Scene voiceover scripts
│
├── media/                            # Output folder for Manim renders
│   ├── images/                       # Rendered static frames
│   └── videos/                       # Rendered MP4 videos (480p, 720p, 1080p)
│
├── src/                              # Core Python source code
│   ├── core/                         # Mathematical and data generators
│   │   ├── __init__.py
│   │   ├── fusion_data.py            # Generates 2D cluster and concentric circle data
│   │   ├── kernels.py                # Mathematical functions for RBF kernel
│   │   └── score_norm.py             # Score normalization algorithms (Min-Max, Z-score)
│   │
│   ├── scenes/                       # Manim visualization scenes (scripts to render)
│   │   ├── __init__.py
│   │   ├── part0_intro.py            # Title drop, XOR trap, 3D lift teaser
│   │   ├── part1_unibiometric.py     # Unimodal distributions, threshold, FAR/FRR
│   │   ├── part1_5_pipeline.py       # Normalization, comparison tables, pipelines
│   │   ├── part2_score_fusion.py     # Score vectorization, linear SVM detour & margin
│   │   ├── part3_kernel_lift.py      # Nonlinear concentric circles, 3D RBF dome, Gamma slider
│   │   └── part4_conclusion.py       # Deployment costs, latency, privacy risks, outro
│   │
│   ├── utils/                        # Visualization and mathematical helper functions
│   │   ├── __init__.py
│   │   ├── math_helpers.py           # Scaled Gaussians and function minimums
│   │   └── visual_helpers.py         # Custom vectors, icons (Genuine, Impostor, Spoof)
│   │
│   ├── constants.py                  # Colors, fonts, and scene aesthetic tokens
│   └── README_TK.md                  # Project README template from Machine Learning course
│
├── .gitignore
├── LICENSE                           # MIT License
├── README.md                         # This file
└── requirements.txt                  # Python dependencies (manim, numpy, scikit-learn, etc.)
```

---

## 5. Getting Started

### Prerequisites

| Tool / Library | Version | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.10+ | Execution environment for scripts |
| **Manim** | 0.18.x | Animation engine (requires System dependencies like FFmpeg and LaTeX/MiKTeX) |
| **NumPy** | 1.22+ | Mathematical calculations |
| **scikit-learn** | 1.0+ | Computing SVM boundaries and margins |
| **SciPy** | 1.8+ | Scientific distributions (Gaussians) |

### Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/ThanhChuong12/svm-manim-visualization.git
cd svm-manim-visualization
```

**Step 2: Set up a virtual environment and install dependencies**
```bash
python -m venv .venv
# On Windows (PowerShell)
.venv\Scripts\activate
# On Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> [!NOTE]
> Since Manim depends on external system packages (FFmpeg, dvisvgm, LaTeX), please ensure they are installed on your path. Follow the [Manim installation guide](https://docs.manim.community/en/stable/installation.html) for detailed setup instructions.

### Rendering the Scenes

To render any of the scenes, make sure your virtual environment is active. You can control the render quality using the flags `-pql` (low, 480p15), `-pqm` (medium, 720p30), or `-pqh` (high, 1080p60).

```bash
# Render Part 0: Introduction
manim -pql src/scenes/part0_intro.py IntroScene

# Render Part 1: Unimodal Biometrics
manim -pql src/scenes/part1_unibiometric.py UnibiometricsScene

# Render Part 1.5: Normalization Pipeline
manim -pql src/scenes/part1_5_pipeline.py PipelineScene

# Render Part 2: Score Fusion & Linear SVM
manim -pql src/scenes/part2_score_fusion.py ScoreCombinationScene

# Render Part 3: The Kernel Trick & RBF SVM
manim -pql src/scenes/part3_kernel_lift.py KernelTrickScene

# Render Part 4: System Limitations & Conclusion
manim -pql src/scenes/part4_conclusion.py SystemLimitationsScene
```

The rendered video files will be saved in the `media/videos/` directory under their respective subfolders.

---

## 6. Contributors

This project was developed by a single student from the *Faculty of Information Technology, VNU-HCM University of Science*.

| Contributor | Student ID | Role | Main Contributions |
| :--- | :---: | :--- | :--- |
| **Lê Hà Thanh Chương** | `23120195` | **Solo Developer & Visual Designer** | Designed the entire project storyboards, narration script, and visual roadmap. Implemented the mathematical core (`fusion_data.py`, `kernels.py`, `score_norm.py`) and programmed all six Manim scenes (`part0_intro.py` to `part4_conclusion.py`). |

---

## 7. License & Acknowledgments

### Academic Acknowledgments

This project is the **Course Project** for the *Pattern Recognition (Nhận dạng)* course (CSC14006) at VNU-HCM University of Science.

We sincerely thank:
- **Instructor:** Nguyễn Thanh Tình (nttinh@fit.hcmus.edu.vn) for guidance and feedback during the course.

### Paper & Book Attribution

- **Primary Textbook Reference:** *Handbook of Biometrics* — edited by Anil K. Jain, Patrick Flynn, and Arun A. Ross. Springer, 2008.
  - Specifically, **Chapter 14: "Introduction to Multibiometrics"** by Arun Ross, Karthik Nandakumar, and Anil K. Jain.
- **Secondary Textbook Reference:** *Handbook of Face Recognition* (2nd Edition) — Stan Z. Li and Anil K. Jain (Eds.). Springer, 2011.
- **Support Vector Machines (SVM) Foundations:** *Support Vector Networks* — Corinna Cortes and Vladimir Vapnik. Machine Learning, 1995.
- **Visualization Engine:** [Manim Community](https://github.com/ManimCommunity/manim) (Mathematics Animation Engine).

### License

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

The source code of this project is distributed under the **MIT License**.
See the `LICENSE` file for full details.

<br>
<p align="center">
  <i>Built by Lê Hà Thanh Chương | University of Science, VNU-HCM | 2026</i>
</p>
