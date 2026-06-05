"""Scene 3 — Score Combination & Linear SVM.

Phase 1 (0:00-0:15): Two 1D scores merge into a 2D vector dot on clean Axes.
Phase 2 (0:15-0:30): 1D overlap crowd explodes into linearly-separable 2D clouds.
Phase 3 (0:30-0:45): Linear SVM hyperplane + animated margin expansion reveals SVs.
Phase 4 (0:45-1:00): XOR spoof attack appears; the linear boundary fails visually.

Mathematical logic for the SVM hyperplane and margin distance is adapted from
svm_animation.py (sklearn SVC linear kernel), keeping full academic accuracy.
"""


import numpy as np
from manim import *

# scikit-learn — only for the optimal SVM fit in Phase 3
from sklearn.svm import SVC

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import (
        GENUINE_COLOR, IMPOSTOR_COLOR, HYPERPLANE_COLOR,
        BG_COLOR, FONT_MAIN, SLATE_GRAY,
        CLASS_A_COLOR, CLASS_B_COLOR, MARGIN_OPACITY,
    )
except ImportError:
    GENUINE_COLOR    = "#2ECC71"
    IMPOSTOR_COLOR   = "#E74C3C"
    HYPERPLANE_COLOR = "#F9DC5C"
    BG_COLOR         = "#0B0C10"
    FONT_MAIN        = "Montserrat"
    SLATE_GRAY       = "#888888"
    CLASS_A_COLOR    = "#FF5E5E"
    CLASS_B_COLOR    = "#00C2D1"
    MARGIN_OPACITY   = 0.30

# ── Scene-local design tokens ─────────────────────────────────────────────────
FONT            = FONT_MAIN          # Primary font used throughout the scene
SUPPORT_RING    = "#FFFFFF"          # Ring around support-vector dots
MARGIN_COLOR    = HYPERPLANE_COLOR   # Dashed margin lines share hyperplane yellow
SPOOF_RED       = "#FF2222"          # Flash colour for the spoof-attack warning
AXIS_OPACITY    = 0.55
AXIS_RANGE      = [0.0, 1.0, 0.25]  # Score axes both run 0 → 1

# ── Reproducible random seeds for each cluster ───────────────────────────────
RNG_SEED_GEN_ORIG = 7     # Genuine cluster seeded for Phase 1 sample point
RNG_SEED_GEN_2D   = 11    # Genuine 2D cloud, Phase 2
RNG_SEED_IMP_2D   = 17    # Impostor 2D cloud, Phase 2
RNG_SEED_SPOOF_TL = 29    # Phase 4 spoof cluster — top-left
RNG_SEED_SPOOF_BR = 37    # Phase 4 spoof cluster — bottom-right

N_CLOUD = 18              # Points per cloud cluster


# ─────────────────────────────────────────────────────────────────────────────
# Helper — generate a Gaussian scatter in score space
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_2d(
    center: tuple[float, float],
    n: int,
    sigma: float,
    seed: int,
    x_clip: tuple[float, float] = (0.0, 1.0),
    y_clip: tuple[float, float] = (0.0, 1.0),
) -> list[tuple[float, float]]:
    """Return *n* (x, y) score-space points clustered around *center*."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center, scale=sigma, size=(n, 2))
    pts[:, 0] = np.clip(pts[:, 0], *x_clip)
    pts[:, 1] = np.clip(pts[:, 1], *y_clip)
    return [(float(x), float(y)) for x, y in pts]


# ─────────────────────────────────────────────────────────────────────────────
# Helper — compute the SVM hyperplane and margin from data
#   Adapted directly from svm_animation.py (sklearn SVC, linear kernel).
#   Returns: (slope, intercept_y, margin_half_width_in_axis_units)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_linear_svm(
    genuine_pts: list[tuple[float, float]],
    impostor_pts: list[tuple[float, float]],
) -> tuple[float, float, float, np.ndarray]:
    """Fit a hard-margin Linear SVM and return boundary parameters.

    Returns
    -------
    slope         : float  — dy/dx of the decision boundary in score-space.
    y_at_x0       : float  — y-intercept (x=0) of the decision boundary.
    margin_dy     : float  — half-margin shift in the y direction, at any x.
    support_vecs  : ndarray shape (k, 2) — support vector coordinates.
    """
    X = np.array(genuine_pts + impostor_pts)
    y = np.array([1] * len(genuine_pts) + [-1] * len(impostor_pts))

    # Replicate svm_animation.py: SVC with linear kernel, C=1
    svm = SVC(kernel="linear", C=1, random_state=42)
    svm.fit(X, y)

    coef      = svm.coef_[0]       # [w1, w2]
    intercept = svm.intercept_[0]  # b

    # Decision boundary: coef[0]*x + coef[1]*y + intercept = 0
    #   => y = (-coef[0]*x - intercept) / coef[1]
    slope    = -coef[0] / coef[1]
    y_at_x0  = -intercept / coef[1]

    # Margin half-width in y-direction (same formula as svm_animation.py):
    #   dist = 1 / (coef[1] * ||w||)
    norm      = np.linalg.norm(coef)
    margin_dy = 1.0 / (coef[1] * norm)

    return slope, y_at_x0, margin_dy, svm.support_vectors_


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build a Line that spans the entire visible axis range
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_line(
    axes: Axes,
    slope: float,
    y_at_x0: float,
    x_min: float,
    x_max: float,
    color: ManimColor,
    stroke_width: float = 3,
    stroke_opacity: float = 1.0,
) -> Line:
    y_min = slope * x_min + y_at_x0
    y_max = slope * x_max + y_at_x0
    return Line(
        axes.c2p(x_min, y_min),
        axes.c2p(x_max, y_max),
        color=color,
        stroke_width=stroke_width,
        stroke_opacity=stroke_opacity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCombinationScene(Scene):
    """Scene 3: Score Vectorisation → 2D Separation → Linear SVM → XOR Failure."""

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Build shared axes ──────────────────────────────────────────
        axes = Axes(
            x_range=AXIS_RANGE,
            y_range=AXIS_RANGE,
            x_length=6.5,
            y_length=5.5,
            axis_config={
                "stroke_width": 2,
                "color": WHITE,
                "stroke_opacity": AXIS_OPACITY,
                "include_ticks": True,
                "tick_size": 0.06,
            },
            tips=True,
        ).shift(DOWN * 0.25)

        x_label = Text(
            "Face Match Score  (s₁)", font=FONT, font_size=16, color=SLATE_GRAY
        ).next_to(axes.x_axis, DOWN, buff=0.25).align_to(axes.x_axis, RIGHT)

        y_label = Text(
            "Fingerprint Score  (s₂)", font=FONT, font_size=16, color=SLATE_GRAY
        ).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.28)

        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        # ── PHASE 1 ───────────────────────────────────────────────────────────
        self._phase1_vectorization(axes)

        # ── PHASE 2 ───────────────────────────────────────────────────────────
        genuine_cloud, impostor_cloud = self._phase2_break_free(axes)

        # ── PHASE 3 ───────────────────────────────────────────────────────────
        hyperplane_line = self._phase3_linear_svm(axes, genuine_cloud, impostor_cloud)

        # ── PHASE 4 ───────────────────────────────────────────────────────────
        self._phase4_xor_dilemma(axes, genuine_cloud, impostor_cloud, hyperplane_line)

    # =========================================================================
    # PHASE 1  (0:00 – 0:15)  The Score Vectorization
    # =========================================================================
    def _phase1_vectorization(self, axes: Axes) -> None:
        """Two floating 1D scores merge into a 2D column vector, then land as a dot."""

        # ── Two floating score values ──────────────────────────────────────────
        # Positioned in screen-space (not axes-space) so they never overlap the Y-axis.
        # s1 sits in the upper-left quadrant of the screen, s2 in the upper-right.
        s1_val = MathTex(r"s_1 = 0.85", color=CLASS_B_COLOR, font_size=40)
        s2_val = MathTex(r"s_2 = 0.92", color=CLASS_B_COLOR, font_size=40)
        s1_val.move_to(UP * 2.8 + LEFT * 3.5)
        s2_val.move_to(UP * 2.8 + RIGHT * 3.5)

        self.play(
            FadeIn(s1_val, shift=UP * 0.3),
            FadeIn(s2_val, shift=UP * 0.3),
            run_time=1.2,
        )
        self.wait(0.8)

        # ── Merge into column vector ───────────────────────────────────────────
        score_matrix = Matrix(
            [["0.85"], ["0.92"]],
            left_bracket="[", right_bracket="]",
            element_to_mobject_config={"font_size": 36, "color": CLASS_B_COLOR},
        )
        prefix = MathTex(r"\mathbf{S} =", color=CLASS_B_COLOR, font_size=40)
        vector_group = VGroup(prefix, score_matrix).arrange(RIGHT, buff=0.2)
        vector_group.move_to(UP * 2.8)  # centred at top, safely above the axes

        self.play(
            ReplacementTransform(s1_val, score_matrix.get_entries()[0]),
            ReplacementTransform(s2_val, score_matrix.get_entries()[1]),
            FadeIn(score_matrix.get_brackets()),
            FadeIn(prefix),
            run_time=1.5,
        )
        self.wait(0.7)

        # Brief Flash then shrink the matrix to a corner badge
        self.play(Flash(score_matrix, color=CLASS_B_COLOR, flash_radius=0.9), run_time=0.6)
        self.play(vector_group.animate.scale(0.58).to_corner(UL, buff=0.35), run_time=0.9)
        self.wait(0.4)

        # ── Morph into a dot at (0.85, 0.92) ─────────────────────────────────
        target_pos = axes.c2p(0.85, 0.92)
        sample_dot = Dot(point=target_pos, color=CLASS_B_COLOR, radius=0.12)

        # Connecting dashed projection guides
        guide_x = DashedLine(
            axes.c2p(0.85, 0.0), axes.c2p(0.85, 0.92),
            color=CLASS_B_COLOR, stroke_opacity=0.45, dash_length=0.08,
        )
        guide_y = DashedLine(
            axes.c2p(0.0, 0.92), axes.c2p(0.85, 0.92),
            color=CLASS_B_COLOR, stroke_opacity=0.45, dash_length=0.08,
        )

        self.play(Create(guide_x), Create(guide_y), run_time=0.9)
        self.play(
            ReplacementTransform(vector_group.copy(), sample_dot),
            run_time=1.1,
        )
        self.play(Flash(sample_dot, color=WHITE, flash_radius=0.35, num_lines=8), run_time=0.5)
        self.wait(0.6)

        # Coordinate annotation
        coord_label = MathTex(
            r"(0.85,\ 0.92)", color=CLASS_B_COLOR, font_size=22
        ).next_to(sample_dot, UR, buff=0.14)
        self.play(FadeIn(coord_label, shift=UP * 0.15), run_time=0.6)
        self.wait(0.9)

        # Clean up Phase 1 ephemeral objects; keep axes and labels alive
        self.play(
            FadeOut(guide_x), FadeOut(guide_y),
            FadeOut(coord_label), FadeOut(sample_dot),
            FadeOut(vector_group),
            run_time=0.9,
        )
        self.wait(0.3)

    # =========================================================================
    # PHASE 2  (0:15 – 0:30)  Breaking Free from 1D
    # =========================================================================
    def _phase2_break_free(
        self, axes: Axes
    ) -> tuple[VGroup, VGroup]:
        """Overlapping 1D crowd expands into a clean 2D linearly-separable layout."""

        # ── Overlap crowd on the X-axis (confusion zone from Scene 2) ─────────
        rng_crowd = np.random.default_rng(99)
        crowd_x = rng_crowd.uniform(0.35, 0.65, N_CLOUD * 2)
        crowd_cols = (
            [GENUINE_COLOR] * N_CLOUD + [IMPOSTOR_COLOR] * N_CLOUD
        )
        crowd_dots = VGroup(
            *[
                Dot(axes.c2p(x, 0.0), color=c, radius=0.09)
                for x, c in zip(crowd_x, crowd_cols)
            ]
        )
        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.5) for d in crowd_dots],
                lag_ratio=0.05,
            ),
            run_time=1.5,
        )

        confusion_tag = Text(
            "1D Overlap (Scene 2)", font=FONT, font_size=18, color=SLATE_GRAY
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(confusion_tag, shift=DOWN * 0.2), run_time=0.7)
        self.wait(0.9)

        # ── Animate expansion into 2D space ────────────────────────────────────
        gen_targets  = _scatter_2d((0.72, 0.75), N_CLOUD, 0.09, RNG_SEED_GEN_2D)
        imp_targets  = _scatter_2d((0.27, 0.25), N_CLOUD, 0.09, RNG_SEED_IMP_2D)
        all_targets  = gen_targets + imp_targets

        expand_anims = [
            d.animate.move_to(axes.c2p(tx, ty))
            for d, (tx, ty) in zip(crowd_dots, all_targets)
        ]
        self.play(
            FadeOut(confusion_tag),
            *expand_anims,
            run_time=2.2,
            rate_func=smooth,
        )
        self.wait(0.6)

        # Split into two named VGroups for later phases
        genuine_cloud  = VGroup(*crowd_dots[:N_CLOUD])
        impostor_cloud = VGroup(*crowd_dots[N_CLOUD:])

        # ── Labels for the two clusters ────────────────────────────────────────
        gen_label = Text("Genuine  ✓", font=FONT, font_size=19, color=GENUINE_COLOR)
        gen_bg = SurroundingRectangle(gen_label, fill_color=BG_COLOR, fill_opacity=0.8, stroke_width=0, buff=0.08)
        gen_group = VGroup(gen_bg, gen_label).next_to(genuine_cloud, UP, buff=0.15)

        imp_label = Text("Impostor  ✗", font=FONT, font_size=19, color=IMPOSTOR_COLOR)
        imp_bg = SurroundingRectangle(imp_label, fill_color=BG_COLOR, fill_opacity=0.8, stroke_width=0, buff=0.08)
        imp_group = VGroup(imp_bg, imp_label).next_to(impostor_cloud, DOWN, buff=0.15)

        self.play(
            FadeIn(gen_group, shift=UP * 0.15),
            FadeIn(imp_group, shift=DOWN * 0.15),
            run_time=0.9,
        )
        self.wait(1.0)
        self.play(FadeOut(gen_group), FadeOut(imp_group), run_time=0.7)
        self.wait(0.3)

        return genuine_cloud, impostor_cloud

    # =========================================================================
    # PHASE 3  (0:30 – 1:00)  Linear SVM — Learning Process + Convergence
    # =========================================================================
    def _phase3_linear_svm(
        self,
        axes: Axes,
        genuine_cloud: VGroup,
        impostor_cloud: VGroup,
    ) -> Line:
        """Fake-Iteration learning: random start → optimal SVM via ValueTracker lerp.

        Technique adapted from svm_animation.py:
          - sklearn SVC gives us w_final and b_final (the ground truth).
          - We define w_start / b_start as deliberately wrong offsets.
          - A ValueTracker alpha ∈ [0, 1] drives always_redraw() to linearly
            interpolate (lerp) every visual element from wrong → correct.
          - This animates the boundary "learning" without re-implementing
            gradient descent — pure visual interpolation, academically honest.
        """

        # ── Subtitle ──────────────────────────────────────────────────────────
        subtitle = Text(
            "Linear Support Vector Machine",
            font=FONT, font_size=24, color=HYPERPLANE_COLOR,
        ).to_edge(UP, buff=0.32)
        underline = Line(
            LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=1.5
        ).set_width(subtitle.width).next_to(subtitle, DOWN, buff=0.06)
        self.play(
            FadeIn(subtitle, shift=DOWN * 0.15),
            Create(underline),
            run_time=1.0,
        )
        self.wait(0.5)

        # ── Recover point coordinates from the animated dots ──────────────────
        gen_pts = [tuple(axes.p2c(d.get_center())[:2]) for d in genuine_cloud]
        imp_pts = [tuple(axes.p2c(d.get_center())[:2]) for d in impostor_cloud]

        # ── Fit optimal Linear SVM (sklearn, same as svm_animation.py) ────────
        X_data = np.array(gen_pts + imp_pts)
        y_data = np.array([1] * len(gen_pts) + [-1] * len(imp_pts))
        clf = SVC(kernel="linear", C=1, random_state=42)
        clf.fit(X_data, y_data)

        coef_final = clf.coef_[0].copy()       # [w1, w2]
        b_final    = clf.intercept_[0]         # scalar b
        support_vecs = clf.support_vectors_

        # Derived final boundary parameters
        slope_final  = -coef_final[0] / coef_final[1]
        y0_final     = -b_final / coef_final[1]
        norm_final   = np.linalg.norm(coef_final)
        margin_dy_final = 1.0 / (coef_final[1] * norm_final)  # half-margin in y

        # ── Define a deliberately wrong starting boundary (Epoch 0) ────────────
        # Offset coefficients so the line starts tilted and shifted away from optimal.
        coef_start = coef_final + np.array([-1.8,  2.2])
        b_start    = b_final - 2.5
        slope_start = -coef_start[0] / coef_start[1]
        y0_start    = -b_start / coef_start[1]
        norm_start  = np.linalg.norm(coef_start)
        margin_dy_start = 1.0 / (coef_start[1] * norm_start)

        x_lo, x_hi = AXIS_RANGE[0], AXIS_RANGE[1]

        # ── ValueTracker: alpha drives the entire lerp (0 = random, 1 = optimal)
        alpha = ValueTracker(0.0)

        # ── Helper: interpolated boundary parameters at current alpha ─────────
        def _lerp_params():
            t = alpha.get_value()
            sl  = slope_start  * (1 - t) + slope_final  * t
            y0  = y0_start     * (1 - t) + y0_final     * t
            mdy = margin_dy_start * (1 - t) + margin_dy_final * t
            return sl, y0, mdy

        # ── always_redraw: main hyperplane ────────────────────────────────────
        def _make_hyperplane():
            sl, y0, _ = _lerp_params()
            t = alpha.get_value()
            # Colour shifts from a dim orange-red (wrong) to HYPERPLANE_COLOR (correct)
            c = interpolate_color(ManimColor("#E87040"), ManimColor(HYPERPLANE_COLOR), t)
            return Line(
                axes.c2p(x_lo, sl * x_lo + y0),
                axes.c2p(x_hi, sl * x_hi + y0),
                color=c, stroke_width=3.5,
            )

        # ── always_redraw: positive margin (dashed, above boundary) ───────────
        def _make_margin_pos():
            sl, y0, mdy = _lerp_params()
            t = alpha.get_value()
            opacity = 0.4 + 0.5 * t  # margin becomes more visible as we converge
            return DashedLine(
                axes.c2p(x_lo, sl * x_lo + y0 + abs(mdy)),
                axes.c2p(x_hi, sl * x_hi + y0 + abs(mdy)),
                color=MARGIN_COLOR, stroke_width=2,
                stroke_opacity=opacity, dash_length=0.1,
            )

        # ── always_redraw: negative margin (dashed, below boundary) ───────────
        def _make_margin_neg():
            sl, y0, mdy = _lerp_params()
            t = alpha.get_value()
            opacity = 0.4 + 0.5 * t
            return DashedLine(
                axes.c2p(x_lo, sl * x_lo + y0 - abs(mdy)),
                axes.c2p(x_hi, sl * x_hi + y0 - abs(mdy)),
                color=MARGIN_COLOR, stroke_width=2,
                stroke_opacity=opacity, dash_length=0.1,
            )

        # ── always_redraw: shaded margin band ─────────────────────────────────
        def _make_margin_band():
            sl, y0, mdy = _lerp_params()
            t = alpha.get_value()
            corners = [
                axes.c2p(x_lo, sl * x_lo + y0 - abs(mdy)),
                axes.c2p(x_hi, sl * x_hi + y0 - abs(mdy)),
                axes.c2p(x_hi, sl * x_hi + y0 + abs(mdy)),
                axes.c2p(x_lo, sl * x_lo + y0 + abs(mdy)),
            ]
            return Polygon(
                *corners,
                fill_color=HYPERPLANE_COLOR,
                fill_opacity=0.08 + 0.10 * t,
                stroke_width=0,
            )

        # ── Epoch counter (always_redraw text) ─────────────────────────────────
        # Maps alpha ∈ [0,1] to a fake "iteration" count 0 → N_ITER_DISPLAY
        N_ITER_DISPLAY = 50
        def _make_epoch_text():
            t = alpha.get_value()
            epoch = int(t * N_ITER_DISPLAY)
            label = Text(
                f"Epoch: {epoch:>3} / {N_ITER_DISPLAY}",
                font=FONT, font_size=18, color=SLATE_GRAY,
            ).to_corner(DR, buff=0.45)
            return label

        # ── Create all live objects and add to scene ───────────────────────────
        margin_band  = always_redraw(_make_margin_band)
        hyperplane   = always_redraw(_make_hyperplane)
        margin_pos_d = always_redraw(_make_margin_pos)
        margin_neg_d = always_redraw(_make_margin_neg)
        epoch_text   = always_redraw(_make_epoch_text)

        self.add(margin_band, margin_pos_d, margin_neg_d, hyperplane, epoch_text)

        # ── Step 1: Epoch 0 — show the random (wrong) boundary ────────────────
        epoch0_caption = Text(
            "Epoch 0 — Ranh giới ngẫu nhiên",
            font=FONT, font_size=18, color=SLATE_GRAY,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(epoch0_caption, shift=UP * 0.15), run_time=0.8)
        self.wait(1.2)

        # ── Step 2: Animate learning — alpha 0 → 1 (the core lerp) ───────────
        # rate_func=smooth gives a natural ease-in / ease-out feel.
        self.play(FadeOut(epoch0_caption), run_time=0.4)

        converge_caption = Text(
            "Đang tối ưu hóa lề (Maximising margin)…",
            font=FONT, font_size=18, color=HYPERPLANE_COLOR,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(converge_caption, shift=UP * 0.1), run_time=0.6)

        # The main learning animation: 4 seconds, smooth easing
        self.play(
            alpha.animate.set_value(1.0),
            run_time=4.0,
            rate_func=smooth,
        )
        self.play(FadeOut(converge_caption), run_time=0.4)
        self.wait(0.5)

        # ── Step 3: Convergence — snap, flash, highlight Support Vectors ───────
        # Remove updaters so the objects are now static at alpha = 1
        self.remove(margin_band, margin_pos_d, margin_neg_d, hyperplane, epoch_text)

        # Rebuild static final versions (clean, no always_redraw overhead)
        final_band = Polygon(
            axes.c2p(x_lo, slope_final * x_lo + y0_final - abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final - abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final + abs(margin_dy_final)),
            axes.c2p(x_lo, slope_final * x_lo + y0_final + abs(margin_dy_final)),
            fill_color=HYPERPLANE_COLOR, fill_opacity=0.18, stroke_width=0,
        )
        final_hp = Line(
            axes.c2p(x_lo, slope_final * x_lo + y0_final),
            axes.c2p(x_hi, slope_final * x_hi + y0_final),
            color=HYPERPLANE_COLOR, stroke_width=4,
        )
        final_margin_pos = DashedLine(
            axes.c2p(x_lo, slope_final * x_lo + y0_final + abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final + abs(margin_dy_final)),
            color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1,
        )
        final_margin_neg = DashedLine(
            axes.c2p(x_lo, slope_final * x_lo + y0_final - abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final - abs(margin_dy_final)),
            color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1,
        )

        self.play(
            FadeIn(final_band),
            Create(final_hp),
            Create(final_margin_pos),
            Create(final_margin_neg),
            run_time=0.7,
        )
        # Snap flash on the hyperplane
        self.play(Flash(final_hp.get_center(), color=HYPERPLANE_COLOR, flash_radius=0.5, num_lines=10), run_time=0.5)

        # ── Support Vector rings — glow on convergence ─────────────────────────
        sv_rings = VGroup()
        for sv in support_vecs:
            ring = Circle(
                radius=0.20, color=WHITE, stroke_width=2.8,
            ).move_to(axes.c2p(sv[0], sv[1]))
            sv_rings.add(ring)

        self.play(
            LaggedStart(*[
                AnimationGroup(Create(r), Flash(r.get_center(), color=WHITE, flash_radius=0.25, num_lines=6))
                for r in sv_rings
            ], lag_ratio=0.25),
            run_time=1.2,
        )

        sv_badge = Text(
            "Support Vectors — Vector hỗ trợ", font=FONT, font_size=19, color=WHITE
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(sv_badge, shift=UP * 0.15), run_time=0.7)
        self.wait(0.8)

        # ── 2/||w|| margin label ───────────────────────────────────────────────
        # Thủ thuật "Đường thẳng ảo": Tạo Line liền nét trong bộ nhớ để mượn hàm tính toán
        calc_line_pos = Line(
            axes.c2p(x_lo, slope_final * x_lo + y0_final + abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final + abs(margin_dy_final))
        )
        calc_line_neg = Line(
            axes.c2p(x_lo, slope_final * x_lo + y0_final - abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final - abs(margin_dy_final))
        )
        
        # Manim có thể thoải mái tính toán trên đường Line liền nét
        pt_on_pos = calc_line_pos.point_from_proportion(0.65) 
        pt_on_neg = calc_line_neg.get_projection(pt_on_pos)
        
        # Mũi tên 2 đầu đo khoảng cách lề vuông góc
        margin_arrow = DoubleArrow(
            pt_on_neg, pt_on_pos, buff=0, 
            color=MARGIN_COLOR, stroke_width=2.5, tip_length=0.15
        )
        margin_text  = MathTex(
            r"\frac{2}{\|\mathbf{w}\|}", font_size=28, color=MARGIN_COLOR
        ).next_to(margin_arrow, RIGHT, buff=0.1)

        # Mũi tên cong chỉ rõ đâu là Support Vector
        sv_pointer = CurvedArrow(
            start_point=sv_badge.get_top() + RIGHT * 1.5,
            end_point=sv_rings[0].get_bottom() + DOWN * 0.05,
            color=WHITE, angle=-PI/3, stroke_width=2, tip_length=0.15
        )

        self.play(
            Create(margin_arrow), Write(margin_text), 
            Create(sv_pointer), run_time=1.0
        )
        self.wait(1.2)

        # ── Clean up Phase 3 overlays; pass hyperplane forward to Phase 4 ─────
        self.play(
            FadeOut(sv_badge), FadeOut(sv_pointer),
            FadeOut(sv_rings),
            FadeOut(margin_arrow), FadeOut(margin_text),
            FadeOut(subtitle), FadeOut(underline),
            FadeOut(final_margin_pos), FadeOut(final_margin_neg),
            FadeOut(final_band),
            run_time=1.0,
        )
        self.wait(0.3)

        return final_hp  # passed to Phase 4

    # =========================================================================
    # PHASE 4  (0:45 – 1:00)  The XOR Spoof Attack Dilemma
    # =========================================================================
    def _phase4_xor_dilemma(
        self,
        axes: Axes,
        genuine_cloud: VGroup,
        impostor_cloud: VGroup,
        hyperplane: Line,
    ) -> None:
        """Inject XOR spoof clusters and animate the SVM boundary's failure."""

        # ── Red screen flash ──────────────────────────────────────────────────
        flash_rect = Rectangle(
            width=config.frame_width + 2,
            height=config.frame_height + 2,
            fill_color=SPOOF_RED, fill_opacity=0.0,
            stroke_width=0,
        )
        self.add(flash_rect)
        self.play(flash_rect.animate.set_fill(opacity=0.32), run_time=0.3, rate_func=there_and_back)
        self.remove(flash_rect)

        # ── Warning text ───────────────────────────────────────────────────────
        warning_vn = Text(
            "Tấn công giả mạo", font=FONT, font_size=22, color=SPOOF_RED,
        )
        warning_en = Text(
            "Spoof Attack", font=FONT, font_size=28, weight=BOLD, color=SPOOF_RED,
        )
        warning_en.to_edge(UP, buff=0.30)
        warning_vn.next_to(warning_en, DOWN, buff=0.10)

        warning_bg = SurroundingRectangle(
            VGroup(warning_en, warning_vn),
            fill_color=BLACK, fill_opacity=0.75,
            stroke_color=SPOOF_RED, stroke_width=1.5,
            corner_radius=0.14, buff=0.18,
        )
        self.play(
            FadeIn(warning_bg),
            Write(warning_en),
            FadeIn(warning_vn, shift=DOWN * 0.1),
            run_time=0.8,
        )
        self.wait(0.5)

        # ── Spoof cluster 1 — Top-Left (high fingerprint, low face) ──────────
        #    Represents: forged fingerprint + genuine face rejected
        spoof_tl_pts = _scatter_2d((0.27, 0.75), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_TL)
        spoof_tl = VGroup(
            *[
                Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09)
                for x, y in spoof_tl_pts
            ]
        )

        # ── Spoof cluster 2 — Bottom-Right (high face, low fingerprint) ───────
        #    Represents: 3D face mask + genuine fingerprint absent
        spoof_br_pts = _scatter_2d((0.72, 0.25), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_BR)
        spoof_br = VGroup(
            *[
                Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09)
                for x, y in spoof_br_pts
            ]
        )

        # Annotation for spoof clusters
        tl_caption = Text(
            "Silicone fingerprint\n(Giả mạo vân tay)", font=FONT, font_size=14, color=IMPOSTOR_COLOR
        ).next_to(spoof_tl, LEFT, buff=0.18)
        br_caption = Text(
            "3D face mask\n(Mặt nạ khuôn mặt)", font=FONT, font_size=14, color=IMPOSTOR_COLOR
        ).next_to(spoof_br, RIGHT, buff=0.18)

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_tl], lag_ratio=0.07),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_br], lag_ratio=0.07),
            run_time=1.0,
        )
        self.play(
            FadeIn(tl_caption, shift=RIGHT * 0.15),
            FadeIn(br_caption, shift=LEFT * 0.15),
            run_time=0.5,
        )
        self.wait(0.5)

        # ── XOR label ─────────────────────────────────────────────────────────
        xor_label = MathTex(
            r"\text{XOR Layout}", color=SPOOF_RED, font_size=26
        ).to_edge(DOWN, buff=0.5)
        xor_sub = Text(
            "No linear boundary can separate this!", font=FONT, font_size=15, color=SLATE_GRAY
        ).next_to(xor_label, DOWN, buff=0.14)

        self.play(FadeIn(xor_label, shift=UP * 0.15), FadeIn(xor_sub), run_time=0.5)
        self.wait(0.3)

        # ── Hyperplane rotates wildly and flashes RED to signal failure ────────
        angle_tracker = ValueTracker(0)
        c_yellow = ManimColor(HYPERPLANE_COLOR)
        c_red    = ManimColor(SPOOF_RED)

        # Rebuild the hyperplane as an updater-driven line
        # We store its current state as a persistent reference via a mutable list
        hp_ref: list[Line] = [hyperplane]

        def _hp_updater(line: Line) -> None:
            a   = angle_tracker.get_value()
            cx  = axes.c2p(0.5, 0.5)  # pivot at centre of score space
            dir_vec = np.array([np.cos(a), np.sin(a), 0.0])
            line.put_start_and_end_on(cx - 3.5 * dir_vec, cx + 3.5 * dir_vec)
            # Colour interpolation: yellow → red with oscillating intensity
            t = (np.sin(a * 6) + 1) / 2
            line.set_color(interpolate_color(c_yellow, c_red, t))
            line.set_stroke(width=3 + 3 * t)

        hyperplane.add_updater(_hp_updater)
        self.play(
            angle_tracker.animate.set_value(PI * 1.4),
            run_time=2.5,
            rate_func=linear,
        )
        hyperplane.remove_updater(_hp_updater)

        # Dim to signal definitive failure
        self.play(
            hyperplane.animate.set_color(SPOOF_RED).set_stroke(opacity=0.35),
            run_time=0.5,
        )

        # Failure caption
        failure_text = Text(
            "⚠  Tuyến tính thất bại!",
            font=FONT, font_size=22, weight=BOLD, color=SPOOF_RED,
        ).move_to(ORIGIN + UP * 0.5)
        failure_bg = SurroundingRectangle(
            failure_text,
            fill_color=BLACK, fill_opacity=0.82,
            stroke_color=SPOOF_RED, stroke_width=1.8,
            corner_radius=0.12, buff=0.18,
        )
        self.play(FadeIn(failure_bg), Write(failure_text), run_time=0.7)
        self.wait(1.2)

        # ── Fade everything to black ───────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)
