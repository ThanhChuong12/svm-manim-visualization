"""Scene 3 — Score Combination & Linear SVM.

Phase 1 (0:00–0:20): Feature extraction split-screen → score vectorisation → dot lands on 2D axes.
Phase 2 (0:20–0:40): 1D overlap crowd "rescued" into clean 2D separation with camera zoom.
Phase 3 (0:40–0:55): Linear SVM hyperplane + animated margin expansion + support vectors.
Phase 4 (0:55–1:10): XOR spoof attack; the linear boundary fails visually.

Uses MovingCameraScene to enable the "Rescue" zoom effect in Phase 2.
SVM hyperplane fitted via sklearn SVC (linear kernel), then visually interpolated.
"""

import numpy as np
from manim import *
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

try:
    from utils.visual_helpers import (
        make_genuine_icon, make_fingerprint_icon,
        make_noisy_icon, make_spoof_icon,
    )
except ImportError:
    # Minimal fallbacks — icons degrade to simple shapes
    def make_genuine_icon(size=0.6):
        return Circle(radius=size * 0.5, color=GENUINE_COLOR, stroke_width=2)

    def make_fingerprint_icon(size=0.6):
        return Circle(radius=size * 0.5, color="#AABBFF", stroke_width=2)

    def make_noisy_icon(size=0.6):
        return Circle(radius=size * 0.5, color="#888888", stroke_width=2)

    def make_spoof_icon(size=0.6):
        return Square(side_length=size, color=IMPOSTOR_COLOR, stroke_width=2)

# ── Scene-local design tokens ─────────────────────────────────────────────────
FONT            = FONT_MAIN
SUPPORT_RING    = "#FFFFFF"
MARGIN_COLOR    = HYPERPLANE_COLOR
SPOOF_RED       = "#FF2222"
AXIS_OPACITY    = 0.55
AXIS_RANGE      = [0.0, 1.0, 0.25]

# Reproducible random seeds for each cluster
RNG_SEED_GEN_2D   = 11
RNG_SEED_IMP_2D   = 17
RNG_SEED_SPOOF_TL = 29
RNG_SEED_SPOOF_BR = 37

N_CLOUD = 18


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
    """Build a Line spanning the visible axis range for a linear boundary."""
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

class ScoreCombinationScene(MovingCameraScene):
    """Scene 3: Score Vectorisation → 2D Separation → Linear SVM → XOR Failure."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # Shared 2D axes (score space 0→1 on both dimensions)
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
            "Face Match Score  (s₁)", font=FONT, font_size=16, color=SLATE_GRAY,
        ).next_to(axes.x_axis, DOWN, buff=0.25).align_to(axes.x_axis, RIGHT)

        y_label = Text(
            "Fingerprint Score  (s₂)", font=FONT, font_size=16, color=SLATE_GRAY,
        ).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.28)

        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label), run_time=1.0)

        self._phase1_vectorization(axes)
        genuine_cloud, impostor_cloud = self._phase2_rescue(axes)
        hyperplane_line = self._phase3_linear_svm(axes, genuine_cloud, impostor_cloud)
        self._phase4_xor_dilemma(axes, genuine_cloud, impostor_cloud, hyperplane_line)

    # =========================================================================
    # PHASE 1 — Feature Extraction & Score Vectorisation
    # =========================================================================
    def _phase1_vectorization(self, axes: Axes) -> None:
        """Split-screen biometric extraction → matrix → dot on 2D axes."""

        # ── Split-screen panels ───────────────────────────────────────────────
        panel_w, panel_h = 2.8, 2.2

        # Left panel: Face recognition
        face_panel_bg = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.12,
            stroke_color=CLASS_B_COLOR, stroke_width=1.5, stroke_opacity=0.5,
            fill_color=BG_COLOR, fill_opacity=0.9,
        ).shift(LEFT * 3.2 + UP * 2.4)

        face_icon = make_genuine_icon(0.45)
        face_icon.move_to(face_panel_bg.get_center() + UP * 0.3)

        face_title = Text(
            "Face Recognition", font=FONT, font_size=13, color=CLASS_B_COLOR,
        ).next_to(face_panel_bg, UP, buff=0.10)

        # Laser scan effect (thin rectangle sweeping down)
        face_laser = Rectangle(
            width=panel_w * 0.75, height=0.04,
            fill_color="#00FFFF", fill_opacity=0.8, stroke_width=0,
        ).move_to(face_panel_bg.get_top() + DOWN * 0.3)

        # Right panel: Fingerprint recognition
        fp_panel_bg = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.12,
            stroke_color="#AABBFF", stroke_width=1.5, stroke_opacity=0.5,
            fill_color=BG_COLOR, fill_opacity=0.9,
        ).shift(RIGHT * 3.2 + UP * 2.4)

        fp_icon = make_fingerprint_icon(0.45)
        fp_icon.move_to(fp_panel_bg.get_center() + UP * 0.3)

        fp_title = Text(
            "Fingerprint Scan", font=FONT, font_size=13, color="#AABBFF",
        ).next_to(fp_panel_bg, UP, buff=0.10)

        fp_laser = Rectangle(
            width=panel_w * 0.75, height=0.04,
            fill_color="#00FFFF", fill_opacity=0.8, stroke_width=0,
        ).move_to(fp_panel_bg.get_top() + DOWN * 0.3)

        # Show panels
        self.play(
            FadeIn(face_panel_bg), FadeIn(fp_panel_bg),
            FadeIn(face_title), FadeIn(fp_title),
            GrowFromCenter(face_icon), GrowFromCenter(fp_icon),
            run_time=1.0,
        )

        # Animate laser scans simultaneously
        scan_dist = panel_h - 0.6
        self.play(
            face_laser.animate.shift(DOWN * scan_dist),
            fp_laser.animate.shift(DOWN * scan_dist),
            FadeIn(face_laser), FadeIn(fp_laser),
            run_time=1.2, rate_func=linear,
        )
        self.play(FadeOut(face_laser), FadeOut(fp_laser), run_time=0.3)

        # ── Score labels appear beneath panels ────────────────────────────────
        s1_val = MathTex(r"s_1 = 0.85", color=CLASS_B_COLOR, font_size=34)
        s1_val.next_to(face_panel_bg, DOWN, buff=0.18)

        s2_val = MathTex(r"s_2 = 0.92", color="#AABBFF", font_size=34)
        s2_val.next_to(fp_panel_bg, DOWN, buff=0.18)

        self.play(
            FadeIn(s1_val, shift=UP * 0.2),
            FadeIn(s2_val, shift=UP * 0.2),
            run_time=0.9,
        )
        self.wait(0.6)

        # ── Merge into column vector ──────────────────────────────────────────
        score_matrix = Matrix(
            [["0.85"], ["0.92"]],
            left_bracket="[", right_bracket="]",
            element_to_mobject_config={"font_size": 32, "color": CLASS_B_COLOR},
        )
        prefix = MathTex(r"\mathbf{S} =", color=CLASS_B_COLOR, font_size=36)
        vector_group = VGroup(prefix, score_matrix).arrange(RIGHT, buff=0.2)
        vector_group.move_to(UP * 2.4)

        self.play(
            FadeOut(face_panel_bg), FadeOut(fp_panel_bg),
            FadeOut(face_icon), FadeOut(fp_icon),
            FadeOut(face_title), FadeOut(fp_title),
            ReplacementTransform(s1_val, score_matrix.get_entries()[0]),
            ReplacementTransform(s2_val, score_matrix.get_entries()[1]),
            FadeIn(score_matrix.get_brackets()),
            FadeIn(prefix),
            run_time=1.4,
        )
        self.wait(0.5)

        # Flash then shrink the matrix to a corner badge
        self.play(
            Flash(score_matrix, color=CLASS_B_COLOR, flash_radius=0.9),
            run_time=0.5,
        )
        self.play(
            vector_group.animate.scale(0.55).to_corner(UL, buff=0.35),
            run_time=0.8,
        )
        self.wait(0.3)

        # ── Morph into a dot at (0.85, 0.92) with Vector Flash ────────────────
        target_pos = axes.c2p(0.85, 0.92)
        sample_dot = Dot(point=target_pos, color=CLASS_B_COLOR, radius=0.12)

        # Projection guides
        guide_x = DashedLine(
            axes.c2p(0.85, 0.0), axes.c2p(0.85, 0.92),
            color=CLASS_B_COLOR, stroke_opacity=0.4, dash_length=0.08,
        )
        guide_y = DashedLine(
            axes.c2p(0.0, 0.92), axes.c2p(0.85, 0.92),
            color=CLASS_B_COLOR, stroke_opacity=0.4, dash_length=0.08,
        )

        self.play(Create(guide_x), Create(guide_y), run_time=0.8)

        # Flying dot with glow trail
        flying_dot = Dot(
            point=vector_group.get_center(), color=CLASS_B_COLOR, radius=0.10,
        )
        glow_trail = TracedPath(
            flying_dot.get_center,
            stroke_color=CLASS_B_COLOR, stroke_width=3, stroke_opacity=0.4,
            dissipating_time=0.6,
        )
        self.add(glow_trail)
        self.play(
            flying_dot.animate.move_to(target_pos),
            run_time=1.0, rate_func=smooth,
        )
        self.remove(glow_trail, flying_dot)
        self.add(sample_dot)

        # Flash at landing point
        self.play(
            Flash(sample_dot, color=WHITE, flash_radius=0.4, num_lines=10),
            run_time=0.5,
        )

        # Coordinate annotation
        coord_label = MathTex(
            r"(0.85,\ 0.92)", color=CLASS_B_COLOR, font_size=22,
        ).next_to(sample_dot, UR, buff=0.14)
        self.play(FadeIn(coord_label, shift=UP * 0.12), run_time=0.5)
        self.wait(0.7)

        # Clean up Phase 1 ephemera
        self.play(
            FadeOut(guide_x), FadeOut(guide_y),
            FadeOut(coord_label), FadeOut(sample_dot),
            FadeOut(vector_group),
            run_time=0.8,
        )
        self.wait(0.2)

    # =========================================================================
    # PHASE 2 — The Rescue (Breaking Free from 1D)
    # =========================================================================
    def _phase2_rescue(self, axes: Axes) -> tuple[VGroup, VGroup]:
        """1D overlap crowd → rescue icons → clean 2D separation with camera zoom."""

        # ── 1D overlap crowd on the X-axis ────────────────────────────────────
        rng_crowd = np.random.default_rng(99)
        crowd_x = rng_crowd.uniform(0.35, 0.65, N_CLOUD * 2)
        crowd_cols = [GENUINE_COLOR] * N_CLOUD + [IMPOSTOR_COLOR] * N_CLOUD
        crowd_dots = VGroup(*[
            Dot(axes.c2p(x, 0.0), color=c, radius=0.09)
            for x, c in zip(crowd_x, crowd_cols)
        ])

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.5) for d in crowd_dots],
                lag_ratio=0.04,
            ),
            run_time=1.2,
        )

        confusion_tag = Text(
            "1D Overlap — Vùng chồng lấn", font=FONT, font_size=18, color=SLATE_GRAY,
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(confusion_tag, shift=DOWN * 0.15), run_time=0.6)
        self.wait(0.7)

        # ── Camera zoom in for the rescue sequence ────────────────────────────
        self.play(
            self.camera.frame.animate.set_width(10),
            run_time=1.0, rate_func=smooth,
        )

        # ── Rescue 1: Noisy Genuine ───────────────────────────────────────────
        noisy_icon = make_noisy_icon(0.35)
        noisy_start = axes.c2p(0.50, 0.02)
        noisy_icon.move_to(noisy_start + UP * 0.6)

        noisy_label = Text(
            "Ảnh nhiễu", font=FONT, font_size=12, color="#888888",
        ).next_to(noisy_icon, DOWN, buff=0.06)

        # The noisy genuine has uncertain face score but HIGH fingerprint score
        noisy_target = axes.c2p(0.50, 0.82)

        rescue_arrow_g = Arrow(
            noisy_start + UP * 0.15, noisy_target + DOWN * 0.15,
            color=GENUINE_COLOR, stroke_width=2, stroke_opacity=0.6,
            max_tip_length_to_length_ratio=0.12,
        )

        self.play(GrowFromCenter(noisy_icon), FadeIn(noisy_label), run_time=0.7)
        self.wait(0.3)

        self.play(
            noisy_icon.animate.move_to(noisy_target + UP * 0.6),
            noisy_label.animate.move_to(noisy_target + UP * 1.1),
            Create(rescue_arrow_g),
            run_time=1.4, rate_func=smooth,
        )

        rescue_text_g = Text(
            "✓ Thoát vùng nhiễu!", font=FONT, font_size=13, color=GENUINE_COLOR,
        ).next_to(noisy_icon, RIGHT, buff=0.15)
        self.play(FadeIn(rescue_text_g, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.5)

        # ── Rescue 2: Spoof Impostor ──────────────────────────────────────────
        spoof_icon = make_spoof_icon(0.30)
        spoof_start = axes.c2p(0.55, 0.02)
        spoof_icon.move_to(spoof_start + UP * 0.6)

        spoof_label = Text(
            "Spoof", font=FONT, font_size=12, color=IMPOSTOR_COLOR,
        ).next_to(spoof_icon, DOWN, buff=0.06)

        # High face score but LOW fingerprint → plummets
        spoof_target_high = axes.c2p(0.78, 0.55)
        spoof_target_low = axes.c2p(0.78, 0.15)

        self.play(GrowFromCenter(spoof_icon), FadeIn(spoof_label), run_time=0.7)
        self.wait(0.3)

        # First lift (high face score)
        self.play(
            spoof_icon.animate.move_to(spoof_target_high + UP * 0.5),
            spoof_label.animate.move_to(spoof_target_high + UP * 1.0),
            run_time=0.8, rate_func=smooth,
        )

        # Then plummet (low fingerprint)
        rescue_arrow_s = Arrow(
            spoof_target_high + DOWN * 0.1, spoof_target_low + UP * 0.15,
            color=IMPOSTOR_COLOR, stroke_width=2, stroke_opacity=0.6,
            max_tip_length_to_length_ratio=0.12,
        )
        self.play(
            spoof_icon.animate.move_to(spoof_target_low + UP * 0.5),
            spoof_label.animate.move_to(spoof_target_low + UP * 1.0),
            Create(rescue_arrow_s),
            run_time=1.2, rate_func=rush_into,
        )

        rescue_text_s = Text(
            "✗ Lộ diện kẻ giả mạo!", font=FONT, font_size=13, color=IMPOSTOR_COLOR,
        ).next_to(spoof_icon, RIGHT, buff=0.15)
        self.play(FadeIn(rescue_text_s, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.6)

        # ── Camera zoom out for full 2D view ──────────────────────────────────
        self.play(
            FadeOut(noisy_icon), FadeOut(noisy_label), FadeOut(rescue_text_g),
            FadeOut(rescue_arrow_g),
            FadeOut(spoof_icon), FadeOut(spoof_label), FadeOut(rescue_text_s),
            FadeOut(rescue_arrow_s),
            FadeOut(confusion_tag),
            self.camera.frame.animate.set_width(14.222),
            run_time=1.0,
        )

        # ── Expand crowd into 2D clusters ─────────────────────────────────────
        gen_targets = _scatter_2d((0.72, 0.75), N_CLOUD, 0.09, RNG_SEED_GEN_2D)
        imp_targets = _scatter_2d((0.27, 0.25), N_CLOUD, 0.09, RNG_SEED_IMP_2D)
        all_targets = gen_targets + imp_targets

        expand_anims = [
            d.animate.move_to(axes.c2p(tx, ty))
            for d, (tx, ty) in zip(crowd_dots, all_targets)
        ]
        self.play(*expand_anims, run_time=2.0, rate_func=smooth)
        self.wait(0.4)

        genuine_cloud = VGroup(*crowd_dots[:N_CLOUD])
        impostor_cloud = VGroup(*crowd_dots[N_CLOUD:])

        # Cluster labels
        gen_label = Text("Genuine  ✓", font=FONT, font_size=19, color=GENUINE_COLOR)
        gen_bg = SurroundingRectangle(
            gen_label, fill_color=BG_COLOR, fill_opacity=0.8,
            stroke_width=0, buff=0.08,
        )
        gen_group = VGroup(gen_bg, gen_label).next_to(genuine_cloud, UP, buff=0.15)

        imp_label = Text("Impostor  ✗", font=FONT, font_size=19, color=IMPOSTOR_COLOR)
        imp_bg = SurroundingRectangle(
            imp_label, fill_color=BG_COLOR, fill_opacity=0.8,
            stroke_width=0, buff=0.08,
        )
        imp_group = VGroup(imp_bg, imp_label).next_to(impostor_cloud, DOWN, buff=0.15)

        self.play(
            FadeIn(gen_group, shift=UP * 0.12),
            FadeIn(imp_group, shift=DOWN * 0.12),
            run_time=0.8,
        )
        self.wait(0.8)
        self.play(FadeOut(gen_group), FadeOut(imp_group), run_time=0.6)
        self.wait(0.3)

        return genuine_cloud, impostor_cloud

    # =========================================================================
    # PHASE 3 — Linear SVM: Learning Process & Convergence
    # =========================================================================
    def _phase3_linear_svm(
        self,
        axes: Axes,
        genuine_cloud: VGroup,
        impostor_cloud: VGroup,
    ) -> Line:
        """ValueTracker lerp: random boundary → sklearn-optimal SVM via fake iteration."""

        # Subtitle
        subtitle = Text(
            "Linear Support Vector Machine",
            font=FONT, font_size=24, color=HYPERPLANE_COLOR,
        ).to_edge(UP, buff=0.32)
        underline = Line(
            LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=1.5,
        ).set_width(subtitle.width).next_to(subtitle, DOWN, buff=0.06)
        self.play(
            FadeIn(subtitle, shift=DOWN * 0.15),
            Create(underline),
            run_time=0.9,
        )
        self.wait(0.4)

        # ── Recover point coordinates from animated dots ──────────────────────
        gen_pts = [tuple(axes.p2c(d.get_center())[:2]) for d in genuine_cloud]
        imp_pts = [tuple(axes.p2c(d.get_center())[:2]) for d in impostor_cloud]

        # ── Fit optimal Linear SVM (sklearn) ──────────────────────────────────
        X_data = np.array(gen_pts + imp_pts)
        y_data = np.array([1] * len(gen_pts) + [-1] * len(imp_pts))
        clf = SVC(kernel="linear", C=1, random_state=42)
        clf.fit(X_data, y_data)

        coef_final = clf.coef_[0].copy()
        b_final = clf.intercept_[0]
        support_vecs = clf.support_vectors_

        slope_final = -coef_final[0] / coef_final[1]
        y0_final = -b_final / coef_final[1]
        norm_final = np.linalg.norm(coef_final)
        margin_dy_final = 1.0 / (coef_final[1] * norm_final)

        # ── Deliberately wrong starting boundary (Epoch 0) ────────────────────
        coef_start = coef_final + np.array([-1.8, 2.2])
        b_start = b_final - 2.5
        slope_start = -coef_start[0] / coef_start[1]
        y0_start = -b_start / coef_start[1]
        norm_start = np.linalg.norm(coef_start)
        margin_dy_start = 1.0 / (coef_start[1] * norm_start)

        x_lo, x_hi = AXIS_RANGE[0], AXIS_RANGE[1]

        # ValueTracker drives the entire lerp
        alpha = ValueTracker(0.0)

        def _lerp_params():
            t = alpha.get_value()
            sl = slope_start * (1 - t) + slope_final * t
            y0 = y0_start * (1 - t) + y0_final * t
            mdy = margin_dy_start * (1 - t) + margin_dy_final * t
            return sl, y0, mdy

        # Hyperplane (colour shifts from dim orange-red to yellow)
        def _make_hyperplane():
            sl, y0, _ = _lerp_params()
            t = alpha.get_value()
            c = interpolate_color(ManimColor("#E87040"), ManimColor(HYPERPLANE_COLOR), t)
            return Line(
                axes.c2p(x_lo, sl * x_lo + y0),
                axes.c2p(x_hi, sl * x_hi + y0),
                color=c, stroke_width=3.5,
            )

        # Positive margin (above boundary)
        def _make_margin_pos():
            sl, y0, mdy = _lerp_params()
            t = alpha.get_value()
            return DashedLine(
                axes.c2p(x_lo, sl * x_lo + y0 + abs(mdy)),
                axes.c2p(x_hi, sl * x_hi + y0 + abs(mdy)),
                color=MARGIN_COLOR, stroke_width=2,
                stroke_opacity=0.4 + 0.5 * t, dash_length=0.1,
            )

        # Negative margin (below boundary)
        def _make_margin_neg():
            sl, y0, mdy = _lerp_params()
            t = alpha.get_value()
            return DashedLine(
                axes.c2p(x_lo, sl * x_lo + y0 - abs(mdy)),
                axes.c2p(x_hi, sl * x_hi + y0 - abs(mdy)),
                color=MARGIN_COLOR, stroke_width=2,
                stroke_opacity=0.4 + 0.5 * t, dash_length=0.1,
            )

        # Shaded margin band with subtle shimmer
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

        # Epoch counter
        N_ITER_DISPLAY = 50

        def _make_epoch_text():
            t = alpha.get_value()
            epoch = int(t * N_ITER_DISPLAY)
            return Text(
                f"Epoch: {epoch:>3} / {N_ITER_DISPLAY}",
                font=FONT, font_size=18, color=SLATE_GRAY,
            ).to_corner(DR, buff=0.45)

        # Create live redraw objects
        margin_band = always_redraw(_make_margin_band)
        hyperplane = always_redraw(_make_hyperplane)
        margin_pos_d = always_redraw(_make_margin_pos)
        margin_neg_d = always_redraw(_make_margin_neg)
        epoch_text = always_redraw(_make_epoch_text)

        self.add(margin_band, margin_pos_d, margin_neg_d, hyperplane, epoch_text)

        # ── Epoch 0 — random (wrong) boundary ────────────────────────────────
        epoch0_caption = Text(
            "Epoch 0 — Ranh giới ngẫu nhiên",
            font=FONT, font_size=18, color=SLATE_GRAY,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(epoch0_caption, shift=UP * 0.12), run_time=0.7)
        self.wait(1.0)

        # ── Learning animation: alpha 0 → 1 ──────────────────────────────────
        self.play(FadeOut(epoch0_caption), run_time=0.3)

        converge_caption = Text(
            "Đang tối ưu hóa lề (Maximising margin)…",
            font=FONT, font_size=18, color=HYPERPLANE_COLOR,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(converge_caption, shift=UP * 0.1), run_time=0.5)

        # Core lerp animation
        self.play(
            alpha.animate.set_value(1.0),
            run_time=4.0, rate_func=smooth,
        )
        self.play(FadeOut(converge_caption), run_time=0.3)
        self.wait(0.4)

        # ── Convergence: replace live objects with static finals ───────────────
        self.remove(margin_band, margin_pos_d, margin_neg_d, hyperplane, epoch_text)

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
        self.play(
            Flash(final_hp.get_center(), color=HYPERPLANE_COLOR,
                  flash_radius=0.5, num_lines=10),
            run_time=0.5,
        )

        # ── Support Vector rings ──────────────────────────────────────────────
        sv_rings = VGroup()
        for sv in support_vecs:
            ring = Circle(
                radius=0.20, color=WHITE, stroke_width=2.8,
            ).move_to(axes.c2p(sv[0], sv[1]))
            sv_rings.add(ring)

        self.play(
            LaggedStart(*[
                AnimationGroup(
                    Create(r),
                    Flash(r.get_center(), color=WHITE,
                          flash_radius=0.25, num_lines=6),
                )
                for r in sv_rings
            ], lag_ratio=0.25),
            run_time=1.2,
        )

        sv_badge = Text(
            "Support Vectors — Vector hỗ trợ",
            font=FONT, font_size=19, color=WHITE,
        ).to_edge(DOWN, buff=0.55)
        self.play(
            FadeIn(sv_badge, shift=UP * 0.15),
            run_time=0.7,
        )
        self.wait(0.6)

        # ── 2/||w|| margin label ──────────────────────────────────────────────
        # Invisible solid lines to leverage Manim's projection math
        calc_line_pos = Line(
            axes.c2p(x_lo, slope_final * x_lo + y0_final + abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final + abs(margin_dy_final)),
        )
        calc_line_neg = Line(
            axes.c2p(x_lo, slope_final * x_lo + y0_final - abs(margin_dy_final)),
            axes.c2p(x_hi, slope_final * x_hi + y0_final - abs(margin_dy_final)),
        )

        pt_on_pos = calc_line_pos.point_from_proportion(0.65)
        pt_on_neg = calc_line_neg.get_projection(pt_on_pos)

        margin_arrow = DoubleArrow(
            pt_on_neg, pt_on_pos, buff=0,
            color=MARGIN_COLOR, stroke_width=2.5, tip_length=0.15,
        )
        margin_text = MathTex(
            r"\frac{2}{\|\mathbf{w}\|}", font_size=28, color=MARGIN_COLOR,
        ).next_to(margin_arrow, RIGHT, buff=0.1)

        sv_pointer = CurvedArrow(
            start_point=sv_badge.get_top() + RIGHT * 1.5,
            end_point=sv_rings[0].get_bottom() + DOWN * 0.05,
            color=WHITE, angle=-PI / 3, stroke_width=2, tip_length=0.15,
        )

        self.play(
            Create(margin_arrow), Write(margin_text),
            Create(sv_pointer),
            run_time=1.0,
        )
        self.wait(1.0)

        # ── Clean up Phase 3 overlays; pass hyperplane to Phase 4 ─────────────
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

        return final_hp

    # =========================================================================
    # PHASE 4 — The XOR Spoof Attack Dilemma
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
        self.play(
            flash_rect.animate.set_fill(opacity=0.32),
            run_time=0.3, rate_func=there_and_back,
        )
        self.remove(flash_rect)

        # ── Warning text ──────────────────────────────────────────────────────
        warning_en = Text(
            "Spoof Attack", font=FONT, font_size=28,
            weight=BOLD, color=SPOOF_RED,
        ).to_edge(UP, buff=0.30)
        warning_vn = Text(
            "Tấn công giả mạo", font=FONT, font_size=22, color=SPOOF_RED,
        ).next_to(warning_en, DOWN, buff=0.10)

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
        self.wait(0.4)

        # ── Spoof cluster 1 — Top-Left (high fingerprint, low face) ──────────
        spoof_tl_pts = _scatter_2d(
            (0.27, 0.75), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_TL,
        )
        spoof_tl = VGroup(*[
            Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09)
            for x, y in spoof_tl_pts
        ])

        # ── Spoof cluster 2 — Bottom-Right (high face, low fingerprint) ──────
        spoof_br_pts = _scatter_2d(
            (0.72, 0.25), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_BR,
        )
        spoof_br = VGroup(*[
            Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09)
            for x, y in spoof_br_pts
        ])

        # Cluster annotations
        tl_caption = Text(
            "Silicone fingerprint\n(Giả mạo vân tay)",
            font=FONT, font_size=14, color=IMPOSTOR_COLOR,
        ).next_to(spoof_tl, LEFT, buff=0.18)
        br_caption = Text(
            "3D face mask\n(Mặt nạ khuôn mặt)",
            font=FONT, font_size=14, color=IMPOSTOR_COLOR,
        ).next_to(spoof_br, RIGHT, buff=0.18)

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.4) for d in spoof_tl], lag_ratio=0.07,
            ),
            LaggedStart(
                *[FadeIn(d, scale=0.4) for d in spoof_br], lag_ratio=0.07,
            ),
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
            r"\text{XOR Layout}", color=SPOOF_RED, font_size=26,
        ).to_edge(DOWN, buff=0.5)
        xor_sub = Text(
            "No linear boundary can separate this!",
            font=FONT, font_size=15, color=SLATE_GRAY,
        ).next_to(xor_label, DOWN, buff=0.14)

        self.play(
            FadeIn(xor_label, shift=UP * 0.15),
            FadeIn(xor_sub),
            run_time=0.5,
        )
        self.wait(0.3)

        # ── Hyperplane rotates wildly — colour shifts yellow → red ────────────
        angle_tracker = ValueTracker(0)
        c_yellow = ManimColor(HYPERPLANE_COLOR)
        c_red = ManimColor(SPOOF_RED)

        def _hp_updater(line: Line) -> None:
            a = angle_tracker.get_value()
            cx = axes.c2p(0.5, 0.5)
            dir_vec = np.array([np.cos(a), np.sin(a), 0.0])
            line.put_start_and_end_on(cx - 3.5 * dir_vec, cx + 3.5 * dir_vec)
            t = (np.sin(a * 6) + 1) / 2
            line.set_color(interpolate_color(c_yellow, c_red, t))
            line.set_stroke(width=3 + 3 * t)

        hyperplane.add_updater(_hp_updater)
        self.play(
            angle_tracker.animate.set_value(PI * 1.4),
            run_time=2.5, rate_func=linear,
        )
        hyperplane.remove_updater(_hp_updater)

        # Dim to signal failure
        self.play(
            hyperplane.animate.set_color(SPOOF_RED).set_stroke(opacity=0.35),
            run_time=0.5,
        )

        # ── Failure icon (abstract ✕ cross) + caption ─────────────────────────
        cross_size = 0.5
        cross = VGroup(
            Line(UL * cross_size, DR * cross_size, color=SPOOF_RED, stroke_width=5),
            Line(UR * cross_size, DL * cross_size, color=SPOOF_RED, stroke_width=5),
        ).move_to(ORIGIN + UP * 0.85)

        failure_text = Text(
            "⚠  Tuyến tính thất bại!",
            font=FONT, font_size=22, weight=BOLD, color=SPOOF_RED,
        ).next_to(cross, DOWN, buff=0.2)

        failure_bg = SurroundingRectangle(
            VGroup(cross, failure_text),
            fill_color=BLACK, fill_opacity=0.82,
            stroke_color=SPOOF_RED, stroke_width=1.8,
            corner_radius=0.12, buff=0.22,
        )
        self.play(
            FadeIn(failure_bg),
            Create(cross),
            Write(failure_text),
            run_time=0.8,
        )
        self.wait(1.2)

        # ── Fade everything to black ──────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)
