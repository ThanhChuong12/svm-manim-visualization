"""Scene 2 — The Limitations of Unibiometrics (1D Thresholding).

Demonstrates why a single biometric score dimension cannot reliably separate
genuine users from impostors when the score distributions overlap, motivating
the transition to a 2D (multi-biometric) feature space.
"""

from manim import *
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import (
    GENUINE_COLOR, IMPOSTOR_COLOR, OVERLAP_COLOR,
    THRESHOLD_COLOR, BG_COLOR, FONT_MAIN, SLATE_GRAY,
    HYPERPLANE_COLOR,
)
from utils.math_helpers import gaussian, min_func

# ── Scene-local constants ─────────────────────────────────────────────────────
SIGMA = 1.0
X_MIN, X_MAX = -7, 7
PEAK_Y = 1.0           # Scaled Gaussian peak height at x = μ is ~1.0
LABEL_LIFT = 0.55      # vertical gap between curve peak and label centre
GAUSSIAN_SCALE = 2.5   # Amplification factor to scale standard PDF peak (~0.398) to ~1.0

# Use a font that is confirmed to be installed in the WSL/Linux environment.
FONT = "Montserrat"


def scaled_gaussian(x: float, mu: float, sigma: float) -> float:
    """Scaled Gaussian probability density function to maximize visual layout."""
    return gaussian(x, mu, sigma) * GAUSSIAN_SCALE


def _make_warning_badge(text_str: str, color: ManimColor) -> VGroup:
    """Wrap a warning label in a dark pill for legibility over colored fills."""
    label = Text(text_str, font=FONT, font_size=18, color=color)
    bg = SurroundingRectangle(
        label, fill_color=BLACK, fill_opacity=0.78,
        stroke_width=0, corner_radius=0.12, buff=0.14,
    )
    return VGroup(bg, label)


class UnibiometricsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── PHASE 1: The Ideal World (0:00 – 0:15) ──────────────────────────

        # Task 1: Increased y_length from 3.5 to 4.5 to expand vertical plotting space.
        axes = Axes(
            x_range=[X_MIN, X_MAX, 1],
            y_range=[0, 1.15, 0.5],
            x_length=9.0, y_length=4.5,
            axis_config={"stroke_width": 2, "color": WHITE, "stroke_opacity": 0.6},
            y_axis_config={"stroke_opacity": 0},
            tips=False,
        ).shift(DOWN * 0.5)

        # Task 2 (from previous turn): X-axis label neatly arranged at bottom right.
        x_axis_label = Text(
            "Face Match Score", font=FONT, font_size=15, color=SLATE_GRAY,
        )
        x_axis_label.next_to(axes.x_axis, DOWN, buff=0.2).align_to(axes.x_axis, RIGHT)

        self.play(FadeIn(axes), FadeIn(x_axis_label), run_time=1)

        mu_impostor = ValueTracker(-3.0)
        mu_genuine = ValueTracker(3.0)

        # Bell curves rebuilt every frame using the scaled Gaussian function.
        impostor_curve = always_redraw(lambda: axes.plot(
            lambda x: scaled_gaussian(x, mu_impostor.get_value(), SIGMA),
            x_range=[X_MIN, X_MAX], color=IMPOSTOR_COLOR, stroke_width=3,
        ))
        genuine_curve = always_redraw(lambda: axes.plot(
            lambda x: scaled_gaussian(x, mu_genuine.get_value(), SIGMA),
            x_range=[X_MIN, X_MAX], color=GENUINE_COLOR, stroke_width=3,
        ))

        impostor_fill = always_redraw(lambda: axes.get_area(
            axes.plot(lambda x: scaled_gaussian(x, mu_impostor.get_value(), SIGMA),
                      x_range=[X_MIN, X_MAX]),
            color=IMPOSTOR_COLOR, opacity=0.25,
        ))
        genuine_fill = always_redraw(lambda: axes.get_area(
            axes.plot(lambda x: scaled_gaussian(x, mu_genuine.get_value(), SIGMA),
                      x_range=[X_MIN, X_MAX]),
            color=GENUINE_COLOR, opacity=0.25,
        ))

        # Top labels with dynamic boundary offsets to prevent overlap as distributions merge.
        # The labels automatically fan outwards smoothly as the peaks approach the center.
        impostor_label = Text(
            "Kẻ mạo danh", font=FONT, font_size=20, color=IMPOSTOR_COLOR,
        )
        impostor_label.add_updater(lambda m: m.move_to(
            axes.c2p(0.6 * mu_impostor.get_value() - 1.6, PEAK_Y + LABEL_LIFT)
        ))

        genuine_label = Text(
            "Người dùng hợp lệ", font=FONT, font_size=20, color=GENUINE_COLOR,
        )
        genuine_label.add_updater(lambda m: m.move_to(
            axes.c2p(0.6 * mu_genuine.get_value() + 1.8, PEAK_Y + LABEL_LIFT)
        ))

        self.play(
            Create(impostor_curve), Create(genuine_curve),
            FadeIn(impostor_fill), FadeIn(genuine_fill),
            FadeIn(impostor_label), FadeIn(genuine_label),
            run_time=2,
        )
        self.wait(1.5)

        # ── PHASE 2: The Overlap / Reality (0:15 – 0:35) ─────────────────────

        reality_tag = Text(
            "Thực tế (Reality)", font=FONT, font_size=18,
            color=HYPERPLANE_COLOR,
        ).to_corner(UR, buff=0.5)
        self.play(FadeIn(reality_tag, shift=LEFT * 0.3), run_time=0.8)

        # Slide distributions inward.
        self.play(
            mu_impostor.animate.set_value(-1.0),
            mu_genuine.animate.set_value(1.0),
            run_time=3, rate_func=smooth,
        )
        self.wait(0.5)

        # Dynamic min(impostor, genuine) overlap shading.
        overlap_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                min_func(
                    lambda x: scaled_gaussian(x, mu_impostor.get_value(), SIGMA),
                    lambda x: scaled_gaussian(x, mu_genuine.get_value(), SIGMA),
                ),
                x_range=[X_MIN, X_MAX],
            ),
            color=OVERLAP_COLOR, opacity=0.45,
        ))
        self.play(FadeIn(overlap_fill), run_time=1)

        # Tính toán điểm giao cắt thực tế của 2 đồ thị khi mu = -1 và 1
        intersection_peak = axes.c2p(0.0, scaled_gaussian(0.0, -1.0, SIGMA))

        # Đặt chữ "Vùng nhiễu" lên vị trí cao, vượt qua hẳn đỉnh đồ thị (PEAK_Y)
        confusion_text = Text(
            "Vùng nhiễu (Overlap Zone)",
            font=FONT, font_size=18, color=OVERLAP_COLOR,
        ).move_to(axes.c2p(0.0, PEAK_Y + 0.4)) 

        # Kéo dài mũi tên từ chữ xuống đúng điểm giao cắt
        confusion_arrow = Arrow(
            start=confusion_text.get_bottom() + DOWN * 0.1,
            end=intersection_peak + UP * 0.05, # Cách điểm giao cắt một chút cho đẹp
            color=OVERLAP_COLOR, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.08, # Trục mũi tên dài nên cần thu nhỏ tỷ lệ đầu mũi tên
        )
        confusion_indicator = VGroup(confusion_text, confusion_arrow)

        self.play(FadeIn(confusion_text), Create(confusion_arrow), run_time=0.8)
        for _ in range(2):
            self.play(
                confusion_indicator.animate.set_opacity(0.25),
                run_time=0.3, rate_func=there_and_back,
            )
        self.wait(1)

        # ── PHASE 3: The Threshold Dilemma (0:35 – 0:50) ─────────────────────

        self.play(FadeOut(confusion_indicator), run_time=0.5)

        eta = ValueTracker(0.0)

        threshold_line = always_redraw(lambda: DashedLine(
            start=axes.c2p(eta.get_value(), 0),
            end=axes.c2p(eta.get_value(), 1.05),
            color=THRESHOLD_COLOR, stroke_width=3, dash_length=0.1,
        ))

        # η floats above the line tip, tracking it in real-time.
        eta_label = MathTex(r"\eta", color=THRESHOLD_COLOR, font_size=36)
        eta_label.add_updater(
            lambda m: m.next_to(axes.c2p(eta.get_value(), 1.05), UP, buff=0.12)
        )

        self.play(Create(threshold_line), FadeIn(eta_label), run_time=1)
        self.wait(0.5)

        # ── False Accept (FAR): Impostors who score RIGHT of η ───────────────
        far_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                lambda x: scaled_gaussian(x, mu_impostor.get_value(), SIGMA),
                x_range=[eta.get_value(), X_MAX],
            ),
            x_range=[eta.get_value(), X_MAX],
            color=IMPOSTOR_COLOR, opacity=0.55,
        ))

        # Badge sits upper-right; arrow tip targets the impostor tail RIGHT of η.
        far_badge = _make_warning_badge("False Accept\n(Nhận nhầm kẻ gian)", IMPOSTOR_COLOR)
        far_badge.move_to(axes.c2p(3.8, 0.82))

        self.play(FadeIn(far_fill), run_time=0.3)
        self.play(eta.animate.set_value(-1.5), run_time=2, rate_func=smooth)

        # Arrow points into the impostor tail strictly right of η.
        ETA_FAR = -1.5
        far_arrow = Arrow(
            start=far_badge.get_bottom(),
            end=axes.c2p(ETA_FAR + 1.2, 0.05),
            color=IMPOSTOR_COLOR, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.15,
        )
        self.play(FadeIn(far_badge), Create(far_arrow), run_time=0.6)
        self.wait(1.2)

        self.play(FadeOut(far_fill), FadeOut(far_badge), FadeOut(far_arrow), run_time=0.5)

        # ── False Reject (FRR): Genuine users who score LEFT of η ────────────
        frr_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                lambda x: scaled_gaussian(x, mu_genuine.get_value(), SIGMA),
                x_range=[X_MIN, eta.get_value()],
            ),
            x_range=[X_MIN, eta.get_value()],
            color=GENUINE_COLOR, opacity=0.55,
        ))

        frr_badge = _make_warning_badge("False Reject\n(Từ chối người thật)", GENUINE_COLOR)
        frr_badge.move_to(axes.c2p(-3.8, 0.82))

        self.play(FadeIn(frr_fill), run_time=0.3)
        self.play(eta.animate.set_value(1.5), run_time=2, rate_func=smooth)

        # Arrow points into the genuine tail strictly left of η.
        ETA_FRR = 1.5
        frr_arrow = Arrow(
            start=frr_badge.get_bottom(),
            end=axes.c2p(ETA_FRR - 1.2, 0.05),
            color=GENUINE_COLOR, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.15,
        )
        self.play(FadeIn(frr_badge), Create(frr_arrow), run_time=0.6)
        self.wait(1.2)

        self.play(FadeOut(frr_fill), FadeOut(frr_badge), FadeOut(frr_arrow), run_time=0.5)

        # Return threshold to centre for visual closure.
        self.play(eta.animate.set_value(0.0), run_time=1, rate_func=smooth)
        self.wait(0.5)

        # ── PHASE 4: Setup for Scene 3 (0:50 – 1:00) ─────────────────────────

        # Detach updaters before fading to prevent orphaned lambda calls.
        impostor_label.clear_updaters()
        genuine_label.clear_updaters()
        eta_label.clear_updaters()

        # Task 3: Clean up all 1D curves, fills, and labels to leave clean space for the Y-axis transition.
        self.play(
            FadeOut(threshold_line), FadeOut(eta_label),
            FadeOut(overlap_fill), FadeOut(reality_tag),
            FadeOut(impostor_curve), FadeOut(genuine_curve),
            FadeOut(impostor_fill), FadeOut(genuine_fill),
            FadeOut(impostor_label), FadeOut(genuine_label),
            run_time=1,
        )

        # Grow a Y-axis from the origin, signalling the 1D → 2D transition.
        y_axis_line = Line(
            start=axes.c2p(0, 0), end=axes.c2p(0, 1.1),
            color=WHITE, stroke_width=2, stroke_opacity=0.6,
        )
        y_axis_arrow = Triangle(
            color=WHITE, fill_color=WHITE, fill_opacity=1,
        ).scale(0.08).rotate(PI).next_to(y_axis_line, UP, buff=0)

        # y-label beautifully rotated and pinned next to the y-axis tip.
        y_label = Text(
            "Fingerprint Score", font=FONT, font_size=16, color=SLATE_GRAY,
        )
        y_label.rotate(PI/2).next_to(y_axis_arrow, LEFT, buff=0.2).shift(DOWN * 0.5)

        self.play(
            Create(y_axis_line), FadeIn(y_axis_arrow),
            FadeIn(y_label, shift=RIGHT * 0.2),
            run_time=1.5,
        )
        self.wait(1)

        # Fade to black.
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(0.5)
