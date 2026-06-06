"""Scene 2 — The Limitations of Unibiometrics (1D Thresholding)  V2.0

Phases
------
Phase 1  (0:00–0:18)  Ideal world    — two well-separated Gaussians; icons above peaks.
Phase 2  (0:18–0:40)  Reality        — noisy genuine + spoofed impostor appear; curves
                                        expand and merge, burying the edge-case icons.
Phase 3  (0:40–0:58)  Threshold trap — η sweeps left (FAR / spoof) then right (FRR / noise).
Phase 4  (0:58–1:10)  2D transition  — Y-axis grows; fade to black for Scene 3.

Icon factory is embedded here (no external SVGs needed).
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
SIGMA_IDEAL   = 1.0     # σ for well-separated ideal case
SIGMA_REAL    = 1.3     # σ after reality sets in (wider spread)
X_MIN, X_MAX  = -7, 7
PEAK_Y        = 1.0     # Approximate Gaussian peak height after scaling
LABEL_LIFT    = 0.60    # Vertical gap: curve peak → label centre
GAUSSIAN_SCALE = 2.5    # Scale factor: raw PDF peak (~0.398) → ~1.0

# Enforce Arial for full Vietnamese Unicode support
FONT = "Arial"


# ─────────────────────────────────────────────────────────────────────────────
# Utility: scaled Gaussian
# ─────────────────────────────────────────────────────────────────────────────

def scaled_gaussian(x: float, mu: float, sigma: float) -> float:
    return gaussian(x, mu, sigma) * GAUSSIAN_SCALE


# ─────────────────────────────────────────────────────────────────────────────
# Icon Factory — all built from Manim primitives; no external assets needed
# ─────────────────────────────────────────────────────────────────────────────

def make_genuine_icon(size: float = 0.6) -> VGroup:
    """Green face with a smile — represents a legitimate user."""
    head  = Ellipse(width=size * 1.2, height=size * 1.5,
                    stroke_color=GENUINE_COLOR, stroke_width=2,
                    fill_color="#1A1D27", fill_opacity=1)
    eye_l = Dot(LEFT  * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=GENUINE_COLOR)
    eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=GENUINE_COLOR)
    mouth = Arc(radius=size * 0.30, start_angle=-10 * DEGREES,
                angle=-160 * DEGREES,
                color=GENUINE_COLOR, stroke_width=2).shift(DOWN * size * 0.20)
    return VGroup(head, eye_l, eye_r, mouth)


def make_impostor_icon(size: float = 0.6) -> VGroup:
    """Red face with a frown — represents an attacker."""
    head  = Ellipse(width=size * 1.2, height=size * 1.5,
                    stroke_color=IMPOSTOR_COLOR, stroke_width=2,
                    fill_color="#1A1D27", fill_opacity=1)
    eye_l = Dot(LEFT  * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=IMPOSTOR_COLOR)
    eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=IMPOSTOR_COLOR)
    mouth = Line(LEFT * size * 0.20, RIGHT * size * 0.30,
                 color=IMPOSTOR_COLOR, stroke_width=2
                 ).shift(DOWN * size * 0.20).rotate(15 * DEGREES)
    return VGroup(head, eye_l, eye_r, mouth)


def make_noisy_icon(size: float = 0.6) -> VGroup:
    """Glitched genuine face — represents sensor noise / low-quality capture."""
    face = make_genuine_icon(size)
    face.set_color(color="#888888")   # desaturate to grey
    noise_lines = VGroup(*[
        Line(LEFT * size * 0.75, RIGHT * size * 0.75,
             color=WHITE, stroke_width=1.2, stroke_opacity=0.65
             ).shift(UP * size * dy)
        for dy in [-0.20, 0.05, 0.32]
    ])
    return VGroup(face, noise_lines)


def make_spoof_icon(size: float = 0.6) -> VGroup:
    """Phone showing a genuine face — represents a photo/video spoof attack."""
    phone  = RoundedRectangle(
        width=size * 1.6, height=size * 2.4, corner_radius=0.1,
        stroke_color=WHITE, stroke_width=2,
        fill_color="#000000", fill_opacity=1,
    )
    screen = Rectangle(
        width=size * 1.35, height=size * 1.9,
        stroke_color="#333333", stroke_width=1,
        fill_color="#111122", fill_opacity=1,
    )
    face_on_screen = make_genuine_icon(size * 0.65)
    return VGroup(phone, screen, face_on_screen)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: warning badge with dark pill background
# ─────────────────────────────────────────────────────────────────────────────

def _make_warning_badge(text_str: str, color: ManimColor) -> VGroup:
    """Wrap warning text in a dark pill for legibility over colored fills."""
    label = Text(text_str, font=FONT, font_size=17, color=color)
    bg    = SurroundingRectangle(
        label, fill_color=BLACK, fill_opacity=0.82,
        stroke_color=color, stroke_width=1.2,
        corner_radius=0.12, buff=0.14,
    )
    return VGroup(bg, label)


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class UnibiometricsScene(Scene):
    """Scene 2 V2.0 — Unibiometric 1D thresholding with icon-driven storytelling."""

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR

        # Shared axes (X only; Y axis added in Phase 4)
        axes = Axes(
            x_range=[X_MIN, X_MAX, 1],
            y_range=[0, 1.20, 0.5],
            x_length=9.2, y_length=4.6,
            axis_config={"stroke_width": 2, "color": WHITE, "stroke_opacity": 0.6},
            y_axis_config={"stroke_opacity": 0},
            tips=False,
        ).shift(DOWN * 0.55)

        x_axis_label = Text(
            "Face Match Score", font=FONT, font_size=15, color=SLATE_GRAY,
        ).next_to(axes.x_axis, DOWN, buff=0.22).align_to(axes.x_axis, RIGHT)

        self.play(FadeIn(axes), FadeIn(x_axis_label), run_time=1.0)

        # ── PHASE 1 ───────────────────────────────────────────────────────────
        mu_imp = ValueTracker(-3.0)
        mu_gen = ValueTracker( 3.0)
        sigma_imp = ValueTracker(SIGMA_IDEAL)
        sigma_gen = ValueTracker(SIGMA_IDEAL)

        self._phase1_ideal(axes, mu_imp, mu_gen, sigma_imp, sigma_gen)

        # ── PHASE 2 ───────────────────────────────────────────────────────────
        (impostor_curve, impostor_fill,
         genuine_curve,  genuine_fill,
         impostor_label, genuine_label,
         overlap_fill) = self._phase2_reality(axes, mu_imp, mu_gen, sigma_imp, sigma_gen)

        # ── PHASE 3 ───────────────────────────────────────────────────────────
        self._phase3_threshold(
            axes, mu_imp, mu_gen, sigma_imp, sigma_gen,
            impostor_curve, impostor_fill, genuine_curve, genuine_fill,
            impostor_label, genuine_label, overlap_fill,
        )

        # ── PHASE 4 ───────────────────────────────────────────────────────────
        self._phase4_transition(axes, x_axis_label)

    # =========================================================================
    # PHASE 1  —  Ideal World
    # =========================================================================
    def _phase1_ideal(
        self,
        axes: Axes,
        mu_imp:   ValueTracker,
        mu_gen:   ValueTracker,
        sigma_imp: ValueTracker,
        sigma_gen: ValueTracker,
    ) -> None:
        """Two separated bell curves; one genuine icon, one impostor icon above peaks."""

        # ── Curves (always_redraw so they follow tracker changes later) ───────
        impostor_curve = always_redraw(lambda: axes.plot(
            lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
            x_range=[X_MIN, X_MAX], color=IMPOSTOR_COLOR, stroke_width=3,
        ))
        genuine_curve = always_redraw(lambda: axes.plot(
            lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
            x_range=[X_MIN, X_MAX], color=GENUINE_COLOR, stroke_width=3,
        ))
        impostor_fill = always_redraw(lambda: axes.get_area(
            axes.plot(lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
                      x_range=[X_MIN, X_MAX]),
            color=IMPOSTOR_COLOR, opacity=0.22,
        ))
        genuine_fill = always_redraw(lambda: axes.get_area(
            axes.plot(lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
                      x_range=[X_MIN, X_MAX]),
            color=GENUINE_COLOR, opacity=0.22,
        ))

        # Store for Phase 2 / 3 access
        self._impostor_curve = impostor_curve
        self._impostor_fill  = impostor_fill
        self._genuine_curve  = genuine_curve
        self._genuine_fill   = genuine_fill

        # ── Floating labels with V-formation anchoring ─────────────────────────
        impostor_label = Text("Kẻ mạo danh", font=FONT, font_size=20, color=IMPOSTOR_COLOR)
        impostor_label.add_updater(lambda m: m.move_to(
            axes.c2p(0.58 * mu_imp.get_value() - 1.5, PEAK_Y + LABEL_LIFT)
        ))
        genuine_label = Text("Người dùng hợp lệ", font=FONT, font_size=20, color=GENUINE_COLOR)
        genuine_label.add_updater(lambda m: m.move_to(
            axes.c2p(0.58 * mu_gen.get_value() + 1.8, PEAK_Y + LABEL_LIFT)
        ))
        self._impostor_label = impostor_label
        self._genuine_label  = genuine_label

        # ── Icons above the peaks ──────────────────────────────────────────────
        # Ideal positions: impostor peak at x = -3, genuine peak at x = +3
        gen_icon  = make_genuine_icon(0.60).move_to(axes.c2p( 3.0, PEAK_Y + 0.82))
        imp_icon  = make_impostor_icon(0.60).move_to(axes.c2p(-3.0, PEAK_Y + 0.82))

        gen_icon_label = Text("Người hợp lệ", font=FONT, font_size=13, color=GENUINE_COLOR
                              ).next_to(gen_icon, DOWN, buff=0.08)
        imp_icon_label = Text("Kẻ tấn công",  font=FONT, font_size=13, color=IMPOSTOR_COLOR
                              ).next_to(imp_icon, DOWN, buff=0.08)

        self._gen_icon  = gen_icon
        self._imp_icon  = imp_icon
        self._gen_icon_label = gen_icon_label
        self._imp_icon_label = imp_icon_label

        # Animate everything in
        self.play(
            Create(impostor_curve), Create(genuine_curve),
            FadeIn(impostor_fill),  FadeIn(genuine_fill),
            FadeIn(impostor_label), FadeIn(genuine_label),
            run_time=2.0,
        )
        self.play(
            GrowFromCenter(gen_icon), GrowFromCenter(imp_icon),
            FadeIn(gen_icon_label),   FadeIn(imp_icon_label),
            run_time=1.2,
        )
        self.wait(1.5)

    # =========================================================================
    # PHASE 2  —  Reality: Edge Cases Appear
    # =========================================================================
    def _phase2_reality(
        self,
        axes: Axes,
        mu_imp:   ValueTracker,
        mu_gen:   ValueTracker,
        sigma_imp: ValueTracker,
        sigma_gen: ValueTracker,
    ):
        """Noisy genuine + spoof impostor appear; curves expand and merge."""

        impostor_curve = self._impostor_curve
        impostor_fill  = self._impostor_fill
        genuine_curve  = self._genuine_curve
        genuine_fill   = self._genuine_fill
        impostor_label = self._impostor_label
        genuine_label  = self._genuine_label

        # ── Reality tag (corner) ───────────────────────────────────────────────
        reality_tag = Text("Thực tế (Reality)", font=FONT, font_size=18,
                           color=HYPERPLANE_COLOR).to_corner(UR, buff=0.5)
        self.play(FadeIn(reality_tag, shift=LEFT * 0.3), run_time=0.8)

        # ── Event 1: Noisy genuine dot (low score – left tail of genuine curve)
        # Score ~0.2 → maps to x ≈ 0.2  (well inside the confusion zone)
        NOISY_X = 0.4   # axes x-coord
        noisy_dot = Dot(axes.c2p(NOISY_X, 0), color=GENUINE_COLOR, radius=0.11)
        noisy_icon = make_noisy_icon(0.58).move_to(axes.c2p(NOISY_X, PEAK_Y + 0.85))
        noisy_lbl  = Text("Ảnh nhiễu (Noise)", font=FONT, font_size=13, color="#888888"
                          ).next_to(noisy_icon, DOWN, buff=0.08)

        self.play(FadeIn(noisy_dot, scale=0.4), run_time=0.5)
        self.play(GrowFromCenter(noisy_icon), FadeIn(noisy_lbl), run_time=0.9)
        self.wait(0.4)

        # ── Event 2: Spoof impostor dot (high score – right tail of impostor curve)
        SPOOF_X = -0.5
        spoof_dot = Dot(axes.c2p(SPOOF_X, 0), color=IMPOSTOR_COLOR, radius=0.11)
        spoof_icon = make_spoof_icon(0.55).move_to(axes.c2p(SPOOF_X, PEAK_Y + 0.95))
        spoof_lbl  = Text("Tấn công giả mạo (Spoof)", font=FONT, font_size=13,
                          color=IMPOSTOR_COLOR).next_to(spoof_icon, DOWN, buff=0.08)

        self.play(FadeIn(spoof_dot, scale=0.4), run_time=0.5)
        self.play(GrowFromCenter(spoof_icon), FadeIn(spoof_lbl), run_time=0.9)
        self.wait(0.5)

        # ── Curves merge inward (distributions now overlap) ────────────────────
        self.play(
            mu_imp.animate.set_value(-1.0),
            mu_gen.animate.set_value( 1.0),
            sigma_imp.animate.set_value(SIGMA_REAL),
            sigma_gen.animate.set_value(SIGMA_REAL),
            # Sink edge-case icon heads as curves move (they stay at fixed coords)
            self._gen_icon.animate.set_opacity(0.25),
            self._imp_icon.animate.set_opacity(0.25),
            self._gen_icon_label.animate.set_opacity(0.25),
            self._imp_icon_label.animate.set_opacity(0.25),
            run_time=2.8, rate_func=smooth,
        )
        self.wait(0.4)

        # ── Overlap fill (dynamic min) ─────────────────────────────────────────
        overlap_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                min_func(
                    lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
                    lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
                ),
                x_range=[X_MIN, X_MAX],
            ),
            color=OVERLAP_COLOR, opacity=0.45,
        ))
        self.play(FadeIn(overlap_fill), run_time=0.9)

        # Overlap zone label + arrow pointing to intersection
        inter_pt = axes.c2p(0.0, scaled_gaussian(0.0, -1.0, SIGMA_REAL))
        confusion_text = Text("Vùng nhiễu (Overlap Zone)", font=FONT,
                              font_size=17, color=OVERLAP_COLOR
                              ).move_to(axes.c2p(0.0, PEAK_Y + 0.48))
        confusion_arrow = Arrow(
            confusion_text.get_bottom() + DOWN * 0.08,
            inter_pt + UP * 0.05,
            color=OVERLAP_COLOR, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.08,
        )
        confusion_group = VGroup(confusion_text, confusion_arrow)
        self.play(FadeIn(confusion_text), Create(confusion_arrow), run_time=0.8)
        for _ in range(2):
            self.play(confusion_group.animate.set_opacity(0.22),
                      run_time=0.35, rate_func=there_and_back)
        self.wait(0.9)
        self.play(FadeOut(confusion_group), run_time=0.5)

        # Fade edge-case dots/icons now that the overlap message is clear
        self.play(
            FadeOut(noisy_dot),  FadeOut(noisy_icon),  FadeOut(noisy_lbl),
            FadeOut(spoof_dot),  FadeOut(spoof_icon),  FadeOut(spoof_lbl),
            FadeOut(self._gen_icon), FadeOut(self._imp_icon),
            FadeOut(self._gen_icon_label), FadeOut(self._imp_icon_label),
            FadeOut(reality_tag),
            run_time=0.9,
        )
        self.wait(0.3)

        return (impostor_curve, impostor_fill,
                genuine_curve,  genuine_fill,
                impostor_label, genuine_label,
                overlap_fill)

    # =========================================================================
    # PHASE 3  —  Threshold Dilemma
    # =========================================================================
    def _phase3_threshold(
        self,
        axes: Axes,
        mu_imp:   ValueTracker,
        mu_gen:   ValueTracker,
        sigma_imp: ValueTracker,
        sigma_gen: ValueTracker,
        impostor_curve, impostor_fill,
        genuine_curve,  genuine_fill,
        impostor_label, genuine_label,
        overlap_fill,
    ) -> None:
        """η threshold sweeps left (FAR/spoof) then right (FRR/noise)."""

        eta = ValueTracker(0.0)

        threshold_line = always_redraw(lambda: DashedLine(
            start=axes.c2p(eta.get_value(), 0),
            end=axes.c2p(eta.get_value(), 1.12),
            color=THRESHOLD_COLOR, stroke_width=3, dash_length=0.10,
        ))
        eta_label = MathTex(r"\eta", color=THRESHOLD_COLOR, font_size=38)
        eta_label.add_updater(
            lambda m: m.next_to(axes.c2p(eta.get_value(), 1.12), UP, buff=0.13)
        )

        self.play(Create(threshold_line), FadeIn(eta_label), run_time=1.0)
        self.wait(0.5)

        # ── FAR scenario: η moves LEFT → impostors slip through (spoof attack) ─
        # Fill = impostor area to the RIGHT of η
        far_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
                x_range=[max(eta.get_value(), X_MIN), X_MAX],
            ),
            x_range=[max(eta.get_value(), X_MIN), X_MAX],
            color=IMPOSTOR_COLOR, opacity=0.58,
        ))

        # Spoof icon at the right tail of the impostor curve (score slightly right of η)
        spoof_tail_icon = make_spoof_icon(0.50)
        spoof_tail_icon.move_to(axes.c2p(1.4, PEAK_Y + 0.82))
        spoof_tail_lbl = Text("Spoof vượt ngưỡng!", font=FONT, font_size=12,
                              color=IMPOSTOR_COLOR).next_to(spoof_tail_icon, DOWN, buff=0.06)

        far_badge = _make_warning_badge("False Accept\n(Nhận nhầm kẻ gian)", IMPOSTOR_COLOR)
        far_badge.move_to(axes.c2p(3.9, 0.88))

        self.play(FadeIn(far_fill), run_time=0.35)
        self.play(eta.animate.set_value(-1.5), run_time=2.2, rate_func=smooth)

        # Arrow tip points into impostor tail strictly RIGHT of η
        ETA_FAR = -1.5
        far_arrow = Arrow(
            far_badge.get_bottom(),
            axes.c2p(ETA_FAR + 1.4, 0.04),
            color=IMPOSTOR_COLOR, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.14,
        )
        self.play(
            FadeIn(far_badge), Create(far_arrow),
            GrowFromCenter(spoof_tail_icon), FadeIn(spoof_tail_lbl),
            run_time=0.8,
        )
        self.wait(1.4)
        self.play(
            FadeOut(far_fill), FadeOut(far_badge), FadeOut(far_arrow),
            FadeOut(spoof_tail_icon), FadeOut(spoof_tail_lbl),
            run_time=0.6,
        )

        # ── FRR scenario: η moves RIGHT → genuine users rejected (noisy capture)
        # Fill = genuine area to the LEFT of η
        frr_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
                x_range=[X_MIN, min(eta.get_value(), X_MAX)],
            ),
            x_range=[X_MIN, min(eta.get_value(), X_MAX)],
            color=GENUINE_COLOR, opacity=0.58,
        ))

        # Noisy icon at the left tail of the genuine curve
        noisy_tail_icon = make_noisy_icon(0.50)
        noisy_tail_icon.move_to(axes.c2p(-1.3, PEAK_Y + 0.82))
        noisy_tail_lbl = Text("Ảnh nhiễu bị từ chối!", font=FONT, font_size=12,
                              color=GENUINE_COLOR).next_to(noisy_tail_icon, DOWN, buff=0.06)

        frr_badge = _make_warning_badge("False Reject\n(Từ chối người thật)", GENUINE_COLOR)
        frr_badge.move_to(axes.c2p(-3.9, 0.88))

        self.play(FadeIn(frr_fill), run_time=0.35)
        self.play(eta.animate.set_value(1.5), run_time=2.2, rate_func=smooth)

        ETA_FRR = 1.5
        frr_arrow = Arrow(
            frr_badge.get_bottom(),
            axes.c2p(ETA_FRR - 1.4, 0.04),
            color=GENUINE_COLOR, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.14,
        )
        self.play(
            FadeIn(frr_badge), Create(frr_arrow),
            GrowFromCenter(noisy_tail_icon), FadeIn(noisy_tail_lbl),
            run_time=0.8,
        )
        self.wait(1.4)
        self.play(
            FadeOut(frr_fill), FadeOut(frr_badge), FadeOut(frr_arrow),
            FadeOut(noisy_tail_icon), FadeOut(noisy_tail_lbl),
            run_time=0.6,
        )

        # ── Reset η to centre for visual closure ───────────────────────────────
        self.play(eta.animate.set_value(0.0), run_time=1.0, rate_func=smooth)
        self.wait(0.5)

        # ── Clear all 1D elements before transition ────────────────────────────
        impostor_label.clear_updaters()
        genuine_label.clear_updaters()
        eta_label.clear_updaters()

        self.play(
            FadeOut(threshold_line), FadeOut(eta_label),
            FadeOut(overlap_fill),
            FadeOut(impostor_curve), FadeOut(impostor_fill),
            FadeOut(genuine_curve),  FadeOut(genuine_fill),
            FadeOut(impostor_label), FadeOut(genuine_label),
            run_time=1.2,
        )

    # =========================================================================
    # PHASE 4  —  2D Transition
    # =========================================================================
    def _phase4_transition(self, axes: Axes, x_axis_label) -> None:
        """Y-axis grows from origin, signalling the 1D → 2D expansion."""

        # Y-axis line + arrowhead
        y_axis_line = Line(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, 1.12),
            color=WHITE, stroke_width=2, stroke_opacity=0.65,
        )
        y_axis_arrow = Triangle(
            color=WHITE, fill_color=WHITE, fill_opacity=1,
        ).scale(0.075).rotate(PI).next_to(y_axis_line, UP, buff=0)

        # Y-axis label, rotated and pinned
        y_label = Text("Fingerprint Score", font=FONT, font_size=15, color=SLATE_GRAY)
        y_label.rotate(PI / 2).next_to(y_axis_arrow, LEFT, buff=0.18).shift(DOWN * 0.45)

        self.play(
            Create(y_axis_line), FadeIn(y_axis_arrow),
            FadeIn(y_label, shift=RIGHT * 0.2),
            run_time=1.6,
        )
        self.wait(1.0)

        # Fade to black — Scene 3 begins
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)
