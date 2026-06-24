"""Unibiometric scene for the single-modal thresholding narrative."""

from manim import *
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import (
        GENUINE_COLOR, IMPOSTOR_COLOR, OVERLAP_COLOR,
        THRESHOLD_COLOR, BG_COLOR, FONT_MAIN, SLATE_GRAY,
        HYPERPLANE_COLOR,
    )
except ImportError:
    GENUINE_COLOR    = "#2ECC71"
    IMPOSTOR_COLOR   = "#E74C3C"
    OVERLAP_COLOR    = "#9B59B6"
    THRESHOLD_COLOR  = "#F39C12"
    BG_COLOR         = "#0B0C10"
    FONT_MAIN        = "Montserrat"
    SLATE_GRAY       = "#888888"
    HYPERPLANE_COLOR = "#F9DC5C"

try:
    from utils.math_helpers import gaussian, min_func
except ImportError:
    def gaussian(x: float, mu: float, sigma: float) -> float:
        """Normalised Gaussian probability density function (PDF)."""
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    def min_func(f, g):
        """Return a callable that evaluates min(f(x), g(x)) point-wise."""
        return lambda x: min(f(x), g(x))

SIGMA_IDEAL    = 1.0
SIGMA_REAL     = 1.3
X_MIN, X_MAX   = -7, 7
PEAK_Y         = 1.0       # Gaussian peak in data-y units
GAUSSIAN_SCALE = 2.5       # raw PDF peak (~0.398) × scale -> ~1.0
FONT           = FONT_MAIN

ICON_LIFT      = 1.10      
LABEL_Y_WORLD  = 3.20      


def scaled_gaussian(x: float, mu: float, sigma: float) -> float:
    """Return a Gaussian PDF scaled to match the scene layout."""
    return gaussian(x, mu, sigma) * GAUSSIAN_SCALE


try:
    from utils.visual_helpers import (
        make_genuine_icon, make_impostor_icon,
        make_noisy_icon, make_spoof_icon,
    )
except ImportError:
    from manim import Ellipse, Dot, Arc, Line, RoundedRectangle, Rectangle, VGroup

    def make_genuine_icon(size=0.6):
        head = Ellipse(
            width=size * 1.2, height=size * 1.5,
            stroke_color=GENUINE_COLOR, stroke_width=2,
            fill_color="#1A1D27", fill_opacity=1,
        )
        eye_l = Dot(LEFT * size * 0.25 + UP * size * 0.20, radius=size * 0.06, color=GENUINE_COLOR)
        eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.20, radius=size * 0.06, color=GENUINE_COLOR)
        mouth = Arc(
            radius=size * 0.30,
            start_angle=-10 * DEGREES,
            angle=-160 * DEGREES,
            color=GENUINE_COLOR,
            stroke_width=2,
        ).shift(DOWN * size * 0.20)
        return VGroup(head, eye_l, eye_r, mouth)

    def make_impostor_icon(size=0.6):
        head = Ellipse(
            width=size * 1.2, height=size * 1.5,
            stroke_color=IMPOSTOR_COLOR, stroke_width=2,
            fill_color="#1A1D27", fill_opacity=1,
        )
        eye_l = Dot(LEFT * size * 0.25 + UP * size * 0.20, radius=size * 0.06, color=IMPOSTOR_COLOR)
        eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.20, radius=size * 0.06, color=IMPOSTOR_COLOR)
        mouth = Line(
            LEFT * size * 0.20, RIGHT * size * 0.30,
            color=IMPOSTOR_COLOR, stroke_width=2,
        ).shift(DOWN * size * 0.20).rotate(15 * DEGREES)
        return VGroup(head, eye_l, eye_r, mouth)

    def make_noisy_icon(size=0.6):
        face = make_genuine_icon(size)
        face.set_color("#888888")
        lines = VGroup(*[
            Line(
                LEFT * size * 0.75, RIGHT * size * 0.75,
                color=WHITE, stroke_width=1.2, stroke_opacity=0.65,
            ).shift(UP * size * dy)
            for dy in [-0.20, 0.05, 0.32]
        ])
        return VGroup(face, lines)

    def make_spoof_icon(size=0.6):
        phone = RoundedRectangle(
            width=size * 1.6, height=size * 2.4, corner_radius=0.10,
            stroke_color=WHITE, stroke_width=2,
            fill_color="#000000", fill_opacity=1,
        )
        screen = Rectangle(
            width=size * 1.35, height=size * 1.90,
            stroke_color="#333333", stroke_width=1,
            fill_color="#111122", fill_opacity=1,
        )
        face_on_screen = make_genuine_icon(size*0.65)
        return VGroup(phone, screen, face_on_screen)


def _make_warning_badge(text_str: str, color: ManimColor) -> VGroup:
    """Create a warning badge with a matching outline."""
    label = Text(text_str, font=FONT, font_size=16, color=color)
    bg = SurroundingRectangle(
        label,
        fill_color=BLACK, fill_opacity=0.85,
        stroke_color=color, stroke_width=1.3,
        corner_radius=0.13, buff=0.15,
    )
    return VGroup(bg, label)


class UnibiometricsScene(Scene):
    """Visualizes the limitations of unibiometric systems using 1D thresholding.

    Shows the transition from an ideal separated state to a realistic overlapping state,
    demonstrating the trade-off between False Accept Rate (FAR) and False Reject Rate (FRR).
    """

    def construct(self) -> None:
        """Build and play the scene in sequence."""
        self.camera.background_color = BG_COLOR

        axes = Axes(
            x_range=[X_MIN, X_MAX, 1],
            y_range=[0, 1.20, 0.5],
            x_length=9.2,
            y_length=3.2,
            axis_config={"stroke_width": 2, "color": WHITE, "stroke_opacity": 0.60},
            y_axis_config={"stroke_opacity": 0},
            tips=False,
        ).shift(DOWN * 1.3)

        x_axis_label = Text(
            "Face Match Score", font=FONT, font_size=15, color=SLATE_GRAY,
        ).next_to(axes.x_axis, DOWN, buff=0.22).align_to(axes.x_axis, RIGHT)

        self.play(FadeIn(axes), FadeIn(x_axis_label), run_time=1.0)
        self.wait(0.2)

        mu_imp    = ValueTracker(-3.0)
        mu_gen    = ValueTracker( 3.0)
        sigma_imp = ValueTracker(SIGMA_IDEAL)
        sigma_gen = ValueTracker(SIGMA_IDEAL)

        self._phase1_ideal(axes, mu_imp, mu_gen, sigma_imp, sigma_gen)

        out = self._phase2_reality(axes, mu_imp, mu_gen, sigma_imp, sigma_gen)
        imp_curve, imp_fill, gen_curve, gen_fill, imp_lbl, gen_lbl, ovlp_fill = out

        self._phase3_threshold(
            axes, mu_imp, mu_gen, sigma_imp, sigma_gen,
            imp_curve, imp_fill, gen_curve, gen_fill, imp_lbl, gen_lbl, ovlp_fill,
        )
        self._phase4_transition(axes, x_axis_label)

    def _phase1_ideal(
        self, axes: Axes,
        mu_imp: ValueTracker, mu_gen: ValueTracker,
        sigma_imp: ValueTracker, sigma_gen: ValueTracker,
    ) -> None:
        """Show the ideal case with clearly separated distributions."""
        imp_curve = always_redraw(lambda: axes.plot(
            lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
            x_range=[X_MIN, X_MAX], color=IMPOSTOR_COLOR, stroke_width=3))
        gen_curve = always_redraw(lambda: axes.plot(
            lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
            x_range=[X_MIN, X_MAX], color=GENUINE_COLOR, stroke_width=3))
        
        imp_fill = always_redraw(lambda: axes.get_area(
            axes.plot(lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
                      x_range=[X_MIN, X_MAX]),
            color=IMPOSTOR_COLOR, opacity=0.22))
        gen_fill = always_redraw(lambda: axes.get_area(
            axes.plot(lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
                      x_range=[X_MIN, X_MAX]),
            color=GENUINE_COLOR, opacity=0.22))

        self._imp_curve = imp_curve
        self._imp_fill  = imp_fill
        self._gen_curve = gen_curve
        self._gen_fill  = gen_fill

        imp_lbl = Text("Impostors", font=FONT, font_size=20, color=IMPOSTOR_COLOR)
        imp_lbl.add_updater(lambda m: m.move_to(
            np.array([axes.c2p(mu_imp.get_value() - 1.2, 0)[0], LABEL_Y_WORLD, 0])
        ))
        
        gen_lbl = Text("Genuine Users", font=FONT, font_size=20, color=GENUINE_COLOR)
        gen_lbl.add_updater(lambda m: m.move_to(
            np.array([axes.c2p(mu_gen.get_value() + 1.2, 0)[0], LABEL_Y_WORLD, 0])
        ))
        
        self._imp_lbl = imp_lbl
        self._gen_lbl = gen_lbl

        icon_size = 0.45
        gen_icon = make_genuine_icon(icon_size)
        gen_icon.move_to(axes.c2p(3.0, PEAK_Y) + UP * ICON_LIFT)
        gen_icon_lbl = Text(
            "Genuine", font=FONT, font_size=13, color=GENUINE_COLOR,
        ).next_to(gen_icon, DOWN, buff=0.08)

        imp_icon = make_impostor_icon(icon_size)
        imp_icon.move_to(axes.c2p(-3.0, PEAK_Y) + UP * ICON_LIFT)
        imp_icon_lbl = Text(
            "Impostor", font=FONT, font_size=13, color=IMPOSTOR_COLOR,
        ).next_to(imp_icon, DOWN, buff=0.08)

        self._gen_icon     = gen_icon
        self._gen_icon_lbl = gen_icon_lbl
        self._imp_icon     = imp_icon
        self._imp_icon_lbl = imp_icon_lbl

        self.play(
            Create(imp_curve), Create(gen_curve),
            FadeIn(imp_fill),  FadeIn(gen_fill),
            FadeIn(imp_lbl),   FadeIn(gen_lbl),
            run_time=2.0,
        )
        self.play(
            GrowFromCenter(gen_icon), GrowFromCenter(imp_icon),
            FadeIn(gen_icon_lbl),    FadeIn(imp_icon_lbl),
            run_time=1.2,
        )
        self.wait(1.5)

    def _phase2_reality(
        self, axes: Axes,
        mu_imp: ValueTracker, mu_gen: ValueTracker,
        sigma_imp: ValueTracker, sigma_gen: ValueTracker,
    ):
        """Introduce noisy and spoofed samples, then widen the overlap."""
        imp_curve = self._imp_curve
        imp_fill  = self._imp_fill
        gen_curve = self._gen_curve
        gen_fill  = self._gen_fill

        reality_tag = Text("Reality", font=FONT, font_size=18,
                           color=HYPERPLANE_COLOR).to_corner(UR, buff=0.5)
        self.play(FadeIn(reality_tag, shift=LEFT * 0.3), run_time=0.8)

        self.play(
            FadeOut(self._gen_icon), FadeOut(self._gen_icon_lbl),
            FadeOut(self._imp_icon), FadeOut(self._imp_icon_lbl),
            run_time=0.5
        )

        noisy_x = 3.5
        spoof_x = -3.5

        noisy_dot  = Dot(axes.c2p(noisy_x, 0), color=GENUINE_COLOR, radius=0.11)
        noisy_ring = Circle(radius=0.18, color=GENUINE_COLOR,
                            stroke_width=1.5, stroke_opacity=0.45
                            ).move_to(axes.c2p(noisy_x, 0))
        noisy_icon = make_noisy_icon(0.40)
        noisy_icon.move_to(axes.c2p(noisy_x, PEAK_Y) + UP * ICON_LIFT)
        noisy_lbl = Text(
            "Noisy Sample", font=FONT, font_size=13, color="#888888",
        ).next_to(noisy_icon, DOWN, buff=0.08)

        self.play(
            FadeIn(noisy_ring, scale=0.5), FadeIn(noisy_dot, scale=0.5), run_time=0.55)
        self.play(GrowFromCenter(noisy_icon), FadeIn(noisy_lbl), run_time=0.90)
        self.wait(0.4)

        spoof_dot  = Dot(axes.c2p(spoof_x, 0), color=IMPOSTOR_COLOR, radius=0.11)
        spoof_ring = Circle(radius=0.18, color=IMPOSTOR_COLOR,
                            stroke_width=1.5, stroke_opacity=0.45
                            ).move_to(axes.c2p(spoof_x, 0))
        spoof_icon = make_spoof_icon(0.40)
        spoof_icon.move_to(axes.c2p(spoof_x, PEAK_Y) + UP * ICON_LIFT)
        spoof_lbl = Text(
            "Spoof Attack", font=FONT, font_size=13,
            color=IMPOSTOR_COLOR,
        ).next_to(spoof_icon, DOWN, buff=0.08)

        self.play(
            FadeIn(spoof_ring, scale=0.5), FadeIn(spoof_dot, scale=0.5), run_time=0.55)
        self.play(GrowFromCenter(spoof_icon), FadeIn(spoof_lbl), run_time=0.90)
        self.wait(0.5)

        self.play(
            mu_imp.animate.set_value(-1.0),
            mu_gen.animate.set_value( 1.0),
            sigma_imp.animate.set_value(SIGMA_REAL),
            sigma_gen.animate.set_value(SIGMA_REAL),
            run_time=2.8, rate_func=smooth,
        )
        self.wait(0.4)

        ovlp_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                min_func(
                    lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
                    lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
                ),
                x_range=[X_MIN, X_MAX],
            ),
            color=OVERLAP_COLOR, opacity=0.45,
        ))
        self.play(FadeIn(ovlp_fill), run_time=0.90)

        inter_world_y = axes.c2p(0.0, scaled_gaussian(0.0, -1.0, SIGMA_REAL))[1]
        confusion_text = Text(
            "Confusion Zone", font=FONT, font_size=15, color=OVERLAP_COLOR,
        ).move_to(axes.c2p(0.0, PEAK_Y + 0.25))
        
        confusion_arrow = Arrow(
            confusion_text.get_bottom() + DOWN * 0.05,
            np.array([axes.c2p(0.0, 0)[0], inter_world_y + 0.05, 0]),
            color=OVERLAP_COLOR, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.15,
        )
        confusion_group = VGroup(confusion_text, confusion_arrow)
        self.play(FadeIn(confusion_text), Create(confusion_arrow), run_time=0.8)
        
        for _ in range(2):
            self.play(
                confusion_group.animate.set_opacity(0.22),
                run_time=0.35,
                rate_func=there_and_back,
            )
        self.wait(0.9)
        self.play(FadeOut(confusion_group), run_time=0.5)

        self.play(
            FadeOut(noisy_dot), FadeOut(noisy_ring),
            FadeOut(noisy_icon), FadeOut(noisy_lbl),
            FadeOut(spoof_dot), FadeOut(spoof_ring),
            FadeOut(spoof_icon), FadeOut(spoof_lbl),
            FadeOut(reality_tag),
            run_time=0.9,
        )
        self.wait(0.3)

        return (imp_curve, imp_fill, gen_curve, gen_fill,
                self._imp_lbl, self._gen_lbl, ovlp_fill)

    def _phase3_threshold(
        self, axes: Axes,
        mu_imp: ValueTracker, mu_gen: ValueTracker,
        sigma_imp: ValueTracker, sigma_gen: ValueTracker,
        imp_curve, imp_fill, gen_curve, gen_fill,
        imp_lbl, gen_lbl, ovlp_fill,
    ) -> None:
        """Explore the decision-threshold trade-off and its failure modes."""
        eta = ValueTracker(0.0)

        threshold_line = always_redraw(lambda: DashedLine(
            axes.c2p(eta.get_value(), 0),
            axes.c2p(eta.get_value(), 1.12),
            color=THRESHOLD_COLOR, stroke_width=3, dash_length=0.10,
        ))
        eta_lbl = MathTex(r"\eta", color=THRESHOLD_COLOR, font_size=38)
        eta_lbl.add_updater(
            lambda m: m.next_to(axes.c2p(eta.get_value(), 1.12), UP, buff=0.14)
        )

        self.play(Create(threshold_line), FadeIn(eta_lbl), run_time=1.0)
        self.wait(0.5)

        ETA_FAR = -1.5

        far_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                lambda x: scaled_gaussian(x, mu_imp.get_value(), sigma_imp.get_value()),
                x_range=[max(eta.get_value(), X_MIN), X_MAX],
            ),
            x_range=[max(eta.get_value(), X_MIN), X_MAX],
            color=IMPOSTOR_COLOR, opacity=0.55,
        ))

        spoof_tail = make_spoof_icon(0.38)
        spoof_tail.move_to(axes.c2p(0.5, PEAK_Y) + UP * ICON_LIFT)
        spoof_tail_lbl = Text("Spoof accepted!", font=FONT, font_size=12,
                               color=IMPOSTOR_COLOR).next_to(spoof_tail, DOWN, buff=0.08)

        far_badge = _make_warning_badge("False Accept\n(FAR Error)", IMPOSTOR_COLOR)
        far_badge.move_to(axes.c2p(5.5, 0.85))

        self.play(FadeIn(far_fill), run_time=0.35)
        self.play(eta.animate.set_value(ETA_FAR), run_time=2.2, rate_func=smooth)

        far_arrow = Arrow(
            far_badge.get_left(),
            axes.c2p(-0.7, 0.08),
            color=IMPOSTOR_COLOR, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.15,
        )
        self.play(
            FadeIn(far_badge), Create(far_arrow),
            GrowFromCenter(spoof_tail), FadeIn(spoof_tail_lbl),
            run_time=0.8,
        )
        self.wait(1.4)
        self.play(
            FadeOut(far_fill), FadeOut(far_badge), FadeOut(far_arrow),
            FadeOut(spoof_tail), FadeOut(spoof_tail_lbl),
            run_time=0.6,
        )

        ETA_FRR = 1.5

        frr_fill = always_redraw(lambda: axes.get_area(
            axes.plot(
                lambda x: scaled_gaussian(x, mu_gen.get_value(), sigma_gen.get_value()),
                x_range=[X_MIN, min(eta.get_value(), X_MAX)],
            ),
            x_range=[X_MIN, min(eta.get_value(), X_MAX)],
            color=GENUINE_COLOR, opacity=0.55,
        ))

        noisy_tail = make_noisy_icon(0.38)
        noisy_tail.move_to(axes.c2p(-0.5, PEAK_Y) + UP * ICON_LIFT)
        noisy_tail_lbl = Text("Genuine rejected!", font=FONT, font_size=12,
                               color=GENUINE_COLOR).next_to(noisy_tail, DOWN, buff=0.08)

        frr_badge = _make_warning_badge("False Reject\n(FRR Error)", GENUINE_COLOR)
        frr_badge.move_to(axes.c2p(-5.5, 0.85))

        self.play(FadeIn(frr_fill), run_time=0.35)
        self.play(eta.animate.set_value(ETA_FRR), run_time=2.2, rate_func=smooth)

        frr_arrow = Arrow(
            frr_badge.get_right(),
            axes.c2p(0.7, 0.08),
            color=GENUINE_COLOR, stroke_width=2.2,
            max_tip_length_to_length_ratio=0.15,
        )
        self.play(
            FadeIn(frr_badge), Create(frr_arrow),
            GrowFromCenter(noisy_tail), FadeIn(noisy_tail_lbl),
            run_time=0.8,
        )
        self.wait(1.4)
        self.play(
            FadeOut(frr_fill), FadeOut(frr_badge), FadeOut(frr_arrow),
            FadeOut(noisy_tail), FadeOut(noisy_tail_lbl),
            run_time=0.6,
        )

        self.play(eta.animate.set_value(0.0), run_time=1.0, rate_func=smooth)
        self.wait(0.5)

        imp_lbl.clear_updaters()
        gen_lbl.clear_updaters()
        eta_lbl.clear_updaters()

        self.play(
            FadeOut(threshold_line), FadeOut(eta_lbl),
            FadeOut(ovlp_fill),
            FadeOut(imp_curve), FadeOut(imp_fill),
            FadeOut(gen_curve),  FadeOut(gen_fill),
            FadeOut(imp_lbl),    FadeOut(gen_lbl),
            run_time=1.2,
        )

    def _phase4_transition(self, axes: Axes, x_axis_label: Text) -> None:
        """Bridge from 1D failure to the full pipeline question.

        Shows that adding a 2nd modality is NOT just stacking axes — we need
        to understand HOW scores are produced and normalized first.
        This transition leads naturally into Part 1.5 (Pipeline Walkthrough).
        """
        # ── Step 1: grow the Y axis to suggest 2D space ───────────────────
        y_axis_arrow = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, 1.22),
            color=WHITE, stroke_width=2.5, buff=0,
            max_tip_length_to_length_ratio=0.10,
        )
        y_label = Text(
            "Fingerprint Score", font=FONT, font_size=15, color=SLATE_GRAY,
        ).rotate(PI / 2).next_to(y_axis_arrow, LEFT, buff=0.25).shift(DOWN * 0.2)

        self.play(
            GrowArrow(y_axis_arrow),
            FadeIn(y_label, shift=RIGHT * 0.15),
            run_time=1.4,
        )
        self.wait(0.4)

        # ── Step 2: a point "pops in" as if it just got both scores ────────
        demo_dot = Dot(axes.c2p(1.5, 0.0), color="#2ECC71", radius=0.09)
        self.play(FadeIn(demo_dot, scale=0.4), run_time=0.5)

        tracking_line = always_redraw(lambda: DashedLine(
            start=axes.c2p(1.5, 0),
            end=demo_dot.get_center(),
            color="#2ECC71", stroke_width=1.4, dash_length=0.05,
        ).set_opacity(0.55))
        self.add(tracking_line)

        self.play(
            demo_dot.animate.move_to(axes.c2p(1.5, 0.88)),
            run_time=1.8, rate_func=smooth,
        )
        self.wait(0.5)

        # ── Step 3: "Wait — but WHERE do these scores come from?" ──────────
        question_box = RoundedRectangle(
            width=7.8, height=1.05, corner_radius=0.18,
            fill_color=BG_COLOR, fill_opacity=0.94,
            stroke_color="#F9DC5C", stroke_width=1.5,
        ).to_edge(UP, buff=0.15)

        q_line1 = Text(
            "But where do these scores come from?",
            font=FONT, font_size=26, color="#F9DC5C",
        ).move_to(question_box.get_center() + UP * 0.16)

        q_line2 = Text(
            "And why normalize before fusion?",
            font=FONT, font_size=17, color=SLATE_GRAY,
        ).move_to(question_box.get_center() + DOWN * 0.22)

        self.play(
            FadeIn(question_box, shift=DOWN * 0.12),
            run_time=0.55,
        )
        self.play(Write(q_line1), run_time=0.9)
        self.play(FadeIn(q_line2, shift=DOWN * 0.10), run_time=0.6)
        self.wait(1.6)

        # ── Step 4: fade everything — cut to Part 1.5 ─────────────────────
        tracking_line.clear_updaters()
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.4,
        )
        self.wait(0.5)