"""System Limitations — Real-world trade-offs of biometric fusion.

Three cinematic infographic segments:
  7A. Hardware Cost   — balance scale tipping under sensor & coin weight.
  7B. Processing Latency — animated progress bars: single-modal vs multi-modal.
  7C. Privacy Risk    — data-stream honeypot visualization with server flash.
"""

import os
import sys

import numpy as np
from manim import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import (
        GENUINE_COLOR, IMPOSTOR_COLOR, HYPERPLANE_COLOR,
        BG_COLOR, FONT_MAIN, SLATE_GRAY,
        CLASS_A_COLOR, CLASS_B_COLOR,
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

try:
    from utils.visual_helpers import make_genuine_icon, make_fingerprint_icon
except ImportError:
    def make_genuine_icon(size=0.6):
        return Circle(radius=size * 0.5, color=GENUINE_COLOR, stroke_width=2)

    def make_fingerprint_icon(size=0.6):
        return Circle(radius=size * 0.5, color="#AABBFF", stroke_width=2)


# ── Scene-local design tokens ─────────────────────────────────────────────────
FONT       = FONT_MAIN
SPOOF_RED  = "#FF2222"
GOLD_COLOR = "#FFD700"
FP_COLOR   = "#AABBFF"


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class SystemLimitationsScene(ThreeDScene):
    """Standalone scene covering three real-world biometric fusion limitations."""

    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        self._show_cost_animation()
        self._show_latency_animation()
        self._show_privacy_animation()

    # =========================================================================
    # 7A — Hardware Cost
    # =========================================================================
    def _show_cost_animation(self) -> None:
        """Balance scale: sensor hardware vs monetary cost."""
        title = Text("Limitation 1 — Hardware Cost",
                     font=FONT, font_size=24, weight=BOLD, color=HYPERPLANE_COLOR)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.5)

        # Balance scale skeleton
        fulcrum    = Triangle(fill_color=SLATE_GRAY, fill_opacity=0.8, stroke_width=0
                              ).scale(0.28).move_to(DOWN * 0.6)
        beam       = Line(LEFT * 2.4 + DOWN * 0.3, RIGHT * 2.4 + DOWN * 0.3,
                          color=SLATE_GRAY, stroke_width=4)
        left_wire  = Line(LEFT  * 2.4 + DOWN * 0.3, LEFT  * 2.4 + DOWN * 1.1,
                          color=SLATE_GRAY, stroke_width=1.5)
        right_wire = Line(RIGHT * 2.4 + DOWN * 0.3, RIGHT * 2.4 + DOWN * 1.1,
                          color=SLATE_GRAY, stroke_width=1.5)
        left_pan   = Line(LEFT  * 2.8 + DOWN * 1.1, LEFT  * 1.95 + DOWN * 1.1,
                          color=SLATE_GRAY, stroke_width=3)
        right_pan  = Line(RIGHT * 1.95 + DOWN * 1.1, RIGHT * 2.8 + DOWN * 1.1,
                          color=SLATE_GRAY, stroke_width=3)

        scale_parts = VGroup(fulcrum, beam, left_wire, right_wire, left_pan, right_pan)
        self.add_fixed_in_frame_mobjects(scale_parts)
        self.play(FadeIn(scale_parts), run_time=0.6)

        # Left pan: biometric sensor icons
        cam = make_genuine_icon(0.35).move_to(LEFT * 2.42 + DOWN * 1.48)
        fp  = make_fingerprint_icon(0.32).move_to(LEFT * 2.0 + DOWN * 1.48)
        self.add_fixed_in_frame_mobjects(cam, fp)
        self.play(FadeIn(cam), FadeIn(fp), run_time=0.5)

        # Right pan: gold coins drop one by one
        coin_targets = [
            RIGHT * 2.1 + DOWN * 1.48,
            RIGHT * 2.5 + DOWN * 1.48,
            RIGHT * 2.3 + DOWN * 1.48,
            RIGHT * 2.1 + DOWN * 1.82,
            RIGHT * 2.5 + DOWN * 1.82,
        ]
        coins = VGroup(*[
            Circle(radius=0.16, fill_color=GOLD_COLOR, fill_opacity=1.0, stroke_width=0)
            .move_to(pos + UP * 2.0)
            for pos in coin_targets
        ])
        self.add_fixed_in_frame_mobjects(coins)
        self.play(
            LaggedStart(*[coins[i].animate.move_to(coin_targets[i])
                          for i in range(len(coin_targets))],
                        lag_ratio=0.2, run_time=1.2, rate_func=rush_into),
        )

        # Scale tips right under the coin weight
        beam_center   = beam.get_center()
        right_balance = VGroup(beam, right_wire, right_pan)
        left_balance  = VGroup(left_wire, left_pan)
        self.play(
            right_balance.animate.rotate(-12 * DEGREES, about_point=beam_center).shift(DOWN * 0.12),
            left_balance.animate.shift(UP * 0.08),
            run_time=0.8, rate_func=smooth,
        )

        note = Text("Hardware Cost scales with number of sensors",
                    font=FONT, font_size=16, color=GOLD_COLOR)
        note.to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(note)
        self.play(FadeIn(note, shift=UP * 0.1), run_time=0.5)
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.8)
        self.remove(title, scale_parts, cam, fp, coins, note)
        self.wait(0.3)

    # =========================================================================
    # 7B — Processing Latency
    # =========================================================================
    def _show_latency_animation(self) -> None:
        """Two progress bars comparing single-modal vs multi-modal pipeline latency."""
        title = Text("Limitation 2 — Processing Latency",
                     font=FONT, font_size=24, weight=BOLD, color=HYPERPLANE_COLOR)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.5)

        bar_w, bar_h = 5.8, 0.52
        y_top, y_bot = 0.6, -0.9

        track_top = Rectangle(width=bar_w, height=bar_h, fill_color="#18191E",
                               fill_opacity=1, stroke_color=SLATE_GRAY, stroke_width=1
                               ).move_to(UP * y_top)
        track_bot = Rectangle(width=bar_w, height=bar_h, fill_color="#18191E",
                               fill_opacity=1, stroke_color=SLATE_GRAY, stroke_width=1
                               ).move_to(UP * y_bot)
        lbl_top = Text("Single Modality", font=FONT, font_size=14, color=GENUINE_COLOR
                        ).next_to(track_top, LEFT, buff=0.25)
        lbl_bot = Text("Multi-modal",     font=FONT, font_size=14, color=IMPOSTOR_COLOR
                        ).next_to(track_bot, LEFT, buff=0.25)

        self.add_fixed_in_frame_mobjects(track_top, track_bot, lbl_top, lbl_bot)
        self.play(FadeIn(track_top), FadeIn(track_bot), FadeIn(lbl_top), FadeIn(lbl_bot),
                  run_time=0.5)

        # Top bar: fill smoothly 0 → full
        bh        = bar_h * 0.82
        bar_top_s = Rectangle(width=0.01, height=bh, fill_color=GENUINE_COLOR,
                               fill_opacity=0.85, stroke_width=0)
        bar_top_s.move_to(np.array([track_top.get_left()[0] + 0.005, y_top, 0]))
        bar_top_f = Rectangle(width=bar_w, height=bh, fill_color=GENUINE_COLOR,
                               fill_opacity=0.85, stroke_width=0)
        bar_top_f.move_to(np.array([track_top.get_left()[0] + bar_w / 2, y_top, 0]))
        self.add_fixed_in_frame_mobjects(bar_top_s)
        self.play(Transform(bar_top_s, bar_top_f), run_time=1.0, rate_func=smooth)

        fast_lbl = Text("✓ 120ms", font=FONT, font_size=14, color=GENUINE_COLOR
                         ).next_to(track_top, RIGHT, buff=0.2)
        self.add_fixed_in_frame_mobjects(fast_lbl)
        self.play(FadeIn(fast_lbl), run_time=0.3)
        self.wait(0.3)

        # Bottom bar: 0 → 50%, stutter, → 100%
        bar_bot_s = Rectangle(width=0.01, height=bh, fill_color=IMPOSTOR_COLOR,
                               fill_opacity=0.85, stroke_width=0)
        bar_bot_s.move_to(np.array([track_bot.get_left()[0] + 0.005, y_bot, 0]))
        bar_bot_h = Rectangle(width=bar_w * 0.5, height=bh, fill_color=IMPOSTOR_COLOR,
                               fill_opacity=0.85, stroke_width=0)
        bar_bot_h.move_to(np.array([track_bot.get_left()[0] + bar_w * 0.25, y_bot, 0]))
        bar_bot_f = Rectangle(width=bar_w, height=bh, fill_color=IMPOSTOR_COLOR,
                               fill_opacity=0.85, stroke_width=0)
        bar_bot_f.move_to(np.array([track_bot.get_left()[0] + bar_w / 2, y_bot, 0]))

        self.add_fixed_in_frame_mobjects(bar_bot_s)
        self.play(Transform(bar_bot_s, bar_bot_h), run_time=1.2, rate_func=smooth)
        self.play(Wiggle(bar_bot_s, scale_value=1.02), run_time=0.4)
        self.wait(0.5)
        self.play(Transform(bar_bot_s, bar_bot_f), run_time=2.5, rate_func=linear)

        slow_lbl = Text("⚠ 380ms", font=FONT, font_size=14, color=IMPOSTOR_COLOR
                         ).next_to(track_bot, RIGHT, buff=0.2)
        self.add_fixed_in_frame_mobjects(slow_lbl)
        self.play(FadeIn(slow_lbl), run_time=0.3)
        self.wait(1.0)

        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.8)
        self.remove(title, track_top, track_bot, lbl_top, lbl_bot,
                    bar_top_s, bar_bot_s, fast_lbl, slow_lbl)
        self.wait(0.3)

    # =========================================================================
    # 7C — Privacy & Security Risk
    # =========================================================================
    def _show_privacy_animation(self) -> None:
        """Two biometric data streams converge into a central server — honeypot risk."""
        title = Text("Limitation 3 — Privacy & Security Risk",
                     font=FONT, font_size=24, weight=BOLD, color=HYPERPLANE_COLOR)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.5)

        # Central server box
        server = Rectangle(width=1.8, height=1.2, fill_color="#0E1420", fill_opacity=1.0,
                           stroke_color=CLASS_B_COLOR, stroke_width=2).move_to(DOWN * 0.2)
        server_lbl = Text("SERVER", font=FONT, font_size=13, color=CLASS_B_COLOR
                           ).move_to(server)
        lights = VGroup(*[
            Dot(radius=0.075, color=c).move_to(
                server.get_top() + DOWN * 0.18 + RIGHT * (i - 1) * 0.32)
            for i, c in enumerate([GENUINE_COLOR, HYPERPLANE_COLOR, SPOOF_RED])
        ])
        self.add_fixed_in_frame_mobjects(server, server_lbl, lights)
        self.play(FadeIn(server), FadeIn(server_lbl), FadeIn(lights), run_time=0.6)

        # Blinking boot sequence
        for _ in range(2):
            self.play(lights.animate.set_opacity(0.15), run_time=0.25)
            self.play(lights.animate.set_opacity(1.0),  run_time=0.25)

        # Data source labels
        face_src = Text("Face Data",        font=FONT, font_size=15, color=CLASS_B_COLOR
                         ).move_to(LEFT * 4.2 + DOWN * 0.2)
        fp_src   = Text("Fingerprint Data", font=FONT, font_size=15, color=FP_COLOR
                         ).move_to(RIGHT * 4.2 + DOWN * 0.2)
        self.add_fixed_in_frame_mobjects(face_src, fp_src)
        self.play(FadeIn(face_src), FadeIn(fp_src), run_time=0.4)

        # Particle streams from both sources converging into the server
        n           = 8
        l_particles = VGroup(*[Dot(LEFT  * 4.2 + DOWN * 0.2, radius=0.07, color=CLASS_B_COLOR) for _ in range(n)])
        r_particles = VGroup(*[Dot(RIGHT * 4.2 + DOWN * 0.2, radius=0.07, color=FP_COLOR)      for _ in range(n)])
        self.add_fixed_in_frame_mobjects(l_particles, r_particles)

        self.play(
            LaggedStart(*[p.animate.move_to(server.get_left())  for p in l_particles], lag_ratio=0.12),
            LaggedStart(*[p.animate.move_to(server.get_right()) for p in r_particles], lag_ratio=0.12),
            run_time=2.5,
        )
        self.play(FadeOut(l_particles), FadeOut(r_particles), run_time=0.15)

        # Server warning flash — honeypot risk signal
        for _ in range(3):
            self.play(server.animate.set_stroke(color=SPOOF_RED, width=4.5), run_time=0.18)
            self.play(server.animate.set_stroke(color=CLASS_B_COLOR, width=2.0), run_time=0.18)

        risk_lbl = Text("Data Honeypot Risk — All biometric data centralized",
                        font=FONT, font_size=16, color=SPOOF_RED)
        risk_bg  = SurroundingRectangle(risk_lbl, fill_color=BG_COLOR, fill_opacity=0.9,
                                        stroke_color=SPOOF_RED, stroke_width=1.5,
                                        corner_radius=0.1, buff=0.14)
        risk_group = VGroup(risk_bg, risk_lbl)
        self.add_fixed_in_frame_mobjects(risk_group)
        risk_group.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(risk_group, shift=UP * 0.1), run_time=0.5)
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=1.2)
        self.remove(title, server, server_lbl, lights, face_src, fp_src,
                    l_particles, r_particles, risk_group)
        self.wait(0.5)
