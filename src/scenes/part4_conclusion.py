"""System Limitations & Conclusion scene."""

import os
import sys
import numpy as np
from manim import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants & Colors ────────────────────────────────────────────────────────
GENUINE_COLOR    = "#2ECC71"
IMPOSTOR_COLOR   = "#E74C3C"
HYPERPLANE_COLOR = "#F9DC5C"
BG_COLOR         = "#0B0C10"
FONT_MAIN        = "Roboto"
SLATE_GRAY       = "#888888"
CLASS_A_COLOR    = "#FF5E5E"  # Red class color matching Part 0
CLASS_B_COLOR    = "#00C2D1"  # Blue class color
SPOOF_RED        = "#FF2222"
GOLD_COLOR       = "#FFD700"
FP_COLOR         = "#AABBFF"
VOICE_COLOR      = "#D291FF"

# ── Helper Icons (Vector drawn for standalone execution) ──────────────────────
def make_genuine_icon(size=0.6):
    head = Ellipse(width=size*1.2, height=size*1.5, stroke_color=CLASS_B_COLOR, stroke_width=2, fill_color="#1A1D27", fill_opacity=1)
    eye_l = Dot(LEFT * size * 0.25 + UP * size * 0.2, radius=size*0.06, color=CLASS_B_COLOR)
    eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.2, radius=size*0.06, color=CLASS_B_COLOR)
    mouth = Arc(radius=size*0.3, start_angle=-10*DEGREES, angle=-160*DEGREES, color=CLASS_B_COLOR, stroke_width=2).shift(DOWN*size*0.2)
    return VGroup(head, eye_l, eye_r, mouth)

def make_fingerprint_icon(size=0.6):
    arcs = VGroup(*[Arc(radius=r, start_angle=180*DEGREES, angle=-270*DEGREES, stroke_color=FP_COLOR, stroke_width=2) 
                    for r in np.linspace(size*0.2, size*0.6, 4)])
    return arcs

def make_mic_icon(size=0.6):
    pill = RoundedRectangle(width=size*0.4, height=size*0.8, corner_radius=size*0.2, stroke_color=VOICE_COLOR, stroke_width=2)
    holder = Arc(radius=size*0.3, start_angle=180*DEGREES, angle=-180*DEGREES, stroke_color=VOICE_COLOR, stroke_width=2).shift(DOWN*size*0.1)
    base = Line(DOWN*size*0.4, DOWN*size*0.7, color=VOICE_COLOR, stroke_width=2)
    return VGroup(pill, holder, base)

def make_stopwatch(size=0.8):
    body = Circle(radius=size, stroke_color=WHITE, stroke_width=3, fill_color="#18191E", fill_opacity=1)
    button = Rectangle(width=size*0.3, height=size*0.15, fill_color=WHITE, fill_opacity=1).next_to(body, UP, buff=0)
    ticks = VGroup(*[Line(UP*size*0.8, UP*size*0.95, color=WHITE, stroke_width=2).rotate(a, about_point=ORIGIN) for a in np.linspace(0, TAU, 12, endpoint=False)])
    hand = Line(ORIGIN, UP*size*0.7, color=HYPERPLANE_COLOR, stroke_width=3)
    dot = Dot(radius=0.08, color=WHITE)
    return VGroup(body, button, ticks, hand, dot), hand

def make_hacker_skull(size=0.7):
    skull = Circle(radius=size*0.5, color=SPOOF_RED, fill_color=SPOOF_RED, fill_opacity=0.2)
    jaw = Rectangle(width=size*0.5, height=size*0.3, color=SPOOF_RED).next_to(skull, DOWN, buff=-size*0.15)
    eyes = VGroup(Line(UL*0.1, DR*0.1, color=WHITE, stroke_width=3), Line(UR*0.1, DL*0.1, color=WHITE, stroke_width=3))
    eyes_l = eyes.copy().move_to(skull.get_center() + LEFT*size*0.2)
    eyes_r = eyes.copy().move_to(skull.get_center() + RIGHT*size*0.2)
    return VGroup(skull, jaw, eyes_l, eyes_r)

# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class SystemLimitationsScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        self._show_cost_animation()
        self._show_latency_animation()
        self._show_privacy_animation()
        self._show_wrap_up()
        self._show_callback_to_part0()

    # =========================================================================
    # Micro-animation 1: Hardware Cost (Balance scale with multiple sensors)
    # =========================================================================
    def _show_cost_animation(self) -> None:
        title = Text("1. High Hardware/Deployment Cost", font=FONT_MAIN, font_size=28, weight=BOLD, color=HYPERPLANE_COLOR).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.15))

        # Balance scale structure setup
        fulcrum = Triangle(fill_color=SLATE_GRAY, fill_opacity=0.8, stroke_width=0).scale(0.4).move_to(DOWN * 1.5)
        # Temporary line to determine exact pivot center
        temp_beam = Line(LEFT * 3, RIGHT * 3).next_to(fulcrum, UP, buff=0)
        pivot = temp_beam.get_center()

        # Static versions for initial presentation
        beam_static = Line(LEFT * 3, RIGHT * 3, color=SLATE_GRAY, stroke_width=5).move_to(pivot)
        left_attach_static = beam_static.point_from_proportion(0.05)
        right_attach_static = beam_static.point_from_proportion(0.95)

        left_side_static = VGroup(
            VGroup(
                Line(left_attach_static, left_attach_static + LEFT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2),
                Line(left_attach_static, left_attach_static + RIGHT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2)
            ),
            Line(left_attach_static + LEFT * 1.0 + DOWN * 1.5, left_attach_static + RIGHT * 1.0 + DOWN * 1.5, stroke_width=4, color=SLATE_GRAY)
        )

        right_side_static = VGroup(
            VGroup(
                Line(right_attach_static, right_attach_static + LEFT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2),
                Line(right_attach_static, right_attach_static + RIGHT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2)
            ),
            Line(right_attach_static + LEFT * 1.0 + DOWN * 1.5, right_attach_static + RIGHT * 1.0 + DOWN * 1.5, stroke_width=4, color=SLATE_GRAY)
        )

        self.play(FadeIn(fulcrum), Create(beam_static), Create(left_side_static), Create(right_side_static))

        # Dynamic State Trackers
        angle_tracker = ValueTracker(0.0)

        # Dynamic Redrawn Objects
        beam = always_redraw(lambda:
            Line(LEFT * 3, RIGHT * 3, color=SLATE_GRAY, stroke_width=5)
            .move_to(pivot)
            .rotate(angle_tracker.get_value(), about_point=pivot)
        )

        left_side = always_redraw(lambda:
            VGroup(
                VGroup(
                    Line(beam.point_from_proportion(0.05), beam.point_from_proportion(0.05) + LEFT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2),
                    Line(beam.point_from_proportion(0.05), beam.point_from_proportion(0.05) + RIGHT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2)
                ),
                Line(beam.point_from_proportion(0.05) + LEFT * 1.0 + DOWN * 1.5, beam.point_from_proportion(0.05) + RIGHT * 1.0 + DOWN * 1.5, stroke_width=4, color=SLATE_GRAY)
            )
        )

        right_side = always_redraw(lambda:
            VGroup(
                VGroup(
                    Line(beam.point_from_proportion(0.95), beam.point_from_proportion(0.95) + LEFT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2),
                    Line(beam.point_from_proportion(0.95), beam.point_from_proportion(0.95) + RIGHT * 0.8 + DOWN * 1.5, color=SLATE_GRAY, stroke_width=2)
                ),
                Line(beam.point_from_proportion(0.95) + LEFT * 1.0 + DOWN * 1.5, beam.point_from_proportion(0.95) + RIGHT * 1.0 + DOWN * 1.5, stroke_width=4, color=SLATE_GRAY)
            )
        )

        # Swap static with dynamic
        self.remove(beam_static, left_side_static, right_side_static)
        self.add(beam, left_side, right_side)

        def get_left_pan_center():
            return beam.point_from_proportion(0.05) + DOWN * 1.5

        def get_right_pan_center():
            return beam.point_from_proportion(0.95) + DOWN * 1.5

        # Drop 1: Camera
        cam_offset_y = ValueTracker(3.0)
        cam = always_redraw(lambda:
            make_genuine_icon(0.4).move_to(
                get_left_pan_center() + LEFT * 0.5 + UP * 0.2 + UP * cam_offset_y.get_value()
            )
        )
        self.add(cam)
        self.play(cam_offset_y.animate.set_value(0.0), run_time=0.5, rate_func=rush_into)
        self.play(angle_tracker.animate.set_value(4 * DEGREES), run_time=0.3)

        # Drop 2: Fingerprint
        fp_offset_y = ValueTracker(3.0)
        fp = always_redraw(lambda:
            make_fingerprint_icon(0.4).move_to(
                get_left_pan_center() + RIGHT * 0.5 + UP * 0.2 + UP * fp_offset_y.get_value()
            )
        )
        self.add(fp)
        self.play(fp_offset_y.animate.set_value(0.0), run_time=0.5, rate_func=rush_into)
        self.play(angle_tracker.animate.set_value(10 * DEGREES), run_time=0.3)

        # Drop 3: Microphone
        mic_offset_y = ValueTracker(3.0)
        mic = always_redraw(lambda:
            make_mic_icon(0.4).move_to(
                get_left_pan_center() + UP * 0.4 + UP * mic_offset_y.get_value()
            )
        )
        self.add(mic)
        self.play(mic_offset_y.animate.set_value(0.0), run_time=0.5, rate_func=rush_into)
        self.play(angle_tracker.animate.set_value(20 * DEGREES), run_time=0.4)

        # Coins setup
        coin_offsets = [
            LEFT * 0.4 + UP * 0.18,
            RIGHT * 0.4 + UP * 0.18,
            ORIGIN + UP * 0.18,
            LEFT * 0.2 + UP * 0.5,
            RIGHT * 0.2 + UP * 0.5,
        ]
        
        coin_trackers = [ValueTracker(3.0) for _ in range(5)]
        coins = VGroup()
        for i in range(5):
            coin_redraw = always_redraw(
                lambda idx=i:
                    VGroup(
                        Circle(radius=0.18, fill_color=GOLD_COLOR, fill_opacity=1, stroke_width=2, stroke_color=WHITE),
                        Text("$", font=FONT_MAIN, font_size=16, color=BLACK)
                    ).move_to(get_right_pan_center() + coin_offsets[idx] + UP * coin_trackers[idx].get_value())
            )
            coins.add(coin_redraw)

        # Drop coins with realistic bounce effects
        for i in range(5):
            self.add(coins[i])
            self.play(coin_trackers[i].animate.set_value(0.0), run_time=0.25, rate_func=rush_into)
            self.play(coin_trackers[i].animate.set_value(0.15), run_time=0.1, rate_func=slow_into)
            self.play(coin_trackers[i].animate.set_value(0.0), run_time=0.1, rate_func=rush_into)

        # Scale balances back to level
        self.play(
            angle_tracker.animate.set_value(0.0),
            run_time=1.2,
            rate_func=smooth
        )

        caption = Text("Multiple Sensors = Higher Hardware Cost", font=FONT_MAIN, font_size=18, color=GOLD_COLOR).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP*0.2))
        self.wait(1.5)
        
        self.play(*[FadeOut(m) for m in self.mobjects])

    # =========================================================================
    # Micro-animation 2: Processing Time & Latency
    # =========================================================================
    def _show_latency_animation(self) -> None:
        title = Text("2. Increased Processing Time", font=FONT_MAIN, font_size=28, weight=BOLD, color=HYPERPLANE_COLOR).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.15))

        # Stopwatch
        watch, hand = make_stopwatch(size=0.6)
        watch.to_edge(RIGHT, buff=1.0).shift(UP*0.5)
        self.play(FadeIn(watch))

        # Progress Bars
        bar_w = 6.0
        bar_h = 0.4
        
        # Unibiometric
        lbl1 = Text("Unibiometric:", font=FONT_MAIN, font_size=16, color=GENUINE_COLOR).to_edge(LEFT, buff=1.0).shift(UP*0.5)
        bg1 = Rectangle(width=bar_w, height=bar_h, stroke_color=SLATE_GRAY, fill_color="#18191E", fill_opacity=1).next_to(lbl1, RIGHT, buff=0.5)
        fill1 = Rectangle(width=0.01, height=bar_h, stroke_width=0, fill_color=GENUINE_COLOR, fill_opacity=1).align_to(bg1, LEFT).align_to(bg1, UP)
        
        # Multibiometric
        lbl2 = Text("Multi + SVM:", font=FONT_MAIN, font_size=16, color=HYPERPLANE_COLOR).to_edge(LEFT, buff=1.0).shift(DOWN*1.0)
        bg2 = Rectangle(width=bar_w, height=bar_h, stroke_color=SLATE_GRAY, fill_color="#18191E", fill_opacity=1).next_to(lbl2, RIGHT, buff=0.5)
        fill2 = Rectangle(width=0.01, height=bar_h, stroke_width=0, fill_color=HYPERPLANE_COLOR, fill_opacity=1).align_to(bg2, LEFT).align_to(bg2, UP)

        self.play(FadeIn(lbl1), FadeIn(bg1), FadeIn(lbl2), FadeIn(bg2))
        self.add(fill1, fill2)

        # Unibiometric — fills quickly
        fill1_full = Rectangle(width=bar_w, height=bar_h, stroke_width=0,
                               fill_color=GENUINE_COLOR, fill_opacity=1
                               ).align_to(bg1, LEFT).align_to(bg1, UP)
        self.play(
            Transform(fill1, fill1_full),
            Rotate(hand, angle=-PI / 2, about_point=watch[0].get_center()),
            run_time=0.5,
        )
        flash1 = Text("Access Granted!", font=FONT_MAIN, font_size=16, weight=BOLD, color=GENUINE_COLOR).next_to(bg1, DOWN)
        self.play(FadeIn(flash1))

        # Reset watch hand
        self.play(Rotate(hand, angle=PI / 2, about_point=watch[0].get_center()), run_time=0.2)

        # Multibiometric — 3 stages
        stage_txt = Text("1. Extracting 3 features...", font=FONT_MAIN, font_size=16, color=WHITE).next_to(bg2, DOWN)

        # Stage 1 (30%)
        fill2_30 = Rectangle(width=bar_w * 0.3, height=bar_h, stroke_width=0,
                              fill_color=HYPERPLANE_COLOR, fill_opacity=1
                              ).align_to(bg2, LEFT).align_to(bg2, UP)
        self.play(
            Transform(fill2, fill2_30),
            Rotate(hand, angle=-PI, about_point=watch[0].get_center()),
            Write(stage_txt),
            run_time=1.0,
        )

        # Stage 2 (60%)
        stage2_txt = Text("2. Score Normalization...", font=FONT_MAIN, font_size=16, color=WHITE).next_to(bg2, DOWN)
        fill2_60 = Rectangle(width=bar_w * 0.6, height=bar_h, stroke_width=0,
                              fill_color=HYPERPLANE_COLOR, fill_opacity=1
                              ).align_to(bg2, LEFT).align_to(bg2, UP)
        self.play(
            Transform(fill2, fill2_60),
            Rotate(hand, angle=-PI, about_point=watch[0].get_center()),
            ReplacementTransform(stage_txt, stage2_txt),
            run_time=0.8,
        )

        # Stage 3 (100%)
        stage3_txt = Text("3. SVM RBF Fusion Processing...", font=FONT_MAIN, font_size=16, color=WHITE).next_to(bg2, DOWN)
        fill2_full = Rectangle(width=bar_w, height=bar_h, stroke_width=0,
                               fill_color=HYPERPLANE_COLOR, fill_opacity=1
                               ).align_to(bg2, LEFT).align_to(bg2, UP)
        self.play(
            Transform(fill2, fill2_full),
            Rotate(hand, angle=-PI * 1.5, about_point=watch[0].get_center()),
            ReplacementTransform(stage2_txt, stage3_txt),
            run_time=1.5,
        )

        flash2 = Text("Access Granted!", font=FONT_MAIN, font_size=16, weight=BOLD, color=GENUINE_COLOR).next_to(bg2, DOWN)
        self.play(ReplacementTransform(stage3_txt, flash2))

        caption = Text("Increased Latency due to Multi-modal Sync", font=FONT_MAIN, font_size=18, color=IMPOSTOR_COLOR).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP * 0.2))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])


    # =========================================================================
    # Micro-animation 3: Privacy & Honeypot Target
    # =========================================================================
    def _show_privacy_animation(self) -> None:
        # Change background tone to indicate threat
        bg_rect = Rectangle(width=config.frame_width, height=config.frame_height, fill_color="#1A0505", fill_opacity=1, stroke_width=0)
        self.add(bg_rect)
        
        title = Text("3. Privacy Risk (Honeypot Target)", font=FONT_MAIN, font_size=28, weight=BOLD, color=SPOOF_RED).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.15))

        # Central Server
        server = Rectangle(width=2.5, height=3.5, fill_color="#0E1420", fill_opacity=1.0, stroke_color=CLASS_B_COLOR, stroke_width=3)
        server_lbl = Text("CENTRAL SERVER\n(Super-profile)", font=FONT_MAIN, font_size=14, color=WHITE).move_to(server)
        self.play(Create(server), Write(server_lbl))

        # Data Streams (Binary)
        face_stream = Text("011001...", font="Monospace", font_size=16, color=CLASS_B_COLOR).move_to(LEFT*4 + UP*1)
        fp_stream = Text("101100...", font="Monospace", font_size=16, color=FP_COLOR).move_to(LEFT*4)
        voice_stream = Text("001011...", font="Monospace", font_size=16, color=VOICE_COLOR).move_to(LEFT*4 + DOWN*1)

        face_icon = make_genuine_icon(0.3).next_to(face_stream, LEFT)
        fp_icon = make_fingerprint_icon(0.3).next_to(fp_stream, LEFT)
        voice_icon = make_mic_icon(0.3).next_to(voice_stream, LEFT)

        self.play(FadeIn(VGroup(face_icon, fp_icon, voice_icon)))
        
        # Streams flowing into server
        self.play(
            face_stream.animate.move_to(server.get_left() + UP*1).set_opacity(0),
            fp_stream.animate.move_to(server.get_left()).set_opacity(0),
            voice_stream.animate.move_to(server.get_left() + DOWN*1).set_opacity(0),
            run_time=1.5, rate_func=linear
        )

        # Expand and glow server to represent a honeypot target
        server_glow = server.copy().set_stroke(color=HYPERPLANE_COLOR, width=10).set_opacity(0.3)
        self.play(
            server.animate.scale(1.2).set_stroke(color=HYPERPLANE_COLOR),
            server_lbl.animate.scale(1.2).set_color(HYPERPLANE_COLOR),
            FadeIn(server_glow),
            run_time=1.0
        )

        # Hacker attack line
        hacker = make_hacker_skull(1.0).move_to(RIGHT*4)
        attack_line = Line(hacker.get_left(), server.get_right(), color=SPOOF_RED, stroke_width=4)
        self.play(FadeIn(hacker, shift=LEFT*0.5), Create(attack_line), run_time=0.5)

        # Camera shake and threat flash
        self.play(
            Wiggle(server, scale_value=1.1, rotation_angle=0.05*TAU),
            bg_rect.animate.set_fill(color="#330000"),
            run_time=0.6
        )

        # Data Leak & Red Crosses
        np.random.seed(42)  # Set seed for deterministic leakage pattern
        leak_particles = VGroup(*[Dot(server.get_center(), color=c) for c in [CLASS_B_COLOR, FP_COLOR, VOICE_COLOR]*5])
        leak_anims = [p.animate.move_to(server.get_center() + np.random.uniform(-4, 4, 3)).set_opacity(0) for p in leak_particles]
        
        crosses = VGroup(*[
            VGroup(Line(UL*0.2, DR*0.2, color=SPOOF_RED, stroke_width=4), Line(UR*0.2, DL*0.2, color=SPOOF_RED, stroke_width=4)).move_to(icon)
            for icon in [face_icon, fp_icon, voice_icon]
        ])

        self.play(
            AnimationGroup(*leak_anims),
            FadeIn(crosses),
            run_time=1.5
        )

        caption = Text("Biometrics cannot be reset like passwords if compromised", font=FONT_MAIN, font_size=18, color=SPOOF_RED).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(caption, shift=UP*0.2))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects])

    # =========================================================================
    # Phase 4: The Wrap-up (Comparison Split Screen)
    # =========================================================================
    def _show_wrap_up(self) -> None:
        self.camera.background_color = BG_COLOR
        
        # Split Screen Line
        divider = Line(UP * 4, DOWN * 4, color=SLATE_GRAY, stroke_opacity=0.5)
        self.play(Create(divider))

        # --- LEFT: 1D Overlap Vulnerability ---
        left_lbl = Text("1D Unimodal: Highly Vulnerable", font=FONT_MAIN, font_size=18, color=IMPOSTOR_COLOR).to_edge(UL, buff=0.5).shift(RIGHT*1)
        
        # 1D Overlap Gaussians (Simplified)
        ax_l = Axes(x_range=[-3, 3], y_range=[0, 1], x_length=4, y_length=2).shift(LEFT*3.5 + DOWN*0.5)
        curve1 = ax_l.plot(lambda x: np.exp(-x**2/0.5), color=IMPOSTOR_COLOR, stroke_width=3)
        curve2 = ax_l.plot(lambda x: np.exp(-(x-1.5)**2/0.5), color=GENUINE_COLOR, stroke_width=3)
        overlap_area = ax_l.get_area(curve1, x_range=[0.75, 3], color=SPOOF_RED, opacity=0.5)
        
        self.play(FadeIn(left_lbl), Create(ax_l), Create(curve1), Create(curve2))
        self.play(FadeIn(overlap_area))
        
        # --- RIGHT: RBF SVM Shield ---
        right_lbl = Text("Multi-modal + SVM RBF: Protective Shield", font=FONT_MAIN, font_size=18, color=GENUINE_COLOR).to_edge(UR, buff=0.5).shift(LEFT*0.5)
        
        # 2D RBF (Simplified rings)
        ax_r = Axes(x_range=[-2, 2], y_range=[-2, 2], x_length=3.5, y_length=3.5).shift(RIGHT*3.5 + DOWN*0.5)
        np.random.seed(42)  # Set seed for deterministic point generation
        green_core = VGroup(*[Dot(ax_r.c2p(np.random.uniform(-0.5,0.5), np.random.uniform(-0.5,0.5)), color=GENUINE_COLOR) for _ in range(15)])
        red_ring = VGroup(*[Dot(ax_r.c2p(1.5*np.cos(a), 1.5*np.sin(a)), color=IMPOSTOR_COLOR) for a in np.linspace(0, TAU, 20, endpoint=False)])
        
        # Shield
        shield_glow = Circle(radius=1.2, color=HYPERPLANE_COLOR, stroke_width=15, stroke_opacity=0.3).move_to(ax_r.c2p(0,0))
        shield_ring = Circle(radius=1.2, color=HYPERPLANE_COLOR, stroke_width=3).move_to(ax_r.c2p(0,0))
        
        self.play(FadeIn(right_lbl), Create(ax_r), FadeIn(green_core), FadeIn(red_ring))
        self.play(Create(shield_ring), FadeIn(shield_glow))
        self.play(Flash(shield_ring, color=HYPERPLANE_COLOR, flash_radius=1.5, num_lines=12))

        self.wait(1.0)

        # --- Transition: Fade left, expand right ---
        self.play(
            FadeOut(left_lbl), FadeOut(ax_l), FadeOut(curve1), FadeOut(curve2), FadeOut(overlap_area), FadeOut(divider),
            right_lbl.animate.move_to(UP*3),
            VGroup(ax_r, green_core, red_ring, shield_glow, shield_ring).animate.move_to(ORIGIN).scale(1.5),
            run_time=1.5, rate_func=smooth
        )

        final_msg = Text("A Worthy Trade-off for Multi-dimensional Security", font=FONT_MAIN, font_size=26, weight=BOLD, color=WHITE).to_edge(DOWN, buff=1.0)
        final_bg = SurroundingRectangle(final_msg, fill_color=BLACK, fill_opacity=0.8, stroke_color=HYPERPLANE_COLOR, stroke_width=2)
        
        self.play(FadeIn(final_bg), Write(final_msg))
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)

    # =========================================================================
    # Phase 5: Narrative Callback to Part 0 (XOR Trap Failure)
    # =========================================================================
    def _show_callback_to_part0(self) -> None:
        self.camera.background_color = BG_COLOR
        
        # 2D Grid / Scatter plot axes for XOR representation
        axes_2d = Axes(
            x_range=[-4, 4], y_range=[-4, 4],
            x_length=5, y_length=5,
            axis_config={"color": SLATE_GRAY, "stroke_width": 2}
        ).shift(UP * 0.5)

        # Re-generating XOR pattern from Part 0
        def local_generate_cluster(center, n=14, sigma=0.15, seed=None):
            rng = np.random.default_rng(seed)
            pts = rng.normal(loc=np.asarray(center)[:2], scale=sigma, size=(n, 2))
            return [np.array([x, y, 0.0]) for x, y in pts]
            
        b1 = local_generate_cluster(center=[ 1.8,  1.8], seed=42)
        b2 = local_generate_cluster(center=[-1.8, -1.8], seed=43)
        r1 = local_generate_cluster(center=[-1.8,  1.8], seed=44)
        r2 = local_generate_cluster(center=[ 1.8, -1.8], seed=45)

        blue_dots = VGroup(*[Dot(axes_2d.c2p(x, y), color=CLASS_B_COLOR, radius=0.08) for x, y, _ in b1 + b2])
        red_dots = VGroup(*[Dot(axes_2d.c2p(x, y), color=CLASS_A_COLOR, radius=0.08) for x, y, _ in r1 + r2])

        # Brief flash of XOR scatter plot (Part 0 throwback)
        self.play(
            Create(axes_2d),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in blue_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in red_dots], lag_ratio=0.04),
            run_time=2.0
        )
        self.wait(1.0)

        # Text closing the narrative loop
        callback_text = Text(
            "Now you know why the straight line always fails.",
            font=FONT_MAIN, font_size=24, color=HYPERPLANE_COLOR
        ).to_edge(DOWN, buff=1.0)

        callback_bg = SurroundingRectangle(
            callback_text,
            fill_color=BG_COLOR, fill_opacity=0.9,
            stroke_color=HYPERPLANE_COLOR, stroke_width=1.5,
            buff=0.3
        )

        self.play(
            FadeIn(callback_bg, shift=UP * 0.2),
            Write(callback_text),
            run_time=1.5
        )
        self.wait(4.0)

        # Fade out everything to black
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=2.0)
        self.wait(1.0)
