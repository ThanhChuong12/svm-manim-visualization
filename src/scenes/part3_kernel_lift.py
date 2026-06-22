"""Scene 4 — The Kernel Trick (The 3D Shield).

Phase 1 (0:00–0:15): Concentric circles (non-linear) + linear SVM failure.
Phase 2 (0:15–0:35): RBF kernel formula + 3D lift (Gaussian peak).
Phase 3 (0:35–0:55): Laser slicing plane descends + hemisphere dome shield.
Phase 4 (0:55–1:15): Projection back to 2D + circular decision boundary.

Uses ThreeDScene (established pattern from part0_intro.py).
Data generated via sklearn.datasets.make_circles, RBF lift z = e^{-γ(x²+y²)}.
"""

import numpy as np
from manim import *
import sys
import os

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
    from utils.visual_helpers import make_spoof_icon
except ImportError:
    def make_spoof_icon(size=0.6):
        return Square(side_length=size, color=IMPOSTOR_COLOR, stroke_width=2)

try:
    from core.kernels import rbf_z as _rbf_z
    from core.fusion_data import generate_circle_data as _generate_circle_data
except ImportError:
    from sklearn.datasets import make_circles

    def _rbf_z(x, y, gamma=2.0):
        return float(np.exp(-gamma * (x ** 2 + y ** 2)))

    def _generate_circle_data(seed=42):
        X, y = make_circles(n_samples=80, noise=0.08, factor=0.4, random_state=seed)
        X = X * 1.6
        inner = [X[i] for i in range(len(y)) if y[i] == 1]
        outer = [X[i] for i in range(len(y)) if y[i] == 0]
        return inner, outer


# ── Scene-local design tokens ─────────────────────────────────────────────────
FONT              = FONT_MAIN
LASER_YELLOW      = HYPERPLANE_COLOR
SHIELD_GREEN      = "#2ECC71"
SPOOF_RED         = "#FF2222"
ERROR_RED         = "#FF3333"
DOME_COLOR        = "#00E676"
LASER_GLOW_COLOR  = "#FFFFAA"

# RBF kernel parameters
GAMMA             = 2.0
LASER_Z_HEIGHT    = 0.45
N_SAMPLES         = 80
CIRCLE_NOISE      = 0.08
CIRCLE_FACTOR     = 0.4

# Axes ranges
XY_RANGE          = [-2.0, 2.0, 0.5]
Z_RANGE           = [-0.2, 1.4, 0.5]
AXIS_LEN_XY       = 5.5
AXIS_LEN_Z        = 3.5
DOT_RADIUS        = 0.07


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class KernelTrickScene(ThreeDScene):
    """Scene 4: Non-linear Siege → Kernel Lift → Laser Shield → Projection."""

    def construct(self):
        self.camera.background_color = BG_COLOR
        # Start with top-down view (flat 2D perspective)
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        # Generate data once for all phases
        inner_pts, outer_pts = _generate_circle_data()

        # Phase orchestration
        axes_2d, green_dots, red_dots = self._phase1_nonlinear_siege(
            inner_pts, outer_pts,
        )
        axes_3d = self._phase2_kernel_lift(axes_2d, green_dots, red_dots)
        self._phase3_laser_shield(axes_3d, green_dots, red_dots)
        self._phase4_projection_back(axes_3d, green_dots, red_dots)

    # =========================================================================
    # PHASE 1 — The Non-linear Siege
    # =========================================================================
    def _phase1_nonlinear_siege(
        self,
        inner_pts: list,
        outer_pts: list,
    ) -> tuple[Axes, VGroup, VGroup]:
        """Concentric circles + rotating linear SVM → failure."""

        # 2D axes (viewed top-down, so appears flat)
        axes_2d = Axes(
            x_range=XY_RANGE, y_range=XY_RANGE,
            x_length=AXIS_LEN_XY, y_length=AXIS_LEN_XY,
            axis_config={
                "stroke_width": 1.8, "color": WHITE,
                "stroke_opacity": 0.4, "include_ticks": True,
                "tick_size": 0.05,
            },
            tips=False,
        ).shift(DOWN * 0.15)

        self.play(FadeIn(axes_2d), run_time=0.8)

        # Title
        title = Text(
            "Non-linearly Separable Data",
            font=FONT, font_size=22, color=SLATE_GRAY,
        )
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.6)

        # Plot genuine (green, inner circle) and impostor (red, outer ring)
        green_dots = VGroup(*[
            Dot3D(
                point=np.array([pt[0], pt[1], 0.0]),
                color=GENUINE_COLOR, radius=DOT_RADIUS,
            ) for pt in inner_pts
        ])
        red_dots = VGroup(*[
            Dot3D(
                point=np.array([pt[0], pt[1], 0.0]),
                color=IMPOSTOR_COLOR, radius=DOT_RADIUS,
            ) for pt in outer_pts
        ])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in green_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in red_dots], lag_ratio=0.04),
            run_time=1.8,
        )
        self.wait(0.4)

        # Legend
        gen_legend = VGroup(
            Dot(radius=0.06, color=GENUINE_COLOR),
            Text("Genuine (Inner)", font=FONT, font_size=14, color=GENUINE_COLOR),
        ).arrange(RIGHT, buff=0.12)
        imp_legend = VGroup(
            Dot(radius=0.06, color=IMPOSTOR_COLOR),
            Text("Impostor (Outer)", font=FONT, font_size=14, color=IMPOSTOR_COLOR),
        ).arrange(RIGHT, buff=0.12)
        legend = VGroup(gen_legend, imp_legend).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_bg = SurroundingRectangle(
            legend, fill_color=BG_COLOR, fill_opacity=0.85,
            stroke_width=0, buff=0.10,
        )
        legend_group = VGroup(legend_bg, legend)
        self.add_fixed_in_frame_mobjects(legend_group)
        legend_group.to_corner(DR, buff=0.4)
        self.play(FadeIn(legend_group), run_time=0.5)
        self.wait(0.3)

        # ── Rotating linear SVM (doomed to fail) ─────────────────────────────
        hp_line = Line(
            start=axes_2d.c2p(-2.2, 0), end=axes_2d.c2p(2.2, 0),
            color=HYPERPLANE_COLOR, stroke_width=3,
        )
        self.play(Create(hp_line), run_time=0.6)

        angle_tracker = ValueTracker(0.0)
        c1 = ManimColor(HYPERPLANE_COLOR)
        c2 = ManimColor(ERROR_RED)

        def _hp_updater(line: Line) -> None:
            a = angle_tracker.get_value()
            start = np.array([-2.5 * np.cos(a), -2.5 * np.sin(a), 0.0])
            end = np.array([2.5 * np.cos(a), 2.5 * np.sin(a), 0.0])
            line.put_start_and_end_on(start, end)
            # Pulsing red to show constant failure
            t = (np.sin(a * 8) + 1) / 2
            line.set_color(interpolate_color(c1, c2, t))
            line.set_stroke(width=3 + 3 * t)

        hp_line.add_updater(_hp_updater)
        self.play(
            angle_tracker.animate.set_value(PI * 1.2),
            run_time=3.0, rate_func=linear,
        )
        hp_line.remove_updater(_hp_updater)

        # Dim to show failure
        self.play(
            hp_line.animate.set_color(ERROR_RED).set_stroke(opacity=0.3),
            run_time=0.4,
        )

        # ── Failure cross icon ────────────────────────────────────────────────
        cross_size = 0.35
        cross = VGroup(
            Line(UL * cross_size, DR * cross_size, color=ERROR_RED, stroke_width=5),
            Line(UR * cross_size, DL * cross_size, color=ERROR_RED, stroke_width=5),
        )
        fail_text = Text(
            "Linear SVM Failed!", font=FONT, font_size=18,
            weight=BOLD, color=ERROR_RED,
        )
        fail_group = VGroup(cross, fail_text).arrange(RIGHT, buff=0.15)
        fail_bg = SurroundingRectangle(
            fail_group, fill_color=BLACK, fill_opacity=0.85,
            stroke_color=ERROR_RED, stroke_width=1.5,
            corner_radius=0.10, buff=0.15,
        )
        fail_block = VGroup(fail_bg, fail_group)
        self.add_fixed_in_frame_mobjects(fail_block)
        fail_block.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(fail_bg), Create(cross), Write(fail_text), run_time=0.7)
        self.wait(0.7)

        # ── Spoof thumbnails on two specific red dots ─────────────────────────
        if len(red_dots) >= 2:
            spoof_1 = make_spoof_icon(0.22)
            spoof_2 = make_spoof_icon(0.22)

            spoof_1_target = red_dots[0].get_center() + UP * 0.55 + RIGHT * 0.3
            spoof_2_target = red_dots[len(red_dots) // 2].get_center() + UP * 0.55 + LEFT * 0.3

            spoof_1.move_to(spoof_1_target)
            spoof_2.move_to(spoof_2_target)

            sp_label_1 = Text(
                "3D Mask", font=FONT, font_size=10, color=IMPOSTOR_COLOR,
            ).next_to(spoof_1, DOWN, buff=0.05)
            sp_label_2 = Text(
                "Silicone", font=FONT, font_size=10, color=IMPOSTOR_COLOR,
            ).next_to(spoof_2, DOWN, buff=0.05)

            spoof_group = VGroup(spoof_1, spoof_2, sp_label_1, sp_label_2)

            self.play(
                GrowFromCenter(spoof_1), GrowFromCenter(spoof_2),
                FadeIn(sp_label_1), FadeIn(sp_label_2),
                run_time=0.8,
            )
            self.wait(0.6)
            self.play(FadeOut(spoof_group), run_time=0.5)

        # Cleanup Phase 1 overlays
        self.play(
            FadeOut(fail_block), FadeOut(hp_line),
            FadeOut(title), FadeOut(legend_group),
            run_time=0.8,
        )
        self.wait(0.2)

        return axes_2d, green_dots, red_dots

    # =========================================================================
    # PHASE 2 — The Kernel Lift
    # =========================================================================
    def _phase2_kernel_lift(
        self,
        axes_2d: Axes,
        green_dots: VGroup,
        red_dots: VGroup,
    ) -> ThreeDAxes:
        """RBF formula display + camera tilt + dot Z-coordinate animation."""

        # ── RBF formula ───────────────────────────────────────────────────────
        formula = MathTex(
            r"K(\mathbf{x}, \mathbf{l}) = e^{-\gamma \|\mathbf{x} - \mathbf{l}\|^2}",
            font_size=34, color=HYPERPLANE_COLOR,
        )
        formula_label = Text(
            "Radial Basis Function (RBF) Kernel",
            font=FONT, font_size=16, color=SLATE_GRAY,
        )
        formula_group = VGroup(formula, formula_label).arrange(DOWN, buff=0.15)
        formula_bg = SurroundingRectangle(
            formula_group, fill_color=BG_COLOR, fill_opacity=0.90,
            stroke_color=HYPERPLANE_COLOR, stroke_width=1.5,
            corner_radius=0.12, buff=0.20,
        )
        formula_block = VGroup(formula_bg, formula_group)

        self.add_fixed_in_frame_mobjects(formula_block)
        formula_block.to_edge(UP, buff=0.30)
        self.play(
            FadeIn(formula_bg),
            Write(formula), FadeIn(formula_label, shift=UP * 0.1),
            run_time=1.2,
        )
        self.wait(1.0)

        # ── Mapping annotation ────────────────────────────────────────────────
        mapping_tex = MathTex(
            r"z = e^{-\gamma(x^2 + y^2)}, \quad \gamma = 2.0",
            font_size=26, color=SLATE_GRAY,
        )
        self.add_fixed_in_frame_mobjects(mapping_tex)
        mapping_tex.next_to(formula_block, DOWN, buff=0.15)
        self.play(FadeIn(mapping_tex, shift=UP * 0.1), run_time=0.6)
        self.wait(0.6)

        # ── Build 3D axes ─────────────────────────────────────────────────────
        axes_3d = ThreeDAxes(
            x_range=XY_RANGE, y_range=XY_RANGE, z_range=Z_RANGE,
            x_length=AXIS_LEN_XY, y_length=AXIS_LEN_XY, z_length=AXIS_LEN_Z,
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.25},
        ).shift(DOWN * 0.15)

        # Z-axis label
        z_label = Text(
            "RBF Mapping (z)", font=FONT, font_size=13, color=SLATE_GRAY,
        )

        # Transform 2D axes → 3D axes
        self.play(ReplacementTransform(axes_2d, axes_3d), run_time=0.8)

        # ── Compute lift targets ──────────────────────────────────────────────
        lift_anims = []
        trace_lines = VGroup()

        for dots_grp, color in [(green_dots, GENUINE_COLOR), (red_dots, IMPOSTOR_COLOR)]:
            for d in dots_grp:
                cx, cy, _ = d.get_center()
                z_target = _rbf_z(cx, cy) * (Z_RANGE[1] - 0.2)  # Scale to axis range
                target = np.array([cx, cy, z_target])
                trace = DashedLine(
                    d.get_center(), target,
                    color=color, stroke_width=1.2, dash_length=0.08,
                ).set_opacity(0.35)
                trace_lines.add(trace)
                lift_anims.append(d.animate.move_to(target))

        # ── Camera tilt + simultaneous lift ───────────────────────────────────
        self.move_camera(
            phi=70 * DEGREES, theta=-40 * DEGREES,
            run_time=3.5, rate_func=smooth,
            added_anims=[
                LaggedStart(
                    *[Create(t) for t in trace_lines],
                    lag_ratio=0.02, run_time=1.5,
                ),
                AnimationGroup(*lift_anims, run_time=3.5, rate_func=smooth),
            ],
        )

        # Z-axis label placement
        z_label.move_to(axes_3d.c2p(0, 0, Z_RANGE[1] + 0.3))
        self.add_fixed_orientation_mobjects(z_label)
        self.play(FadeIn(z_label), run_time=0.5)

        # Cleanup traces
        self.play(FadeOut(trace_lines), run_time=0.5)
        self.wait(0.3)

        # ── Slow camera rotation to appreciate the 3D shape ───────────────────
        self.move_camera(theta=-25 * DEGREES, run_time=1.5)
        self.move_camera(theta=15 * DEGREES, run_time=2.0, rate_func=smooth)
        self.move_camera(theta=-35 * DEGREES, run_time=1.5, rate_func=smooth)

        # Cleanup formula
        self.play(
            FadeOut(formula_block), FadeOut(mapping_tex), FadeOut(z_label),
            run_time=0.6,
        )
        self.wait(0.2)

        return axes_3d

    # =========================================================================
    # PHASE 3 — The Laser Shield
    # =========================================================================
    def _phase3_laser_shield(
        self,
        axes_3d: ThreeDAxes,
        green_dots: VGroup,
        red_dots: VGroup,
    ) -> None:
        """Laser slicing plane descends + hemisphere dome shield."""

        # ── Title ─────────────────────────────────────────────────────────────
        shield_title = Text(
            "The Laser Plane — Hyperplane in Feature Space",
            font=FONT, font_size=20, color=HYPERPLANE_COLOR,
        )
        self.add_fixed_in_frame_mobjects(shield_title)
        shield_title.to_edge(UP, buff=0.30)
        self.play(FadeIn(shield_title, shift=DOWN * 0.12), run_time=0.6)

        # ── Laser Plane (descending from above) ──────────────────────────────
        plane_half_size = 2.8
        laser_z = ValueTracker(Z_RANGE[1] + 0.2)

        def _make_laser_plane():
            z = laser_z.get_value()
            return Polygon(
                np.array([-plane_half_size, -plane_half_size, z]),
                np.array([plane_half_size, -plane_half_size, z]),
                np.array([plane_half_size, plane_half_size, z]),
                np.array([-plane_half_size, plane_half_size, z]),
                fill_color=LASER_YELLOW, fill_opacity=0.22,
                stroke_color=LASER_YELLOW, stroke_width=2.5,
                stroke_opacity=0.85,
            )

        laser_plane = always_redraw(_make_laser_plane)
        self.add(laser_plane)

        # Glow edge effect (a slightly larger, dimmer version)
        def _make_glow_plane():
            z = laser_z.get_value()
            s = plane_half_size + 0.08
            return Polygon(
                np.array([-s, -s, z]),
                np.array([s, -s, z]),
                np.array([s, s, z]),
                np.array([-s, s, z]),
                fill_color=LASER_GLOW_COLOR, fill_opacity=0.08,
                stroke_color=LASER_GLOW_COLOR, stroke_width=5,
                stroke_opacity=0.25,
            )

        glow_plane = always_redraw(_make_glow_plane)
        self.add(glow_plane)

        # Z-height label
        def _make_z_label():
            z = laser_z.get_value()
            return Text(
                f"z = {z:.2f}", font=FONT, font_size=14, color=HYPERPLANE_COLOR,
            ).move_to(np.array([plane_half_size + 0.5, 0, z]))

        z_height_label = always_redraw(_make_z_label)
        self.add_fixed_orientation_mobjects(z_height_label)
        self.add(z_height_label)

        # ── Descend the laser plane ───────────────────────────────────────────
        target_z = LASER_Z_HEIGHT * (Z_RANGE[1] - 0.2)
        self.play(
            laser_z.animate.set_value(target_z),
            run_time=3.0, rate_func=smooth,
        )
        self.wait(0.4)

        # ── Replace live plane with static ────────────────────────────────────
        self.remove(laser_plane, glow_plane, z_height_label)

        static_plane = Polygon(
            np.array([-plane_half_size, -plane_half_size, target_z]),
            np.array([plane_half_size, -plane_half_size, target_z]),
            np.array([plane_half_size, plane_half_size, target_z]),
            np.array([-plane_half_size, plane_half_size, target_z]),
            fill_color=LASER_YELLOW, fill_opacity=0.22,
            stroke_color=LASER_YELLOW, stroke_width=2.5,
            stroke_opacity=0.85,
        )
        static_glow = Polygon(
            np.array([-(plane_half_size + 0.08), -(plane_half_size + 0.08), target_z]),
            np.array([(plane_half_size + 0.08), -(plane_half_size + 0.08), target_z]),
            np.array([(plane_half_size + 0.08), (plane_half_size + 0.08), target_z]),
            np.array([-(plane_half_size + 0.08), (plane_half_size + 0.08), target_z]),
            fill_color=LASER_GLOW_COLOR, fill_opacity=0.08,
            stroke_color=LASER_GLOW_COLOR, stroke_width=5,
            stroke_opacity=0.25,
        )
        self.add(static_glow, static_plane)

        # ── Hemisphere dome (protective shield over genuine peak) ─────────────
        dome_radius = 1.0
        dome_center_z = target_z

        dome = Surface(
            lambda u, v: np.array([
                dome_radius * np.cos(u) * np.sin(v),
                dome_radius * np.sin(u) * np.sin(v),
                dome_center_z + dome_radius * np.cos(v) * 0.5,
            ]),
            u_range=[0, TAU],
            v_range=[0, PI / 2],
            resolution=(24, 12),
            fill_color=DOME_COLOR, fill_opacity=0.18,
            stroke_color=DOME_COLOR, stroke_width=1.0,
            stroke_opacity=0.4,
            checkerboard_colors=False,
        )
        dome.set_color(DOME_COLOR)

        self.play(FadeIn(dome, run_time=1.5))

        # ── Flash the protected genuine dots ──────────────────────────────────
        self.play(
            *[Flash(d.get_center(), color=GENUINE_COLOR,
                    flash_radius=0.15, num_lines=4) for d in green_dots[:6]],
            run_time=0.8,
        )

        # ── Reject markers on impostor dots below the plane ───────────────────
        reject_markers = VGroup()
        for d in red_dots:
            if d.get_center()[2] < target_z:
                marker_size = 0.08
                marker = VGroup(
                    Line(UL * marker_size, DR * marker_size,
                         color=SPOOF_RED, stroke_width=2),
                    Line(UR * marker_size, DL * marker_size,
                         color=SPOOF_RED, stroke_width=2),
                ).move_to(d.get_center() + UP * 0.15)
                reject_markers.add(marker)

        if len(reject_markers) > 0:
            self.play(
                LaggedStart(
                    *[Create(m) for m in reject_markers],
                    lag_ratio=0.05,
                ),
                run_time=1.0,
            )

        # ── "Shield Active" badge ─────────────────────────────────────────────
        shield_badge = Text(
            "✓ Security Shield Active",
            font=FONT, font_size=18, color=DOME_COLOR,
        )
        shield_badge_bg = SurroundingRectangle(
            shield_badge, fill_color=BLACK, fill_opacity=0.80,
            stroke_color=DOME_COLOR, stroke_width=1.5,
            corner_radius=0.10, buff=0.12,
        )
        shield_badge_block = VGroup(shield_badge_bg, shield_badge)
        self.add_fixed_in_frame_mobjects(shield_badge_block)
        shield_badge_block.to_edge(DOWN, buff=0.45)
        self.play(
            FadeIn(shield_badge_bg), FadeIn(shield_badge, shift=UP * 0.1),
            run_time=0.6,
        )
        self.wait(1.0)

        # Slow rotation to appreciate the shield
        self.move_camera(theta=-50 * DEGREES, run_time=2.0, rate_func=smooth)
        self.move_camera(theta=-20 * DEGREES, run_time=1.5, rate_func=smooth)
        self.wait(0.5)

        # Store for Phase 4 cleanup
        self._static_plane = static_plane
        self._static_glow = static_glow
        self._dome = dome
        self._reject_markers = reject_markers
        self._shield_badge_block = shield_badge_block
        self._shield_title = shield_title
        self._target_z = target_z

    # =========================================================================
    # PHASE 4: The XOR Trap
    # =========================================================================
    def _phase4_xor_trap(self, axes, genuine_cloud, impostor_cloud, hyperplane):
        flash_rect = Rectangle(
            width=config.frame_width + 2, height=config.frame_height + 2, 
            fill_color=SPOOF_RED, fill_opacity=0.0, stroke_width=0
        )
        self.add(flash_rect)
        self.play(flash_rect.animate.set_fill(opacity=0.30), run_time=0.3, rate_func=there_and_back)
        self.remove(flash_rect)

        warning_en = Text("Spoof Attack", font=FONT, font_size=28, weight=BOLD, color=SPOOF_RED).to_edge(UP, buff=0.30)
        warning_vn = Text("XOR Trap!", font=FONT, font_size=22, color=SPOOF_RED).next_to(warning_en, DOWN, buff=0.10)
        warning_bg = SurroundingRectangle(
            VGroup(warning_en, warning_vn), fill_color=BLACK, fill_opacity=0.75, 
            stroke_color=SPOOF_RED, stroke_width=1.5, corner_radius=0.14, buff=0.18
        )
        self.play(FadeIn(warning_bg), Write(warning_en), FadeIn(warning_vn, shift=DOWN * 0.1), run_time=0.8)
        self.wait(0.4)

        spoof_tl_pts = _scatter_2d((0.27, 0.75), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_TL)
        spoof_tl = VGroup(*[Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in spoof_tl_pts])
        
        spoof_br_pts = _scatter_2d((0.72, 0.25), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_BR)
        spoof_br = VGroup(*[Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in spoof_br_pts])

        tl_caption = Text("Silicone fingerprint", font=FONT, font_size=13, color=IMPOSTOR_COLOR)
        tl_caption_bg = SurroundingRectangle(tl_caption, fill_color=BG_COLOR, fill_opacity=0.85, stroke_width=0, buff=0.06)
        tl_cap_group = VGroup(tl_caption_bg, tl_caption).next_to(spoof_tl, RIGHT, buff=0.15)
        
        br_caption = Text("3D face mask", font=FONT, font_size=13, color=IMPOSTOR_COLOR)
        br_caption_bg = SurroundingRectangle(br_caption, fill_color=BG_COLOR, fill_opacity=0.85, stroke_width=0, buff=0.06)
        br_cap_group = VGroup(br_caption_bg, br_caption).next_to(spoof_br, LEFT, buff=0.15)

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_tl], lag_ratio=0.07), 
            LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_br], lag_ratio=0.07), 
            run_time=1.0
        )
        self.play(FadeIn(tl_cap_group, shift=LEFT * 0.12), FadeIn(br_cap_group, shift=RIGHT * 0.12), run_time=0.5)
        self.wait(0.5)

        angle_tracker = ValueTracker(0)
        c_yellow = ManimColor(HYPERPLANE_COLOR)
        c_red = ManimColor(SPOOF_RED)

        def _hp_updater(line):
            a = angle_tracker.get_value()
            c, s = np.cos(a), np.sin(a)
            pts = []
            if abs(c) > 1e-5:
                y0 = 0.5 + (0 - 0.5) * s / c
                if 0 <= y0 <= 1: pts.append((0.0, y0))
                y1 = 0.5 + (1 - 0.5) * s / c
                if 0 <= y1 <= 1: pts.append((1.0, y1))
            if abs(s) > 1e-5:
                x0 = 0.5 + (0 - 0.5) * c / s
                if 0 <= x0 <= 1: pts.append((x0, 0.0))
                x1 = 0.5 + (1 - 0.5) * c / s
                if 0 <= x1 <= 1: pts.append((x1, 1.0))
            
            unique_pts = []
            for p in pts:
                if not any(np.linalg.norm(np.array(p) - np.array(up)) < 1e-4 for up in unique_pts):
                    unique_pts.append(p)
            
            if len(unique_pts) >= 2:
                line.put_start_and_end_on(axes.c2p(*unique_pts[0]), axes.c2p(*unique_pts[1]))
                
            t = (np.sin(a * 6) + 1) / 2
            line.set_color(interpolate_color(c_yellow, c_red, t))
            line.set_stroke(width=3 + 4 * t)

        hyperplane.add_updater(_hp_updater)
        self.play(angle_tracker.animate.set_value(PI * 1.4), run_time=2.5, rate_func=linear)
        hyperplane.remove_updater(_hp_updater)
        
        # ── Hiệu ứng Focus Dimming: Làm tối dữ liệu để làm nổi bật thông báo lỗi ──
        self.play(
            hyperplane.animate.set_color(SPOOF_RED).set_stroke(opacity=0.5),
            VGroup(genuine_cloud, impostor_cloud, spoof_tl, spoof_br, axes).animate.set_opacity(0.3),
            run_time=0.5
        )

        # ── Cinematic Error Modal ──
        cross_size = 0.35
        cross = VGroup(
            Line(UL * cross_size, DR * cross_size, color=SPOOF_RED, stroke_width=5),
            Line(UR * cross_size, DL * cross_size, color=SPOOF_RED, stroke_width=5)
        )
        failure_text = Text("Linear Model Fails!", font=FONT, font_size=24, weight=BOLD, color=SPOOF_RED)
        modal_content = VGroup(cross, failure_text).arrange(RIGHT, buff=0.25)
        
        failure_bg = SurroundingRectangle(
            modal_content, fill_color=BLACK, fill_opacity=0.95, 
            stroke_color=SPOOF_RED, stroke_width=2.5, corner_radius=0.1, buff=0.25
        )
        failure_modal = VGroup(failure_bg, modal_content).move_to(axes.c2p(0.5, 0.5))

        self.play(
            FadeIn(failure_bg, scale=1.1),
            Create(cross),
            Write(failure_text),
            run_time=0.6
        )
        self.play(
            Flash(failure_modal, color=SPOOF_RED, flash_radius=2.2, num_lines=12),
            Wiggle(warning_vn, scale_value=1.2), # Nhấn mạnh chữ XOR Trap!
            run_time=0.6
        )
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(0.5)