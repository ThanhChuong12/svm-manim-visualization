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
            checkerboard_colors=None,
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
    # PHASE 4 — Projection Back to 2D
    # =========================================================================
    def _phase4_projection_back(
        self,
        axes_3d: ThreeDAxes,
        green_dots: VGroup,
        red_dots: VGroup,
    ) -> None:
        """Camera back to top-down + circular boundary projection."""

        target_z = self._target_z

        # ── Clean up 3D overlays ──────────────────────────────────────────────
        self.play(
            FadeOut(self._dome),
            FadeOut(self._static_plane),
            FadeOut(self._static_glow),
            FadeOut(self._reject_markers),
            FadeOut(self._shield_badge_block),
            FadeOut(self._shield_title),
            run_time=1.0,
        )

        # ── Project dots back to z=0 ──────────────────────────────────────────
        flatten_anims = []
        for d in [*green_dots, *red_dots]:
            cx, cy, _ = d.get_center()
            flatten_anims.append(d.animate.move_to(np.array([cx, cy, 0.0])))

        # Camera back to top-down
        self.move_camera(
            phi=0, theta=-90 * DEGREES,
            run_time=3.0, rate_func=smooth,
            added_anims=[
                AnimationGroup(*flatten_anims, run_time=3.0, rate_func=smooth),
            ],
        )
        self.wait(0.4)

        # ── Circular decision boundary ────────────────────────────────────────
        # The intersection of z = exp(-γr²) with z = target_z_normalized gives:
        #   r = sqrt(-ln(z_norm) / γ)
        z_norm = target_z / (Z_RANGE[1] - 0.2)  # Undo the visual scaling
        if z_norm > 0 and z_norm < 1:
            r_boundary = np.sqrt(-np.log(z_norm) / GAMMA)
        else:
            r_boundary = 0.7

        # Scale to match the coordinate system
        boundary_circle = Circle(
            radius=r_boundary * 1.6,  # Match the data scaling
            color=HYPERPLANE_COLOR, stroke_width=3.5,
        ).move_to(ORIGIN)

        # Dashed version for elegance
        boundary_dashed = DashedVMobject(
            boundary_circle, num_dashes=40,
        )

        # Title
        proj_title = Text(
            "Circular Decision Boundary (Projected)",
            font=FONT, font_size=20, color=HYPERPLANE_COLOR,
        )
        self.add_fixed_in_frame_mobjects(proj_title)
        proj_title.to_edge(UP, buff=0.30)
        self.play(
            FadeIn(proj_title, shift=DOWN * 0.12),
            Create(boundary_dashed),
            run_time=1.2,
        )
        self.wait(0.4)

        # ── Highlight support vectors (closest dots to the boundary) ──────────
        def _dist_to_boundary(dot):
            cx, cy, _ = dot.get_center()
            r = np.sqrt(cx ** 2 + cy ** 2)
            return abs(r - r_boundary * 1.6)

        all_dots = [*green_dots, *red_dots]
        distances = [(d, _dist_to_boundary(d)) for d in all_dots]
        distances.sort(key=lambda x: x[1])
        sv_count = min(6, len(distances))

        sv_rings = VGroup()
        for d, _ in distances[:sv_count]:
            ring = Circle(
                radius=0.18, color=WHITE, stroke_width=2.5,
            ).move_to(d.get_center())
            sv_rings.add(ring)

        self.play(
            LaggedStart(*[
                AnimationGroup(
                    Create(r),
                    Flash(r.get_center(), color=WHITE,
                          flash_radius=0.2, num_lines=5),
                )
                for r in sv_rings
            ], lag_ratio=0.20),
            run_time=1.2,
        )

        # SV label
        sv_label = Text(
            "Support Vectors", font=FONT, font_size=18, color=WHITE,
        )
        sv_bg = SurroundingRectangle(
            sv_label, fill_color=BG_COLOR, fill_opacity=0.85,
            stroke_width=0, buff=0.10,
        )
        sv_block = VGroup(sv_bg, sv_label)
        self.add_fixed_in_frame_mobjects(sv_block)
        sv_block.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(sv_block, shift=UP * 0.12), run_time=0.6)
        self.wait(0.5)

        # ── Final explanation annotation ──────────────────────────────────────
        rbf_note = MathTex(
            r"\text{RBF SVM} \rightarrow \text{Circular boundary in 2D}",
            font_size=24, color=SLATE_GRAY,
        )
        self.add_fixed_in_frame_mobjects(rbf_note)
        rbf_note.next_to(sv_block, UP, buff=0.3)
        self.play(FadeIn(rbf_note, shift=UP * 0.1), run_time=0.6)
        self.wait(1.2)

        # ── Success badge ─────────────────────────────────────────────────────
        success_text = Text(
            "✓ Kernel Trick — Perfect Separation!",
            font=FONT, font_size=20, weight=BOLD, color=GENUINE_COLOR,
        )
        success_bg = SurroundingRectangle(
            success_text, fill_color=BLACK, fill_opacity=0.85,
            stroke_color=GENUINE_COLOR, stroke_width=2,
            corner_radius=0.12, buff=0.18,
        )
        success_block = VGroup(success_bg, success_text)
        self.add_fixed_in_frame_mobjects(success_block)
        success_block.move_to(ORIGIN)
        self.play(
            FadeIn(success_bg), Write(success_text),
            run_time=0.8,
        )
        self.wait(1.5)

        # ── Fade everything to black ──────────────────────────────────────────
        all_fixed = [proj_title, sv_block, rbf_note, success_block]
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            *[FadeOut(f) for f in all_fixed],
            run_time=1.5,
        )
        self.wait(0.5)
