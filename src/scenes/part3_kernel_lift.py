"""The Kernel Trick & RBF SVM.

Phase 1: Non-linear siege — concentric data, linear SVM fails.
Phase 2: RBF kernel formula + 3D Gaussian lift.
Phase 3: Laser slicing plane descends + hemisphere dome shield.
Phase 4: Project back to 2D + circular decision boundary.
Phase 5: Gamma slider — underfitting vs overfitting.
Phase 6: Split-screen Linear vs RBF comparison.

System limitations (Cost / Latency / Privacy) are in part4_conclusion.py.
"""

import os
import sys

import numpy as np
from manim import *
from sklearn.datasets import make_circles

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
    from utils.visual_helpers import make_spoof_icon, make_genuine_icon, make_fingerprint_icon
except ImportError:
    def make_spoof_icon(size=0.6):
        return Square(side_length=size, color=IMPOSTOR_COLOR, stroke_width=2)
    def make_genuine_icon(size=0.6):
        return Circle(radius=size * 0.5, color=GENUINE_COLOR, stroke_width=2)
    def make_fingerprint_icon(size=0.6):
        return Circle(radius=size * 0.5, color="#AABBFF", stroke_width=2)

try:
    from core.kernels import rbf_z as _rbf_z
    from core.fusion_data import generate_circle_data as _generate_circle_data
except ImportError:
    def _rbf_z(x, y, gamma=2.0):
        return float(np.exp(-gamma * (x ** 2 + y ** 2)))

    def _generate_circle_data(seed=42):
        X, y = make_circles(n_samples=80, noise=0.08, factor=0.4, random_state=seed)
        X = X * 1.6
        inner = [X[i] for i in range(len(y)) if y[i] == 1]
        outer = [X[i] for i in range(len(y)) if y[i] == 0]
        return inner, outer


# ── Scene-local design tokens ─────────────────────────────────────────────────
FONT             = FONT_MAIN
LASER_YELLOW     = HYPERPLANE_COLOR
SPOOF_RED        = "#FF2222"
ERROR_RED        = "#FF3333"
DOME_COLOR       = "#00E676"
LASER_GLOW_COLOR = "#FFFFAA"
GOLD_COLOR       = "#FFD700"
FP_COLOR         = "#AABBFF"

# RBF / axis parameters
GAMMA            = 2.0
LASER_Z_HEIGHT   = 0.45   # fractional z threshold for the laser plane

XY_RANGE         = [-2.0, 2.0, 0.5]
Z_RANGE          = [-0.2, 1.4, 0.5]
AXIS_LEN_XY      = 5.5
AXIS_LEN_Z       = 3.5
DOT_RADIUS       = 0.07

_Z_SCALE  = Z_RANGE[1] - 0.2          # 1.2  — max z after scaling
_Z_THRESH = LASER_Z_HEIGHT * _Z_SCALE  # 0.54 — laser plane world z-height


def _rbf_decision_radius(gamma: float) -> float:
    """Analytic radius of the RBF decision boundary at the laser threshold."""
    return float(np.sqrt(-np.log(LASER_Z_HEIGHT) / gamma))


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class KernelTrickScene(ThreeDScene):
    """Non-linear Siege → Kernel Lift → Laser Shield → 2D Projection → Gamma → Compare → Limits."""

    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        inner_pts, outer_pts = _generate_circle_data()

        # Truyền dữ liệu tĩnh (inner_pts, outer_pts) đi xuyên suốt các Phase để khóa tọa độ toán học
        axes_2d, green_dots, red_dots = self._phase1_nonlinear_siege(inner_pts, outer_pts)
        axes_3d = self._phase2_kernel_lift(axes_2d, green_dots, red_dots, inner_pts, outer_pts)
        self._phase3_laser_shield(axes_3d, green_dots, red_dots, inner_pts, outer_pts)
        self._phase4_projection_back(axes_3d, green_dots, red_dots, inner_pts, outer_pts)
        
        self._phase5_gamma_slider(inner_pts, outer_pts)
        self._phase6_split_comparison(inner_pts, outer_pts)


    # =========================================================================
    # PHASE 1 — The Non-linear Siege
    # =========================================================================
    def _phase1_nonlinear_siege(self, inner_pts: list, outer_pts: list) -> tuple:
        # FIX: Dời trục vừa phải (DOWN * 0.5) để 2D và 3D đồng bộ tâm khung hình
        axes_2d = Axes(
            x_range=XY_RANGE, y_range=XY_RANGE,
            x_length=AXIS_LEN_XY, y_length=AXIS_LEN_XY,
            axis_config={"stroke_width": 1.8, "color": WHITE, "stroke_opacity": 0.4,
                         "include_ticks": True, "tick_size": 0.05},
            tips=False,
        ).shift(DOWN * 0.5 + RIGHT * 0.5)
        self.play(FadeIn(axes_2d), run_time=0.8)

        title = Text("Non-linearly Separable Data", font=FONT, font_size=22, color=SLATE_GRAY)
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=DOWN * 0.15), run_time=0.6)

        green_dots = VGroup(*[
            Dot3D(point=axes_2d.c2p(pt[0], pt[1], 0.0), color=GENUINE_COLOR, radius=DOT_RADIUS)
            for pt in inner_pts
        ])
        red_dots = VGroup(*[
            Dot3D(point=axes_2d.c2p(pt[0], pt[1], 0.0), color=IMPOSTOR_COLOR, radius=DOT_RADIUS)
            for pt in outer_pts
        ])
        
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in green_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in red_dots],   lag_ratio=0.04),
            run_time=1.8,
        )
        self.wait(0.4)

        # Legend
        gen_legend = VGroup(Dot(radius=0.06, color=GENUINE_COLOR), Text("Genuine (Inner)", font=FONT, font_size=14, color=GENUINE_COLOR)).arrange(RIGHT, buff=0.12)
        imp_legend = VGroup(Dot(radius=0.06, color=IMPOSTOR_COLOR), Text("Impostor (Outer)", font=FONT, font_size=14, color=IMPOSTOR_COLOR)).arrange(RIGHT, buff=0.12)
        legend = VGroup(gen_legend, imp_legend).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_bg = SurroundingRectangle(legend, fill_color=BG_COLOR, fill_opacity=0.85, stroke_width=0, buff=0.10)
        legend_group = VGroup(legend_bg, legend)
        self.add_fixed_in_frame_mobjects(legend_group)
        legend_group.to_corner(DR, buff=0.4)
        self.play(FadeIn(legend_group), run_time=0.5)
        self.wait(0.3)

        # Rotating linear SVM (doomed to fail)
        hp_line = Line(start=axes_2d.c2p(-2.2, 0), end=axes_2d.c2p(2.2, 0), color=HYPERPLANE_COLOR, stroke_width=3)
        self.play(Create(hp_line), run_time=0.6)

        angle_tracker = ValueTracker(0.0)
        c1, c2 = ManimColor(HYPERPLANE_COLOR), ManimColor(ERROR_RED)

        def _hp_updater(line: Line) -> None:
            a = angle_tracker.get_value()
            start = axes_2d.c2p(-2.5 * np.cos(a), -2.5 * np.sin(a), 0.0)
            end   = axes_2d.c2p( 2.5 * np.cos(a),  2.5 * np.sin(a), 0.0)
            line.put_start_and_end_on(start, end)
            t = (np.sin(a * 8) + 1) / 2
            line.set_color(interpolate_color(c1, c2, t))
            line.set_stroke(width=3 + 3 * t)

        hp_line.add_updater(_hp_updater)
        self.play(angle_tracker.animate.set_value(PI * 1.2), run_time=3.0, rate_func=linear)
        hp_line.clear_updaters()
        self.play(hp_line.animate.set_color(ERROR_RED).set_stroke(opacity=0.3), run_time=0.4)

        # Failure modal
        cross_s = 0.35
        cross = VGroup(Line(UL * cross_s, DR * cross_s, color=ERROR_RED, stroke_width=5), Line(UR * cross_s, DL * cross_s, color=ERROR_RED, stroke_width=5))
        fail_text = Text("Linear SVM Failed!", font=FONT, font_size=18, weight=BOLD, color=ERROR_RED)
        fail_group = VGroup(cross, fail_text).arrange(RIGHT, buff=0.15)
        fail_bg = SurroundingRectangle(fail_group, fill_color=BLACK, fill_opacity=0.85, stroke_color=ERROR_RED, stroke_width=1.5, corner_radius=0.10, buff=0.15)
        fail_block = VGroup(fail_bg, fail_group)
        self.add_fixed_in_frame_mobjects(fail_block)
        fail_block.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(fail_bg), Create(cross), Write(fail_text), run_time=0.7)
        self.wait(0.7)

        self.play(FadeOut(fail_block), FadeOut(hp_line), FadeOut(title), FadeOut(legend_group), run_time=0.8)
        self.wait(0.2)

        return axes_2d, green_dots, red_dots

    # =========================================================================
    # PHASE 2 — The Kernel Lift
    # =========================================================================
    def _phase2_kernel_lift(self, axes_2d, green_dots, red_dots, inner_pts, outer_pts) -> ThreeDAxes:
        formula = MathTex(r"K(\mathbf{x}, \mathbf{l}) = e^{-\gamma \|\mathbf{x} - \mathbf{l}\|^2}", font_size=34, color=HYPERPLANE_COLOR)
        formula_label = Text("Radial Basis Function (RBF) Kernel", font=FONT, font_size=16, color=SLATE_GRAY)
        formula_group = VGroup(formula, formula_label).arrange(DOWN, buff=0.15)
        formula_bg = SurroundingRectangle(formula_group, fill_color=BG_COLOR, fill_opacity=0.90, stroke_color=HYPERPLANE_COLOR, stroke_width=1.5, corner_radius=0.12, buff=0.20)
        formula_block = VGroup(formula_bg, formula_group)
        
        self.add_fixed_in_frame_mobjects(formula_block)
        formula_block.to_corner(UL, buff=0.35)
        self.play(FadeIn(formula_bg), Write(formula), FadeIn(formula_label, shift=UP * 0.1), run_time=1.2)

        mapping_tex = MathTex(r"z = e^{-\gamma(x^2 + y^2)}, \quad \gamma = 2.0", font_size=26, color=SLATE_GRAY)
        self.add_fixed_in_frame_mobjects(mapping_tex)
        mapping_tex.next_to(formula_block, DOWN, aligned_edge=LEFT, buff=0.2)
        self.play(FadeIn(mapping_tex, shift=UP * 0.1), run_time=0.6)
        self.wait(0.6)

        # Keep same shift as axes_2d so ReplacementTransform is seamless.
        # The camera look-at (frame_center) is adjusted in move_camera to recenter the 3D view.
        axes_3d = ThreeDAxes(
            x_range=XY_RANGE, y_range=XY_RANGE, z_range=Z_RANGE,
            x_length=AXIS_LEN_XY, y_length=AXIS_LEN_XY, z_length=AXIS_LEN_Z,
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.25},
        ).shift(DOWN * 0.5 + RIGHT * 0.5)

        self.play(ReplacementTransform(axes_2d, axes_3d), run_time=0.8)

        # FIX TOÁN HỌC TỐI THƯỢNG: Zip trực tiếp mảng gốc (inner_pts, outer_pts) với Dot
        # Chấm dứt hoàn toàn sự sai lệch do nội suy ngược p2c
        lift_anims = []
        trace_lines = VGroup()
        for dots_grp, pts_grp, color in [(green_dots, inner_pts, GENUINE_COLOR), (red_dots, outer_pts, IMPOSTOR_COLOR)]:
            for d, pt in zip(dots_grp, pts_grp):
                z_math = _rbf_z(pt[0], pt[1], GAMMA) * _Z_SCALE
                target_pos = axes_3d.c2p(pt[0], pt[1], z_math)
                
                trace = DashedLine(d.get_center(), target_pos, color=color, stroke_width=1.2, dash_length=0.08).set_opacity(0.35)
                trace_lines.add(trace)
                lift_anims.append(d.animate.move_to(target_pos))

        # frame_center shifts the camera's look-at point to the visual midpoint of the
        # lifted cloud (approx. math z=0.7), pulling the scene into vertical center.
        # axes_3d is at [0.5, -0.5, 0] world; z=0.55 in world ≈ math z=0.85 on the z-axis.
        _fc = axes_3d.get_center() + np.array([0.0, 0.0, 0.55])
        self.move_camera(
            phi=70 * DEGREES, theta=-40 * DEGREES,
            frame_center=_fc,
            run_time=3.5, rate_func=smooth,
            added_anims=[
                LaggedStart(*[Create(t) for t in trace_lines], lag_ratio=0.02, run_time=1.5),
                AnimationGroup(*lift_anims, run_time=3.5, rate_func=smooth),
            ],
        )

        z_label = Text("RBF Mapping (z)", font=FONT, font_size=13, color=SLATE_GRAY)
        # Place label diagonally off the z-axis tip to avoid overlapping the arrow.
        z_label.move_to(axes_3d.c2p(0.7, 0.7, 1.3))
        self.add_fixed_orientation_mobjects(z_label)
        self.play(FadeIn(z_label), run_time=0.5)
        self.play(FadeOut(trace_lines), run_time=0.5)
        self.wait(0.3)

        self.move_camera(theta=-25 * DEGREES, run_time=1.5)
        self.move_camera(theta=15  * DEGREES, run_time=2.0, rate_func=smooth)
        self.move_camera(theta=-35 * DEGREES, run_time=1.5, rate_func=smooth)

        # Clear UI gọn gàng để nhường chỗ cho Phase 3
        self.play(FadeOut(formula_block), FadeOut(mapping_tex), FadeOut(z_label), run_time=0.6)
        self.wait(0.2)

        return axes_3d

    # =========================================================================
    # PHASE 3 — The Laser Shield
    # =========================================================================
    def _phase3_laser_shield(self, axes_3d, green_dots, red_dots, inner_pts, outer_pts) -> None:
        shield_title = Text("The Hyperplane in Feature Space", font=FONT, font_size=20, color=HYPERPLANE_COLOR)
        self.add_fixed_in_frame_mobjects(shield_title)
        shield_title.to_edge(UP, buff=0.35)
        self.play(FadeIn(shield_title, shift=DOWN * 0.12), run_time=0.6)

        plane_half = 2.8
        laser_z = ValueTracker(1.2)

        def _make_laser_plane():
            z_math = laser_z.get_value()
            p1, p2 = axes_3d.c2p(-plane_half, -plane_half, z_math), axes_3d.c2p(plane_half, -plane_half, z_math)
            p3, p4 = axes_3d.c2p(plane_half, plane_half, z_math), axes_3d.c2p(-plane_half, plane_half, z_math)
            return Polygon(p1, p2, p3, p4, fill_color=LASER_YELLOW, fill_opacity=0.22, stroke_color=LASER_YELLOW, stroke_width=2.5, stroke_opacity=0.85)

        laser_plane = always_redraw(_make_laser_plane)
        self.add(laser_plane)
        
        target_z_math = _Z_THRESH
        self.play(laser_z.animate.set_value(target_z_math), run_time=3.0, rate_func=smooth)
        self.wait(0.4)

        self.remove(laser_plane)
        
        h = plane_half
        static_plane = Polygon(
            axes_3d.c2p(-h, -h, target_z_math), axes_3d.c2p(h, -h, target_z_math), 
            axes_3d.c2p(h, h, target_z_math), axes_3d.c2p(-h, h, target_z_math), 
            fill_color=LASER_YELLOW, fill_opacity=0.15, stroke_color=LASER_YELLOW, stroke_width=2.5, stroke_opacity=0.85
        )
        self.add(static_plane)

        # Dome: base radius is derived from the actual data spread (not just the analytic
        # RBF boundary) to guarantee it visually encloses ALL genuine (inner) dots.
        # A 15 % margin and a slight base-lift separate it cleanly from the laser plane.
        inner_max_r = max(np.sqrt(pt[0]**2 + pt[1]**2) for pt in inner_pts)
        dome_base_r = max(inner_max_r, _rbf_decision_radius(GAMMA)) * 1.18
        dome_peak   = _Z_SCALE                   # apex at top of z-range
        dome_base_z = target_z_math + 0.05       # float base above laser plane

        dome = Surface(
            lambda u, v: axes_3d.c2p(
                dome_base_r * np.cos(u) * np.sin(v),
                dome_base_r * np.sin(u) * np.sin(v),
                dome_base_z + (dome_peak - dome_base_z) * np.cos(v),
            ),
            u_range=[0, TAU], v_range=[0, PI / 2],
            resolution=(32, 16),
            fill_color=DOME_COLOR, fill_opacity=0.20,
            stroke_color=DOME_COLOR, stroke_width=0.7, stroke_opacity=0.60,
            checkerboard_colors=False,
        ).set_color(DOME_COLOR)

        self.play(FadeIn(dome), run_time=1.5)
        self.play(*[Flash(d.get_center(), color=GENUINE_COLOR, flash_radius=0.2, num_lines=5) for d in green_dots[:8]], run_time=0.8)

        # Markers centered directly on each 3D dot (no positional offset).
        reject_markers = VGroup()
        for d, pt in zip(red_dots, outer_pts):
            z_math = _rbf_z(pt[0], pt[1], GAMMA) * _Z_SCALE
            if z_math < target_z_math:
                marker = VGroup(
                    Line(UL * 0.09, DR * 0.09, color=SPOOF_RED, stroke_width=2.5),
                    Line(UR * 0.09, DL * 0.09, color=SPOOF_RED, stroke_width=2.5),
                ).move_to(d.get_center())
                reject_markers.add(marker)
        
        if len(reject_markers) > 0:
            self.play(LaggedStart(*[Create(m) for m in reject_markers], lag_ratio=0.05), run_time=1.0)

        shield_badge = Text("Security Shield Active", font=FONT, font_size=18, color=DOME_COLOR)
        shield_badge_bg = SurroundingRectangle(
            shield_badge, fill_color=BLACK, fill_opacity=0.80,
            stroke_color=DOME_COLOR, stroke_width=1.5, corner_radius=0.10, buff=0.12,
        )
        shield_badge_block = VGroup(shield_badge_bg, shield_badge)
        self.add_fixed_in_frame_mobjects(shield_badge_block)
        shield_badge_block.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(shield_badge_block, shift=UP * 0.1), run_time=0.6)
        self.wait(1.0)

        self.move_camera(theta=-50 * DEGREES, run_time=2.0, rate_func=smooth)
        self.move_camera(theta=-20 * DEGREES, run_time=1.5, rate_func=smooth)
        self.wait(0.5)

        self._static_plane = static_plane
        self._dome = dome
        self._reject_markers = reject_markers
        self._shield_badge_block = shield_badge_block
        self._shield_title = shield_title

    # =========================================================================
    # PHASE 4 — Projection Back to 2D
    # =========================================================================
    def _phase4_projection_back(self, axes_3d, green_dots, red_dots, inner_pts, outer_pts) -> None:
        proj_title = Text("Project Decision Boundary to 2D", font=FONT, font_size=22, color=HYPERPLANE_COLOR).to_edge(UP, buff=0.35)
        self.add_fixed_in_frame_mobjects(proj_title)
        
        # FIX TEXT: Fade mượt mà tránh đè chữ
        self.play(FadeOut(self._shield_title), FadeIn(proj_title), run_time=0.8)

        # Reset frame_center: going toward top-down so look at axes XY floor.
        _fc_floor = axes_3d.get_center()
        self.move_camera(
            phi=10 * DEGREES, theta=-90 * DEGREES,
            frame_center=_fc_floor,
            run_time=2.5, rate_func=smooth,
        )

        # FIX TOÁN HỌC: Lấy lại tọa độ gốc ép thẳng xuống z=0
        proj_anims = []
        trace_lines = VGroup()
        for dots_grp, pts_grp, color in [(green_dots, inner_pts, GENUINE_COLOR), (red_dots, outer_pts, IMPOSTOR_COLOR)]:
            for d, pt in zip(dots_grp, pts_grp):
                z_math = _rbf_z(pt[0], pt[1], GAMMA) * _Z_SCALE
                floor_pos = axes_3d.c2p(pt[0], pt[1], 0.0)
                if abs(z_math) > 0.01:
                    trace = DashedLine(d.get_center(), floor_pos, color=color, stroke_width=1.0, dash_length=0.06).set_opacity(0.3)
                    trace_lines.add(trace)
                proj_anims.append(d.animate.move_to(floor_pos))

        if len(trace_lines) > 0:
            self.play(Create(trace_lines), run_time=0.5)

        self.play(AnimationGroup(*proj_anims, run_time=1.5, rate_func=smooth), FadeOut(self._dome), run_time=1.5)
        self.play(FadeOut(trace_lines), FadeOut(self._static_plane), run_time=0.4)

        center_pos = axes_3d.c2p(0, 0, 0)
        math_r  = _rbf_decision_radius(GAMMA)
        
        edge_pos = axes_3d.c2p(math_r, 0, 0)
        r_scene = np.linalg.norm(edge_pos - center_pos)

        glow_circle = Circle(radius=r_scene, color=HYPERPLANE_COLOR, stroke_width=14, stroke_opacity=0.15).move_to(center_pos)
        boundary_circle = Circle(radius=r_scene, color=HYPERPLANE_COLOR, stroke_width=3.5, stroke_opacity=0.95).move_to(center_pos)

        self.play(Create(glow_circle), Create(boundary_circle), run_time=1.5)
        self.play(Flash(center_pos, color=HYPERPLANE_COLOR, flash_radius=r_scene * 1.2, num_lines=16), run_time=0.8)

        boundary_label = Text("Decision Boundary (RBF)", font=FONT, font_size=18, weight=BOLD, color=HYPERPLANE_COLOR).to_edge(DOWN, buff=0.45)
        self.add_fixed_in_frame_mobjects(boundary_label)
        
        # FIX TEXT: Dọn sạch Label cũ
        self.play(FadeOut(self._shield_badge_block), FadeIn(boundary_label), run_time=0.8)
        self.wait(1.5)

        self.move_camera(
            phi=0, theta=-90 * DEGREES,
            frame_center=axes_3d.get_center(),
            run_time=1.0,
        )
        self.wait(0.5)

        fade_targets = [proj_title, boundary_label, boundary_circle, glow_circle, axes_3d, green_dots, red_dots]
        if len(self._reject_markers) > 0: fade_targets.append(self._reject_markers)
        self.play(*[FadeOut(m) for m in fade_targets], run_time=1.0)
        self.wait(0.3)
    # =========================================================================
    # PHASE 5 — Gamma Slider
    # =========================================================================
    def _phase5_gamma_slider(self, inner_pts: list, outer_pts: list) -> None:
        """Visual slider demonstrating how γ controls boundary tightness."""
        axes = Axes(
            x_range=XY_RANGE, y_range=XY_RANGE,
            x_length=AXIS_LEN_XY, y_length=AXIS_LEN_XY,
            axis_config={"stroke_width": 1.5, "color": WHITE, "stroke_opacity": 0.35},
            tips=False,
        ).shift(DOWN * 0.4 + RIGHT * 0.3)

        green_dots = VGroup(*[
            Dot(axes.c2p(pt[0], pt[1]), color=GENUINE_COLOR, radius=DOT_RADIUS)
            for pt in inner_pts
        ])
        red_dots = VGroup(*[
            Dot(axes.c2p(pt[0], pt[1]), color=IMPOSTOR_COLOR, radius=DOT_RADIUS)
            for pt in outer_pts
        ])

        phase_title = Text("Effect of Hyperparameter γ (Gamma)",
                           font=FONT, font_size=22, color=WHITE)
        self.add_fixed_in_frame_mobjects(phase_title)
        phase_title.to_edge(UP, buff=0.3)
        self.play(FadeIn(axes), FadeIn(green_dots), FadeIn(red_dots),
                  FadeIn(phase_title), run_time=1.0)

        # Slider track UI
        slider_w = 4.5
        slider_y = -3.3
        sl_l, sl_r = -slider_w / 2, slider_w / 2

        slider_track = Line(np.array([sl_l, slider_y, 0]), np.array([sl_r, slider_y, 0]),
                            color=SLATE_GRAY, stroke_width=3)
        lbl_l = Text("γ = 0.5", font=FONT, font_size=13, color=SLATE_GRAY).next_to(
            slider_track, LEFT, buff=0.15)
        lbl_r = Text("γ = 8.0", font=FONT, font_size=13, color=SLATE_GRAY).next_to(
            slider_track, RIGHT, buff=0.15)

        self.add_fixed_in_frame_mobjects(slider_track, lbl_l, lbl_r)
        self.play(FadeIn(slider_track), FadeIn(lbl_l), FadeIn(lbl_r), run_time=0.5)

        # Compute initial boundary at γ=2.0
        ax_center = axes.c2p(0, 0)
        unit_scale = AXIS_LEN_XY / (XY_RANGE[1] - XY_RANGE[0])
        r0 = _rbf_decision_radius(2.0) * unit_scale

        boundary = Circle(radius=r0, color=HYPERPLANE_COLOR,
                          stroke_width=3, stroke_opacity=0.9).move_to(ax_center)
        glow     = Circle(radius=r0, color=HYPERPLANE_COLOR,
                          stroke_width=10, stroke_opacity=0.15).move_to(ax_center)
        self.play(Create(boundary), Create(glow), run_time=0.8)

        # Initial handle
        handle     = Dot(np.array([0.0, slider_y, 0]), radius=0.13, color=HYPERPLANE_COLOR)
        gamma_text = Text("γ = 2.0  |  Optimal", font=FONT, font_size=18,
                          weight=BOLD, color=HYPERPLANE_COLOR)
        gamma_text.move_to(np.array([0, slider_y - 0.48, 0]))
        self.add_fixed_in_frame_mobjects(handle, gamma_text)
        self.play(FadeIn(handle), FadeIn(gamma_text), run_time=0.5)
        self.wait(0.5)

        gamma_steps = [
            (0.5,  sl_l, "Underfitting", "#FF8C00"),
            (2.0,  0.0,  "Optimal",      HYPERPLANE_COLOR),
            (8.0,  sl_r, "Overfitting",  SPOOF_RED),
        ]

        for gamma_val, hx, status, color in gamma_steps:
            r = _rbf_decision_radius(gamma_val) * unit_scale

            new_boundary = Circle(radius=r, color=color, stroke_width=3,
                                  stroke_opacity=0.9).move_to(ax_center)
            new_glow     = Circle(radius=r, color=color, stroke_width=10,
                                  stroke_opacity=0.15).move_to(ax_center)
            new_handle   = Dot(np.array([hx, slider_y, 0]), radius=0.13, color=color)
            new_text     = Text(f"γ = {gamma_val:.1f}  |  {status}", font=FONT,
                                font_size=18, weight=BOLD, color=color)
            new_text.move_to(np.array([0, slider_y - 0.48, 0]))

            self.add_fixed_in_frame_mobjects(new_handle, new_text)
            self.remove(handle, gamma_text)

            self.play(
                Transform(boundary, new_boundary),
                Transform(glow, new_glow),
                FadeIn(new_handle),
                FadeIn(new_text),
                run_time=1.2, rate_func=smooth,
            )
            self.wait(0.8)
            handle, gamma_text = new_handle, new_text

        self.wait(0.5)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.8)
        self.remove(phase_title, slider_track, lbl_l, lbl_r, handle, gamma_text)
        self.wait(0.3)

    # =========================================================================
    # PHASE 6 — Split-Screen Comparison
    # =========================================================================
    def _phase6_split_comparison(self, inner_pts: list, outer_pts: list) -> None:
        """Side-by-side: Linear SVM failure on XOR data vs RBF circular boundary."""
        # Reset camera projection offset left over from previous phases
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, gamma=0, frame_center=ORIGIN)

        divider = Line(UP * 3.8, DOWN * 3.8, color=SLATE_GRAY,
                       stroke_width=1.5, stroke_opacity=0.5)
        self.add_fixed_in_frame_mobjects(divider)
        self.play(Create(divider), run_time=0.4)

        # Panel positions (center of each screen half)
        left_center = LEFT * 3.5
        right_center = RIGHT * 3.5

        # Shift both panels down to create clear top margin for titles.
        la = Axes(
            x_range=[0, 1, 0.5], y_range=[0, 1, 0.5],
            x_length=3.6, y_length=3.2,
            axis_config={"stroke_width": 1.2, "color": WHITE, "stroke_opacity": 0.3},
            tips=False,
        ).move_to(left_center + DOWN * 0.5)

        xor_green = [(0.22, 0.78), (0.18, 0.83), (0.28, 0.73),
                     (0.75, 0.22), (0.70, 0.28), (0.80, 0.17)]
        xor_red   = [(0.22, 0.22), (0.17, 0.17), (0.28, 0.28),
                     (0.78, 0.78), (0.72, 0.73), (0.83, 0.83)]

        lg = VGroup(*[Dot(la.c2p(x, y), color=GENUINE_COLOR,  radius=0.09) for x, y in xor_green])
        lr = VGroup(*[Dot(la.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in xor_red])
        l_line  = DashedLine(la.c2p(0.0, 0.95), la.c2p(1.0, 0.05),
                             color=SPOOF_RED, stroke_width=2.5, stroke_opacity=0.8)
        l_cross = VGroup(
            Line(UL * 0.2, DR * 0.2, color=SPOOF_RED, stroke_width=4),
            Line(UR * 0.2, DL * 0.2, color=SPOOF_RED, stroke_width=4),
        ).move_to(la.c2p(0.5, 0.5))

        left_title = Text("Linear SVM", font=FONT, font_size=18, weight=BOLD, color=SPOOF_RED)
        left_title.move_to(left_center + UP * 1.8)
        self.add_fixed_in_frame_mobjects(left_title)

        # ── Right Panel: concentric data + RBF boundary ───────────────────────
        ra = Axes(
            x_range=XY_RANGE, y_range=XY_RANGE,
            x_length=3.6, y_length=3.2,
            axis_config={"stroke_width": 1.2, "color": WHITE, "stroke_opacity": 0.3},
            tips=False,
        ).move_to(right_center + DOWN * 0.5)

        rg = VGroup(*[Dot(ra.c2p(pt[0], pt[1]), color=GENUINE_COLOR,  radius=0.06) for pt in inner_pts])
        rr = VGroup(*[Dot(ra.c2p(pt[0], pt[1]), color=IMPOSTOR_COLOR, radius=0.06) for pt in outer_pts])

        r_data  = _rbf_decision_radius(GAMMA)
        r_scene = r_data * (3.6 / (XY_RANGE[1] - XY_RANGE[0]))
        r_center = ra.c2p(0, 0)
        rb      = Circle(radius=r_scene, color=HYPERPLANE_COLOR,
                         stroke_width=3).move_to(r_center)
        rg_glow = Circle(radius=r_scene, color=HYPERPLANE_COLOR,
                         stroke_width=10, stroke_opacity=0.15).move_to(r_center)
        checkmark = Text("✓", font=FONT, font_size=32, color=GENUINE_COLOR).move_to(r_center)

        right_title = Text("RBF Kernel SVM", font=FONT, font_size=18, weight=BOLD, color=GENUINE_COLOR)
        right_title.move_to(right_center + UP * 1.8)
        self.add_fixed_in_frame_mobjects(right_title)

        self.play(
            FadeIn(la), FadeIn(lg), FadeIn(lr),
            FadeIn(ra), FadeIn(rg), FadeIn(rr),
            FadeIn(left_title), FadeIn(right_title),
            run_time=1.2,
        )
        self.play(Create(l_line), Create(l_cross), run_time=0.7)
        self.play(Create(rg_glow), Create(rb), run_time=1.0)
        self.play(FadeIn(checkmark, scale=0.5), run_time=0.5)
        self.wait(0.5)

        conclusion = Text(
            "Kernel Trick: Map to Higher Dimensions → Linearly Separable",
            font=FONT, font_size=19, weight=BOLD, color=HYPERPLANE_COLOR,
        )
        conclusion_bg = SurroundingRectangle(
            conclusion, fill_color=BG_COLOR, fill_opacity=0.92,
            stroke_color=HYPERPLANE_COLOR, stroke_width=1.5, corner_radius=0.12, buff=0.18,
        )
        conclusion_group = VGroup(conclusion_bg, conclusion)
        self.add_fixed_in_frame_mobjects(conclusion_group)
        conclusion_group.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(conclusion_group, shift=UP * 0.15), run_time=0.8)
        self.wait(2.5)

        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=1.0)
        self.remove(divider, left_title, right_title, conclusion_group)
        self.wait(0.3)
