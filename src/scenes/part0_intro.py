"""Intro scene for the multibiometric fusion and SVM visualization.

The scene opens with a biometric scanning motif, transitions into a 2D XOR
example, lifts the data into 3D, and ends with the title card and roadmap.
"""

import sys
import os
import numpy as np
from manim import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import (
        CLASS_A_COLOR, CLASS_B_COLOR, HYPERPLANE_COLOR,
        BG_COLOR, FONT_MAIN, SLATE_GRAY,
    )
    from core.fusion_data import generate_cluster
except ImportError:
    CLASS_A_COLOR    = "#FF5E5E"
    CLASS_B_COLOR    = "#00C2D1"
    HYPERPLANE_COLOR = "#F9DC5C"
    BG_COLOR         = "#0B0C10"
    FONT_MAIN        = "Roboto"
    SLATE_GRAY       = "#888888"

    def generate_cluster(center, n=14, sigma=0.15, seed=None):
        rng = np.random.default_rng(seed)
        pts = rng.normal(loc=np.asarray(center)[:2], scale=sigma, size=(n, 2))
        return [np.array([x, y, 0.0]) for x, y in pts]


# Scene-local design tokens
FONT_TITLE  = FONT_MAIN
FONT_BODY   = FONT_MAIN
ERROR_RED   = "#FF3333"
LASER_COLOR = "#00FFFF"  # Cyan laser
SCAN_GREEN  = "#39FF14"  # Neon green for scan readout
ICON_STROKE = "#AAAAAA"
ICON_FILL   = "#1A1D27"


def make_face_icon(size: float = 0.8) -> VGroup:
    """Create a minimalist face icon."""
    head = Ellipse(
        width=size * 1.2, height=size * 1.5,
        stroke_color=ICON_STROKE, stroke_width=2,
        fill_color=ICON_FILL, fill_opacity=1
    )
    eye_l = Dot(LEFT * size * 0.26 + UP * size * 0.22, radius=size * 0.07, color=LASER_COLOR)
    eye_r = Dot(RIGHT * size * 0.26 + UP * size * 0.22, radius=size * 0.07, color=LASER_COLOR)
    mouth = Arc(
        radius=size * 0.28, start_angle=-5 * DEGREES, angle=-170 * DEGREES,
        color=ICON_STROKE, stroke_width=2
    ).move_to(DOWN * size * 0.22)
    return VGroup(head, eye_l, eye_r, mouth)


def make_fingerprint_icon(size: float = 0.8) -> VGroup:
    """Create a fingerprint-style loop icon."""
    arcs = VGroup(*[
        Arc(
            radius=r, start_angle=200 * DEGREES, angle=-220 * DEGREES,
            stroke_color=ICON_STROKE, stroke_width=1.8 - i * 0.2,
            stroke_opacity=0.9 - i * 0.05
        )
        for i, r in enumerate(np.linspace(size * 0.12, size * 0.58, 5))
    ])
    bg = Circle(
        radius=size * 0.65, fill_color=ICON_FILL, fill_opacity=1,
        stroke_color=ICON_STROKE, stroke_width=2
    )
    return VGroup(bg, arcs)


def make_iris_icon(size: float = 0.8) -> VGroup:
    """Create an iris icon with spokes and a pupil."""
    outer = Circle(
        radius=size * 0.6, stroke_color=ICON_STROKE, stroke_width=2,
        fill_color=ICON_FILL, fill_opacity=1
    )
    iris_r = Circle(
        radius=size * 0.4, stroke_color=CLASS_B_COLOR,
        stroke_width=1.5, fill_opacity=0
    )
    pupil = Circle(
        radius=size * 0.18, fill_color="#111111", fill_opacity=1, stroke_width=0
    )
    spokes = VGroup(*[
        Line(
            size * 0.22 * np.array([np.cos(a), np.sin(a), 0]),
            size * 0.38 * np.array([np.cos(a), np.sin(a), 0]),
            stroke_color=ICON_STROKE, stroke_width=1, stroke_opacity=0.6
        )
        for angle_deg in range(0, 360, 30) for a in [angle_deg * DEGREES]
    ])
    return VGroup(outer, iris_r, spokes, pupil)


def make_voice_icon(size: float = 0.8) -> VGroup:
    """Create a voice waveform icon."""
    bg = RoundedRectangle(
        width=size * 1.3, height=size * 1.1, corner_radius=size * 0.1,
        fill_color=ICON_FILL, fill_opacity=1,
        stroke_color=ICON_STROKE, stroke_width=2
    )
    heights = [0.25, 0.55, 0.85, 1.0, 0.85, 0.55, 0.25]
    xs = np.linspace(-size * 0.42, size * 0.42, len(heights))
    bars = VGroup(*[
        Rectangle(
            width=size * 0.07, height=size * h * 0.7,
            fill_color=CLASS_B_COLOR, fill_opacity=0.9, stroke_width=0
        ).move_to([x, 0, 0])
        for x, h in zip(xs, heights)
    ])
    return VGroup(bg, bars)


class IntroScene(ThreeDScene):
    """
    Real-world hook -> XOR trap -> 3D kernel lift -> title -> roadmap.
    Demonstrates the transition from linear constraints to 3D SVM capabilities.
    """

    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        self._phase0_biometric_scan()
        blue_dots, red_dots, axes, x_label, y_label, hyperplane = self._phase1_xor_trap()
        slicing_group = self._phase2_kernel_lift(axes, blue_dots, red_dots)
        dark_overlay, title_group = self._phase3_title_drop()
        self._phase4_roadmap_and_cut(
            axes, blue_dots, red_dots, slicing_group, dark_overlay, title_group
        )

    def _phase0_biometric_scan(self) -> None:
        """Build the biometric grid and run the laser scan reveal."""
        cell_spacing = 1.9
        icon_size = 0.72

        offsets = {
            "face": np.array([-cell_spacing / 2, cell_spacing / 2, 0]),
            "fingerprint": np.array([cell_spacing / 2, cell_spacing / 2, 0]),
            "iris": np.array([-cell_spacing / 2, -cell_spacing / 2, 0]),
            "voice": np.array([cell_spacing / 2, -cell_spacing / 2, 0]),
        }

        face_icon = make_face_icon(icon_size).move_to(offsets["face"])
        fp_icon = make_fingerprint_icon(icon_size).move_to(offsets["fingerprint"])
        iris_icon = make_iris_icon(icon_size).move_to(offsets["iris"])
        voice_icon = make_voice_icon(icon_size).move_to(offsets["voice"])

        h_sep = Line(LEFT * 1.8, RIGHT * 1.8, stroke_color=SLATE_GRAY, stroke_width=0.8, stroke_opacity=0.35)
        v_sep = Line(UP * 1.8, DOWN * 1.8, stroke_color=SLATE_GRAY, stroke_width=0.8, stroke_opacity=0.35)

        corners = VGroup()
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = Line(ORIGIN, RIGHT * 0.3 * sx, stroke_color=LASER_COLOR, stroke_width=2, stroke_opacity=0.7)
            by = Line(ORIGIN, UP * 0.3 * sy, stroke_color=LASER_COLOR, stroke_width=2, stroke_opacity=0.7)
            corners.add(VGroup(bx, by).move_to(np.array([sx * 2.1, sy * 2.05, 0])))

        def _icon_label(text: str, pos: np.ndarray) -> Text:
            return Text(text, font=FONT_BODY, font_size=13, color=SLATE_GRAY).move_to(pos + DOWN * 0.75)

        icon_labels = VGroup(
            _icon_label("FACE", offsets["face"]),
            _icon_label("FINGERPRINT", offsets["fingerprint"]),
            _icon_label("IRIS", offsets["iris"]),
            _icon_label("VOICE", offsets["voice"]),
        )

        self.play(
            LaggedStart(FadeIn(h_sep), FadeIn(v_sep), *[FadeIn(c) for c in corners], lag_ratio=0.15),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                FadeIn(face_icon, shift=DOWN * 0.2),
                FadeIn(fp_icon, shift=DOWN * 0.2),
                FadeIn(iris_icon, shift=UP * 0.2),
                FadeIn(voice_icon, shift=UP * 0.2),
                lag_ratio=0.18,
            ),
            FadeIn(icon_labels),
            run_time=1.2,
        )
        self.wait(0.3)

        # Scanning laser effect
        laser_y_start, laser_y_end = 2.05, -2.05
        laser_base = Line(LEFT * 2.2, RIGHT * 2.2)
        
        laser = laser_base.copy().set_color(LASER_COLOR).set_stroke(width=3, opacity=0.9)
        laser_glow = laser_base.copy().set_color(LASER_COLOR).set_stroke(width=12, opacity=0.18)
        laser_group = VGroup(laser_glow, laser).shift(UP * laser_y_start)
        self.add(laser_group)

        laser_tracker = ValueTracker(laser_y_start)
        laser_group.add_updater(
            lambda mob: mob.move_to(UP * laser_tracker.get_value())
        )

        self.play(laser_tracker.animate.set_value(laser_y_end), run_time=2.6, rate_func=linear)
        laser_group.clear_updaters()

        scan_flashes = [
            Flash(
                icon.get_center(), color=LASER_COLOR, flash_radius=0.55, 
                num_lines=10, line_stroke_width=2
            )
            for icon in [face_icon, fp_icon, iris_icon, voice_icon]
        ]
        self.play(LaggedStart(*scan_flashes, lag_ratio=0.2), run_time=1.0)
        self.wait(0.3)

        # Cache variables for subsequent phases
        self._face_icon = face_icon
        self._fp_icon = fp_icon
        self._iris_icon = iris_icon
        self._voice_icon = voice_icon
        self._icon_labels = icon_labels
        self._laser_group = laser_group
        self._grid_chrome = VGroup(h_sep, v_sep, corners)

    def _phase1_xor_trap(self) -> tuple[VGroup, VGroup, Axes, Text, Text, Line]:
        """Morph biometric icons into axes and introduce the XOR pattern."""
        self.play(
            FadeOut(self._iris_icon),
            FadeOut(self._voice_icon),
            FadeOut(self._icon_labels),
            FadeOut(self._laser_group),
            FadeOut(self._grid_chrome),
            run_time=0.9,
        )
        self.wait(0.2)

        axes_2d = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=6.0,
            y_length=5.5,
            axis_config={"stroke_width": 1.8, "color": WHITE, "stroke_opacity": 0.55},
            tips=True,
        ).shift(DOWN * 0.15)

        x_label = Text(
            "Face Match Score", font=FONT_BODY, font_size=16, color=SLATE_GRAY,
        ).next_to(axes_2d.x_axis.get_end(), DR, buff=0.15)

        y_label = Text(
            "Fingerprint Score", font=FONT_BODY, font_size=16, color=SLATE_GRAY,
        ).rotate(PI / 2).next_to(axes_2d.y_axis, LEFT, buff=0.5).shift(UP * 1.8)

        face_icon_copy = self._face_icon.copy()
        fp_icon_copy = self._fp_icon.copy()

        self.play(
            face_icon_copy.animate.scale(0.3).next_to(x_label, RIGHT, buff=0.2),
            fp_icon_copy.animate.scale(0.3).next_to(y_label, UP, buff=0.2),
            Create(axes_2d),
            FadeOut(self._face_icon),
            FadeOut(self._fp_icon),
            run_time=1.4,
        )
        self.play(FadeIn(x_label), FadeIn(y_label), run_time=0.6)
        self.wait(0.4)

        self.play(FadeOut(face_icon_copy), FadeOut(fp_icon_copy), run_time=0.5)

        # Generate clustered points representing XOR data distribution
        b1 = generate_cluster(center=[ 1.8,  1.8], seed=42)
        b2 = generate_cluster(center=[-1.8, -1.8], seed=43)
        r1 = generate_cluster(center=[-1.8,  1.8], seed=44)
        r2 = generate_cluster(center=[ 1.8, -1.8], seed=45)

        blue_dots = VGroup(*[Dot(axes_2d.c2p(x, y), color=CLASS_B_COLOR, radius=0.08) for x, y, _ in b1 + b2])
        red_dots = VGroup(*[Dot(axes_2d.c2p(x, y), color=CLASS_A_COLOR, radius=0.08) for x, y, _ in r1 + r2])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in blue_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in red_dots], lag_ratio=0.04),
            run_time=2.0,
        )
        self.wait(0.5)

        # Teaser banner: The linear failure
        question_bg = Rectangle(
            width=7.4, height=0.74,
            fill_color=BG_COLOR, fill_opacity=0.92,
            stroke_color=HYPERPLANE_COLOR, stroke_width=1.2, stroke_opacity=0.60,
        ).to_edge(UP, buff=0.20)

        question_text = Text(
            "Why does the line always fail?",
            font=FONT_BODY, font_size=26, color=HYPERPLANE_COLOR,
        ).move_to(question_bg)

        self.play(
            FadeIn(question_bg, shift=DOWN * 0.12),
            Write(question_text),
            run_time=1.0,
        )
        self.wait(0.4)

        # Introduce a rotating hyperplane to demonstrate failed linear separation
        hyperplane_line = Line(
            start=axes_2d.c2p(-4, 0), end=axes_2d.c2p(4, 0),
            color=HYPERPLANE_COLOR, stroke_width=3,
        )
        self.play(Create(hyperplane_line), run_time=0.7)

        angle_tracker = ValueTracker(0.0)
        c1_hp = ManimColor(HYPERPLANE_COLOR)
        c2_err = ManimColor(ERROR_RED)

        def _update_line(line: Line) -> None:
            a = angle_tracker.get_value()
            center = axes_2d.c2p(0, 0)
            d_vec = np.array([np.cos(a), np.sin(a), 0.0])
            line.put_start_and_end_on(center - 4.5 * d_vec, center + 4.5 * d_vec)

            # Visually encode error when hyperplane crosses clusters (at 45deg angles)
            mod_a = a % PI
            dist = min(abs(mod_a - PI / 4), abs(mod_a - 3 * PI / 4))
            t = max(0.0, 1.0 - dist / 0.18)
            line.set_color(interpolate_color(c1_hp, c2_err, t))
            line.set_stroke(width=3 + 4 * t)

        hyperplane_line.add_updater(_update_line)
        self.play(
            angle_tracker.animate.set_value(PI * 1.25),
            run_time=3.5, rate_func=linear,
        )
        hyperplane_line.clear_updaters()

        self.play(
            hyperplane_line.animate.set_color(HYPERPLANE_COLOR).set_stroke(opacity=0.22),
            run_time=0.6,
        )
        self.wait(0.8)

        self.play(
            FadeOut(question_bg), FadeOut(question_text),
            FadeOut(x_label), FadeOut(y_label),
            FadeOut(hyperplane_line),
            run_time=0.7,
        )
        self.wait(0.2)

        return blue_dots, red_dots, axes_2d, x_label, y_label, hyperplane_line

    def _phase2_kernel_lift(
        self, axes_2d: Axes, blue_dots: VGroup, red_dots: VGroup,
    ) -> VGroup:
        """Map points into 3D space to solve the XOR problem."""
        axes3d = ThreeDAxes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            z_range=[-2, 2, 1],
            x_length=6.0, y_length=5.5, z_length=3.5,
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.20},
        ).shift(DOWN * 0.15)

        self.play(ReplacementTransform(axes_2d, axes3d), run_time=0.7)
        self.add_fixed_orientation_mobjects(*blue_dots, *red_dots)

        lift_anims = []
        trace_lines = VGroup()

        for dots_grp, color in [(blue_dots, CLASS_B_COLOR), (red_dots, CLASS_A_COLOR)]:
            for dot in dots_grp:
                cx, cy, _ = dot.get_center()
                z_target = cx * cy * 0.4
                target = np.array([cx, cy, z_target])
                
                trace = DashedLine(
                    dot.get_center(), target,
                    color=color, stroke_width=1.4, dash_length=0.08,
                ).set_opacity(0.45)
                trace_lines.add(trace)
                lift_anims.append(dot.animate.move_to(target))

        self.move_camera(
            phi=80 * DEGREES, theta=-45 * DEGREES,
            run_time=3.2, rate_func=smooth,
            added_anims=[
                LaggedStart(*[Create(t) for t in trace_lines], lag_ratio=0.02, run_time=1.5),
                AnimationGroup(*lift_anims, run_time=3.2, rate_func=smooth),
            ],
        )

        self.play(FadeOut(trace_lines), run_time=0.5)
        self.wait(0.4)

        z_label = Text(
            "Quality Score / Kernel Mapping",
            font=FONT_BODY, font_size=13, color=SLATE_GRAY,
        ).move_to(axes3d.c2p(0, 0, 2.7))
        
        self.add_fixed_orientation_mobjects(z_label)
        self.play(FadeIn(z_label), run_time=0.7)

        # Cinematic camera panning
        self.move_camera(theta=-30 * DEGREES, run_time=1.5)
        self.move_camera(theta=25 * DEGREES, run_time=2.8, rate_func=smooth)
        self.move_camera(theta=-25 * DEGREES, run_time=2.0, rate_func=smooth)

        # Form a 3D slicing hyperplane to separate the clusters
        slicing_plane = Polygon(
            np.array([-3.8, -3.8, 0]),
            np.array([ 3.8, -3.8, 0]),
            np.array([ 3.8,  3.8, 0]),
            np.array([-3.8,  3.8, 0]),
            fill_color=HYPERPLANE_COLOR, fill_opacity=0.18,
            stroke_color=HYPERPLANE_COLOR, stroke_width=2.5, stroke_opacity=0.85,
        )
        slicing_group = VGroup(slicing_plane)
        self.play(DrawBorderThenFill(slicing_plane), run_time=2.0)
        self.move_camera(theta=-25 * DEGREES, run_time=1.5)
        self.wait(1.0)

        self.play(FadeOut(z_label), run_time=0.4)
        self._axes3d = axes3d
        return slicing_group

    def _phase3_title_drop(self) -> tuple[Mobject, VGroup]:
        """Present the main title and credits over a dark overlay."""
        self.begin_ambient_camera_rotation(rate=0.055)

        dark_overlay = Rectangle(
            width=20, height=20,
            color=BLACK, fill_opacity=0.90, stroke_width=0,
        )
        self.add_fixed_in_frame_mobjects(dark_overlay)
        self.remove(dark_overlay)
        self.play(FadeIn(dark_overlay), run_time=1.4)

        title_line1 = Text(
            "Multibiometric Fusion & SVM",
            font=FONT_TITLE, weight=BOLD, font_size=48, color=WHITE,
        )
        title_line2 = Text(
            "Unlocking The Kernel Trick",
            font=FONT_TITLE, font_size=36, slant=ITALIC, color=HYPERPLANE_COLOR,
        ).next_to(title_line1, DOWN, buff=0.28)
        
        title_vgroup = VGroup(title_line1, title_line2)

        divider_line = Line(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=2)
        divider_line.width = title_vgroup.width
        divider_line.next_to(title_vgroup, DOWN, buff=0.38)

        credits_text = VGroup(
            Text("Created by: Lê Hà Thanh Chương", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
            Text("Faculty of Information Technology, VNU-HCM University of Science", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
            Text("Course: Pattern Recognition", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        
        credits_text.next_to(divider_line, DOWN, buff=0.38).align_to(divider_line, LEFT)

        title_group = VGroup(title_vgroup, divider_line, credits_text).move_to(ORIGIN)

        self.add_fixed_in_frame_mobjects(title_group)
        self.remove(title_group)

        self.play(Write(title_line1), run_time=1.5)
        self.play(FadeIn(title_line2, shift=UP * 0.2), run_time=1.0)
        self.play(GrowFromCenter(divider_line), run_time=0.7)
        self.play(FadeIn(credits_text, shift=UP * 0.2), run_time=1.2)
        self.wait(2.0)

        return dark_overlay, title_group

    def _phase4_roadmap_and_cut(
        self, axes_2d: Axes, blue_dots: VGroup, red_dots: VGroup, slicing_group: VGroup,
        dark_overlay: Rectangle, title_group: VGroup,
    ) -> None:
        """Present the roadmap and transition out of the intro scene."""
        self.play(
            title_group.animate.scale(0.52).to_corner(UL, buff=0.65),
            run_time=1.4,
        )
        self.wait(0.2)

        roadmap_strings = [
            "① Unimodal Limitations — FAR vs FRR",
            "② Pipeline: From Raw Image to Fusion Score",
            "③ Score-Level Fusion & Linear SVM",
            "④ Kernel Trick: Mapping into 3D Space",
            "⑤ Linear vs RBF Comparison · Pros & Cons",
        ]

        roadmap_items = VGroup()
        for label_str in roadmap_strings:
            bullet = Dot(radius=0.065, color=HYPERPLANE_COLOR)
            label = Text(label_str, font=FONT_BODY, font_size=24, color=WHITE).next_to(bullet, RIGHT, buff=0.25)
            roadmap_items.add(VGroup(bullet, label))

        roadmap_items.arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        roadmap_items.next_to(title_group, DOWN, buff=0.85).align_to(title_group, LEFT).shift(RIGHT * 0.28)

        self.add_fixed_in_frame_mobjects(roadmap_items)
        self.remove(roadmap_items)

        self.play(
            LaggedStart(
                *[FadeIn(item, shift=RIGHT * 0.4) for item in roadmap_items],
                lag_ratio=0.40,
            ),
            run_time=2.8,
        )
        self.wait(1.8)

        item_focus = roadmap_items[0]
        items_other = VGroup(*roadmap_items[1:])

        self.stop_ambient_camera_rotation()
        self.play(FadeOut(items_other), run_time=0.7)

        self.remove(item_focus)
        item_focus_static = item_focus.copy().set_color(HYPERPLANE_COLOR).scale(1.55).move_to(ORIGIN + UP * 0.3)
        
        self.add_fixed_in_frame_mobjects(item_focus_static)
        self.play(FadeIn(item_focus_static, scale=0.8), run_time=0.9)
        self.wait(1.2)

        self.play(
            FadeOut(item_focus_static),
            FadeOut(title_group),
            FadeOut(dark_overlay),
            FadeOut(slicing_group),
            FadeOut(blue_dots),
            FadeOut(red_dots),
            FadeOut(self._axes3d),
            run_time=1.5,
        )
        self.wait(0.5)