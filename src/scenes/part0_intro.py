"""part0_intro.py  —  IntroScene V2.0: From Reality to Multi-Dimensional Space.

Phases
------
Phase 0  (0:00 – 0:10)  Real-world hook  — biometric grid + laser scan.
Phase 1  (0:10 – 0:25)  Abstraction      — icons morph into axis labels; XOR trap.
Phase 2  (0:25 – 0:40)  3D Kernel lift   — camera tilts; z = x·y·0.4; slicer plane.
Phase 3  (0:40 – 0:55)  Architect drop   — dark overlay; titles; credits.
Phase 4  (0:55 – 1:10)  Roadmap + cut    — roadmap items; focus #1; fade to black.
"""

import numpy as np
from manim import *
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import (
        CLASS_A_COLOR, CLASS_B_COLOR, HYPERPLANE_COLOR,
        BG_COLOR, FONT_MAIN, SLATE_GRAY,
    )
except ImportError:
    CLASS_A_COLOR    = "#FF5E5E"
    CLASS_B_COLOR    = "#00C2D1"
    HYPERPLANE_COLOR = "#F9DC5C"
    BG_COLOR         = "#0B0C10"
    FONT_MAIN        = "Montserrat"
    SLATE_GRAY       = "#888888"

# ── Design tokens (scene-local) ───────────────────────────────────────────────
FONT_TITLE       = FONT_MAIN
FONT_BODY        = FONT_MAIN
ERROR_RED        = "#FF3333"
LASER_COLOR      = "#00FFFF"   # Cyan laser
SCAN_GREEN       = "#39FF14"   # Neon green for scan readout
ICON_STROKE      = "#AAAAAA"
ICON_FILL        = "#1A1D27"

POINTS_PER_CLUSTER = 14
CLUSTER_SIGMA      = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_cluster(center, n=POINTS_PER_CLUSTER, sigma=CLUSTER_SIGMA, seed=None):
    """Return *n* 3-D Gaussian points around *center* on the z = 0 plane."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=np.asarray(center)[:2], scale=sigma, size=(n, 2))
    return [np.array([x, y, 0.0]) for x, y in pts]


# ── Stylised biometric icon factories ─────────────────────────────────────────

def make_face_icon(size: float = 0.8) -> VGroup:
    """Minimalist face: oval outline + two eye dots + arc mouth."""
    s = size
    head  = Ellipse(width=s * 1.2, height=s * 1.5,
                    stroke_color=ICON_STROKE, stroke_width=2,
                    fill_color=ICON_FILL, fill_opacity=1)
    eye_l = Dot(LEFT  * s * 0.26 + UP * s * 0.22, radius=s * 0.07, color=LASER_COLOR)
    eye_r = Dot(RIGHT * s * 0.26 + UP * s * 0.22, radius=s * 0.07, color=LASER_COLOR)
    mouth = Arc(radius=s * 0.28, start_angle=-5 * DEGREES,
                angle=-170 * DEGREES, color=ICON_STROKE, stroke_width=2)
    mouth.move_to(DOWN * s * 0.22)
    return VGroup(head, eye_l, eye_r, mouth)


def make_fingerprint_icon(size: float = 0.8) -> VGroup:
    """Concentric arcs simulating a fingerprint loop."""
    s = size
    arcs = VGroup()
    for i, r in enumerate(np.linspace(s * 0.12, s * 0.58, 5)):
        arc = Arc(radius=r,
                  start_angle=200 * DEGREES,
                  angle=-220 * DEGREES,
                  stroke_color=ICON_STROKE,
                  stroke_width=1.8 - i * 0.2,
                  stroke_opacity=0.9 - i * 0.05)
        arcs.add(arc)
    bg = Circle(radius=s * 0.65,
                fill_color=ICON_FILL, fill_opacity=1,
                stroke_color=ICON_STROKE, stroke_width=2)
    return VGroup(bg, arcs)


def make_iris_icon(size: float = 0.8) -> VGroup:
    """Iris: outer ring + pupil + radiating spokes."""
    s = size
    outer  = Circle(radius=s * 0.6, stroke_color=ICON_STROKE, stroke_width=2,
                    fill_color=ICON_FILL, fill_opacity=1)
    iris_r = Circle(radius=s * 0.4, stroke_color=CLASS_B_COLOR,
                    stroke_width=1.5, fill_opacity=0)
    pupil  = Circle(radius=s * 0.18, fill_color="#111111",
                    fill_opacity=1, stroke_width=0)
    spokes = VGroup()
    for angle_deg in range(0, 360, 30):
        a = angle_deg * DEGREES
        spoke = Line(
            s * 0.22 * np.array([np.cos(a), np.sin(a), 0]),
            s * 0.38 * np.array([np.cos(a), np.sin(a), 0]),
            stroke_color=ICON_STROKE, stroke_width=1, stroke_opacity=0.6,
        )
        spokes.add(spoke)
    return VGroup(outer, iris_r, spokes, pupil)


def make_voice_icon(size: float = 0.8) -> VGroup:
    """Voice waveform: vertical bars of varying height."""
    s = size
    bg = RoundedRectangle(width=s * 1.3, height=s * 1.1, corner_radius=s * 0.1,
                          fill_color=ICON_FILL, fill_opacity=1,
                          stroke_color=ICON_STROKE, stroke_width=2)
    heights = [0.25, 0.55, 0.85, 1.0, 0.85, 0.55, 0.25]
    bars    = VGroup()
    xs      = np.linspace(-s * 0.42, s * 0.42, len(heights))
    for x, h in zip(xs, heights):
        bar = Rectangle(width=s * 0.07, height=s * h * 0.7,
                        fill_color=CLASS_B_COLOR, fill_opacity=0.9,
                        stroke_width=0).move_to([x, 0, 0])
        bars.add(bar)
    return VGroup(bg, bars)


# ── Scan-readout number rain ──────────────────────────────────────────────────

def make_scan_readout(n_chars: int = 6) -> Text:
    """One line of random hex / binary noise for the scanning overlay."""
    rng = np.random.default_rng()
    chars = "".join(rng.choice(list("0123456789ABCDEF"), n_chars))
    return Text(chars, font="Courier New", font_size=14, color=SCAN_GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class IntroScene(ThreeDScene):
    """V2.0: Real-world hook → XOR trap → 3D kernel lift → title → roadmap."""

    def construct(self):
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        # ── PHASE 0: The Real-World Hook (0:00 – 0:10) ───────────────────────
        self._phase0_biometric_scan()

        # ── PHASE 1: Abstraction & XOR Trap (0:10 – 0:25) ───────────────────
        (blue_dots, red_dots,
         axes, x_label, y_label,
         hyperplane_line) = self._phase1_xor_trap()

        # ── PHASE 2: 3D Kernel Revelation (0:25 – 0:40) ──────────────────────
        slicing_group = self._phase2_kernel_lift(axes, blue_dots, red_dots)

        # ── PHASE 3: Architect & Title Drop (0:40 – 0:55) ───────────────────
        dark_overlay, title_group = self._phase3_title_drop()

        # ── PHASE 4: Roadmap & Scene Cut (0:55 – 1:10) ───────────────────────
        self._phase4_roadmap_and_cut(
            axes, blue_dots, red_dots, slicing_group,
            dark_overlay, title_group,
        )

    # =========================================================================
    # PHASE 0  —  Biometric Grid + Laser Scan
    # =========================================================================
    def _phase0_biometric_scan(self) -> None:
        """Black screen → 2×2 biometric grid → cyan laser sweeps + scan FX."""

        CELL = 1.9          # cell spacing (centre-to-centre)
        ICON_SIZE = 0.72

        # ── Icon grid layout ──────────────────────────────────────────────────
        offsets = {
            "face":         np.array([-CELL / 2,  CELL / 2, 0]),
            "fingerprint":  np.array([ CELL / 2,  CELL / 2, 0]),
            "iris":         np.array([-CELL / 2, -CELL / 2, 0]),
            "voice":        np.array([ CELL / 2, -CELL / 2, 0]),
        }

        face_icon = make_face_icon(ICON_SIZE).move_to(offsets["face"])
        fp_icon   = make_fingerprint_icon(ICON_SIZE).move_to(offsets["fingerprint"])
        iris_icon = make_iris_icon(ICON_SIZE).move_to(offsets["iris"])
        voice_icon = make_voice_icon(ICON_SIZE).move_to(offsets["voice"])

        # Grid separator lines (subtle)
        h_sep = Line(LEFT * 1.8, RIGHT * 1.8,
                     stroke_color=SLATE_GRAY, stroke_width=0.8, stroke_opacity=0.35)
        v_sep = Line(UP * 1.8, DOWN * 1.8,
                     stroke_color=SLATE_GRAY, stroke_width=0.8, stroke_opacity=0.35)

        # Corner brackets for a high-tech HUD feel
        corners = VGroup()
        for sx, sy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            bx = Line(ORIGIN, RIGHT * 0.3 * sx, stroke_color=LASER_COLOR,
                      stroke_width=2, stroke_opacity=0.7)
            by = Line(ORIGIN, UP * 0.3 * sy, stroke_color=LASER_COLOR,
                      stroke_width=2, stroke_opacity=0.7)
            corner = VGroup(bx, by).move_to(np.array([sx * 2.1, sy * 2.05, 0]))
            corners.add(corner)

        def _icon_label(text: str, pos: np.ndarray) -> Text:
            return Text(text, font=FONT_BODY, font_size=13, color=SLATE_GRAY
                        ).move_to(pos + DOWN * 0.88)

        lbl_face = _icon_label("FACE",        offsets["face"])
        lbl_fp   = _icon_label("FINGERPRINT", offsets["fingerprint"])
        lbl_iris = _icon_label("IRIS",        offsets["iris"])
        lbl_voice = _icon_label("VOICE",      offsets["voice"])
        icon_labels = VGroup(lbl_face, lbl_fp, lbl_iris, lbl_voice)

        # Appear
        self.play(
            LaggedStart(
                FadeIn(h_sep), FadeIn(v_sep),
                *[FadeIn(c) for c in corners],
                lag_ratio=0.15,
            ),
            run_time=0.8,
        )
        self.play(
            LaggedStart(
                FadeIn(face_icon,  shift=DOWN * 0.2),
                FadeIn(fp_icon,    shift=DOWN * 0.2),
                FadeIn(iris_icon,  shift=UP   * 0.2),
                FadeIn(voice_icon, shift=UP   * 0.2),
                lag_ratio=0.18,
            ),
            FadeIn(icon_labels),
            run_time=1.2,
        )
        self.wait(0.3)

        # ── Laser scan line ────────────────────────────────────────────────────
        laser_y_start = 2.05
        laser_y_end   = -2.05
        laser = Line(
            LEFT * 2.2 + UP * laser_y_start,
            RIGHT * 2.2 + UP * laser_y_start,
            color=LASER_COLOR, stroke_width=3,
        ).set_stroke(opacity=0.9)
        
        laser_glow = Line(
            LEFT * 2.2 + UP * laser_y_start,
            RIGHT * 2.2 + UP * laser_y_start,
            color=LASER_COLOR, stroke_width=12,
        ).set_stroke(opacity=0.18)

        laser_group = VGroup(laser_glow, laser)
        self.add(laser_group)

        laser_tracker = ValueTracker(laser_y_start)

        def _laser_updater(mob: VGroup) -> None:
            y = laser_tracker.get_value()
            for line in mob:
                line.put_start_and_end_on(
                    LEFT * 2.2 + UP * y + OUT * 0,
                    RIGHT * 2.2 + UP * y + OUT * 0,
                )

        laser_group.add_updater(_laser_updater)

        # Sweep laser
        self.play(
            laser_tracker.animate.set_value(laser_y_end),
            run_time=2.6,
            rate_func=linear,
        )
        laser_group.remove_updater(_laser_updater)

        # Flash icons
        scan_flashes = []
        for icon in [face_icon, fp_icon, iris_icon, voice_icon]:
            scan_flashes.append(
                Flash(icon.get_center(), color=LASER_COLOR,
                      flash_radius=0.55, num_lines=10, line_stroke_width=2)
            )

        self.play(LaggedStart(*scan_flashes, lag_ratio=0.2), run_time=1.0)
        self.wait(0.3)

        # Store references
        self._face_icon   = face_icon
        self._fp_icon     = fp_icon
        self._iris_icon   = iris_icon
        self._voice_icon  = voice_icon
        self._icon_labels = icon_labels
        self._laser_group = laser_group
        self._grid_chrome = VGroup(h_sep, v_sep, corners)

    # =========================================================================
    # PHASE 1  —  Abstraction & the XOR Trap
    # =========================================================================
    def _phase1_xor_trap(self):
        """Icons morph into axis labels → 2D XOR data appears → linear failure."""

        face_icon  = self._face_icon
        fp_icon    = self._fp_icon
        iris_icon  = self._iris_icon
        voice_icon = self._voice_icon

        self.play(
            FadeOut(iris_icon),
            FadeOut(voice_icon),
            FadeOut(self._icon_labels),
            FadeOut(self._laser_group),
            FadeOut(self._grid_chrome),
            run_time=0.9,
        )
        self.wait(0.2)

        # ── Build the 2D axes ──────────────────────────────────────────────────
        axes_2d = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            x_length=6.0,
            y_length=5.5,
            axis_config={
                "stroke_width": 1.8,
                "color": WHITE,
                "stroke_opacity": 0.55,
            },
            tips=True,
        ).shift(DOWN * 0.15)

        x_label = Text(
            "Face Match Score", font=FONT_BODY, font_size=16, color=SLATE_GRAY,
        ).next_to(axes_2d.x_axis.get_end(), DR, buff=0.15)

        # Căn lề Y rộng hơn (buff=0.65) để không bị đè lên trục Y
        y_label = Text(
            "Fingerprint Score", font=FONT_BODY, font_size=16, color=SLATE_GRAY,
        ).rotate(PI / 2).next_to(axes_2d.y_axis, LEFT, buff=0.65).shift(UP * 0.5)

        # Morph
        face_icon_copy = face_icon.copy()
        fp_icon_copy   = fp_icon.copy()

        self.play(
            face_icon_copy.animate.scale(0.35).move_to(x_label.get_center()),
            fp_icon_copy.animate.scale(0.35).rotate(PI / 2).move_to(y_label.get_center()),
            FadeOut(face_icon),
            FadeOut(fp_icon),
            run_time=1.4,
            rate_func=smooth,
        )
        self.play(
            ReplacementTransform(face_icon_copy, x_label),
            ReplacementTransform(fp_icon_copy,   y_label),
            FadeIn(axes_2d),
            run_time=1.2,
        )
        self.wait(0.5)

        # ── XOR data cloud ────────────────────────────────────────────────────
        SCALE = 1.5 
        blue_centers_2d = [
            np.array([ SCALE,  SCALE, 0.0]),
            np.array([-SCALE, -SCALE, 0.0]),
        ]
        red_centers_2d = [
            np.array([-SCALE,  SCALE, 0.0]),
            np.array([ SCALE, -SCALE, 0.0]),
        ]

        blue_pts_2d = [p for i, c in enumerate(blue_centers_2d)
                       for p in generate_cluster(c, seed=42 + i)]
        red_pts_2d  = [p for i, c in enumerate(red_centers_2d)
                       for p in generate_cluster(c, seed=100 + i)]

        blue_dots = VGroup(
            *[Dot3D(point=p, color=CLASS_B_COLOR, radius=0.07) for p in blue_pts_2d]
        )
        red_dots = VGroup(
            *[Dot3D(point=p, color=CLASS_A_COLOR, radius=0.07) for p in red_pts_2d]
        )

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.5) for d in blue_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.5) for d in red_dots],  lag_ratio=0.04),
            run_time=2.0,
        )
        self.wait(0.5)

        hyperplane_line = Line(
            start=axes_2d.c2p(-4, 0), end=axes_2d.c2p(4, 0),
            color=HYPERPLANE_COLOR, stroke_width=3,
        )
        self.play(Create(hyperplane_line), run_time=0.7)

        angle_tracker = ValueTracker(0.0)
        c1_hp  = ManimColor(HYPERPLANE_COLOR)
        c2_err = ManimColor(ERROR_RED)

        def _update_line(line: Line) -> None:
            a = angle_tracker.get_value()
            start_pt = axes_2d.c2p(-4 * np.cos(a), -4 * np.sin(a))
            end_pt   = axes_2d.c2p( 4 * np.cos(a),  4 * np.sin(a))
            line.put_start_and_end_on(start_pt, end_pt)
            
            mod_a = a % PI
            dist  = min(abs(mod_a - PI / 4), abs(mod_a - 3 * PI / 4))
            t     = max(0.0, 1.0 - dist / 0.18)
            line.set_color(interpolate_color(c1_hp, c2_err, t))
            line.set_stroke(width=3 + 4 * t)

        hyperplane_line.add_updater(_update_line)
        self.play(
            angle_tracker.animate.set_value(PI * 1.25),
            run_time=3.5, rate_func=linear,
        )
        hyperplane_line.remove_updater(_update_line)

        self.play(
            hyperplane_line.animate.set_color(HYPERPLANE_COLOR).set_stroke(opacity=0.22),
            run_time=0.6,
        )
        self.wait(0.8)

        self.play(
            FadeOut(x_label), FadeOut(y_label),
            FadeOut(hyperplane_line),
            run_time=0.7,
        )
        self.wait(0.2)

        return blue_dots, red_dots, axes_2d, x_label, y_label, hyperplane_line

    # =========================================================================
    # PHASE 2  —  3D Kernel Revelation
    # =========================================================================
    def _phase2_kernel_lift(
        self, axes_2d: Axes, blue_dots: VGroup, red_dots: VGroup,
    ) -> VGroup:

        axes3d = ThreeDAxes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            z_range=[-2, 2, 1],
            x_length=6.0, y_length=5.5, z_length=3.5,
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.20},
        ).shift(DOWN * 0.15)

        self.play(ReplacementTransform(axes_2d, axes3d), run_time=0.7)

        lift_anims  = []
        trace_lines = VGroup()

        for dots_grp, color in [(blue_dots, CLASS_B_COLOR), (red_dots, CLASS_A_COLOR)]:
            for d in dots_grp:
                cx, cy, _ = d.get_center()
                z_target  = cx * cy * 0.4
                target    = np.array([cx, cy, z_target])
                trace     = DashedLine(
                    d.get_center(), target,
                    color=color, stroke_width=1.4, dash_length=0.08,
                ).set_opacity(0.45)
                trace_lines.add(trace)
                lift_anims.append(d.animate.move_to(target))

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

        # ── Nâng Z-Axis Label theo tọa độ tuyệt đối 3D ──
        z_label = Text(
            "Quality Score / Kernel Mapping",
            font=FONT_BODY, font_size=13, color=SLATE_GRAY,
        )
        # Đặt chữ lơ lửng tại z = 2.7 (cao hơn mũi tên z=2.0)
        z_label.move_to(axes3d.c2p(0, 0, 2.7))
        
        self.add_fixed_orientation_mobjects(z_label)
        self.play(FadeIn(z_label), run_time=0.7)

        self.move_camera(theta=-30 * DEGREES, run_time=1.5)
        self.move_camera(theta=25 * DEGREES,  run_time=2.8, rate_func=smooth)
        self.move_camera(theta=-25 * DEGREES, run_time=2.0, rate_func=smooth)

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

    # =========================================================================
    # PHASE 3  —  Architect & Title Drop
    # =========================================================================
    def _phase3_title_drop(self):
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
        )
        title_line2.next_to(title_line1, DOWN, buff=0.28)
        title_vgroup = VGroup(title_line1, title_line2)

        divider_line = Line(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=2)
        divider_line.width = title_vgroup.width
        divider_line.next_to(title_vgroup, DOWN, buff=0.38)

        credits_text = VGroup(
            Text("Created by: Lê Hà Thanh Chương", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
            Text("Faculty of Information Technology, VNU-HCM University of Science", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
            Text("Course: Pattern Recognition", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        credits_text.next_to(divider_line, DOWN, buff=0.38)
        credits_text.align_to(divider_line, LEFT)

        title_group = VGroup(title_vgroup, divider_line, credits_text)
        title_group.move_to(ORIGIN)

        self.add_fixed_in_frame_mobjects(title_group)
        self.remove(title_group)

        self.play(Write(title_line1),                          run_time=1.5)
        self.play(FadeIn(title_line2, shift=UP * 0.2),         run_time=1.0)
        self.play(GrowFromCenter(divider_line),                run_time=0.7)
        self.play(FadeIn(credits_text, shift=UP * 0.2),        run_time=1.2)
        self.wait(2.0)

        return dark_overlay, title_group

    # =========================================================================
    # PHASE 4  —  Roadmap + Focus Effect + Cinematic Cut
    # =========================================================================
    def _phase4_roadmap_and_cut(
        self, axes_2d: Axes, blue_dots: VGroup, red_dots: VGroup, slicing_group: VGroup,
        dark_overlay, title_group: VGroup,
    ) -> None:
        
        self.play(
            title_group.animate.scale(0.52).to_corner(UL, buff=0.65),
            run_time=1.4,
        )
        self.wait(0.2)

        roadmap_strings = [
            "Hạn chế của hệ thống đơn lẻ (Unibiometrics)",
            "Kết hợp điểm số và mô hình tuyến tính",
            "Kernel Trick: ánh xạ vào không gian bảo mật 3D",
            "So sánh ranh giới phân loại (Linear vs RBF)",
        ]

        roadmap_items = VGroup()
        for label_str in roadmap_strings:
            bullet = Dot(radius=0.065, color=HYPERPLANE_COLOR)
            label  = Text(label_str, font=FONT_BODY, font_size=24, color=WHITE)
            label.next_to(bullet, RIGHT, buff=0.25)
            roadmap_items.add(VGroup(bullet, label))

        roadmap_items.arrange(DOWN, aligned_edge=LEFT, buff=0.42)
        roadmap_items.next_to(title_group, DOWN, buff=0.85)
        roadmap_items.align_to(title_group, LEFT)
        roadmap_items.shift(RIGHT * 0.28)

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

        # ── Focus Effect: highlight item #1 ───────────────────────────────────
        item_focus  = roadmap_items[0]
        items_other = VGroup(*roadmap_items[1:])

        self.stop_ambient_camera_rotation()

        self.play(FadeOut(items_other), run_time=0.7)

        self.remove(item_focus)
        item_focus_static = item_focus.copy()
        item_focus_static.set_color(HYPERPLANE_COLOR)
        item_focus_static.scale(1.55)
        item_focus_static.move_to(ORIGIN + UP * 0.3)
        self.add(item_focus_static)
        self.play(FadeIn(item_focus_static, scale=0.8), run_time=0.9)
        self.wait(1.2)

        # Dọn dẹp màn hình và fade to black gọn gàng
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