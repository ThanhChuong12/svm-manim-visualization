from manim import *
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import CLASS_A_COLOR, CLASS_B_COLOR, HYPERPLANE_COLOR
except ImportError:
    CLASS_A_COLOR = "#FF5E5E"
    CLASS_B_COLOR = "#00C2D1"
    HYPERPLANE_COLOR = "#F9DC5C"

# ── Design tokens ─────────────────────────────────────────────────────────────
FONT_TITLE = "Montserrat"
FONT_BODY = "Montserrat"
BG_COLOR = "#0B0C10"
SLATE_GRAY = "#888888"
ERROR_RED = "#FF3333"
POINTS_PER_CLUSTER = 16
CLUSTER_SIGMA = 0.15


def generate_cluster(center, n=POINTS_PER_CLUSTER, sigma=CLUSTER_SIGMA, seed=None):
    """Sample *n* 2D Gaussian points around *center*, returned on the z = 0 plane."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center[:2], scale=sigma, size=(n, 2))
    return [np.array([x, y, 0.0]) for x, y in pts]


class IntroScene(ThreeDScene):
    def construct(self):
        # ── PHASE 1: The Cybersecurity Pain Point (0:00 – 0:15) ───────────────
        self.camera.background_color = BG_COLOR
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        # Reduced lengths create breathing room at frame edges.
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 1],
            x_length=5.5, y_length=5, z_length=3.5,
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.15},
        ).set_opacity(0.15)
        self.play(FadeIn(axes), run_time=1)

        # Multi-line labels at axis tips; horizontal orientation for readability.
        x_label = Text("Normalized\nFace Deviation", font=FONT_BODY, font_size=14, color=SLATE_GRAY)
        x_label.next_to(axes.x_axis.get_end(), DR, buff=0.1)

        y_label = Text("Normalized\nFingerprint Deviation", font=FONT_BODY, font_size=14, color=SLATE_GRAY)
        y_label.next_to(axes.y_axis.get_end(), UR, buff=0.1)

        self.play(FadeIn(x_label), FadeIn(y_label), run_time=0.8)

        # XOR spoofing matrix in normalised deviation space:
        #   Blue (Genuine) — correlated deviations (both high or both low).
        #   Red (Impostor) — anti-correlated (one faked, one genuine).
        blue_centers = [np.array([1.5, 1.5, 0.0]), np.array([-1.5, -1.5, 0.0])]
        red_centers = [np.array([1.5, -1.5, 0.0]), np.array([-1.5, 1.5, 0.0])]

        blue_pts = [p for i, c in enumerate(blue_centers) for p in generate_cluster(c, seed=42 + i)]
        red_pts = [p for i, c in enumerate(red_centers) for p in generate_cluster(c, seed=100 + i)]

        blue_dots = VGroup(*[Dot3D(point=p, color=CLASS_B_COLOR, radius=0.08) for p in blue_pts])
        red_dots = VGroup(*[Dot3D(point=p, color=CLASS_A_COLOR, radius=0.08) for p in red_pts])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in blue_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in red_dots], lag_ratio=0.04),
            run_time=2.5,
        )
        self.wait(0.5)

        # Task 1.3: Rotating line anchored at the origin, demonstrating that no
        # line through (0,0) can separate the XOR clusters.
        hyperplane_line = Line(
            start=LEFT * 4, end=RIGHT * 4,
            color=HYPERPLANE_COLOR, stroke_width=3,
        ).move_to(ORIGIN)
        self.play(Create(hyperplane_line), run_time=0.8)

        angle_tracker = ValueTracker(0)
        c1_hp = ManimColor(HYPERPLANE_COLOR)
        c2_err = ManimColor(ERROR_RED)

        def update_line(line):
            a = angle_tracker.get_value()
            # Rebuild line from origin each frame to guarantee origin-anchored rotation.
            new_dir = np.array([np.cos(a), np.sin(a), 0.0])
            line.put_start_and_end_on(-4 * new_dir, 4 * new_dir)
            # Flash red at XOR failure angles (45° / 135°).
            mod_a = a % PI
            dist = min(abs(mod_a - PI / 4), abs(mod_a - 3 * PI / 4))
            t = max(0.0, 1.0 - dist / 0.18)
            line.set_color(interpolate_color(c1_hp, c2_err, t))
            line.set_stroke(width=3 + 4 * t)

        hyperplane_line.add_updater(update_line)
        self.play(angle_tracker.animate.set_value(PI * 1.25), run_time=4, rate_func=linear)
        hyperplane_line.remove_updater(update_line)

        # Hold the failure state so the viewer absorbs the impossibility.
        self.play(hyperplane_line.animate.set_color(HYPERPLANE_COLOR).set_opacity(0.25), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(hyperplane_line), run_time=0.8)
        self.wait(0.5)

        # ── PHASE 2: The Kernel Revelation (0:15 – 0:30) ─────────────────────

        self.play(FadeOut(x_label), FadeOut(y_label), run_time=0.5)

        # Lift map: z = x·y × 0.4 (dampened to preserve cluster cohesion).
        lift_anims = []
        trace_lines = VGroup()

        for dots_group, color in [(blue_dots, CLASS_B_COLOR), (red_dots, CLASS_A_COLOR)]:
            for d in dots_group:
                cx, cy, _ = d.get_center()
                z_target = cx * cy * 0.4
                target = np.array([cx, cy, z_target])
                trace = DashedLine(
                    start=d.get_center(), end=target,
                    color=color, stroke_width=1.5, dash_length=0.08,
                ).set_opacity(0.5)
                trace_lines.add(trace)
                lift_anims.append(d.animate.move_to(target))

        # Camera unlock and data lift in parallel for cinematic parallax.
        self.move_camera(
            phi=80 * DEGREES, theta=-45 * DEGREES,
            run_time=3.5, rate_func=smooth,
            added_anims=[
                LaggedStart(*[Create(t) for t in trace_lines], lag_ratio=0.02, run_time=1.5),
                AnimationGroup(*lift_anims, run_time=3.5, rate_func=smooth),
            ],
        )

        self.play(FadeOut(trace_lines), run_time=0.5)
        self.wait(0.5)

        # Task 1.2: Z-axis label near the tip, fixed orientation so it always faces camera.
        z_label = Text("Quality Score / Kernel Mapping", font=FONT_BODY, font_size=14, color=SLATE_GRAY)
        z_label.next_to(axes.z_axis.get_end(), UP, buff=0.15)
        self.add_fixed_orientation_mobjects(z_label)
        self.play(FadeIn(z_label), run_time=0.8)

        # Task 3.1: Slow orbit around Z-axis to reveal the vertical gap between classes.
        self.move_camera(phi=80 * DEGREES, theta=-30 * DEGREES, run_time=1.5)
        self.move_camera(theta=30 * DEGREES, run_time=3, rate_func=smooth)
        self.move_camera(theta=-30 * DEGREES, run_time=2, rate_func=smooth)

        # Hyperplane at z = 0 — gridless translucent surface with glowing edge.
        slicing_plane = Polygon(
            np.array([-3.5, -3.5, 0]),
            np.array([3.5, -3.5, 0]),
            np.array([3.5, 3.5, 0]),
            np.array([-3.5, 3.5, 0]),
            fill_color=HYPERPLANE_COLOR, fill_opacity=0.2,
            stroke_color=HYPERPLANE_COLOR, stroke_width=3, stroke_opacity=0.85,
        )
        slicing_group = VGroup(slicing_plane)
        self.play(DrawBorderThenFill(slicing_plane), run_time=2)

        self.move_camera(theta=-25 * DEGREES, run_time=2)
        self.wait(1.5)

        # ── PHASE 3: The Architect Drop (0:30 – 0:45) ────────────────────────

        self.begin_ambient_camera_rotation(rate=0.06)
        self.play(FadeOut(z_label), run_time=0.5)

        dark_overlay = Rectangle(width=20, height=20, color=BLACK, fill_opacity=0.88)
        self.add_fixed_in_frame_mobjects(dark_overlay)
        self.remove(dark_overlay)
        self.play(FadeIn(dark_overlay), run_time=1.5)

        title_line1 = Text(
            "Multibiometric Fusion & SVM",
            font=FONT_TITLE, weight=BOLD, font_size=48, color=WHITE,
        )
        title_line2 = Text(
            "Unlocking The Kernel Trick",
            font=FONT_TITLE, font_size=36, slant=ITALIC, color=HYPERPLANE_COLOR,
        )
        title_line2.next_to(title_line1, DOWN, buff=0.25)
        title_vgroup = VGroup(title_line1, title_line2)

        divider_line = Line(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=2)
        divider_line.width = title_vgroup.width
        divider_line.next_to(title_vgroup, DOWN, buff=0.35)

        credits_text = VGroup(
            Text("Created by: Lê Hà Thanh Chương", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
            Text("Faculty of Information Technology, VNU-HCM University of Science", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
            Text("Course: Pattern Recognition", font=FONT_BODY, font_size=19, color=SLATE_GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        credits_text.next_to(divider_line, DOWN, buff=0.35)
        credits_text.align_to(divider_line, LEFT)

        title_group = VGroup(title_vgroup, divider_line, credits_text)
        title_group.move_to(ORIGIN)

        self.add_fixed_in_frame_mobjects(title_group)
        self.remove(title_group)

        self.play(Write(title_line1), run_time=1.5)
        self.play(FadeIn(title_line2, shift=UP * 0.2), run_time=1)
        self.play(GrowFromCenter(divider_line), run_time=0.7)
        self.play(FadeIn(credits_text, shift=UP * 0.2), run_time=1.2)
        self.wait(2)

        # ── PHASE 4: The Roadmap (0:45 – 1:00) ──────────────────────────────

        self.play(title_group.animate.scale(0.55).to_corner(UL, buff=0.7), run_time=1.5)

        # Task 4: Bullet-point list with yellow dot markers instead of numbers.
        roadmap_strings = [
            "Hạn chế của hệ thống đơn lẻ (Unibiometrics)",
            "Kết hợp điểm số và mô hình tuyến tính",
            "Kernel Trick: ánh xạ vào không gian bảo mật 3D",
            "So sánh ranh giới phân loại (Linear vs RBF)",
        ]
        roadmap_items = VGroup()
        for label_str in roadmap_strings:
            bullet = Dot(radius=0.06, color=HYPERPLANE_COLOR)
            label = Text(label_str, font=FONT_BODY, font_size=26, color=WHITE)
            label.next_to(bullet, RIGHT, buff=0.25)
            roadmap_items.add(VGroup(bullet, label))

        roadmap_items.arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        roadmap_items.next_to(title_group, DOWN, buff=0.9)
        roadmap_items.align_to(title_group, LEFT)
        roadmap_items.shift(RIGHT * 0.3)

        self.add_fixed_in_frame_mobjects(roadmap_items)
        self.remove(roadmap_items)

        self.play(
            LaggedStart(*[FadeIn(item, shift=RIGHT * 0.4) for item in roadmap_items], lag_ratio=0.45),
            run_time=3,
        )
        self.wait(2)

        # ── Outro ─────────────────────────────────────────────────────────────
        self.stop_ambient_camera_rotation()
        self.play(
            FadeOut(dark_overlay),
            FadeOut(title_group),
            FadeOut(roadmap_items),
            FadeOut(axes),
            FadeOut(blue_dots),
            FadeOut(red_dots),
            FadeOut(slicing_group),
            run_time=1.5,
        )
        self.wait(1)
