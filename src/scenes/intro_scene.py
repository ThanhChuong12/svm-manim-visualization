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
FONT_TITLE = "Inter"
FONT_BODY = "Inter"
BG_COLOR = "#0B0C10"
SLATE_GRAY = "#888888"
ERROR_RED = "#FF3333"
POINTS_PER_CLUSTER = 16
# Tight σ keeps z = x·y from stretching clusters into vertical pillars in 3D.
CLUSTER_SIGMA = 0.12


def generate_cluster(center, n=POINTS_PER_CLUSTER, sigma=CLUSTER_SIGMA, seed=None):
    """Sample *n* 2D Gaussian points around *center* on the z = 0 plane.

    Uses a fixed seed for deterministic renders across quality levels.
    """
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center[:2], scale=sigma, size=(n, 2))
    return [np.array([x, y, 0.0]) for x, y in pts]


class IntroScene(ThreeDScene):
    def construct(self):
        # ── PHASE 1: 2D Stalemate (0:00 – 0:15) ──────────────────────────────
        self.camera.background_color = BG_COLOR

        # phi=0, theta=-90° collapses the 3D camera to a flat top-down 2D view.
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)

        # Low opacity keeps axes as a spatial reference, not a visual focal point.
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            z_length=4,
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.15},
        ).set_opacity(0.15)
        self.play(FadeIn(axes), run_time=1)

        # XOR layout: same-label points share a quadrant diagonal (not a half-plane).
        # Blue: Q1 (++), Q3 (--) | Red: Q2 (-+), Q4 (+-)
        blue_centers = [np.array([1.5, 1.5, 0.0]), np.array([-1.5, -1.5, 0.0])]
        red_centers = [np.array([1.5, -1.5, 0.0]), np.array([-1.5, 1.5, 0.0])]

        blue_points_data = [
            p for i, c in enumerate(blue_centers)
            for p in generate_cluster(c, seed=42 + i)
        ]
        red_points_data = [
            p for i, c in enumerate(red_centers)
            for p in generate_cluster(c, seed=100 + i)
        ]

        blue_dots = VGroup(*[Dot3D(point=p, color=CLASS_B_COLOR, radius=0.08) for p in blue_points_data])
        red_dots = VGroup(*[Dot3D(point=p, color=CLASS_A_COLOR, radius=0.08) for p in red_points_data])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in blue_dots], lag_ratio=0.04),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in red_dots], lag_ratio=0.04),
            run_time=2.5,
        )
        self.wait(0.5)

        # The candidate separator sweeps 225° and flashes red whenever it crosses
        # the XOR failure angles (45° / 135°), where same-class points mix.
        hyperplane_line = Line(start=LEFT * 4, end=RIGHT * 4, color=HYPERPLANE_COLOR, stroke_width=3)
        self.play(Create(hyperplane_line), run_time=0.8)

        angle_tracker = ValueTracker(0)
        c1_hp = ManimColor(HYPERPLANE_COLOR)
        c2_err = ManimColor(ERROR_RED)

        def update_line(line):
            a = angle_tracker.get_value()
            line.set_angle(a)
            # Proximity to a failure angle [0, 1]: 1 = fully misclassifying.
            mod_a = a % PI
            dist = min(abs(mod_a - PI / 4), abs(mod_a - 3 * PI / 4))
            t = max(0.0, 1.0 - dist / 0.18)
            line.set_color(interpolate_color(c1_hp, c2_err, t))
            line.set_stroke(width=3 + 4 * t)

        hyperplane_line.add_updater(update_line)
        self.play(angle_tracker.animate.set_value(PI * 1.25), run_time=4, rate_func=linear)
        hyperplane_line.remove_updater(update_line)

        # Dim and hold: force the viewer to sit with the mathematical impossibility.
        self.play(hyperplane_line.animate.set_color(HYPERPLANE_COLOR).set_opacity(0.25), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(hyperplane_line), run_time=0.8)
        self.wait(0.5)

        # ── PHASE 2: Kernel Trick Revelation (0:15 – 0:30) ───────────────────

        # Lift map: z = x·y (dampened by 0.4 to prevent cluster elongation).
        # Blue dots land at z > 0 (product of same-sign coords), Red at z < 0.
        lift_anims = []
        trace_lines = VGroup()

        for dot, color in [(blue_dots, CLASS_B_COLOR), (red_dots, CLASS_A_COLOR)]:
            for d in dot:
                cx, cy, _ = d.get_center()
                z_target = cx * cy * 0.4
                target = np.array([cx, cy, z_target])
                trace = DashedLine(
                    start=d.get_center(), end=target,
                    color=color, stroke_width=1.5, dash_length=0.08,
                ).set_opacity(0.5)
                trace_lines.add(trace)
                lift_anims.append(d.animate.move_to(target))

        # Camera unlock and data lift run in parallel for a single cinematic motion.
        self.move_camera(
            phi=75 * DEGREES,
            theta=-45 * DEGREES,
            run_time=3.5,
            rate_func=smooth,
            added_anims=[
                LaggedStart(*[Create(t) for t in trace_lines], lag_ratio=0.02, run_time=1.5),
                AnimationGroup(*lift_anims, run_time=3.5, rate_func=smooth),
            ],
        )

        self.play(FadeOut(trace_lines), run_time=0.5)
        self.wait(0.5)

        # Near-horizontal angle maximises the visible vertical gap between classes,
        # making the hyperplane's entrance at z = 0 immediately legible.
        self.move_camera(phi=80 * DEGREES, theta=-30 * DEGREES, run_time=1.5)

        slicing_plane = Polygon(
            np.array([-3.5, -3.5, 0]),
            np.array([3.5, -3.5, 0]),
            np.array([3.5, 3.5, 0]),
            np.array([-3.5, 3.5, 0]),
            fill_color=HYPERPLANE_COLOR,
            fill_opacity=0.2,
            stroke_color=HYPERPLANE_COLOR,
            stroke_width=3,
            stroke_opacity=0.85,
        )
        slicing_group = VGroup(slicing_plane)
        self.play(DrawBorderThenFill(slicing_plane), run_time=2)

        self.move_camera(theta=-25 * DEGREES, run_time=2)
        self.wait(1.5)

        # ── PHASE 3: Title Card (0:30 – 0:45) ────────────────────────────────

        # Slow ambient rotation keeps the 3D scene alive behind the title overlay.
        self.begin_ambient_camera_rotation(rate=0.06)

        # add_fixed_in_frame_mobjects + immediate remove lets FadeIn control entry.
        dark_overlay = Rectangle(width=20, height=20, color=BLACK, fill_opacity=0.88)
        self.add_fixed_in_frame_mobjects(dark_overlay)
        self.remove(dark_overlay)
        self.play(FadeIn(dark_overlay), run_time=1.5)

        # Two-line hierarchy: primary (bold white) + secondary (italic yellow).
        title_line1 = Text("Support Vector Machines", font=FONT_TITLE, weight=BOLD, font_size=48, color=WHITE)
        title_line2 = Text("& The Kernel Trick", font=FONT_TITLE, font_size=36, slant=ITALIC, color=HYPERPLANE_COLOR)
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

        # ── PHASE 4: Roadmap (0:45 – 1:00) ───────────────────────────────────

        self.play(title_group.animate.scale(0.55).to_corner(UL, buff=0.7), run_time=1.5)

        # Numbered items: index in HYPERPLANE_COLOR, label in white.
        roadmap_data = [
            ("1.", "Phân loại tuyến tính & Maximum Margin"),
            ("2.", "Bài toán XOR (Giới hạn của tuyến tính)"),
            ("3.", "Kernel Trick: Từ 2D lên không gian đa chiều"),
            ("4.", "So sánh các loại Kernel (Linear, Poly, RBF)"),
        ]
        roadmap_items = VGroup()
        for num_str, label_str in roadmap_data:
            num = Text(num_str, font=FONT_BODY, font_size=28, weight=BOLD, color=HYPERPLANE_COLOR)
            label = Text(label_str, font=FONT_BODY, font_size=28, color=WHITE)
            label.next_to(num, RIGHT, buff=0.25)
            roadmap_items.add(VGroup(num, label))

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
