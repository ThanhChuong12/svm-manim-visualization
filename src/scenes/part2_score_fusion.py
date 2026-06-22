"""Scene 3 — Score Combination & Linear SVM (Director's Cut V2.0).

Phase 1 (0:00-0:20): Biometric Vectorization (Split-screen -> Score Vector)
Phase 2 (0:20-0:50): The "Math Class" Detour (Abstract dataset, iterative SVM learning)
Phase 3 (0:50-1:10): Application & Rescue (Restore biometric axes, 1D->2D rescue, snap SVM)
Phase 4 (1:10-1:25): The XOR Trap (Spoof attack defeats linear model)
"""

import numpy as np
from manim import *
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import (
        GENUINE_COLOR, IMPOSTOR_COLOR, HYPERPLANE_COLOR,
        BG_COLOR, FONT_MAIN, SLATE_GRAY,
        CLASS_A_COLOR, CLASS_B_COLOR, MARGIN_OPACITY,
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
    MARGIN_OPACITY   = 0.30

try:
    from utils.visual_helpers import (
        make_genuine_icon, make_fingerprint_icon,
        make_noisy_icon, make_spoof_icon,
    )
except ImportError:
    def make_genuine_icon(size=0.6):
        return Circle(radius=size * 0.5, color=GENUINE_COLOR, stroke_width=2)
    def make_fingerprint_icon(size=0.6):
        return Circle(radius=size * 0.5, color="#AABBFF", stroke_width=2)
    def make_noisy_icon(size=0.6):
        return Circle(radius=size * 0.5, color="#888888", stroke_width=2)
    def make_spoof_icon(size=0.6):
        return Square(side_length=size, color=IMPOSTOR_COLOR, stroke_width=2)

try:
    from core.fusion_data import scatter_2d as _scatter_2d
except ImportError:
    def _scatter_2d(center, n, sigma, seed, x_clip=(0.05, 0.95), y_clip=(0.05, 0.95)):
        rng = np.random.default_rng(seed)
        pts = rng.normal(loc=center, scale=sigma, size=(n, 2))
        pts[:, 0] = np.clip(pts[:, 0], *x_clip)
        pts[:, 1] = np.clip(pts[:, 1], *y_clip)
        return [(float(x), float(y)) for x, y in pts]

FONT            = FONT_MAIN
SUPPORT_RING    = "#FFFFFF"
MARGIN_COLOR    = HYPERPLANE_COLOR
SPOOF_RED       = "#FF2222"
AXIS_OPACITY    = 0.55
AXIS_RANGE      = [0.0, 1.0, 0.25]
FP_COLOR        = "#AABBFF"
LASER_CYAN      = "#00FFFF"

N_CLOUD = 18
RNG_SEED_GEN_2D   = 11
RNG_SEED_IMP_2D   = 17
RNG_SEED_SPOOF_TL = 29
RNG_SEED_SPOOF_BR = 37

class ScoreCombinationScene(MovingCameraScene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # Phase 1: Biometric Vectorization
        axes, x_label, y_label = self._build_biometric_axes()
        biometric_group = VGroup(axes, x_label, y_label)
        self._phase1_vectorization(axes, x_label, y_label)
        
        # Phase 2: Math Detour
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
        self._phase2_math_detour()
        
        # Restore empty scene after detour
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
        
        # Phase 3: Application & Rescue
        self.play(FadeIn(biometric_group))
        genuine_cloud, impostor_cloud, svm_line = self._phase3_application(axes)
        
        # Phase 4: XOR Trap
        self._phase4_xor_trap(axes, genuine_cloud, impostor_cloud, svm_line)

    def _build_biometric_axes(self):
        axes = Axes(
            x_range=AXIS_RANGE, y_range=AXIS_RANGE,
            x_length=6.5, y_length=5.5,
            axis_config={
                "stroke_width": 2, "color": WHITE,
                "stroke_opacity": AXIS_OPACITY, "include_ticks": True, "tick_size": 0.06,
            },
            tips=True,
        ).shift(DOWN * 0.25)
        x_label = Text("Face Match Score  (s₁)", font=FONT, font_size=16, color=SLATE_GRAY).next_to(axes.x_axis, DOWN, buff=0.25).align_to(axes.x_axis, RIGHT)
        y_label = Text("Fingerprint Score  (s₂)", font=FONT, font_size=16, color=SLATE_GRAY).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.28)
        return axes, x_label, y_label

    def _create_scanner_panel(self, icon_factory, title_text, accent_color, position):
        panel_w, panel_h = 2.8, 2.2
        panel_bg = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.12,
            stroke_color=accent_color, stroke_width=1.5, stroke_opacity=0.5,
            fill_color=BG_COLOR, fill_opacity=0.9,
        ).move_to(position)
        icon = icon_factory(0.45).move_to(panel_bg.get_center() + UP * 0.3)
        title = Text(title_text, font=FONT, font_size=13, color=accent_color).next_to(panel_bg, UP, buff=0.10)
        laser = Rectangle(width=panel_w * 0.75, height=0.04, fill_color=LASER_CYAN, fill_opacity=0.8, stroke_width=0).move_to(panel_bg.get_top() + DOWN * 0.3)
        return panel_bg, icon, title, laser

    def _phase1_vectorization(self, axes, x_label, y_label):
        face_bg, face_icon, face_title, face_laser = self._create_scanner_panel(make_genuine_icon, "Face Recognition", CLASS_B_COLOR, LEFT * 3.2 + UP * 2.4)
        fp_bg, fp_icon, fp_title, fp_laser = self._create_scanner_panel(make_fingerprint_icon, "Fingerprint Scan", FP_COLOR, RIGHT * 3.2 + UP * 2.4)
        
        self.play(FadeIn(face_bg), FadeIn(fp_bg), FadeIn(face_title), FadeIn(fp_title), GrowFromCenter(face_icon), GrowFromCenter(fp_icon), run_time=1.0)
        
        scan_dist = 2.2 - 0.6
        self.play(face_laser.animate.shift(DOWN * scan_dist), fp_laser.animate.shift(DOWN * scan_dist), FadeIn(face_laser), FadeIn(fp_laser), run_time=1.2, rate_func=linear)
        self.play(FadeOut(face_laser), FadeOut(fp_laser), run_time=0.3)
        
        s1_val = MathTex(r"s_1 = 0.85", color=CLASS_B_COLOR, font_size=34).next_to(face_bg, DOWN, buff=0.18)
        s2_val = MathTex(r"s_2 = 0.92", color=FP_COLOR, font_size=34).next_to(fp_bg, DOWN, buff=0.18)
        self.play(FadeIn(s1_val, shift=UP * 0.2), FadeIn(s2_val, shift=UP * 0.2), run_time=0.9)
        self.wait(0.5)

        score_matrix = Matrix([["0.85"], ["0.92"]], left_bracket="[", right_bracket="]", element_to_mobject_config={"font_size": 32, "color": CLASS_B_COLOR})
        prefix = MathTex(r"\mathbf{S} =", color=CLASS_B_COLOR, font_size=36)
        vector_group = VGroup(prefix, score_matrix).arrange(RIGHT, buff=0.2).move_to(UP * 2.4)
        
        self.play(
            FadeOut(face_bg), FadeOut(fp_bg), FadeOut(face_icon), FadeOut(fp_icon), FadeOut(face_title), FadeOut(fp_title),
            ReplacementTransform(s1_val, score_matrix.get_entries()[0]), ReplacementTransform(s2_val, score_matrix.get_entries()[1]),
            FadeIn(score_matrix.get_brackets()), FadeIn(prefix), run_time=1.4
        )
        self.wait(0.4)
        self.play(Flash(score_matrix, color=CLASS_B_COLOR, flash_radius=0.9), run_time=0.5)
        self.play(vector_group.animate.scale(0.55).to_corner(UL, buff=0.35), run_time=0.8)
        self.wait(0.3)

        target_pos = axes.c2p(0.85, 0.92)
        sample_dot = Dot(point=target_pos, color=CLASS_B_COLOR, radius=0.12)
        guide_x = DashedLine(axes.c2p(0.85, 0.0), axes.c2p(0.85, 0.92), color=CLASS_B_COLOR, stroke_opacity=0.4, dash_length=0.08)
        guide_y = DashedLine(axes.c2p(0.0, 0.92), axes.c2p(0.85, 0.92), color=CLASS_B_COLOR, stroke_opacity=0.4, dash_length=0.08)
        
        flying_dot = Dot(point=vector_group.get_center(), color=CLASS_B_COLOR, radius=0.10)
        glow_trail = TracedPath(flying_dot.get_center, stroke_color=CLASS_B_COLOR, stroke_width=3, stroke_opacity=0.4, dissipating_time=0.6)
        self.add(glow_trail)
        
        self.play(FadeIn(axes), FadeIn(x_label), FadeIn(y_label), flying_dot.animate.move_to(target_pos), Create(guide_x), Create(guide_y), run_time=1.5, rate_func=smooth)
        self.remove(glow_trail, flying_dot)
        self.add(sample_dot)
        self.play(Flash(sample_dot, color=WHITE, flash_radius=0.4, num_lines=10), run_time=0.5)
        
        coord_label = MathTex(r"(0.85,\ 0.92)", color=CLASS_B_COLOR, font_size=22).next_to(sample_dot, UR, buff=0.14)
        self.play(FadeIn(coord_label, shift=UP * 0.12), run_time=0.5)
        self.wait(0.6)
        
        self.play(FadeOut(guide_x), FadeOut(guide_y), FadeOut(coord_label), FadeOut(sample_dot), FadeOut(vector_group), run_time=0.8)
        self.wait(0.2)

    def _phase2_math_detour(self):
        """
        Phase 2: The "Math Class" Detour
        Dạy học SVM bằng dữ liệu trừu tượng với không gian toán học chuẩn (Grid).
        Phục dựng hiệu ứng Iteration (thử và sai) kết hợp mở rộng lề.
        """
        # ── 1. Khởi tạo Grid Toán học & Dữ liệu tĩnh (Toy Dataset) ────────────
        blue_points = np.array([
            [-3.0, -1.0], [-2.5, -2.0], [-1.5, -2.5], [-2.0, -0.5], 
            [-1.0, -1.5], [-0.5, -3.0], [-4.0, -1.5], [-3.5, -3.0],
            [-0.5, -0.5] # <-- Support Vector Xanh
        ])
        red_points = np.array([
            [1.0, 3.0], [2.0, 2.5], [1.5, 4.0], [2.5, 3.5],
            [3.5, 2.0], [3.0, 1.5], [4.0, 3.0], [3.0, 4.0],
            [0.5, 1.5], # <-- Support Vector Đỏ 1
            [1.5, 0.5]  # <-- Support Vector Đỏ 2
        ])
        X = np.vstack((blue_points, red_points))
        y = np.array([0] * len(blue_points) + [1] * len(red_points))
        
        # Tạo lưới toạ độ (Isotropic grid)
        plane = NumberPlane(
            x_range=[-6, 6, 1], y_range=[-4, 4, 1], 
            x_length=9.0, y_length=6.0, 
            background_line_style={"stroke_opacity": 0.4}
        ).shift(DOWN * 0.5)
        
        self.play(Create(plane), run_time=1.5)
        
        x_label = plane.get_x_axis_label("x_1", direction=DOWN)
        y_label = plane.get_y_axis_label("x_2", direction=LEFT)
        self.play(Create(x_label), Create(y_label))
        
        # Vẽ các chấm dữ liệu
        dots = VGroup(*[
            Dot(point=plane.c2p(pt[0], pt[1]), color=CLASS_A_COLOR if y[i]==0 else CLASS_B_COLOR, radius=0.08) 
            for i, pt in enumerate(X)
        ])
        self.play(Create(dots), run_time=1.2)
        self.wait(0.5)

        # Tiêu đề chuyển lên góc UL (vùng mù)
        title = Text("Searching for the optimal margin", font=FONT, font_size=28, color=WHITE).to_corner(UL)
        self.play(Write(title))
        self.wait(0.5)

        # ── 2. Tính toán mô hình SVM tối ưu bằng scikit-learn ────────────────
        svm = SVC(kernel="linear", C=1000).fit(X, y)
        w_opt = svm.coef_[0]
        b_opt = svm.intercept_[0]
        svs = svm.support_vectors_
        
        opt_angle = np.arctan2(-w_opt[0], w_opt[1])
        opt_margin = 1.0 / np.linalg.norm(w_opt)
        
        if abs(w_opt[1]) > 1e-5:
            cx, cy = 0.0, -b_opt / w_opt[1]
        else:
            cx, cy = -b_opt / w_opt[0], 0.0

        # ── 3. Thiết lập ValueTrackers cho hiệu ứng tìm lề (Searching) ───────
        angle_tracker = ValueTracker(opt_angle + PI / 4) # Bắt đầu lệch 45 độ
        margin_tracker = ValueTracker(0.0)               # Lề bắt đầu bằng 0
        
        db_line = Line(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=3)
        mp_line = DashedLine(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_opacity=0.6)
        mn_line = DashedLine(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_opacity=0.6)
        margin_band = Polygon(ORIGIN, ORIGIN, ORIGIN, ORIGIN, fill_color=HYPERPLANE_COLOR, fill_opacity=0.2, stroke_width=0)
        
        def _get_lines():
            a = angle_tracker.get_value()
            m = margin_tracker.get_value()
            
            dir_vec = np.array([np.cos(a), np.sin(a)])
            perp_vec = np.array([-np.sin(a), np.cos(a)])
            
            p_center = plane.c2p(cx, cy)
            p_dir = plane.c2p(cx + dir_vec[0], cy + dir_vec[1]) - p_center
            
            length = 20 
            db_s = p_center - length * p_dir
            db_e = p_center + length * p_dir
            
            p_perp = plane.c2p(cx + perp_vec[0], cy + perp_vec[1]) - p_center
            m_offset = m * p_perp
            
            return db_s, db_e, m_offset
            
        db_line.add_updater(lambda mob: mob.put_start_and_end_on(_get_lines()[0], _get_lines()[1]))
        mp_line.add_updater(lambda mob: mob.put_start_and_end_on(_get_lines()[0] + _get_lines()[2], _get_lines()[1] + _get_lines()[2]))
        mn_line.add_updater(lambda mob: mob.put_start_and_end_on(_get_lines()[0] - _get_lines()[2], _get_lines()[1] - _get_lines()[2]))
        margin_band.add_updater(lambda mob: mob.set_points_as_corners([
            _get_lines()[0] + _get_lines()[2], _get_lines()[1] + _get_lines()[2],
            _get_lines()[1] - _get_lines()[2], _get_lines()[0] - _get_lines()[2],
            _get_lines()[0] + _get_lines()[2]
        ]))

        self.add(margin_band, db_line, mp_line, mn_line)
        
        # Bộ đếm Iteration
        iter_num = 1
        iter_text = Text(f"Iteration: {iter_num}", font=FONT, font_size=24, color=YELLOW).to_corner(UR).shift(DOWN * 0.5)
        self.play(FadeIn(iter_text))

        # ── 4. Hoạt ảnh: Cỗ máy quét giật cục (Discrete Iterations) ──────────
        test_angles = [
            opt_angle + 30 * DEGREES, 
            opt_angle - 15 * DEGREES, 
            opt_angle + 5 * DEGREES, 
            opt_angle
        ]
        
        for angle in test_angles:
            iter_num += 1
            new_iter_text = Text(f"Iteration: {iter_num}", font=FONT, font_size=24, color=YELLOW).to_corner(UR).shift(DOWN * 0.5)
            
            self.play(
                angle_tracker.animate.set_value(angle),
                # FIX: Dùng ReplacementTransform để gỡ bỏ triệt để text cũ khỏi màn hình
                ReplacementTransform(iter_text, new_iter_text),
                run_time=0.6,
                rate_func=linear if angle != opt_angle else smooth
            )
            self.wait(0.2)
            iter_text = new_iter_text # Cập nhật con trỏ

        # Khi đã tìm đúng góc, Lề (Margin) bắt đầu nở ra
        final_iter_text = Text("Optimal Found!", font=FONT, font_size=24, color=GENUINE_COLOR).to_corner(UR).shift(DOWN * 0.5)
        
        self.play(
            margin_tracker.animate.set_value(opt_margin),
            # FIX: Tương tự, dùng ReplacementTransform
            ReplacementTransform(iter_text, final_iter_text),
            run_time=1.5,
            rate_func=smooth
        )
        
        # ── 5. Khi đụng lề chuẩn, SVs phát sáng và xuất hiện vòng tròn ───────
        sv_rings = VGroup(*[
            Circle(radius=0.15, color=YELLOW, stroke_width=4).move_to(plane.c2p(pt[0], pt[1])) 
            for pt in svs
        ])
        self.play(
            FadeIn(sv_rings, scale=1.5), 
            *[Flash(plane.c2p(pt[0], pt[1]), color=YELLOW) for pt in svs], 
            run_time=0.8
        )
        
        # ── 6. Cinematic Focus: Làm mờ điểm thừa ─────────────────────────────
        non_sv_dots = VGroup(*[d for i, d in enumerate(dots) if i not in svm.support_])
        self.play(non_sv_dots.animate.set_opacity(0.15), run_time=1.0)
        
        db_line.clear_updaters()
        mp_line.clear_updaters()
        mn_line.clear_updaters()
        margin_band.clear_updaters()
        
        # Căn chỉnh chú thích 2/||w||
        sv_label = Text("Support Vectors", font=FONT, font_size=22, color=YELLOW).to_corner(UR, buff=0.5).shift(DOWN * 1.5)
        arrow = Arrow(sv_label.get_bottom(), sv_rings[-1].get_top(), color=YELLOW, buff=0.1)
        formula = MathTex(r"\frac{2}{\|\mathbf{w}\|}").scale(1.2).next_to(sv_label, DOWN, buff=0.3)
        
        self.play(Write(sv_label), Create(arrow), Write(formula), run_time=1.0)
        self.wait(2.5)

    # =========================================================================
    # PHASE 3: Application & Rescue
    # =========================================================================
    def _phase3_application(self, axes):
        rng_crowd = np.random.default_rng(99)
        crowd_x = rng_crowd.uniform(0.35, 0.65, N_CLOUD * 2)
        crowd_cols = [GENUINE_COLOR] * N_CLOUD + [IMPOSTOR_COLOR] * N_CLOUD
        crowd_dots = VGroup(*[Dot(axes.c2p(x, 0.0), color=c, radius=0.09) for x, c in zip(crowd_x, crowd_cols)])
        
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in crowd_dots], lag_ratio=0.04), run_time=1.2)
        
        confusion_tag = Text("1D Overlap", font=FONT, font_size=18, color=SLATE_GRAY).to_edge(UP, buff=0.35)
        self.play(FadeIn(confusion_tag, shift=DOWN * 0.15), run_time=0.6)
        self.wait(0.6)

        self.play(self.camera.frame.animate.set_width(10), run_time=1.0, rate_func=smooth)
        
        # ── Rescue Noisy Genuine ──
        noisy_icon = make_noisy_icon(0.35)
        noisy_start_pos = axes.c2p(0.45, 0.08)
        noisy_icon.move_to(noisy_start_pos + UP * 0.55)
        noisy_label = Text("Noisy Face", font=FONT, font_size=12, color="#888888").next_to(noisy_icon, UP, buff=0.1)
        noisy_target_pos = axes.c2p(0.45, 0.82)
        noisy_target_center = noisy_target_pos + UP * 0.55

        self.play(GrowFromCenter(noisy_icon), FadeIn(noisy_label), run_time=0.7)
        self.wait(0.3)
        
        rescue_arrow_g = Arrow(noisy_start_pos + UP * 0.20, noisy_target_center + DOWN * 0.40, color=GENUINE_COLOR, stroke_width=2, stroke_opacity=0.6, buff=0.1, max_tip_length_to_length_ratio=0.12)
        move_vector = noisy_target_center - noisy_icon.get_center()
        
        self.play(noisy_icon.animate.move_to(noisy_target_center), noisy_label.animate.shift(move_vector), Create(rescue_arrow_g), run_time=1.4, rate_func=smooth)
        rescue_text_g = Text("Saved!", font=FONT, font_size=13, color=GENUINE_COLOR).next_to(noisy_icon, RIGHT, buff=0.15)
        self.play(FadeIn(rescue_text_g, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.4)

        # ── Rescue Spoof Impostor ──
        spoof_icon = make_spoof_icon(0.30)
        spoof_start_pos = axes.c2p(0.60, 0.25)
        spoof_icon.move_to(spoof_start_pos + UP * 0.5)
        spoof_label = Text("Spoof", font=FONT, font_size=12, color=IMPOSTOR_COLOR).next_to(spoof_icon, DOWN, buff=0.1)
        
        spoof_mid_pos = axes.c2p(0.78, 0.55)
        spoof_mid_center = spoof_mid_pos + UP * 0.45
        spoof_final_pos = axes.c2p(0.78, 0.18)
        spoof_final_center = spoof_final_pos + UP * 0.45

        self.play(GrowFromCenter(spoof_icon), FadeIn(spoof_label), run_time=0.7)
        self.wait(0.3)
        self.play(spoof_icon.animate.move_to(spoof_mid_center), spoof_label.animate.next_to(Dot(spoof_mid_center), DOWN, buff=0.1), run_time=0.8, rate_func=smooth)
        rescue_arrow_s = Arrow(spoof_mid_center + DOWN * 0.2, spoof_final_center + UP * 0.45, color=IMPOSTOR_COLOR, stroke_width=2, stroke_opacity=0.6, buff=0.1, max_tip_length_to_length_ratio=0.12)
        self.play(spoof_icon.animate.move_to(spoof_final_center), spoof_label.animate.next_to(Dot(spoof_final_center), DOWN, buff=0.1), Create(rescue_arrow_s), run_time=1.2, rate_func=rush_into)
        rescue_text_s = Text("Rejected!", font=FONT, font_size=13, color=IMPOSTOR_COLOR).next_to(spoof_icon, RIGHT, buff=0.15)
        self.play(FadeIn(rescue_text_s, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.5)

        self.play(
            FadeOut(noisy_icon), FadeOut(noisy_label), FadeOut(rescue_text_g), FadeOut(rescue_arrow_g),
            FadeOut(spoof_icon), FadeOut(spoof_label), FadeOut(rescue_text_s), FadeOut(rescue_arrow_s), FadeOut(confusion_tag),
            self.camera.frame.animate.set_width(14.222), run_time=1.0,
        )

        # ── Expand Crowd ──
        gen_targets = _scatter_2d((0.72, 0.75), N_CLOUD, 0.09, RNG_SEED_GEN_2D)
        imp_targets = _scatter_2d((0.27, 0.25), N_CLOUD, 0.09, RNG_SEED_IMP_2D)
        all_targets = gen_targets + imp_targets
        
        self.play(*[d.animate.move_to(axes.c2p(tx, ty)) for d, (tx, ty) in zip(crowd_dots, all_targets)], run_time=2.0, rate_func=smooth)
        self.wait(0.4)

        genuine_cloud = VGroup(*crowd_dots[:N_CLOUD])
        impostor_cloud = VGroup(*crowd_dots[N_CLOUD:])

        # ── Snap Final SVM (FIX: Trimming lines and margin strictly inside axes bounds) ──
        X_data = np.array([axes.p2c(d.get_center())[:2] for d in genuine_cloud] + [axes.p2c(d.get_center())[:2] for d in impostor_cloud])
        y_data = np.array([1]*N_CLOUD + [-1]*N_CLOUD)
        clf = SVC(kernel="linear", C=1e6, random_state=42).fit(X_data, y_data)
        
        w_final = clf.coef_[0].copy()
        b_final = clf.intercept_[0]

        # Thuật toán cắt tọa độ: Trả về chính xác các giao điểm của một đường thẳng với hình vuông lưới [0, 1]
        def get_clipped_pts(w, b, offset=0):
            pts = []
            for x_val in [0.0, 1.0]:
                y_val = (offset - b - w[0]*x_val) / w[1]
                if -1e-4 <= y_val <= 1.0 + 1e-4: pts.append((x_val, y_val))
            for y_val in [0.0, 1.0]:
                x_val = (offset - b - w[1]*y_val) / w[0]
                if -1e-4 <= x_val <= 1.0 + 1e-4: pts.append((x_val, y_val))
            unique_pts = []
            for p in pts:
                if not any(np.linalg.norm(np.array(p) - np.array(up)) < 1e-4 for up in unique_pts):
                    unique_pts.append(p)
            unique_pts.sort(key=lambda p: p[0])
            return unique_pts

        hp_pts = get_clipped_pts(w_final, b_final, 0)
        mp_pts = get_clipped_pts(w_final, b_final, 1)
        mn_pts = get_clipped_pts(w_final, b_final, -1)

        final_hp = Line(axes.c2p(*hp_pts[0]), axes.c2p(*hp_pts[1]), color=HYPERPLANE_COLOR, stroke_width=4)
        final_mp = DashedLine(axes.c2p(*mp_pts[0]), axes.c2p(*mp_pts[1]), color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1)
        final_mn = DashedLine(axes.c2p(*mn_pts[0]), axes.c2p(*mn_pts[1]), color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1)

        # Polygon Clipping: Gom các điểm mép lề + các góc của trục tọa độ lọt thỏm bên trong lề
        band_pts_data = []
        band_pts_data.extend(mp_pts)
        band_pts_data.extend(mn_pts[::-1]) # Đảo ngược danh sách lề âm để tạo vòng tròn kín
        
        for c in [(0.,0.), (1.,0.), (1.,1.), (0.,1.)]:
            val = w_final[0]*c[0] + w_final[1]*c[1] + b_final
            if -1.0 + 1e-4 < val < 1.0 - 1e-4:
                band_pts_data.append(c)
                
        # Sắp xếp các điểm theo chiều kim đồng hồ để tạo mảng Polygon không bị lỗi xoắn
        center = np.mean(band_pts_data, axis=0)
        band_pts_data.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        final_band = Polygon(*[axes.c2p(*p) for p in band_pts_data], fill_color=HYPERPLANE_COLOR, fill_opacity=0.15, stroke_width=0)
        
        self.play(FadeIn(final_band), Create(final_hp), Create(final_mp), Create(final_mn), run_time=1.5)
        self.play(Flash(final_hp.get_center(), color=HYPERPLANE_COLOR, flash_radius=0.5, num_lines=10), run_time=0.5)
        self.wait(1.0)
        
        return genuine_cloud, impostor_cloud, final_hp

    # =========================================================================
    # PHASE 4: The XOR Trap
    # =========================================================================
    def _phase4_xor_trap(self, axes, genuine_cloud, impostor_cloud, hyperplane):
        # ── 1. System Alert Flash ────────────────────────────────────────────
        flash_rect = Rectangle(
            width=config.frame_width + 2, height=config.frame_height + 2, 
            fill_color=SPOOF_RED, fill_opacity=0.0, stroke_width=0
        )
        self.add(flash_rect)
        self.play(flash_rect.animate.set_fill(opacity=0.30), run_time=0.3, rate_func=there_and_back)
        self.remove(flash_rect)

        # ── 2. Top Warning Banner ────────────────────────────────────────────
        warning_en = Text("Spoof Attack", font=FONT, font_size=28, weight=BOLD, color=SPOOF_RED).to_edge(UP, buff=0.25)
        warning_vn = Text("XOR Trap!", font=FONT, font_size=20, color=SPOOF_RED).next_to(warning_en, DOWN, buff=0.10)
        warning_bg = SurroundingRectangle(
            VGroup(warning_en, warning_vn), fill_color=BLACK, fill_opacity=0.85, 
            stroke_color=SPOOF_RED, stroke_width=2, corner_radius=0.1, buff=0.2
        )
        warning_group = VGroup(warning_bg, warning_en, warning_vn)
        
        self.play(FadeIn(warning_bg), Write(warning_en), FadeIn(warning_vn, shift=DOWN * 0.1), run_time=0.7)
        self.play(Wiggle(warning_group, scale_value=1.05), run_time=0.5)

        # ── 3. Inject Spoof Clusters (XOR Layout) ────────────────────────────
        spoof_tl_pts = _scatter_2d((0.27, 0.75), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_TL)
        spoof_tl = VGroup(*[Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in spoof_tl_pts])
        
        spoof_br_pts = _scatter_2d((0.72, 0.25), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_BR)
        spoof_br = VGroup(*[Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in spoof_br_pts])

        # Negative Space Labeling
        tl_caption = Text("Silicone fingerprint\n(Giả mạo vân tay)", font=FONT, font_size=13, color=IMPOSTOR_COLOR)
        tl_caption_bg = SurroundingRectangle(tl_caption, fill_color=BG_COLOR, fill_opacity=0.85, stroke_width=0, buff=0.08)
        tl_cap_group = VGroup(tl_caption_bg, tl_caption).next_to(spoof_tl, LEFT, buff=0.25)
        
        br_caption = Text("3D face mask\n(Mặt nạ khuôn mặt)", font=FONT, font_size=13, color=IMPOSTOR_COLOR)
        br_caption_bg = SurroundingRectangle(br_caption, fill_color=BG_COLOR, fill_opacity=0.85, stroke_width=0, buff=0.08)
        br_cap_group = VGroup(br_caption_bg, br_caption).next_to(spoof_br, RIGHT, buff=0.25)

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_tl], lag_ratio=0.05), 
            LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_br], lag_ratio=0.05), 
            run_time=1.0
        )
        self.play(FadeIn(tl_cap_group, shift=RIGHT * 0.1), FadeIn(br_cap_group, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.5)

        # ── 4. Hyperplane Failure Animation ──────────────────────────────────
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
            line.set_stroke(width=3 + 5 * t)

        hyperplane.add_updater(_hp_updater)
        self.play(angle_tracker.animate.set_value(PI * 1.5), run_time=2.5, rate_func=linear)
        hyperplane.remove_updater(_hp_updater)
        
        # ── 5. Background Dimming & Center Error Modal ───────────────────────
        self.play(
            hyperplane.animate.set_color(SPOOF_RED).set_stroke(width=5, opacity=0.6),
            VGroup(genuine_cloud, impostor_cloud, spoof_tl, spoof_br, axes, tl_cap_group, br_cap_group).animate.set_opacity(0.25),
            run_time=0.4
        )
        
        cross_size = 0.35
        cross = VGroup(
            Line(UL * cross_size, DR * cross_size, color=SPOOF_RED, stroke_width=5),
            Line(UR * cross_size, DL * cross_size, color=SPOOF_RED, stroke_width=5)
        )
        failure_text = Text("Linear Model Fails!", font=FONT, font_size=26, weight=BOLD, color=SPOOF_RED)
        modal_content = VGroup(cross, failure_text).arrange(RIGHT, buff=0.3)
        
        failure_bg = SurroundingRectangle(
            modal_content, fill_color=BLACK, fill_opacity=0.95, 
            stroke_color=SPOOF_RED, stroke_width=2.5, corner_radius=0.15, buff=0.3
        )
        
        # Centering the modal directly in the middle of the axes space
        failure_modal = VGroup(failure_bg, modal_content).move_to(axes.c2p(0.5, 0.5))
        
        self.play(FadeIn(failure_modal, scale=1.2), run_time=0.6)
        self.play(
            Flash(failure_modal, color=SPOOF_RED, flash_radius=2.5, num_lines=15),
            Wiggle(warning_group, scale_value=1.1),
            run_time=0.6
        )
        self.wait(2.0)

        # ── 6. Scene Cleanup ─────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(0.5)