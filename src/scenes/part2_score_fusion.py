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
        Minh họa quá trình mô hình tuyến tính "đảo" (rotate) để tìm điểm Support Vectors.
        """
        # ── 1. Khởi tạo Grid Toán học & Dữ liệu trừu tượng ───────────────────
        np.random.seed(42)
        X, y = make_classification(n_samples=40, n_features=2, n_redundant=0, n_classes=2, random_state=42)
        
        # Tạo lưới toạ độ (Isotropic grid để margin vuông góc)
        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-5, 5, 1], 
            x_length=7.5, y_length=7.5, 
            background_line_style={"stroke_opacity": 0.4}
        )
        self.play(Create(plane), run_time=1.5)
        
        x_label = plane.get_x_axis_label("x_1", direction=DOWN)
        y_label = plane.get_y_axis_label("x_2", direction=LEFT)
        self.play(Create(x_label), Create(y_label))
        
        # Chấm đỏ / xanh mộc mạc
        dots = VGroup(*[
            Dot(point=plane.c2p(pt[0], pt[1]), color=CLASS_A_COLOR if y[i]==0 else CLASS_B_COLOR, radius=0.08) 
            for i, pt in enumerate(X)
        ])
        self.play(Create(dots), run_time=1.2)
        self.wait(0.5)

        title = Text("Searching for the optimal margin", font=FONT, font_size=28, color=WHITE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # ── 2. Tính toán mô hình SVM tối ưu bằng scikit-learn ────────────────
        svm = SVC(kernel="linear", C=1, random_state=42).fit(X, y)
        w_opt = svm.coef_[0]
        b_opt = svm.intercept_[0]
        svs = svm.support_vectors_
        
        opt_angle = np.arctan2(-w_opt[0], w_opt[1])
        opt_margin = 1.0 / np.linalg.norm(w_opt)
        
        # Điểm tâm (center point) của đường phân chia để làm tâm xoay
        if abs(w_opt[1]) > 1e-5:
            cx, cy = 0.0, -b_opt / w_opt[1]
        else:
            cx, cy = -b_opt / w_opt[0], 0.0

        # ── 3. Thiết lập ValueTrackers cho hiệu ứng tìm lề (Breathing) ───────
        angle_tracker = ValueTracker(opt_angle + PI / 3.5) # Bắt đầu lệch góc
        margin_tracker = ValueTracker(0.0)                 # Lề bắt đầu từ 0
        
        db_line = Line(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=3)
        mp_line = DashedLine(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_opacity=0.6)
        mn_line = DashedLine(LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_opacity=0.6)
        margin_band = Polygon(ORIGIN, ORIGIN, ORIGIN, ORIGIN, fill_color=HYPERPLANE_COLOR, fill_opacity=0.2, stroke_width=0)
        
        def _get_lines():
            """Tính toán tọa độ các đường dựa trên góc và khoảng cách lề hiện tại."""
            a = angle_tracker.get_value()
            m = margin_tracker.get_value()
            
            dir_vec = np.array([np.cos(a), np.sin(a)])
            perp_vec = np.array([-np.sin(a), np.cos(a)])
            
            p_center = plane.c2p(cx, cy)
            p_dir = plane.c2p(cx + dir_vec[0], cy + dir_vec[1]) - p_center
            
            length = 15
            db_s = p_center - length * p_dir
            db_e = p_center + length * p_dir
            
            p_perp = plane.c2p(cx + perp_vec[0], cy + perp_vec[1]) - p_center
            m_offset = m * p_perp
            
            return db_s, db_e, m_offset
            
        # Gắn Updaters
        db_line.add_updater(lambda mob: mob.put_start_and_end_on(_get_lines()[0], _get_lines()[1]))
        mp_line.add_updater(lambda mob: mob.put_start_and_end_on(_get_lines()[0] + _get_lines()[2], _get_lines()[1] + _get_lines()[2]))
        mn_line.add_updater(lambda mob: mob.put_start_and_end_on(_get_lines()[0] - _get_lines()[2], _get_lines()[1] - _get_lines()[2]))
        margin_band.add_updater(lambda mob: mob.set_points_as_corners([
            _get_lines()[0] + _get_lines()[2], _get_lines()[1] + _get_lines()[2],
            _get_lines()[1] - _get_lines()[2], _get_lines()[0] - _get_lines()[2],
            _get_lines()[0] + _get_lines()[2]
        ]))

        self.add(margin_band, db_line, mp_line, mn_line)
        
        # ── 4. Hoạt ảnh: "Đảo" liên tục và mở rộng lề ────────────────────────
        # Đường thẳng xoay từ sai lệch về vị trí tối ưu
        self.play(
            angle_tracker.animate.set_value(opt_angle - PI/6),
            margin_tracker.animate.set_value(opt_margin * 0.5),
            run_time=1.5, rate_func=there_and_back_with_pause
        )
        self.play(
            angle_tracker.animate.set_value(opt_angle),
            margin_tracker.animate.set_value(opt_margin),
            run_time=2.0, rate_func=smooth
        )
        
        # Hiệu ứng Breathing Margin (Nhấp nháy lớn nhỏ và "đụng" vào các chấm)
        self.play(margin_tracker.animate.set_value(opt_margin * 1.25), run_time=0.4, rate_func=rush_into)
        self.play(margin_tracker.animate.set_value(opt_margin), run_time=0.4, rate_func=rush_from)
        self.play(margin_tracker.animate.set_value(opt_margin * 1.1), run_time=0.3, rate_func=rush_into)
        self.play(margin_tracker.animate.set_value(opt_margin), run_time=0.3, rate_func=rush_from)
        
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
        
        # Gỡ updaters để tối ưu hóa hiệu suất vẽ
        db_line.clear_updaters()
        mp_line.clear_updaters()
        mn_line.clear_updaters()
        margin_band.clear_updaters()
        
        # Hiển thị chú thích Support Vectors
        sv_label = Text("Support Vectors", font=FONT, font_size=22, color=YELLOW).to_corner(UR, buff=0.5)
        arrow = Arrow(sv_label.get_left(), sv_rings[-1].get_right(), color=YELLOW, buff=0.1)
        formula = MathTex(r"\frac{2}{\|\mathbf{w}\|}").scale(1.2).next_to(sv_label, DOWN, buff=0.3)
        
        self.play(Write(sv_label), Create(arrow), Write(formula), run_time=1.0)
        self.wait(2.0)

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
        
        # Rescue Noisy Genuine
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

        # Rescue Spoof Impostor
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

        gen_targets = _scatter_2d((0.72, 0.75), N_CLOUD, 0.09, RNG_SEED_GEN_2D)
        imp_targets = _scatter_2d((0.27, 0.25), N_CLOUD, 0.09, RNG_SEED_IMP_2D)
        all_targets = gen_targets + imp_targets
        
        self.play(*[d.animate.move_to(axes.c2p(tx, ty)) for d, (tx, ty) in zip(crowd_dots, all_targets)], run_time=2.0, rate_func=smooth)
        self.wait(0.4)

        genuine_cloud = VGroup(*crowd_dots[:N_CLOUD])
        impostor_cloud = VGroup(*crowd_dots[N_CLOUD:])

        # Snap SVM
        X_data = np.array([axes.p2c(d.get_center())[:2] for d in genuine_cloud] + [axes.p2c(d.get_center())[:2] for d in impostor_cloud])
        y_data = np.array([1]*N_CLOUD + [-1]*N_CLOUD)
        clf = SVC(kernel="linear", C=1e6, random_state=42).fit(X_data, y_data)
        
        w_final = clf.coef_[0].copy()
        b_final = clf.intercept_[0]
        margin_dist = 1.0 / np.linalg.norm(w_final)
        
        x_lo, x_hi = AXIS_RANGE[0], AXIS_RANGE[1]
        y_s = (-w_final[0] * x_lo - b_final) / w_final[1]
        y_e = (-w_final[0] * x_hi - b_final) / w_final[1]
        
        line_vec = np.array([x_hi - x_lo, y_e - y_s])
        line_len = np.linalg.norm(line_vec)
        perp = np.array([-line_vec[1], line_vec[0]]) / line_len
        
        pos_sx, pos_sy = x_lo + perp[0]*margin_dist, y_s + perp[1]*margin_dist
        pos_ex, pos_ey = x_hi + perp[0]*margin_dist, y_e + perp[1]*margin_dist
        neg_sx, neg_sy = x_lo - perp[0]*margin_dist, y_s - perp[1]*margin_dist
        neg_ex, neg_ey = x_hi - perp[0]*margin_dist, y_e - perp[1]*margin_dist
        
        final_hp = Line(axes.c2p(x_lo, y_s), axes.c2p(x_hi, y_e), color=HYPERPLANE_COLOR, stroke_width=4)
        final_mp = DashedLine(axes.c2p(pos_sx, pos_sy), axes.c2p(pos_ex, pos_ey), color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1)
        final_mn = DashedLine(axes.c2p(neg_sx, neg_sy), axes.c2p(neg_ex, neg_ey), color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1)
        final_band = Polygon(axes.c2p(pos_sx, pos_sy), axes.c2p(pos_ex, pos_ey), axes.c2p(neg_ex, neg_ey), axes.c2p(neg_sx, neg_sy), fill_color=HYPERPLANE_COLOR, fill_opacity=0.15, stroke_width=0)
        
        self.play(FadeIn(final_band), Create(final_hp), Create(final_mp), Create(final_mn), run_time=1.5)
        self.play(Flash(final_hp.get_center(), color=HYPERPLANE_COLOR, flash_radius=0.5, num_lines=10), run_time=0.5)
        self.wait(1.0)
        
        return genuine_cloud, impostor_cloud, final_hp

    def _phase4_xor_trap(self, axes, genuine_cloud, impostor_cloud, hyperplane):
        flash_rect = Rectangle(width=config.frame_width + 2, height=config.frame_height + 2, fill_color=SPOOF_RED, fill_opacity=0.0, stroke_width=0)
        self.add(flash_rect)
        self.play(flash_rect.animate.set_fill(opacity=0.30), run_time=0.3, rate_func=there_and_back)
        self.remove(flash_rect)

        warning_en = Text("Spoof Attack", font=FONT, font_size=28, weight=BOLD, color=SPOOF_RED).to_edge(UP, buff=0.30)
        warning_vn = Text("XOR Trap!", font=FONT, font_size=22, color=SPOOF_RED).next_to(warning_en, DOWN, buff=0.10)
        warning_bg = SurroundingRectangle(VGroup(warning_en, warning_vn), fill_color=BLACK, fill_opacity=0.75, stroke_color=SPOOF_RED, stroke_width=1.5, corner_radius=0.14, buff=0.18)
        self.play(FadeIn(warning_bg), Write(warning_en), FadeIn(warning_vn, shift=DOWN * 0.1), run_time=0.8)
        self.wait(0.4)

        spoof_tl_pts = _scatter_2d((0.27, 0.75), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_TL)
        spoof_tl = VGroup(*[Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in spoof_tl_pts])
        spoof_br_pts = _scatter_2d((0.72, 0.25), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_BR)
        spoof_br = VGroup(*[Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09) for x, y in spoof_br_pts])

        tl_caption = Text("Silicone fingerprint", font=FONT, font_size=13, color=IMPOSTOR_COLOR)
        tl_caption_bg = SurroundingRectangle(tl_caption, fill_color=BG_COLOR, fill_opacity=0.7, stroke_width=0, buff=0.06)
        tl_cap_group = VGroup(tl_caption_bg, tl_caption).next_to(spoof_tl, RIGHT, buff=0.15)
        
        br_caption = Text("3D face mask", font=FONT, font_size=13, color=IMPOSTOR_COLOR)
        br_caption_bg = SurroundingRectangle(br_caption, fill_color=BG_COLOR, fill_opacity=0.7, stroke_width=0, buff=0.06)
        br_cap_group = VGroup(br_caption_bg, br_caption).next_to(spoof_br, LEFT, buff=0.15)

        self.play(LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_tl], lag_ratio=0.07), LaggedStart(*[FadeIn(d, scale=0.4) for d in spoof_br], lag_ratio=0.07), run_time=1.0)
        self.play(FadeIn(tl_cap_group, shift=LEFT * 0.12), FadeIn(br_cap_group, shift=RIGHT * 0.12), run_time=0.5)
        self.wait(0.5)

        angle_tracker = ValueTracker(0)
        c_yellow = ManimColor(HYPERPLANE_COLOR)
        c_red = ManimColor(SPOOF_RED)

        def _hp_updater(line):
            a = angle_tracker.get_value()
            cx = axes.c2p(0.5, 0.5)
            dir_vec = np.array([np.cos(a), np.sin(a), 0.0])
            line.put_start_and_end_on(cx - 3.5 * dir_vec, cx + 3.5 * dir_vec)
            t = (np.sin(a * 6) + 1) / 2
            line.set_color(interpolate_color(c_yellow, c_red, t))
            line.set_stroke(width=3 + 3 * t)

        hyperplane.add_updater(_hp_updater)
        self.play(angle_tracker.animate.set_value(PI * 1.4), run_time=2.5, rate_func=linear)
        hyperplane.remove_updater(_hp_updater)
        self.play(hyperplane.animate.set_color(SPOOF_RED).set_stroke(opacity=0.35), run_time=0.5)
        
        failure_text = Text("Linear model fails!", font=FONT, font_size=20, weight=BOLD, color=SPOOF_RED).to_edge(DOWN)
        self.play(Write(failure_text))
        self.wait(1.5)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
        self.wait(0.5)