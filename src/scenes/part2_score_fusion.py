"""Scene 3 — Score Combination & Linear SVM (Final Polish V1.0).

Phase 1 (0:00–0:20): Feature extraction split-screen → score vector → delayed axes reveal.
Phase 2 (0:20–0:40): 1D overlap crowd rescued into clean 2D separation with camera zoom.
Phase 3 (0:40–0:55): Hard-margin SVM + perpendicular margin math + support vectors.
Phase 4 (0:55–1:10): XOR spoof attack; the linear boundary fails visually.

Uses MovingCameraScene for the Phase 2 zoom effect.
Hard-margin SVM fitted via sklearn SVC (C=1e6), margin computed using perpendicular
unit vectors for academic precision (adapted from code.py reference).
"""

import numpy as np
from manim import *
from sklearn.svm import SVC
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

# ── Scene-local design tokens ─────────────────────────────────────────────────
FONT            = FONT_MAIN
SUPPORT_RING    = "#FFFFFF"
MARGIN_COLOR    = HYPERPLANE_COLOR
SPOOF_RED       = "#FF2222"
AXIS_OPACITY    = 0.55
AXIS_RANGE      = [0.0, 1.0, 0.25]
FP_COLOR        = "#AABBFF"           # Fingerprint accent colour
LASER_CYAN      = "#00FFFF"

# Reproducible random seeds
RNG_SEED_GEN_2D   = 11
RNG_SEED_IMP_2D   = 17
RNG_SEED_SPOOF_TL = 29
RNG_SEED_SPOOF_BR = 37

N_CLOUD = 18


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_2d(
    center: tuple[float, float],
    n: int,
    sigma: float,
    seed: int,
    x_clip: tuple[float, float] = (0.05, 0.95),
    y_clip: tuple[float, float] = (0.05, 0.95),
) -> list[tuple[float, float]]:
    """Return *n* (x, y) score-space points clustered around *center*."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=center, scale=sigma, size=(n, 2))
    pts[:, 0] = np.clip(pts[:, 0], *x_clip)
    pts[:, 1] = np.clip(pts[:, 1], *y_clip)
    return [(float(x), float(y)) for x, y in pts]


# ─────────────────────────────────────────────────────────────────────────────
# Main Scene
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCombinationScene(MovingCameraScene):
    """Scene 3: Score Vectorisation → 2D Separation → Linear SVM → XOR Failure."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # Build axes (not yet added to scene — delayed reveal in Phase 1)
        axes = Axes(
            x_range=AXIS_RANGE,
            y_range=AXIS_RANGE,
            x_length=6.5,
            y_length=5.5,
            axis_config={
                "stroke_width": 2,
                "color": WHITE,
                "stroke_opacity": AXIS_OPACITY,
                "include_ticks": True,
                "tick_size": 0.06,
            },
            tips=True,
        ).shift(DOWN * 0.25)

        x_label = Text(
            "Face Match Score  (s₁)", font=FONT, font_size=16, color=SLATE_GRAY,
        ).next_to(axes.x_axis, DOWN, buff=0.25).align_to(axes.x_axis, RIGHT)

        y_label = Text(
            "Fingerprint Score  (s₂)", font=FONT, font_size=16, color=SLATE_GRAY,
        ).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.28)

        # Phase orchestration
        self._phase1_vectorization(axes, x_label, y_label)
        genuine_cloud, impostor_cloud = self._phase2_rescue(axes)
        hyperplane_line = self._phase3_linear_svm(axes, genuine_cloud, impostor_cloud)
        self._phase4_xor_dilemma(axes, genuine_cloud, impostor_cloud, hyperplane_line)

    # ─────────────────────────────────────────────────────────────────────────
    # SOLID Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _create_scanner_panel(
        self,
        icon_factory,
        title_text: str,
        accent_color: str,
        position: np.ndarray,
    ) -> tuple[VGroup, VGroup, Text, Rectangle]:
        """Build one scanner panel: border, icon, title, laser line.

        Returns (panel_bg, icon, title, laser).
        """
        panel_w, panel_h = 2.8, 2.2

        panel_bg = RoundedRectangle(
            width=panel_w, height=panel_h, corner_radius=0.12,
            stroke_color=accent_color, stroke_width=1.5, stroke_opacity=0.5,
            fill_color=BG_COLOR, fill_opacity=0.9,
        ).move_to(position)

        icon = icon_factory(0.45)
        icon.move_to(panel_bg.get_center() + UP * 0.3)

        title = Text(
            title_text, font=FONT, font_size=13, color=accent_color,
        ).next_to(panel_bg, UP, buff=0.10)

        laser = Rectangle(
            width=panel_w * 0.75, height=0.04,
            fill_color=LASER_CYAN, fill_opacity=0.8, stroke_width=0,
        ).move_to(panel_bg.get_top() + DOWN * 0.3)

        return panel_bg, icon, title, laser

    def _draw_svm_margins(
        self, axes: Axes, w: np.ndarray, b: float,
        x_start: float, x_end: float,
    ) -> tuple[Line, DashedLine, DashedLine, Polygon, float]:
        """Compute perpendicular margin geometry and build static Manim objects.

        Uses the perpendicular unit vector method (adapted from code.py reference)
        to ensure margin lines pass exactly through support vector centres.

        Returns (decision_line, margin_pos, margin_neg, margin_band, margin_distance).
        """
        # Decision boundary endpoints
        y_start = (-w[0] * x_start - b) / w[1]
        y_end   = (-w[0] * x_end   - b) / w[1]

        # Margin distance = 1 / ||w||
        margin_distance = 1.0 / np.linalg.norm(w)

        # Perpendicular unit vector to the boundary direction
        line_vec = np.array([x_end - x_start, y_end - y_start])
        line_len = np.linalg.norm(line_vec)
        perp = np.array([-line_vec[1], line_vec[0]]) / line_len

        # Positive margin (shifted towards genuine cluster)
        pos_sx = x_start + perp[0] * margin_distance
        pos_sy = y_start + perp[1] * margin_distance
        pos_ex = x_end   + perp[0] * margin_distance
        pos_ey = y_end   + perp[1] * margin_distance

        # Negative margin (shifted towards impostor cluster)
        neg_sx = x_start - perp[0] * margin_distance
        neg_sy = y_start - perp[1] * margin_distance
        neg_ex = x_end   - perp[0] * margin_distance
        neg_ey = y_end   - perp[1] * margin_distance

        decision_line = Line(
            axes.c2p(x_start, y_start), axes.c2p(x_end, y_end),
            color=HYPERPLANE_COLOR, stroke_width=4,
        )
        margin_pos = DashedLine(
            axes.c2p(pos_sx, pos_sy), axes.c2p(pos_ex, pos_ey),
            color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1,
        )
        margin_neg = DashedLine(
            axes.c2p(neg_sx, neg_sy), axes.c2p(neg_ex, neg_ey),
            color=MARGIN_COLOR, stroke_width=2.2, dash_length=0.1,
        )
        margin_band = Polygon(
            axes.c2p(pos_sx, pos_sy), axes.c2p(pos_ex, pos_ey),
            axes.c2p(neg_ex, neg_ey), axes.c2p(neg_sx, neg_sy),
            fill_color=HYPERPLANE_COLOR, fill_opacity=0.15, stroke_width=0,
        )

        return decision_line, margin_pos, margin_neg, margin_band, margin_distance

    def _highlight_support_vectors(
        self, axes: Axes, support_vecs: np.ndarray,
    ) -> VGroup:
        """Create white rings around support vectors with flash effects."""
        sv_rings = VGroup()
        for sv in support_vecs:
            ring = Circle(
                radius=0.20, color=WHITE, stroke_width=2.8,
            ).move_to(axes.c2p(sv[0], sv[1]))
            sv_rings.add(ring)

        self.play(
            LaggedStart(*[
                AnimationGroup(
                    Create(r),
                    Flash(r.get_center(), color=WHITE,
                          flash_radius=0.25, num_lines=6),
                )
                for r in sv_rings
            ], lag_ratio=0.25),
            run_time=1.2,
        )
        return sv_rings

    def _handle_xor_failure(self) -> VGroup:
        """Create the failure icon + text at top-left with high-contrast background."""
        cross_size = 0.35
        cross = VGroup(
            Line(UL * cross_size, DR * cross_size, color=SPOOF_RED, stroke_width=5),
            Line(UR * cross_size, DL * cross_size, color=SPOOF_RED, stroke_width=5),
        )

        failure_text = Text(
            "⚠  Tuyến tính thất bại!",
            font=FONT, font_size=20, weight=BOLD, color=SPOOF_RED,
        )

        failure_group = VGroup(cross, failure_text).arrange(RIGHT, buff=0.2)
        failure_bg = SurroundingRectangle(
            failure_group,
            fill_color=BLACK, fill_opacity=0.85,
            stroke_color=SPOOF_RED, stroke_width=2,
            corner_radius=0.12, buff=0.20,
        )
        failure_block = VGroup(failure_bg, failure_group)
        failure_block.to_corner(UL, buff=0.4)

        self.play(FadeIn(failure_bg), Create(cross), Write(failure_text), run_time=0.8)
        return failure_block

    # =========================================================================
    # PHASE 1 — Feature Extraction & Score Vectorisation (Delayed Axes Reveal)
    # =========================================================================
    def _phase1_vectorization(
        self, axes: Axes, x_label: Text, y_label: Text,
    ) -> None:
        """Split-screen biometric extraction → matrix → delayed axes reveal + dot landing."""

        # ── Build scanner panels (clean black screen — no axes yet) ───────────
        face_bg, face_icon, face_title, face_laser = self._create_scanner_panel(
            make_genuine_icon, "Face Recognition", CLASS_B_COLOR,
            position=LEFT * 3.2 + UP * 2.4,
        )
        fp_bg, fp_icon, fp_title, fp_laser = self._create_scanner_panel(
            make_fingerprint_icon, "Fingerprint Scan", FP_COLOR,
            position=RIGHT * 3.2 + UP * 2.4,
        )

        self.play(
            FadeIn(face_bg), FadeIn(fp_bg),
            FadeIn(face_title), FadeIn(fp_title),
            GrowFromCenter(face_icon), GrowFromCenter(fp_icon),
            run_time=1.0,
        )

        # Laser scan animation
        scan_dist = 2.2 - 0.6  # panel_h - offset
        self.play(
            face_laser.animate.shift(DOWN * scan_dist),
            fp_laser.animate.shift(DOWN * scan_dist),
            FadeIn(face_laser), FadeIn(fp_laser),
            run_time=1.2, rate_func=linear,
        )
        self.play(FadeOut(face_laser), FadeOut(fp_laser), run_time=0.3)

        # ── Score labels beneath panels ───────────────────────────────────────
        s1_val = MathTex(r"s_1 = 0.85", color=CLASS_B_COLOR, font_size=34)
        s1_val.next_to(face_bg, DOWN, buff=0.18)

        s2_val = MathTex(r"s_2 = 0.92", color=FP_COLOR, font_size=34)
        s2_val.next_to(fp_bg, DOWN, buff=0.18)

        self.play(
            FadeIn(s1_val, shift=UP * 0.2),
            FadeIn(s2_val, shift=UP * 0.2),
            run_time=0.9,
        )
        self.wait(0.5)

        # ── Merge into column vector ──────────────────────────────────────────
        score_matrix = Matrix(
            [["0.85"], ["0.92"]],
            left_bracket="[", right_bracket="]",
            element_to_mobject_config={"font_size": 32, "color": CLASS_B_COLOR},
        )
        prefix = MathTex(r"\mathbf{S} =", color=CLASS_B_COLOR, font_size=36)
        vector_group = VGroup(prefix, score_matrix).arrange(RIGHT, buff=0.2)
        vector_group.move_to(UP * 2.4)

        self.play(
            FadeOut(face_bg), FadeOut(fp_bg),
            FadeOut(face_icon), FadeOut(fp_icon),
            FadeOut(face_title), FadeOut(fp_title),
            ReplacementTransform(s1_val, score_matrix.get_entries()[0]),
            ReplacementTransform(s2_val, score_matrix.get_entries()[1]),
            FadeIn(score_matrix.get_brackets()),
            FadeIn(prefix),
            run_time=1.4,
        )
        self.wait(0.4)

        # Flash then shrink to corner badge
        self.play(Flash(score_matrix, color=CLASS_B_COLOR, flash_radius=0.9), run_time=0.5)
        self.play(vector_group.animate.scale(0.55).to_corner(UL, buff=0.35), run_time=0.8)
        self.wait(0.3)

        # ── Flying dot → DELAYED AXES REVEAL ──────────────────────────────────
        target_pos = axes.c2p(0.85, 0.92)
        sample_dot = Dot(point=target_pos, color=CLASS_B_COLOR, radius=0.12)

        # Projection guides (added with axes)
        guide_x = DashedLine(
            axes.c2p(0.85, 0.0), axes.c2p(0.85, 0.92),
            color=CLASS_B_COLOR, stroke_opacity=0.4, dash_length=0.08,
        )
        guide_y = DashedLine(
            axes.c2p(0.0, 0.92), axes.c2p(0.85, 0.92),
            color=CLASS_B_COLOR, stroke_opacity=0.4, dash_length=0.08,
        )

        # Flying dot with glow trail
        flying_dot = Dot(
            point=vector_group.get_center(), color=CLASS_B_COLOR, radius=0.10,
        )
        glow_trail = TracedPath(
            flying_dot.get_center,
            stroke_color=CLASS_B_COLOR, stroke_width=3, stroke_opacity=0.4,
            dissipating_time=0.6,
        )
        self.add(glow_trail)

        # Axes appear simultaneously as the dot flies in
        self.play(
            FadeIn(axes), FadeIn(x_label), FadeIn(y_label),
            flying_dot.animate.move_to(target_pos),
            Create(guide_x), Create(guide_y),
            run_time=1.5, rate_func=smooth,
        )
        self.remove(glow_trail, flying_dot)
        self.add(sample_dot)

        # Flash at landing point
        self.play(
            Flash(sample_dot, color=WHITE, flash_radius=0.4, num_lines=10),
            run_time=0.5,
        )

        # Coordinate annotation
        coord_label = MathTex(
            r"(0.85,\ 0.92)", color=CLASS_B_COLOR, font_size=22,
        ).next_to(sample_dot, UR, buff=0.14)
        self.play(FadeIn(coord_label, shift=UP * 0.12), run_time=0.5)
        self.wait(0.6)

        # Clean up Phase 1 ephemera
        self.play(
            FadeOut(guide_x), FadeOut(guide_y),
            FadeOut(coord_label), FadeOut(sample_dot),
            FadeOut(vector_group),
            run_time=0.8,
        )
        self.wait(0.2)

    # =========================================================================
    # PHASE 2 — The Rescue (Breaking Free from 1D)
    # =========================================================================
    def _phase2_rescue(self, axes: Axes) -> tuple[VGroup, VGroup]:
        """1D overlap crowd → rescue icons → clean 2D separation with camera zoom."""

        # ── 1D overlap crowd on X-axis ────────────────────────────────────────
        rng_crowd = np.random.default_rng(99)
        crowd_x = rng_crowd.uniform(0.35, 0.65, N_CLOUD * 2)
        crowd_cols = [GENUINE_COLOR] * N_CLOUD + [IMPOSTOR_COLOR] * N_CLOUD
        crowd_dots = VGroup(*[
            Dot(axes.c2p(x, 0.0), color=c, radius=0.09)
            for x, c in zip(crowd_x, crowd_cols)
        ])

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.5) for d in crowd_dots],
                lag_ratio=0.04,
            ),
            run_time=1.2,
        )

        confusion_tag = Text(
            "1D Overlap — Vùng chồng lấn", font=FONT, font_size=18, color=SLATE_GRAY,
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(confusion_tag, shift=DOWN * 0.15), run_time=0.6)
        self.wait(0.6)

        # ── Camera zoom in ───────────────────────────────────────────────────
        self.play(
            self.camera.frame.animate.set_width(10),
            run_time=1.0, rate_func=smooth,
        )

        # ── Rescue 1: Noisy Genuine ───────────────────────────────────────────
        noisy_icon = make_noisy_icon(0.35)
        noisy_start_pos = axes.c2p(0.45, 0.08)
        noisy_icon.move_to(noisy_start_pos + UP * 0.55)

        noisy_label = Text(
            "Ảnh nhiễu", font=FONT, font_size=12, color="#888888",
        ).next_to(noisy_icon, UP, buff=0.1)

        # High fingerprint score → escapes confusion zone
        noisy_target_pos = axes.c2p(0.45, 0.82)
        noisy_target_center = noisy_target_pos + UP * 0.55

        self.play(GrowFromCenter(noisy_icon), FadeIn(noisy_label), run_time=0.7)
        self.wait(0.3)

        # Arrow starts above axis, ends exactly at the BOTTOM of the icon
        rescue_arrow_g = Arrow(
            noisy_start_pos + UP * 0.20,
            noisy_target_center + DOWN * 0.40,
            color=GENUINE_COLOR, stroke_width=2, stroke_opacity=0.6,
            buff=0.1, max_tip_length_to_length_ratio=0.12,
        )

        # Tính toán vector tịnh tiến để di chuyển cả icon và text đồng bộ
        move_vector = noisy_target_center - noisy_icon.get_center()

        self.play(
            noisy_icon.animate.move_to(noisy_target_center),
            noisy_label.animate.shift(move_vector), # Tịnh tiến text theo icon
            Create(rescue_arrow_g),
            run_time=1.4, rate_func=smooth,
        )

        rescue_text_g = Text(
            "✓ Thoát vùng nhiễu!", font=FONT, font_size=13, color=GENUINE_COLOR,
        ).next_to(noisy_icon, RIGHT, buff=0.15)
        self.play(FadeIn(rescue_text_g, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.4)

        # ── Rescue 2: Spoof Impostor ──────────────────────────────────────────
        spoof_icon = make_spoof_icon(0.30)
        spoof_start_pos = axes.c2p(0.60, 0.25)
        spoof_icon.move_to(spoof_start_pos + UP * 0.5)

        # FIX: Safe to keep label DOWN since the arrow comes from ABOVE
        spoof_label = Text(
            "Spoof", font=FONT, font_size=12, color=IMPOSTOR_COLOR,
        ).next_to(spoof_icon, DOWN, buff=0.1)

        # High face score, but LOW fingerprint → plummets
        spoof_mid_pos = axes.c2p(0.78, 0.55)
        spoof_mid_center = spoof_mid_pos + UP * 0.45
        
        spoof_final_pos = axes.c2p(0.78, 0.18)
        spoof_final_center = spoof_final_pos + UP * 0.45

        self.play(GrowFromCenter(spoof_icon), FadeIn(spoof_label), run_time=0.7)
        self.wait(0.3)

        # Lift (high face score)
        self.play(
            spoof_icon.animate.move_to(spoof_mid_center),
            spoof_label.animate.next_to(
                Dot(spoof_mid_center), DOWN, buff=0.1,
            ),
            run_time=0.8, rate_func=smooth,
        )

        # FIX: Arrow points from mid-air to exactly the TOP of the final icon position
        rescue_arrow_s = Arrow(
            spoof_mid_center + DOWN * 0.2,
            spoof_final_center + UP * 0.45,
            color=IMPOSTOR_COLOR, stroke_width=2, stroke_opacity=0.6,
            buff=0.1, max_tip_length_to_length_ratio=0.12,
        )
        self.play(
            spoof_icon.animate.move_to(spoof_final_center),
            spoof_label.animate.next_to(
                Dot(spoof_final_center), DOWN, buff=0.1,
            ),
            Create(rescue_arrow_s),
            run_time=1.2, rate_func=rush_into,
        )

        rescue_text_s = Text(
            "✗ Lộ diện kẻ giả mạo!", font=FONT, font_size=13, color=IMPOSTOR_COLOR,
        ).next_to(spoof_icon, RIGHT, buff=0.15)
        self.play(FadeIn(rescue_text_s, shift=LEFT * 0.1), run_time=0.5)
        self.wait(0.5)

        # ── Camera zoom out + clean rescue visuals ────────────────────────────
        self.play(
            FadeOut(noisy_icon), FadeOut(noisy_label), FadeOut(rescue_text_g),
            FadeOut(rescue_arrow_g),
            FadeOut(spoof_icon), FadeOut(spoof_label), FadeOut(rescue_text_s),
            FadeOut(rescue_arrow_s),
            FadeOut(confusion_tag),
            self.camera.frame.animate.set_width(14.222),
            run_time=1.0,
        )

        # ── Expand crowd into 2D clusters ─────────────────────────────────────
        gen_targets = _scatter_2d((0.72, 0.75), N_CLOUD, 0.09, RNG_SEED_GEN_2D)
        imp_targets = _scatter_2d((0.27, 0.25), N_CLOUD, 0.09, RNG_SEED_IMP_2D)
        all_targets = gen_targets + imp_targets

        expand_anims = [
            d.animate.move_to(axes.c2p(tx, ty))
            for d, (tx, ty) in zip(crowd_dots, all_targets)
        ]
        self.play(*expand_anims, run_time=2.0, rate_func=smooth)
        self.wait(0.4)

        genuine_cloud = VGroup(*crowd_dots[:N_CLOUD])
        impostor_cloud = VGroup(*crowd_dots[N_CLOUD:])

        # V-formation cluster labels (Genuine top-right, Impostor top-left)
        gen_label = Text("Genuine  ✓", font=FONT, font_size=19, color=GENUINE_COLOR)
        gen_bg = SurroundingRectangle(
            gen_label, fill_color=BG_COLOR, fill_opacity=0.8,
            stroke_width=0, buff=0.08,
        )
        gen_group = VGroup(gen_bg, gen_label).next_to(genuine_cloud, UR, buff=0.12)

        imp_label = Text("Impostor  ✗", font=FONT, font_size=19, color=IMPOSTOR_COLOR)
        imp_bg = SurroundingRectangle(
            imp_label, fill_color=BG_COLOR, fill_opacity=0.8,
            stroke_width=0, buff=0.08,
        )
        imp_group = VGroup(imp_bg, imp_label).next_to(impostor_cloud, UL, buff=0.12)

        self.play(
            FadeIn(gen_group, shift=DOWN * 0.1),
            FadeIn(imp_group, shift=DOWN * 0.1),
            run_time=0.8,
        )
        self.wait(0.8)
        self.play(FadeOut(gen_group), FadeOut(imp_group), run_time=0.6)
        self.wait(0.3)

        return genuine_cloud, impostor_cloud
    # =========================================================================
    # PHASE 3 — Linear SVM: Learning Process & Convergence
    # =========================================================================
    def _phase3_linear_svm(
        self,
        axes: Axes,
        genuine_cloud: VGroup,
        impostor_cloud: VGroup,
    ) -> Line:
        """ValueTracker lerp: random boundary → Hard-Margin SVM via fake iteration."""

        # Subtitle
        subtitle = Text(
            "Linear Support Vector Machine",
            font=FONT, font_size=24, color=HYPERPLANE_COLOR,
        ).to_edge(UP, buff=0.32)
        # Sửa lỗi depreciation cho set_width()
        underline = Line(
            LEFT, RIGHT, color=HYPERPLANE_COLOR, stroke_width=1.5,
        ).set(width=subtitle.width).next_to(subtitle, DOWN, buff=0.06)
        self.play(
            FadeIn(subtitle, shift=DOWN * 0.15),
            Create(underline),
            run_time=0.9,
        )
        self.wait(0.4)

        # ── Extract point coordinates ─────────────────────────────────────────
        gen_pts = [tuple(axes.p2c(d.get_center())[:2]) for d in genuine_cloud]
        imp_pts = [tuple(axes.p2c(d.get_center())[:2]) for d in impostor_cloud]

        # ── Hard-Margin SVM (C=1e6 forces exact fit through SVs) ──────────────
        X_data = np.array(gen_pts + imp_pts)
        y_data = np.array([1] * len(gen_pts) + [-1] * len(imp_pts))
        clf = SVC(kernel="linear", C=1e6, random_state=42)
        clf.fit(X_data, y_data)

        w_final = clf.coef_[0].copy()
        b_final = clf.intercept_[0]
        support_vecs = clf.support_vectors_

        # Final boundary parameters for lerp target
        slope_final = -w_final[0] / w_final[1]
        y0_final = -b_final / w_final[1]

        # Starting (wrong) boundary for lerp origin
        w_start = w_final + np.array([-1.8, 2.2])
        b_start = b_final - 2.5
        slope_start = -w_start[0] / w_start[1]
        y0_start = -b_start / w_start[1]

        # Margin distances for lerp
        margin_dist_final = 1.0 / np.linalg.norm(w_final)
        margin_dist_start = 1.0 / np.linalg.norm(w_start)

        x_lo, x_hi = AXIS_RANGE[0], AXIS_RANGE[1]

        # ValueTracker drives the lerp (0 = random, 1 = optimal)
        alpha = ValueTracker(0.0)

        def _lerp_boundary():
            """Interpolated boundary and margin parameters at current alpha."""
            t = alpha.get_value()
            w_t = w_start * (1 - t) + w_final * t
            b_t = b_start * (1 - t) + b_final * t
            sl = -w_t[0] / w_t[1]
            y0 = -b_t / w_t[1]
            md = 1.0 / max(np.linalg.norm(w_t), 1e-6)
            return sl, y0, w_t, b_t, md, t

        def _make_hyperplane():
            sl, y0, _, _, _, t = _lerp_boundary()
            c = interpolate_color(ManimColor("#E87040"), ManimColor(HYPERPLANE_COLOR), t)
            return Line(
                axes.c2p(x_lo, sl * x_lo + y0),
                axes.c2p(x_hi, sl * x_hi + y0),
                color=c, stroke_width=3.5,
            )

        def _make_margin_line(sign: float):
            """Build a dashed margin line (sign=+1 for positive, -1 for negative)."""
            sl, y0, w_t, b_t, md, t = _lerp_boundary()
            y_s = sl * x_lo + y0
            y_e = sl * x_hi + y0
            line_vec = np.array([x_hi - x_lo, y_e - y_s])
            line_len = max(np.linalg.norm(line_vec), 1e-6)
            perp = np.array([-line_vec[1], line_vec[0]]) / line_len
            return DashedLine(
                axes.c2p(x_lo + sign * perp[0] * md, y_s + sign * perp[1] * md),
                axes.c2p(x_hi + sign * perp[0] * md, y_e + sign * perp[1] * md),
                color=MARGIN_COLOR, stroke_width=2,
                stroke_opacity=0.4 + 0.5 * t, dash_length=0.1,
            )

        def _make_margin_band():
            sl, y0, w_t, b_t, md, t = _lerp_boundary()
            y_s = sl * x_lo + y0
            y_e = sl * x_hi + y0
            line_vec = np.array([x_hi - x_lo, y_e - y_s])
            line_len = max(np.linalg.norm(line_vec), 1e-6)
            perp = np.array([-line_vec[1], line_vec[0]]) / line_len
            return Polygon(
                axes.c2p(x_lo + perp[0] * md, y_s + perp[1] * md),
                axes.c2p(x_hi + perp[0] * md, y_e + perp[1] * md),
                axes.c2p(x_hi - perp[0] * md, y_e - perp[1] * md),
                axes.c2p(x_lo - perp[0] * md, y_s - perp[1] * md),
                fill_color=HYPERPLANE_COLOR,
                fill_opacity=0.06 + 0.12 * t,
                stroke_width=0,
            )

        N_ITER_DISPLAY = 50

        def _make_epoch_text():
            t = alpha.get_value()
            epoch = int(t * N_ITER_DISPLAY)
            return Text(
                f"Epoch: {epoch:>3} / {N_ITER_DISPLAY}",
                font=FONT, font_size=18, color=SLATE_GRAY,
            ).to_corner(DR, buff=0.45)

        # Live objects
        margin_band = always_redraw(_make_margin_band)
        hyperplane = always_redraw(_make_hyperplane)
        margin_pos = always_redraw(lambda: _make_margin_line(+1))
        margin_neg = always_redraw(lambda: _make_margin_line(-1))
        epoch_text = always_redraw(_make_epoch_text)

        self.add(margin_band, margin_pos, margin_neg, hyperplane, epoch_text)

        # Epoch 0 caption
        epoch0_cap = Text(
            "Epoch 0 — Ranh giới ngẫu nhiên",
            font=FONT, font_size=18, color=SLATE_GRAY,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(epoch0_cap, shift=UP * 0.12), run_time=0.7)
        self.wait(1.0)

        # Learning animation
        self.play(FadeOut(epoch0_cap), run_time=0.3)
        converge_cap = Text(
            "Đang tối ưu hóa lề (Maximising margin)…",
            font=FONT, font_size=18, color=HYPERPLANE_COLOR,
        ).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(converge_cap, shift=UP * 0.1), run_time=0.5)

        self.play(alpha.animate.set_value(1.0), run_time=4.0, rate_func=smooth)
        self.play(FadeOut(converge_cap), run_time=0.3)
        self.wait(0.4)

        # ── Replace live objects with static finals (perpendicular math) ──────
        self.remove(margin_band, margin_pos, margin_neg, hyperplane, epoch_text)

        final_hp, final_mp, final_mn, final_band, margin_dist = (
            self._draw_svm_margins(axes, w_final, b_final, x_lo, x_hi)
        )

        self.play(
            FadeIn(final_band), Create(final_hp),
            Create(final_mp), Create(final_mn),
            run_time=0.7,
        )
        self.play(
            Flash(final_hp.get_center(), color=HYPERPLANE_COLOR,
                  flash_radius=0.5, num_lines=10),
            run_time=0.5,
        )

        # ── Support Vectors ───────────────────────────────────────────────────
        sv_rings = self._highlight_support_vectors(axes, support_vecs)

        sv_badge = Text(
            "Support Vectors", font=FONT, font_size=19, color=WHITE,
        )
        sv_badge_bg = SurroundingRectangle(
            sv_badge, fill_color=BG_COLOR, fill_opacity=0.8,
            stroke_width=0, buff=0.08,
        )
        sv_badge_group = VGroup(sv_badge_bg, sv_badge)
        sv_badge_group.next_to(axes, RIGHT, buff=0.4).shift(DOWN * 0.8)

        self.play(FadeIn(sv_badge_group, shift=LEFT * 0.15), run_time=0.7)
        self.wait(0.5)

        # ── Margin annotation (right side of graph) ───────────────────────────
        meas_x = x_lo + 0.75 * (x_hi - x_lo)
        meas_y = (-w_final[0] * meas_x - b_final) / w_final[1]

        line_vec = np.array([x_hi - x_lo,
                             (-w_final[0] * x_hi - b_final) / w_final[1]
                             - (-w_final[0] * x_lo - b_final) / w_final[1]])
        line_len = np.linalg.norm(line_vec)
        perp = np.array([-line_vec[1], line_vec[0]]) / line_len

        pt_pos = axes.c2p(meas_x + perp[0] * margin_dist,
                          meas_y + perp[1] * margin_dist)
        pt_neg = axes.c2p(meas_x - perp[0] * margin_dist,
                          meas_y - perp[1] * margin_dist)

        margin_arrow = DoubleArrow(
            pt_neg, pt_pos, buff=0,
            color=MARGIN_COLOR, stroke_width=2.5, tip_length=0.12,
        )
        margin_text = MathTex(
            r"\frac{2}{\|\mathbf{w}\|}", font_size=26, color=MARGIN_COLOR,
        ).next_to(margin_arrow, RIGHT, buff=0.12)

        sv_pointer = CurvedArrow(
            start_point=sv_badge_group.get_left() + LEFT * 0.1,
            end_point=sv_rings[0].get_right() + RIGHT * 0.05,
            color=WHITE, angle=PI / 4, stroke_width=1.8, tip_length=0.12,
        )

        self.play(
            Create(margin_arrow), Write(margin_text),
            Create(sv_pointer),
            run_time=1.0,
        )
        self.wait(1.0)

        # ── Clean up Phase 3 ──────────────────────────────────────────────────
        self.play(
            FadeOut(sv_badge_group), FadeOut(sv_pointer), FadeOut(sv_rings),
            FadeOut(margin_arrow), FadeOut(margin_text),
            FadeOut(subtitle), FadeOut(underline),
            FadeOut(final_mp), FadeOut(final_mn), FadeOut(final_band),
            run_time=1.0,
        )
        self.wait(0.3)

        return final_hp

    # =========================================================================
    # PHASE 4 — The XOR Spoof Attack Dilemma
    # =========================================================================
    def _phase4_xor_dilemma(
        self,
        axes: Axes,
        genuine_cloud: VGroup,
        impostor_cloud: VGroup,
        hyperplane: Line,
    ) -> None:
        """Inject XOR spoof clusters and animate the SVM boundary's failure."""

        # ── Red screen flash ──────────────────────────────────────────────────
        flash_rect = Rectangle(
            width=config.frame_width + 2,
            height=config.frame_height + 2,
            fill_color=SPOOF_RED, fill_opacity=0.0, stroke_width=0,
        )
        self.add(flash_rect)
        self.play(
            flash_rect.animate.set_fill(opacity=0.30),
            run_time=0.3, rate_func=there_and_back,
        )
        self.remove(flash_rect)

        # ── Warning text ──────────────────────────────────────────────────────
        warning_en = Text(
            "Spoof Attack", font=FONT, font_size=28,
            weight=BOLD, color=SPOOF_RED,
        ).to_edge(UP, buff=0.30)
        warning_vn = Text(
            "Tấn công giả mạo", font=FONT, font_size=22, color=SPOOF_RED,
        ).next_to(warning_en, DOWN, buff=0.10)
        warning_bg = SurroundingRectangle(
            VGroup(warning_en, warning_vn),
            fill_color=BLACK, fill_opacity=0.75,
            stroke_color=SPOOF_RED, stroke_width=1.5,
            corner_radius=0.14, buff=0.18,
        )
        self.play(
            FadeIn(warning_bg), Write(warning_en),
            FadeIn(warning_vn, shift=DOWN * 0.1),
            run_time=0.8,
        )
        self.wait(0.4)

        # ── Spoof cluster 1 — Top-Left (high fingerprint, low face) ──────────
        spoof_tl_pts = _scatter_2d(
            (0.27, 0.75), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_TL,
        )
        spoof_tl = VGroup(*[
            Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09)
            for x, y in spoof_tl_pts
        ])

        # ── Spoof cluster 2 — Bottom-Right (high face, low fingerprint) ──────
        spoof_br_pts = _scatter_2d(
            (0.72, 0.25), N_CLOUD // 2, 0.08, RNG_SEED_SPOOF_BR,
        )
        spoof_br = VGroup(*[
            Dot(axes.c2p(x, y), color=IMPOSTOR_COLOR, radius=0.09)
            for x, y in spoof_br_pts
        ])

        # Cluster captions — side of each cluster, clear of axes
        tl_caption = Text(
            "Silicone fingerprint\n(Giả mạo vân tay)",
            font=FONT, font_size=13, color=IMPOSTOR_COLOR,
        )
        tl_caption_bg = SurroundingRectangle(
            tl_caption, fill_color=BG_COLOR, fill_opacity=0.7,
            stroke_width=0, buff=0.06,
        )
        tl_cap_group = VGroup(tl_caption_bg, tl_caption).next_to(spoof_tl, RIGHT, buff=0.15)

        br_caption = Text(
            "3D face mask\n(Mặt nạ khuôn mặt)",
            font=FONT, font_size=13, color=IMPOSTOR_COLOR,
        )
        br_caption_bg = SurroundingRectangle(
            br_caption, fill_color=BG_COLOR, fill_opacity=0.7,
            stroke_width=0, buff=0.06,
        )
        br_cap_group = VGroup(br_caption_bg, br_caption).next_to(spoof_br, LEFT, buff=0.15)

        self.play(
            LaggedStart(
                *[FadeIn(d, scale=0.4) for d in spoof_tl], lag_ratio=0.07,
            ),
            LaggedStart(
                *[FadeIn(d, scale=0.4) for d in spoof_br], lag_ratio=0.07,
            ),
            run_time=1.0,
        )
        self.play(
            FadeIn(tl_cap_group, shift=LEFT * 0.12),
            FadeIn(br_cap_group, shift=RIGHT * 0.12),
            run_time=0.5,
        )
        self.wait(0.5)

        # ── XOR label at bottom with dark background ──────────────────────────
        xor_label = MathTex(
            r"\text{XOR Layout}", color=SPOOF_RED, font_size=26,
        )
        xor_sub = Text(
            "No linear boundary can separate this!",
            font=FONT, font_size=15, color=SLATE_GRAY,
        )
        xor_group = VGroup(xor_label, xor_sub).arrange(DOWN, buff=0.10)
        xor_bg = SurroundingRectangle(
            xor_group, fill_color=BG_COLOR, fill_opacity=0.75,
            stroke_width=0, buff=0.12,
        )
        xor_block = VGroup(xor_bg, xor_group).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(xor_block, shift=UP * 0.15), run_time=0.5)
        self.wait(0.3)

        # ── Hyperplane rotates wildly (yellow → red) ──────────────────────────
        angle_tracker = ValueTracker(0)
        c_yellow = ManimColor(HYPERPLANE_COLOR)
        c_red = ManimColor(SPOOF_RED)

        def _hp_updater(line: Line) -> None:
            a = angle_tracker.get_value()
            cx = axes.c2p(0.5, 0.5)
            dir_vec = np.array([np.cos(a), np.sin(a), 0.0])
            line.put_start_and_end_on(cx - 3.5 * dir_vec, cx + 3.5 * dir_vec)
            t = (np.sin(a * 6) + 1) / 2
            line.set_color(interpolate_color(c_yellow, c_red, t))
            line.set_stroke(width=3 + 3 * t)

        hyperplane.add_updater(_hp_updater)
        self.play(
            angle_tracker.animate.set_value(PI * 1.4),
            run_time=2.5, rate_func=linear,
        )
        hyperplane.remove_updater(_hp_updater)

        # Dim to signal failure
        self.play(
            hyperplane.animate.set_color(SPOOF_RED).set_stroke(opacity=0.35),
            run_time=0.5,
        )

        # ── Failure message (top-left, high contrast) ─────────────────────────
        failure_block = self._handle_xor_failure()
        self.wait(1.2)

        # ── Fade everything to black ──────────────────────────────────────────
        self.play(
            *[FadeOut(m) for m in self.mobjects],
            run_time=1.5,
        )
        self.wait(0.5)