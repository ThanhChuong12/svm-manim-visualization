"""
Part 1.5 — Biometric Score Pipeline Walkthrough
================================================
Covers the FULL pipeline:
  Raw Input → Feature Extraction → Matcher → Raw Score
  → Score Normalization → Score-Level Fusion → SVM Decision

Also explains WHY score-level fusion is chosen over sensor / feature / decision
fusion, and WHY normalization matters before combining heterogeneous scores.

Visual style: 3Blue1Brown-inspired cinematic pedagogy on dark background.
"""

from __future__ import annotations

import numpy as np
from manim import *

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Colour tokens (mirrors constants.py, with safe fallback) ─────────────────
try:
    from constants import (
        CLASS_A_COLOR, CLASS_B_COLOR, HYPERPLANE_COLOR,
        BG_COLOR, FONT_MAIN, SLATE_GRAY,
        GENUINE_COLOR, IMPOSTOR_COLOR,
    )
except ImportError:
    CLASS_A_COLOR    = "#FF5E5E"
    CLASS_B_COLOR    = "#00C2D1"
    HYPERPLANE_COLOR = "#F9DC5C"
    BG_COLOR         = "#0B0C10"
    FONT_MAIN        = "Montserrat"
    SLATE_GRAY       = "#888888"
    GENUINE_COLOR    = "#2ECC71"
    IMPOSTOR_COLOR   = "#E74C3C"

FONT         = FONT_MAIN
ACCENT       = HYPERPLANE_COLOR          # gold
NODE_FILL    = "#12151F"
NODE_STROKE  = "#3A3F55"
PIPE_COLOR   = "#3A3F55"
ARROW_COLOR  = "#5A6080"
HIGHLIGHT    = "#00C2D1"                 # cyan for active node
NORM_COLOR   = "#A855F7"                 # purple for normalization
FUSION_COLOR = HYPERPLANE_COLOR
DECISION_OK  = GENUINE_COLOR
DECISION_BAD = IMPOSTOR_COLOR

# ── Helper: rounded pipeline node ────────────────────────────────────────────

def make_node(
    label: str,
    sub: str = "",
    width: float = 2.4,
    height: float = 0.88,
    stroke: str = NODE_STROKE,
    fill: str = NODE_FILL,
    label_size: int = 20,
    sub_size: int = 14,
) -> VGroup:
    """Create a rounded-rect node with an optional subtitle."""
    box = RoundedRectangle(
        width=width, height=height, corner_radius=0.18,
        fill_color=fill, fill_opacity=1.0,
        stroke_color=stroke, stroke_width=2.0,
    )
    lbl = Text(label, font=FONT, font_size=label_size, color=WHITE)
    grp = VGroup(box, lbl)
    if sub:
        sub_text = Text(sub, font=FONT, font_size=sub_size, color=SLATE_GRAY)
        sub_text.next_to(lbl, DOWN, buff=0.10)
        grp.add(sub_text)
    grp.move_to(ORIGIN)
    return grp


def make_arrow(start: np.ndarray, end: np.ndarray) -> Arrow:
    return Arrow(
        start, end,
        color=ARROW_COLOR,
        stroke_width=2.2,
        buff=0.12,
        max_tip_length_to_length_ratio=0.18,
    )


def highlight_node(node: VGroup, color: str = HIGHLIGHT, scene: Scene = None) -> None:
    """Flash a node's border to the highlight colour."""
    box = node[0]
    if scene:
        scene.play(
            box.animate.set_stroke(color=color, width=3.2),
            run_time=0.35,
        )


# ── Fusion level comparison table data ───────────────────────────────────────

FUSION_LEVELS = [
    ("Sensor",    "Ghép raw data",      "❌ Khó tương thích", "#888888"),
    ("Feature",   "Ghép feature vector","⚠ Chiều cao, phức tạp", "#F39C12"),
    ("Score ✓",  "Ghép matching scores","✅ Đơn giản, hiệu quả", GENUINE_COLOR),
    ("Decision",  "Ghép Có / Không",    "⚠ Mất thông tin xác suất", "#F39C12"),
]


# ═══════════════════════════════════════════════════════════════════════════════
class PipelineScene(Scene):
    """Full biometric pipeline walkthrough with normalization and fusion levels."""

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR

        self._phase0_title()
        self._phase1_pipeline_overview()
        self._phase2_fusion_levels()
        self._phase3_normalization()
        self._phase4_fusion_node_and_bridge()

    # ──────────────────────────────────────────────────────────────────────────
    def _phase0_title(self) -> None:
        """Quick title card — 2.5 s total."""
        line1 = Text(
            "Từ Ảnh Thô đến Quyết Định",
            font=FONT, weight=BOLD, font_size=44, color=WHITE,
        )
        line2 = Text(
            "Pipeline Sinh Trắc Học End-to-End",
            font=FONT, font_size=28, slant=ITALIC, color=ACCENT,
        ).next_to(line1, DOWN, buff=0.28)

        divider = Line(LEFT, RIGHT, color=ACCENT, stroke_width=1.8)
        divider.width = line1.width
        divider.next_to(line2, DOWN, buff=0.30)

        title_group = VGroup(line1, line2, divider).move_to(ORIGIN)

        self.play(Write(line1), run_time=1.1)
        self.play(FadeIn(line2, shift=UP * 0.15), GrowFromCenter(divider), run_time=0.9)
        self.wait(1.2)
        self.play(FadeOut(title_group), run_time=0.7)

    # ──────────────────────────────────────────────────────────────────────────
    def _phase1_pipeline_overview(self) -> None:
        """
        Animate the 7-step pipeline left → right, lighting each node as we
        explain its role with a brief caption.

        Pipeline:
          [Raw Input] → [Feature Extraction] → [Matcher] → [Raw Score]
                                ↓ (second modality, same path)
          [Normalization] → [Score Fusion] → [SVM Decision]
        """

        # ── Build nodes ──────────────────────────────────────────────────────
        n_raw    = make_node("Raw Input",       "Ảnh / Âm thanh",  width=2.3, stroke=CLASS_B_COLOR)
        n_feat   = make_node("Feature\nExtract","Minutiae / MFCC",  width=2.4, label_size=19)
        n_match  = make_node("Matcher",         "Cosine / DTW",     width=2.2)
        n_score  = make_node("Raw Score",       "∈ [0, 100]",       width=2.2, stroke=NORM_COLOR)

        # Row 1: pipeline up to raw score
        row1 = VGroup(n_raw, n_feat, n_match, n_score)
        row1.arrange(RIGHT, buff=0.80)
        row1.shift(UP * 0.9)

        # Row 2: normalization → fusion → decision (centered under row1 gap)
        n_norm   = make_node("Normalize",       "Min-Max / Z-score", width=2.4, stroke=NORM_COLOR)
        n_fusion = make_node("Score Fusion",    "Combine scores",    width=2.4, stroke=ACCENT)
        n_decide = make_node("SVM Decision",    "Accept / Reject",   width=2.4, stroke=GENUINE_COLOR)

        row2 = VGroup(n_norm, n_fusion, n_decide)
        row2.arrange(RIGHT, buff=0.80)
        row2.next_to(row1, DOWN, buff=1.05)
        row2.align_to(row1, LEFT).shift(RIGHT * 0.55)

        all_nodes = VGroup(row1, row2)
        all_nodes.move_to(ORIGIN + DOWN * 0.25)

        # ── Arrows row 1 ─────────────────────────────────────────────────────
        def _h_arrow(a: VGroup, b: VGroup) -> Arrow:
            return make_arrow(a[0].get_right(), b[0].get_left())

        arr_r1_01 = _h_arrow(n_raw,   n_feat)
        arr_r1_12 = _h_arrow(n_feat,  n_match)
        arr_r1_23 = _h_arrow(n_match, n_score)

        # Vertical drop from n_score down to n_norm
        arr_drop = Arrow(
            n_score[0].get_bottom() + DOWN * 0.05,
            n_norm[0].get_top()     + UP   * 0.05,
            color=NORM_COLOR, stroke_width=2.2, buff=0.0,
            max_tip_length_to_length_ratio=0.18,
        )

        arr_r2_01 = _h_arrow(n_norm,   n_fusion)
        arr_r2_12 = _h_arrow(n_fusion, n_decide)

        # Second-modality label on the drop arrow
        second_lbl = Text(
            "Modality 2\n(cùng quy trình)", font=FONT, font_size=13, color=SLATE_GRAY,
        ).next_to(arr_drop, RIGHT, buff=0.10)

        all_arrows = VGroup(
            arr_r1_01, arr_r1_12, arr_r1_23,
            arr_drop,
            arr_r2_01, arr_r2_12,
        )

        # ── "Modality 1" label above row 1 ───────────────────────────────────
        mod1_badge = Text("Modality 1  (VD: Face)", font=FONT, font_size=15, color=CLASS_B_COLOR)
        mod1_badge.next_to(row1, UP, buff=0.25)

        # ── Section heading ───────────────────────────────────────────────────
        heading = Text(
            "Pipeline tổng quan", font=FONT, font_size=22, color=ACCENT,
        ).to_edge(UP, buff=0.25)

        self.play(FadeIn(heading, shift=DOWN * 0.1), run_time=0.6)

        # ── Animate nodes & arrows sequentially ──────────────────────────────
        pipeline_seq = [
            (n_raw,   arr_r1_01, "Đầu vào: ảnh khuôn mặt hoặc âm thanh giọng nói",  CLASS_B_COLOR),
            (n_feat,  arr_r1_12, "Trích đặc trưng: minutiae vân tay, MFCC giọng nói",  WHITE),
            (n_match, arr_r1_23, "Matcher so sánh và cho ra điểm số tương đồng",        WHITE),
            (n_score, arr_drop,  "Raw score — đơn vị khác nhau mỗi hệ thống!",        NORM_COLOR),
        ]

        caption_box = RoundedRectangle(
            width=8.6, height=0.72, corner_radius=0.14,
            fill_color="#0D0F18", fill_opacity=0.92,
            stroke_color=SLATE_GRAY, stroke_width=1.0,
        ).to_edge(DOWN, buff=0.22)
        caption_text = Text("", font=FONT, font_size=18, color=WHITE).move_to(caption_box)

        self.play(FadeIn(caption_box), run_time=0.4)
        self.play(FadeIn(mod1_badge), run_time=0.5)

        current_caption = caption_text

        for node, arrow, cap_str, hi_col in pipeline_seq:
            new_cap = Text(cap_str, font=FONT, font_size=18, color=WHITE).move_to(caption_box)
            self.play(
                FadeIn(node, scale=0.85),
                node[0].animate.set_stroke(color=hi_col, width=3.0),
                Transform(current_caption, new_cap),
                run_time=0.75,
            )
            self.play(Create(arrow), run_time=0.55)
            self.play(
                node[0].animate.set_stroke(color=NODE_STROKE, width=2.0),
                run_time=0.30,
            )
            self.wait(0.20)

        # Drop label
        self.play(FadeIn(second_lbl), run_time=0.4)
        self.wait(0.3)

        # Row 2
        row2_seq = [
            (n_norm,   arr_r2_01, "Chuẩn hóa: đưa tất cả scores về cùng thang đo [0,1]", NORM_COLOR),
            (n_fusion, arr_r2_12, "Dung hợp: kết hợp scores từ nhiều modality thành vector", ACCENT),
            (n_decide, None,      "SVM phân loại vector cuối → Accept hoặc Reject",         GENUINE_COLOR),
        ]

        for node, arrow, cap_str, hi_col in row2_seq:
            new_cap = Text(cap_str, font=FONT, font_size=18, color=WHITE).move_to(caption_box)
            self.play(
                FadeIn(node, scale=0.85),
                node[0].animate.set_stroke(color=hi_col, width=3.0),
                Transform(current_caption, new_cap),
                run_time=0.75,
            )
            if arrow:
                self.play(Create(arrow), run_time=0.55)
            self.play(
                node[0].animate.set_stroke(color=NODE_STROKE, width=2.0),
                run_time=0.30,
            )
            self.wait(0.25)

        self.wait(1.0)

        # ── Pulse the full pipeline once ─────────────────────────────────────
        self.play(
            LaggedStart(
                *[node[0].animate(rate_func=there_and_back).set_stroke(color=ACCENT, width=3.5)
                  for node in [n_raw, n_feat, n_match, n_score, n_norm, n_fusion, n_decide]],
                lag_ratio=0.18,
            ),
            run_time=2.0,
        )
        self.wait(0.8)

        # ── Store for next phase fade ─────────────────────────────────────────
        self._pipeline_group = VGroup(
            all_nodes, all_arrows, second_lbl, mod1_badge,
            caption_box, current_caption, heading,
        )
        self.play(FadeOut(self._pipeline_group), run_time=1.0)
        self.wait(0.3)

    # ──────────────────────────────────────────────────────────────────────────
    def _phase2_fusion_levels(self) -> None:
        """
        Animated comparison table — 4 fusion levels.
        Highlight Score-level as the chosen approach.
        """
        heading = Text(
            "Tại sao chọn Score-Level Fusion?",
            font=FONT, weight=BOLD, font_size=30, color=ACCENT,
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading, shift=DOWN * 0.12), run_time=0.7)

        # ── Build rows ───────────────────────────────────────────────────────
        ROW_H  = 0.88
        COL_W  = [1.8, 3.0, 3.6]   # Level | Method | Verdict
        HEADER_COLOR = SLATE_GRAY

        def _cell(txt: str, w: float, h: float = ROW_H,
                  fc: str = NODE_FILL, sc: str = NODE_STROKE,
                  tc: str = WHITE, fs: int = 17) -> VGroup:
            box  = Rectangle(width=w, height=h,
                             fill_color=fc, fill_opacity=1.0,
                             stroke_color=sc, stroke_width=1.2)
            label = Text(txt, font=FONT, font_size=fs, color=tc)
            label.move_to(box)
            return VGroup(box, label)

        # Header row
        h0 = _cell("Level",    COL_W[0], 0.72, fc="#181A28", sc=SLATE_GRAY, tc=HEADER_COLOR)
        h1 = _cell("Phương pháp", COL_W[1], 0.72, fc="#181A28", sc=SLATE_GRAY, tc=HEADER_COLOR)
        h2 = _cell("Đánh giá", COL_W[2], 0.72, fc="#181A28", sc=SLATE_GRAY, tc=HEADER_COLOR)
        header_row = VGroup(h0, h1, h2).arrange(RIGHT, buff=0)

        data_rows = []
        for level, method, verdict, color in FUSION_LEVELS:
            is_score = "✓" in level
            bg_fc = "#1A2A1A" if is_score else NODE_FILL
            bg_sc = color if is_score else NODE_STROKE
            sw    = 2.2 if is_score else 1.2
            r0 = _cell(level,   COL_W[0], ROW_H, fc=bg_fc, sc=bg_sc, tc=color, fs=16)
            r1 = _cell(method,  COL_W[1], ROW_H, fc=bg_fc, sc=bg_sc, tc=WHITE, fs=15)
            r2 = _cell(verdict, COL_W[2], ROW_H, fc=bg_fc, sc=bg_sc, tc=color, fs=15)
            for cell in [r0, r1, r2]:
                cell[0].set_stroke(width=sw)
            row = VGroup(r0, r1, r2).arrange(RIGHT, buff=0)
            data_rows.append(row)

        table = VGroup(header_row, *data_rows).arrange(DOWN, buff=0)
        table.move_to(ORIGIN + DOWN * 0.3)

        # ── Animate ──────────────────────────────────────────────────────────
        self.play(FadeIn(header_row, shift=DOWN * 0.1), run_time=0.55)

        for i, row in enumerate(data_rows):
            self.play(FadeIn(row, shift=RIGHT * 0.25), run_time=0.45)
            self.wait(0.15)

        self.wait(0.7)

        # Spotlight Score row with glow
        score_row = data_rows[2]
        glow = SurroundingRectangle(
            score_row,
            color=GENUINE_COLOR, stroke_width=2.8,
            buff=0.04, corner_radius=0.05,
        )
        check_label = Text(
            "← Chúng ta chọn cấp độ này",
            font=FONT, font_size=17, color=GENUINE_COLOR,
        ).next_to(score_row, RIGHT, buff=0.22)

        self.play(
            Create(glow),
            FadeIn(check_label, shift=LEFT * 0.2),
            run_time=0.8,
        )
        self.wait(1.5)

        # Rationale bullets (appear below table)
        reasons = [
            "✓  Không cần truy cập raw data sau quá trình matching",
            "✓  Dễ kết hợp các hệ thống matcher khác nhau",
            "✓  Giữ thông tin xác suất (soft decision)",
        ]
        bullets = VGroup(*[
            Text(r, font=FONT, font_size=16, color=GENUINE_COLOR)
            for r in reasons
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        bullets.next_to(table, DOWN, buff=0.42)

        self.play(
            LaggedStart(
                *[FadeIn(b, shift=RIGHT * 0.3) for b in bullets],
                lag_ratio=0.35,
            ),
            run_time=1.4,
        )
        self.wait(1.8)

        self.play(FadeOut(VGroup(heading, table, glow, check_label, bullets)), run_time=1.0)
        self.wait(0.3)

    # ──────────────────────────────────────────────────────────────────────────
    def _phase3_normalization(self) -> None:
        """
        Show WHY score normalization is critical before fusion.
        Side-by-side: raw heterogeneous scores vs normalized [0,1] scores.
        Then animate min-max formula.
        """
        heading = Text(
            "Tại sao cần chuẩn hóa điểm số?",
            font=FONT, weight=BOLD, font_size=30, color=NORM_COLOR,
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading, shift=DOWN * 0.1), run_time=0.6)

        # ── Left panel: raw heterogeneous scores ─────────────────────────────
        axes_left = Axes(
            x_range=[0, 4, 1], y_range=[0, 110, 25],
            x_length=3.6, y_length=3.2,
            axis_config={"stroke_width": 1.8, "color": WHITE, "stroke_opacity": 0.55},
            tips=False,
        ).shift(LEFT * 3.3 + DOWN * 0.4)

        raw_scores = {
            "Face\n(0-100)":        (1, 78,  CLASS_B_COLOR),
            "Fingerprint\n(0-100)": (2, 45,  GENUINE_COLOR),
            "Voice\n(0-1)×100":     (3, 6,   "#A855F7"),
        }

        raw_bars   = VGroup()
        raw_labels = VGroup()
        for key, (xi, yi, col) in raw_scores.items():
            bar = Rectangle(
                width=0.55, height=axes_left.c2p(0, yi)[1] - axes_left.c2p(0, 0)[1],
                fill_color=col, fill_opacity=0.85, stroke_width=0,
            )
            bar.align_to(axes_left.c2p(xi, 0), DOWN)
            bar.align_to(axes_left.c2p(xi, 0), LEFT)
            bar.shift(LEFT * 0.28)

            lbl = Text(key, font=FONT, font_size=12, color=col)
            lbl.next_to(bar, DOWN, buff=0.12)

            val_lbl = Text(f"{yi}", font=FONT, font_size=14, color=col)
            val_lbl.next_to(bar, UP, buff=0.08)

            raw_bars.add(VGroup(bar, val_lbl))
            raw_labels.add(lbl)

        left_title = Text("Raw scores (bất đồng đơn vị)", font=FONT, font_size=16,
                          color=SLATE_GRAY).next_to(axes_left, UP, buff=0.18)
        problem_badge = Text("⚠  Voice score bị lấn át!", font=FONT, font_size=15,
                             color=IMPOSTOR_COLOR).next_to(axes_left, DOWN, buff=0.15)

        # ── Right panel: normalized scores ───────────────────────────────────
        axes_right = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.1, 0.25],
            x_length=3.6, y_length=3.2,
            axis_config={"stroke_width": 1.8, "color": WHITE, "stroke_opacity": 0.55},
            tips=False,
        ).shift(RIGHT * 3.3 + DOWN * 0.4)

        norm_vals = {
            "Face":        (1, 0.78, CLASS_B_COLOR),
            "Fingerprint": (2, 0.45, GENUINE_COLOR),
            "Voice":       (3, 0.62, "#A855F7"),
        }

        norm_bars   = VGroup()
        norm_labels = VGroup()
        for key, (xi, yi, col) in norm_vals.items():
            bar = Rectangle(
                width=0.55, height=axes_right.c2p(0, yi)[1] - axes_right.c2p(0, 0)[1],
                fill_color=col, fill_opacity=0.85, stroke_width=0,
            )
            bar.align_to(axes_right.c2p(xi, 0), DOWN)
            bar.align_to(axes_right.c2p(xi, 0), LEFT)
            bar.shift(LEFT * 0.28)

            lbl = Text(key, font=FONT, font_size=12, color=col)
            lbl.next_to(bar, DOWN, buff=0.12)

            val_lbl = Text(f"{yi:.2f}", font=FONT, font_size=14, color=col)
            val_lbl.next_to(bar, UP, buff=0.08)

            norm_bars.add(VGroup(bar, val_lbl))
            norm_labels.add(lbl)

        right_title = Text("Sau chuẩn hóa Min-Max [0, 1]", font=FONT, font_size=16,
                           color=NORM_COLOR).next_to(axes_right, UP, buff=0.18)
        ok_badge = Text("✓  Cùng thang đo — so sánh công bằng!", font=FONT, font_size=15,
                        color=GENUINE_COLOR).next_to(axes_right, DOWN, buff=0.15)

        # Arrow between panels
        transform_arrow = Arrow(
            axes_left.get_right() + RIGHT * 0.1,
            axes_right.get_left() + LEFT  * 0.1,
            color=NORM_COLOR, stroke_width=2.8, buff=0.05,
            max_tip_length_to_length_ratio=0.20,
        )
        norm_label_mid = Text("Normalize", font=FONT, font_size=16, color=NORM_COLOR)
        norm_label_mid.next_to(transform_arrow, UP, buff=0.10)

        # ── Play left panel ──────────────────────────────────────────────────
        self.play(FadeIn(axes_left), FadeIn(left_title), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in raw_bars], lag_ratio=0.25),
            FadeIn(raw_labels),
            run_time=1.2,
        )
        self.play(FadeIn(problem_badge, shift=UP * 0.1), run_time=0.55)
        self.wait(0.7)

        # Transform arrow
        self.play(
            GrowArrow(transform_arrow),
            FadeIn(norm_label_mid),
            run_time=0.8,
        )

        # Right panel
        self.play(FadeIn(axes_right), FadeIn(right_title), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.3) for b in norm_bars], lag_ratio=0.25),
            FadeIn(norm_labels),
            run_time=1.2,
        )
        self.play(FadeIn(ok_badge, shift=UP * 0.1), run_time=0.55)
        self.wait(0.8)

        # ── Min-Max formula reveal ────────────────────────────────────────────
        formula_bg = RoundedRectangle(
            width=5.4, height=1.1, corner_radius=0.18,
            fill_color="#0D0F1E", fill_opacity=0.95,
            stroke_color=NORM_COLOR, stroke_width=1.8,
        ).to_edge(DOWN, buff=0.18)

        formula = MathTex(
            r"s'_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}",
            color=WHITE, font_size=34,
        ).move_to(formula_bg)

        formula_note = Text(
            "Min-Max Normalization   →   s'ᵢ ∈ [0, 1]",
            font=FONT, font_size=14, color=NORM_COLOR,
        ).next_to(formula_bg, DOWN, buff=0.10)

        self.play(FadeIn(formula_bg), run_time=0.4)
        self.play(Write(formula), run_time=1.3)
        self.play(FadeIn(formula_note), run_time=0.5)
        self.wait(1.8)

        panel_group = VGroup(
            heading, axes_left, left_title, raw_bars, raw_labels, problem_badge,
            axes_right, right_title, norm_bars, norm_labels, ok_badge,
            transform_arrow, norm_label_mid,
            formula_bg, formula, formula_note,
        )
        self.play(FadeOut(panel_group), run_time=1.1)
        self.wait(0.3)

    # ──────────────────────────────────────────────────────────────────────────
    def _phase4_fusion_node_and_bridge(self) -> None:
        """
        Show the fusion node receiving normalized scores, forming the 2D
        feature vector [s_face, s_finger], then zoom into the 2D plane to
        bridge naturally to Part 2 (Score Fusion & Linear SVM).
        """
        heading = Text(
            "Vector dung hợp → đầu vào SVM",
            font=FONT, weight=BOLD, font_size=30, color=ACCENT,
        ).to_edge(UP, buff=0.32)
        self.play(FadeIn(heading, shift=DOWN * 0.1), run_time=0.6)

        # ── Two score streams ─────────────────────────────────────────────────
        s1_lbl = Text("Face score:        s₁ = 0.78", font=FONT, font_size=22, color=CLASS_B_COLOR)
        s2_lbl = Text("Fingerprint score: s₂ = 0.45", font=FONT, font_size=22, color=GENUINE_COLOR)
        streams = VGroup(s1_lbl, s2_lbl).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        streams.shift(LEFT * 3.2 + UP * 0.2)

        self.play(
            LaggedStart(
                FadeIn(s1_lbl, shift=RIGHT * 0.3),
                FadeIn(s2_lbl, shift=RIGHT * 0.3),
                lag_ratio=0.5,
            ),
            run_time=1.0,
        )
        self.wait(0.3)

        # ── Fusion node ───────────────────────────────────────────────────────
        fusion_node = make_node("Fusion Node", width=2.6, height=1.0,
                                stroke=ACCENT, fill="#1A1608")
        fusion_node.move_to(ORIGIN + UP * 0.2)

        arr1 = Arrow(s1_lbl.get_right() + RIGHT * 0.05,
                     fusion_node[0].get_left() + LEFT * 0.05 + UP * 0.22,
                     color=CLASS_B_COLOR, stroke_width=2.2, buff=0.08,
                     max_tip_length_to_length_ratio=0.18)
        arr2 = Arrow(s2_lbl.get_right() + RIGHT * 0.05,
                     fusion_node[0].get_left() + LEFT * 0.05 + DOWN * 0.22,
                     color=GENUINE_COLOR, stroke_width=2.2, buff=0.08,
                     max_tip_length_to_length_ratio=0.18)

        self.play(
            GrowArrow(arr1), GrowArrow(arr2),
            FadeIn(fusion_node, scale=0.85),
            run_time=0.9,
        )
        self.wait(0.3)

        # ── Output vector ─────────────────────────────────────────────────────
        vec_label = MathTex(
            r"\mathbf{x} = \begin{bmatrix} 0.78 \\ 0.45 \end{bmatrix}",
            color=WHITE, font_size=36,
        ).shift(RIGHT * 3.2 + UP * 0.2)

        arr_out = Arrow(
            fusion_node[0].get_right() + RIGHT * 0.05,
            vec_label.get_left() + LEFT * 0.08,
            color=ACCENT, stroke_width=2.4, buff=0.08,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(GrowArrow(arr_out), Write(vec_label), run_time=1.0)
        self.wait(0.5)

        dim_note = Text(
            "Vector 2D — một điểm trong không gian đặc trưng",
            font=FONT, font_size=17, color=SLATE_GRAY,
        ).next_to(vec_label, DOWN, buff=0.25)
        self.play(FadeIn(dim_note, shift=UP * 0.1), run_time=0.6)
        self.wait(1.0)

        # ── Mini axes preview — teaser of Part 2 ─────────────────────────────
        mini_axes = Axes(
            x_range=[0, 1, 0.5], y_range=[0, 1, 0.5],
            x_length=3.0, y_length=2.6,
            axis_config={"stroke_width": 1.5, "color": WHITE, "stroke_opacity": 0.50},
            tips=True,
        ).to_edge(DOWN, buff=0.30).shift(RIGHT * 0.5)

        xa = Text("Face score", font=FONT, font_size=13, color=SLATE_GRAY)
        xa.next_to(mini_axes.x_axis.get_end(), DR, buff=0.10)
        ya = Text("Fingerprint", font=FONT, font_size=13, color=SLATE_GRAY)
        ya.rotate(PI / 2).next_to(mini_axes.y_axis, LEFT, buff=0.18).shift(UP * 0.5)

        self.play(FadeIn(mini_axes), FadeIn(xa), FadeIn(ya), run_time=0.7)

        # Plot the example vector point
        vec_dot = Dot(mini_axes.c2p(0.78, 0.45), color=CLASS_B_COLOR, radius=0.11)
        dot_lbl = MathTex(r"\mathbf{x}", color=CLASS_B_COLOR, font_size=26)
        dot_lbl.next_to(vec_dot, UR, buff=0.10)

        self.play(FadeIn(vec_dot, scale=0.4), FadeIn(dot_lbl), run_time=0.6)

        # Scatter a few more sample points for genuine/impostor
        sample_pts_genuine = [
            (0.72, 0.58), (0.81, 0.49), (0.68, 0.55), (0.85, 0.62),
        ]
        sample_pts_impostor = [
            (0.22, 0.18), (0.30, 0.25), (0.15, 0.32), (0.28, 0.12),
        ]
        gen_dots = VGroup(*[
            Dot(mini_axes.c2p(*p), color=GENUINE_COLOR, radius=0.07)
            for p in sample_pts_genuine
        ])
        imp_dots = VGroup(*[
            Dot(mini_axes.c2p(*p), color=IMPOSTOR_COLOR, radius=0.07)
            for p in sample_pts_impostor
        ])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in gen_dots], lag_ratio=0.12),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in imp_dots], lag_ratio=0.12),
            run_time=1.0,
        )
        self.wait(0.5)

        # Bridge text
        bridge_box = RoundedRectangle(
            width=7.6, height=0.80, corner_radius=0.16,
            fill_color="#0D0F1E", fill_opacity=0.95,
            stroke_color=ACCENT, stroke_width=1.5,
        ).to_edge(DOWN, buff=0.12).shift(LEFT * 1.0)

        bridge_text = Text(
            "Tiếp theo → SVM tìm đường phân chia tối ưu trong không gian này",
            font=FONT, font_size=18, color=ACCENT,
        ).move_to(bridge_box)

        self.play(FadeIn(bridge_box, shift=UP * 0.1), Write(bridge_text), run_time=1.1)
        self.wait(2.0)

        # ── Final fade out ────────────────────────────────────────────────────
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.4)
        self.wait(0.5)
