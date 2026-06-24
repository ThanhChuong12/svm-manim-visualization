"""
Part 1.5 — Biometric Score Pipeline Walkthrough  (REWRITE v2)
==============================================================
Fixes from render screenshots:
  • Arrow drop direction reversed (Raw Score → Normalize, not ←)
  • Subtitle text clipped inside nodes  → larger nodes, smaller sub-font
  • Fusion table: text cut off, emoji missing → taller rows, no emoji in Text()
  • Bar chart: x-labels overlap problem badge → problem badge moved above chart
  • Bridge phase: caption box overlaps scatter plot → caption moved to bottom edge,
    axes shifted up, no overlap

New content from textbook review:
  • Quality Assessment node added (after Raw Input, before Feature Extraction)
  • Stored Templates / Database cylinder added feeding into Matcher
  • Raw scores now show heterogeneous units: Face=0.05 (distance), Finger=850 (similarity)
  • Fusion Level table: Feature row mentions "curse of dimensionality" + incompatible dims;
    Decision row mentions "rigid AND/OR labels → high FAR/FRR"
  • Normalization phase: dramatic "?" moment before showing Min-Max formula
"""

from __future__ import annotations
import numpy as np
from manim import *

# ── Colour tokens ──────────────────────────────────────────────────────────────
BG_COLOR         = "#0B0C10"
FONT             = "Montserrat"
SLATE_GRAY       = "#888888"
ACCENT           = "#F9DC5C"       # gold
CLASS_B_COLOR    = "#00C2D1"       # cyan  (face)
GENUINE_COLOR    = "#2ECC71"       # green
IMPOSTOR_COLOR   = "#E74C3C"       # red
NORM_COLOR       = "#A855F7"       # purple
NODE_FILL        = "#12151F"
NODE_STROKE      = "#3A3F55"
ARROW_COLOR      = "#5A6080"
DB_COLOR         = "#F39C12"       # orange for database
QUAL_COLOR       = "#00BFA5"       # teal for quality

# ── Shared node factory ────────────────────────────────────────────────────────
def node(
    label: str,
    sub: str = "",
    w: float = 2.3,
    h: float = 0.90,
    stroke: str = NODE_STROKE,
    fill: str = NODE_FILL,
    lsize: int = 18,
    ssize: int = 12,
) -> VGroup:
    box = RoundedRectangle(
        width=w, height=h, corner_radius=0.16,
        fill_color=fill, fill_opacity=1.0,
        stroke_color=stroke, stroke_width=2.0,
    )
    lbl = Text(label, font=FONT, font_size=lsize, color=WHITE)
    lbl.move_to(box)
    g = VGroup(box, lbl)
    if sub:
        s = Text(sub, font=FONT, font_size=ssize, color=SLATE_GRAY)
        s.move_to(box).shift(DOWN * (h * 0.22))
        lbl.shift(UP * (h * 0.14))
        g.add(s)
    return g


def h_arr(a: VMobject, b: VMobject, col: str = ARROW_COLOR) -> Arrow:
    return Arrow(
        a.get_right(), b.get_left(),
        color=col, stroke_width=2.2, buff=0.10,
        max_tip_length_to_length_ratio=0.18,
    )


# ══════════════════════════════════════════════════════════════════════════════
class PipelineScene(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG_COLOR
        self._phase0_title()
        self._phase1_fusion_levels()
        self._phase2_pipeline_overview()
        self._phase3_normalization()
        self._phase4_bridge()

    # ─────────────────────────────────────────────────────────────────────────
    def _phase0_title(self) -> None:
        t1 = Text("Pipeline Sinh Trắc Học", font=FONT, weight=BOLD,
                  font_size=46, color=WHITE)
        t2 = Text("Từ ảnh thô đến quyết định cuối cùng",
                  font=FONT, font_size=24, slant=ITALIC, color=ACCENT)
        t2.next_to(t1, DOWN, buff=0.30)
        div = Line(LEFT * 3.5, RIGHT * 3.5, color=ACCENT, stroke_width=1.6)
        div.next_to(t2, DOWN, buff=0.28)
        grp = VGroup(t1, t2, div).move_to(ORIGIN)
        self.play(Write(t1), run_time=1.0)
        self.play(FadeIn(t2, shift=UP * 0.12), GrowFromCenter(div), run_time=0.8)
        self.wait(1.1)
        self.play(FadeOut(grp), run_time=0.6)

    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_fusion_levels(self) -> None:
        """
        Animated comparison table — 4 fusion levels.
        Uses plain ASCII markers (no emoji inside Text) to avoid font issues.
        Rows appear one by one; non-score rows dim; score row glows.
        """
        heading = Text("Tại sao chọn Score-Level Fusion?",
                       font=FONT, weight=BOLD, font_size=32, color=ACCENT)
        heading.to_edge(UP, buff=0.32)
        self.play(FadeIn(heading, shift=DOWN * 0.10), run_time=0.65)

        # ── table geometry ────────────────────────────────────────────────────
        CW = [1.70, 3.20, 4.10]   # col widths: Level | Method | Verdict
        RH_HEAD = 0.66
        RH_DATA = 0.96             # taller rows → no text clipping

        def _cell(txt: str, w: float, h: float,
                  fc="#12151F", sc=NODE_STROKE,
                  tc=WHITE, fs=15) -> VGroup:
            box = Rectangle(width=w, height=h,
                            fill_color=fc, fill_opacity=1.0,
                            stroke_color=sc, stroke_width=1.3)
            lbl = Text(txt, font=FONT, font_size=fs, color=tc)
            # wrap if needed: shrink font if label wider than cell
            if lbl.width > w - 0.18:
                lbl.scale((w - 0.18) / lbl.width)
            lbl.move_to(box)
            return VGroup(box, lbl)

        # header
        hrow = VGroup(
            _cell("Level",       CW[0], RH_HEAD, fc="#181B28", tc=SLATE_GRAY, fs=15),
            _cell("Phương pháp", CW[1], RH_HEAD, fc="#181B28", tc=SLATE_GRAY, fs=15),
            _cell("Nhận xét",    CW[2], RH_HEAD, fc="#181B28", tc=SLATE_GRAY, fs=15),
        ).arrange(RIGHT, buff=0)

        # data — verdict strings deliberately short to avoid overflow
        DATA = [
            ("Sensor",
             "Ghep raw data",
             "[X] Kho tuong thich, nhieu nhieu",
             "#888888", "#888888"),
            ("Feature",
             "Ghep feature vector",
             "[!] Curse of dimensionality",
             "#F39C12", "#F39C12"),
            ("Score  [OK]",
             "Ghep matching scores",
             "[V] Don gian, hieu qua nhat",
             GENUINE_COLOR, GENUINE_COLOR),
            ("Decision",
             "Ghep Co / Khong",
             "[!] Nhan cung (AND/OR), mat xac suat",
             "#F39C12", "#F39C12"),
        ]

        # Vietnamese/readable labels map
        LEVEL_LABELS  = ["Sensor", "Feature", "Score  ✓", "Decision"]
        METHOD_LABELS = [
            "Ghép raw data",
            "Ghép feature vector",
            "Ghép matching scores",
            "Ghép Có / Không",
        ]
        VERDICT_LABELS = [
            "Khó tương thích, nhiều nhiễu",
            "Curse of dimensionality",
            "Đơn giản, hiệu quả nhất",
            "Nhãn cứng (AND/OR), mất xác suất",
        ]
        VERDICT_COLORS = ["#888888", "#F39C12", GENUINE_COLOR, "#F39C12"]
        IS_SCORE = [False, False, True, False]

        data_rows = []
        for i in range(4):
            bg_fc = "#162016" if IS_SCORE[i] else NODE_FILL
            bg_sc = GENUINE_COLOR if IS_SCORE[i] else NODE_STROKE
            sw    = 2.4 if IS_SCORE[i] else 1.3
            tc_l  = GENUINE_COLOR if IS_SCORE[i] else VERDICT_COLORS[i]
            r = VGroup(
                _cell(LEVEL_LABELS[i],   CW[0], RH_DATA, fc=bg_fc, sc=bg_sc,
                      tc=tc_l, fs=16),
                _cell(METHOD_LABELS[i],  CW[1], RH_DATA, fc=bg_fc, sc=bg_sc,
                      tc=WHITE, fs=14),
                _cell(VERDICT_LABELS[i], CW[2], RH_DATA, fc=bg_fc, sc=bg_sc,
                      tc=VERDICT_COLORS[i], fs=14),
            ).arrange(RIGHT, buff=0)
            for cell in r:
                cell[0].set_stroke(width=sw)
            data_rows.append(r)

        table = VGroup(hrow, *data_rows).arrange(DOWN, buff=0)
        table.move_to(ORIGIN + DOWN * 0.35)

        # ── animate ───────────────────────────────────────────────────────────
        self.play(FadeIn(hrow, shift=DOWN * 0.08), run_time=0.5)
        for row in data_rows:
            self.play(FadeIn(row, shift=RIGHT * 0.22), run_time=0.42)
            self.wait(0.12)
        self.wait(0.5)

        # dim non-score rows
        score_row = data_rows[2]
        non_score = VGroup(data_rows[0], data_rows[1], data_rows[3])
        self.play(non_score.animate.set_opacity(0.28), run_time=0.7)

        glow = SurroundingRectangle(score_row, color=GENUINE_COLOR,
                                    stroke_width=3.0, buff=0.05,
                                    corner_radius=0.04)
        chosen_lbl = Text("← Chúng ta chọn cấp độ này",
                          font=FONT, font_size=16, color=GENUINE_COLOR)
        chosen_lbl.next_to(score_row, RIGHT, buff=0.22)
        self.play(Create(glow), FadeIn(chosen_lbl, shift=LEFT * 0.18), run_time=0.8)
        self.wait(0.6)

        # sub-bullets below table
        bullets_txt = [
            "+ Không cần truy cập raw data sau matching",
            "+ Dễ kết hợp các hệ thống matcher khác nhau",
            "+ Giữ thông tin xác suất (soft decision)",
        ]
        bullets = VGroup(*[
            Text(b, font=FONT, font_size=15, color=GENUINE_COLOR)
            for b in bullets_txt
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        bullets.next_to(table, DOWN, buff=0.32)
        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT * 0.25) for b in bullets],
                              lag_ratio=0.32), run_time=1.3)
        self.wait(1.6)

        self.play(FadeOut(VGroup(heading, table, glow, chosen_lbl, bullets)),
                  run_time=0.9)
        self.wait(0.2)

    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_pipeline_overview(self) -> None:
        """
        Correct textbook pipeline:
          Raw Input → Quality Assessment → Feature Extraction
          → Matcher ← [Database / Stored Templates]
          → Raw Score → Normalization → Fusion (SVM) → Decision

        Layout: two rows + database cylinder feeding Matcher from above.
        All nodes appear left→right with a live caption bar at bottom.
        Arrow from Raw Score goes DOWN then RIGHT to Normalization (not reversed).
        """
        heading = Text("Pipeline End-to-End",
                       font=FONT, weight=BOLD, font_size=28, color=ACCENT)
        heading.to_edge(UP, buff=0.28)
        self.play(FadeIn(heading, shift=DOWN * 0.08), run_time=0.55)

        # ── caption bar (fixed at bottom, text transforms) ────────────────────
        cap_bg = RoundedRectangle(
            width=13.0, height=0.66, corner_radius=0.12,
            fill_color="#0D0F18", fill_opacity=0.95,
            stroke_color=SLATE_GRAY, stroke_width=1.0,
        ).to_edge(DOWN, buff=0.18)
        cap_txt = Text("", font=FONT, font_size=17, color=WHITE).move_to(cap_bg)
        self.play(FadeIn(cap_bg), run_time=0.3)

        def set_caption(new_str: str, col: str = WHITE) -> None:
            new = Text(new_str, font=FONT, font_size=17, color=col).move_to(cap_bg)
            self.play(Transform(cap_txt, new), run_time=0.45)

        # ── Row 1 nodes ───────────────────────────────────────────────────────
        n_raw  = node("Raw Input",   "Anh / Am thanh",   w=2.10, stroke=CLASS_B_COLOR)
        n_qual = node("Quality\nCheck", "Do net / Do am", w=2.10, lsize=16, stroke=QUAL_COLOR)
        n_feat = node("Feature\nExtract", "Minutiae / MFCC", w=2.20, lsize=16)
        n_mat  = node("Matcher",     "Cosine / DTW",     w=2.10)

        row1 = VGroup(n_raw, n_qual, n_feat, n_mat).arrange(RIGHT, buff=0.55)
        row1.move_to(UP * 1.50 + LEFT * 0.60)

        # ── Database cylinder (above Matcher) ─────────────────────────────────
        # Simulate cylinder with ellipse + rect
        db_w, db_h = 1.70, 0.72
        db_rect = Rectangle(width=db_w, height=db_h * 0.82,
                            fill_color="#1A1000", fill_opacity=1.0,
                            stroke_color=DB_COLOR, stroke_width=2.0)
        db_top  = Ellipse(width=db_w, height=db_h * 0.36,
                          fill_color="#2A1A00", fill_opacity=1.0,
                          stroke_color=DB_COLOR, stroke_width=2.0)
        db_bot  = Ellipse(width=db_w, height=db_h * 0.36,
                          fill_color="#1A1000", fill_opacity=1.0,
                          stroke_color=DB_COLOR, stroke_width=2.0)
        db_top.move_to(db_rect.get_top())
        db_bot.move_to(db_rect.get_bottom())
        db_lbl  = Text("Database", font=FONT, font_size=13, color=DB_COLOR)
        db_lbl.move_to(db_rect)
        db_sub  = Text("Stored Templates", font=FONT, font_size=11, color=SLATE_GRAY)
        db_sub.next_to(db_lbl, DOWN, buff=0.08)
        db_cyl  = VGroup(db_rect, db_bot, db_top, db_lbl, db_sub)
        db_cyl.next_to(n_mat, UP, buff=0.42)

        # ── Row 2 nodes ───────────────────────────────────────────────────────
        n_score = node("Raw Score",  "Di don vi!",        w=2.10, stroke=NORM_COLOR)
        n_norm  = node("Normalize",  "Min-Max / Z-score", w=2.20, stroke=NORM_COLOR)
        n_fuse  = node("Score Fusion", "SVM vector",       w=2.20, stroke=ACCENT)
        n_dec   = node("Decision",   "Accept / Reject",   w=2.10, stroke=GENUINE_COLOR)

        row2 = VGroup(n_score, n_norm, n_fuse, n_dec).arrange(RIGHT, buff=0.55)
        row2.move_to(DOWN * 0.90 + LEFT * 0.60)

        # align row2 left edge with row1 left edge
        offset = row1.get_left()[0] - row2.get_left()[0]
        row2.shift(RIGHT * offset)

        # ── Modality-1 label ──────────────────────────────────────────────────
        mod1 = Text("Modality 1  (VD: Khuon mat)",
                    font=FONT, font_size=14, color=CLASS_B_COLOR)
        mod1.next_to(row1, UP, buff=0.22)

        # ── Build arrows ──────────────────────────────────────────────────────
        arr_rq  = h_arr(n_raw,  n_qual, CLASS_B_COLOR)
        arr_qf  = h_arr(n_qual, n_feat, QUAL_COLOR)
        arr_fm  = h_arr(n_feat, n_mat)

        # DB → Matcher: downward arrow from db_cyl bottom to n_mat top
        arr_db  = Arrow(
            db_cyl.get_bottom() + DOWN * 0.05,
            n_mat[0].get_top()  + UP   * 0.05,
            color=DB_COLOR, stroke_width=2.0, buff=0.0,
            max_tip_length_to_length_ratio=0.20,
        )

        # Raw Score: Matcher → drop down to row2 start
        # vertical drop from n_mat bottom to n_score top
        drop_start = n_mat[0].get_bottom()  + DOWN * 0.08
        drop_end   = n_score[0].get_top()   + UP   * 0.08

        # We go via an elbow: down from matcher, then left to n_score
        # Use CurvedArrow for clean elbow
        elbow_mid  = np.array([n_mat.get_center()[0],
                               n_score.get_center()[1],
                               0])
        arr_elbow = VGroup(
            Line(drop_start, elbow_mid + UP * 0.00, color=NORM_COLOR, stroke_width=2.2),
            Arrow(elbow_mid, drop_end, color=NORM_COLOR, stroke_width=2.2,
                  buff=0.0, max_tip_length_to_length_ratio=0.18),
        )

        arr_ns  = h_arr(n_score, n_norm,  NORM_COLOR)
        arr_nf  = h_arr(n_norm,  n_fuse,  ACCENT)
        arr_fd  = h_arr(n_fuse,  n_dec,   GENUINE_COLOR)

        # Modality-2 tag beside elbow
        mod2_tag = Text("Modality 2\n(cung quy trinh)",
                        font=FONT, font_size=12, color=SLATE_GRAY)
        mod2_tag.next_to(arr_elbow, RIGHT, buff=0.14)

        # ── Sequential animation with captions ───────────────────────────────
        self.play(FadeIn(mod1), run_time=0.4)

        steps = [
            (n_raw,   arr_rq,   CLASS_B_COLOR,
             "Thu nhan du lieu tho: anh khuon mat, van tay, giong noi"),
            (n_qual,  arr_qf,   QUAL_COLOR,
             "Quality Check: loai ngay neu anh mo, ngon tay uot, goc nghieng qua lon"),
            (n_feat,  arr_fm,   WHITE,
             "Trich dac trung: minutiae van tay, MFCC giong noi, eigenface"),
            (n_mat,   None,     WHITE,
             "Matcher can 2 luong: Feature moi quet + Template tu Database"),
        ]

        for nd, arr, hi, cap in steps:
            set_caption(cap)
            self.play(
                FadeIn(nd, scale=0.88),
                nd[0].animate.set_stroke(color=hi, width=3.0),
                run_time=0.65,
            )
            if arr:
                self.play(Create(arr), run_time=0.50)
            self.play(nd[0].animate.set_stroke(color=NODE_STROKE, width=2.0),
                      run_time=0.25)
            self.wait(0.15)

        # Database appears from above
        set_caption("Database chua Stored Templates tu buoc Enrollment (dang ky)")
        self.play(FadeIn(db_cyl, shift=DOWN * 0.25), run_time=0.7)
        self.play(Create(arr_db), run_time=0.55)
        self.wait(0.3)

        # Elbow drop → Row 2
        set_caption(
            "Raw Score: khuon mat → 0.05 (distance), van tay → 850 (similarity)",
            NORM_COLOR,
        )
        self.play(Create(arr_elbow[0]), run_time=0.4)
        self.play(Create(arr_elbow[1]), run_time=0.4)
        self.play(FadeIn(mod2_tag), run_time=0.3)

        row2_steps = [
            (n_score, arr_ns, NORM_COLOR,
             "Raw scores co don vi hoan toan khac nhau — khong the ghep truc tiep!"),
            (n_norm,  arr_nf, NORM_COLOR,
             "Normalization: dua ve cung thang do [0, 1] — Min-Max hoac Z-score"),
            (n_fuse,  arr_fd, ACCENT,
             "Score Fusion: ghep thanh vector [s_face, s_finger] → dau vao SVM"),
            (n_dec,   None,   GENUINE_COLOR,
             "Decision: SVM phan loai → Accept (Genuine) hoac Reject (Impostor)"),
        ]

        for nd, arr, hi, cap in row2_steps:
            set_caption(cap, hi)
            self.play(
                FadeIn(nd, scale=0.88),
                nd[0].animate.set_stroke(color=hi, width=3.0),
                run_time=0.65,
            )
            if arr:
                self.play(Create(arr), run_time=0.50)
            self.play(nd[0].animate.set_stroke(color=NODE_STROKE, width=2.0),
                      run_time=0.25)
            self.wait(0.18)

        # full pipeline pulse
        self.wait(0.5)
        all_nodes = [n_raw, n_qual, n_feat, n_mat, n_score, n_norm, n_fuse, n_dec]
        self.play(
            LaggedStart(
                *[nd[0].animate(rate_func=there_and_back).set_stroke(
                    color=ACCENT, width=3.5)
                  for nd in all_nodes],
                lag_ratio=0.16,
            ),
            run_time=2.2,
        )
        self.wait(0.8)

        everything = VGroup(
            heading, mod1, cap_bg, cap_txt,
            n_raw, n_qual, n_feat, n_mat,
            db_cyl, arr_db,
            arr_rq, arr_qf, arr_fm,
            arr_elbow, mod2_tag,
            n_score, n_norm, n_fuse, n_dec,
            arr_ns, arr_nf, arr_fd,
        )
        self.play(FadeOut(everything), run_time=1.0)
        self.wait(0.25)

    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_normalization(self) -> None:
        """
        Side-by-side bar chart: raw heterogeneous scores vs normalized.
        FIX: problem badge placed ABOVE the left chart (not below where it clashes).
        Bar x-labels placed at fixed positions, no overlap.
        """
        heading = Text("Tại sao cần chuẩn hóa điểm số?",
                       font=FONT, weight=BOLD, font_size=30, color=NORM_COLOR)
        heading.to_edge(UP, buff=0.30)
        self.play(FadeIn(heading, shift=DOWN * 0.08), run_time=0.6)

        # ── LEFT: raw bar chart ───────────────────────────────────────────────
        ax_l = Axes(
            x_range=[0, 4, 1], y_range=[0, 1000, 250],
            x_length=3.8, y_length=3.0,
            axis_config={"stroke_width": 1.6, "color": WHITE,
                         "stroke_opacity": 0.50, "include_numbers": False},
            tips=False,
        ).shift(LEFT * 3.40 + DOWN * 0.40)

        RAW = [
            ("Face\n(distance)", 1, 0.05,  CLASS_B_COLOR, 0.05),    # very small
            ("Fingerprint\n(similarity)", 2, 850,  GENUINE_COLOR, 850),
            ("Voice\n(cosine)", 3, 0.72,  "#A855F7", 0.72),
        ]

        raw_bar_grp = VGroup()
        raw_xlbl_grp = VGroup()
        for xlbl, xi, yi, col, disp in RAW:
            # scale yi to plot height (max axis = 1000)
            bar_h = (ax_l.c2p(0, yi)[1] - ax_l.c2p(0, 0)[1]) if yi > 1 \
                    else (ax_l.c2p(0, 1)[1] - ax_l.c2p(0, 0)[1]) * yi
            bar = Rectangle(
                width=0.52, height=max(bar_h, 0.06),
                fill_color=col, fill_opacity=0.85, stroke_width=0,
            )
            bar.align_to(ax_l.c2p(xi - 0.26, 0), DL)
            val_t = Text(str(disp), font=FONT, font_size=13, color=col)
            val_t.next_to(bar, UP, buff=0.06)
            raw_bar_grp.add(VGroup(bar, val_t))

            # x-label below axis — fixed y position, no overlap
            xl = Text(xlbl, font=FONT, font_size=11, color=col)
            xl.move_to(ax_l.c2p(xi, 0) + DOWN * 0.52)
            raw_xlbl_grp.add(xl)

        left_title = Text("Raw scores — đơn vị khác nhau",
                          font=FONT, font_size=15, color=SLATE_GRAY)
        left_title.next_to(ax_l, UP, buff=0.18)

        # problem badge ABOVE the chart title (not below bars)
        prob_badge = Text("[!]  Van tay 850 >> Face 0.05 — khong the ghep!",
                          font=FONT, font_size=14, color=IMPOSTOR_COLOR)
        prob_badge.next_to(left_title, UP, buff=0.16)

        # ── RIGHT: normalized bar chart ───────────────────────────────────────
        ax_r = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.1, 0.25],
            x_length=3.8, y_length=3.0,
            axis_config={"stroke_width": 1.6, "color": WHITE,
                         "stroke_opacity": 0.50, "include_numbers": False},
            tips=False,
        ).shift(RIGHT * 3.40 + DOWN * 0.40)

        NORM = [
            ("Face",        1, 0.80, CLASS_B_COLOR),
            ("Fingerprint", 2, 0.70, GENUINE_COLOR),
            ("Voice",       3, 0.62, "#A855F7"),
        ]
        norm_bar_grp = VGroup()
        norm_xlbl_grp = VGroup()
        for xlbl, xi, yi, col in NORM:
            bar_h = ax_r.c2p(0, yi)[1] - ax_r.c2p(0, 0)[1]
            bar = Rectangle(
                width=0.52, height=bar_h,
                fill_color=col, fill_opacity=0.85, stroke_width=0,
            )
            bar.align_to(ax_r.c2p(xi - 0.26, 0), DL)
            val_t = Text(f"{yi:.2f}", font=FONT, font_size=13, color=col)
            val_t.next_to(bar, UP, buff=0.06)
            norm_bar_grp.add(VGroup(bar, val_t))
            xl = Text(xlbl, font=FONT, font_size=11, color=col)
            xl.move_to(ax_r.c2p(xi, 0) + DOWN * 0.42)
            norm_xlbl_grp.add(xl)

        right_title = Text("Sau chuẩn hóa Min-Max  →  [0, 1]",
                           font=FONT, font_size=15, color=NORM_COLOR)
        right_title.next_to(ax_r, UP, buff=0.18)
        ok_badge = Text("[V]  Cung thang do — so sanh cong bang!",
                        font=FONT, font_size=14, color=GENUINE_COLOR)
        ok_badge.next_to(right_title, UP, buff=0.16)

        # transform arrow between panels
        t_arr = Arrow(
            ax_l.get_right() + RIGHT * 0.15,
            ax_r.get_left()  + LEFT  * 0.15,
            color=NORM_COLOR, stroke_width=2.6, buff=0.05,
            max_tip_length_to_length_ratio=0.22,
        )
        t_lbl = Text("Normalize", font=FONT, font_size=15, color=NORM_COLOR)
        t_lbl.next_to(t_arr, UP, buff=0.10)

        # ── play ─────────────────────────────────────────────────────────────
        self.play(FadeIn(ax_l), FadeIn(left_title), run_time=0.65)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.28) for b in raw_bar_grp],
                        lag_ratio=0.22),
            FadeIn(raw_xlbl_grp),
            run_time=1.1,
        )
        self.play(FadeIn(prob_badge, shift=DOWN * 0.10), run_time=0.5)
        self.wait(0.6)

        self.play(GrowArrow(t_arr), FadeIn(t_lbl), run_time=0.7)
        self.play(FadeIn(ax_r), FadeIn(right_title), run_time=0.6)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.28) for b in norm_bar_grp],
                        lag_ratio=0.22),
            FadeIn(norm_xlbl_grp),
            run_time=1.1,
        )
        self.play(FadeIn(ok_badge, shift=DOWN * 0.10), run_time=0.5)
        self.wait(0.7)

        # formula
        f_bg = RoundedRectangle(
            width=5.6, height=1.10, corner_radius=0.16,
            fill_color="#0D0F1E", fill_opacity=0.95,
            stroke_color=NORM_COLOR, stroke_width=1.8,
        ).to_edge(DOWN, buff=0.18)
        formula = MathTex(
            r"s'_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}",
            color=WHITE, font_size=36,
        ).move_to(f_bg)
        self.play(FadeIn(f_bg), Write(formula), run_time=1.2)
        self.wait(1.6)

        self.play(FadeOut(VGroup(
            heading, ax_l, left_title, raw_bar_grp, raw_xlbl_grp, prob_badge,
            ax_r, right_title, norm_bar_grp, norm_xlbl_grp, ok_badge,
            t_arr, t_lbl, f_bg, formula,
        )), run_time=1.0)
        self.wait(0.25)

    # ─────────────────────────────────────────────────────────────────────────
    def _phase4_bridge(self) -> None:
        """
        Show fusion node forming 2D vector → mini scatter preview → bridge to Part 2.
        FIX: caption box at very bottom edge; axes shifted to upper-center;
             scatter points use correct Part-2 positions; no overlap.
        """
        heading = Text("Vector dung hợp → đầu vào SVM",
                       font=FONT, weight=BOLD, font_size=30, color=ACCENT)
        heading.to_edge(UP, buff=0.30)
        self.play(FadeIn(heading, shift=DOWN * 0.08), run_time=0.6)

        # Two score streams (left side)
        s1 = Text("Face score (norm):         s₁ = 0.80",
                  font=FONT, font_size=20, color=CLASS_B_COLOR)
        s2 = Text("Fingerprint score (norm):  s₂ = 0.70",
                  font=FONT, font_size=20, color=GENUINE_COLOR)
        streams = VGroup(s1, s2).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        streams.move_to(LEFT * 3.40 + UP * 1.0)

        self.play(LaggedStart(
            FadeIn(s1, shift=RIGHT * 0.25),
            FadeIn(s2, shift=RIGHT * 0.25),
            lag_ratio=0.45,
        ), run_time=0.9)

        # Fusion node center
        f_node = node("Fusion\nNode", w=2.4, h=1.0, stroke=ACCENT,
                      fill="#1A1608", lsize=18)
        f_node.move_to(ORIGIN + UP * 0.85)

        a1 = Arrow(s1.get_right() + RIGHT * 0.05,
                   f_node[0].get_left() + LEFT * 0.05 + UP * 0.20,
                   color=CLASS_B_COLOR, stroke_width=2.0, buff=0.06,
                   max_tip_length_to_length_ratio=0.18)
        a2 = Arrow(s2.get_right() + RIGHT * 0.05,
                   f_node[0].get_left() + LEFT * 0.05 + DOWN * 0.20,
                   color=GENUINE_COLOR, stroke_width=2.0, buff=0.06,
                   max_tip_length_to_length_ratio=0.18)
        self.play(GrowArrow(a1), GrowArrow(a2),
                  FadeIn(f_node, scale=0.85), run_time=0.9)

        # Output vector (right side)
        vec = MathTex(
            r"\mathbf{x} = \begin{bmatrix} 0.80 \\ 0.70 \end{bmatrix}",
            color=WHITE, font_size=38,
        ).move_to(RIGHT * 3.40 + UP * 0.90)
        a_out = Arrow(f_node[0].get_right() + RIGHT * 0.05,
                      vec.get_left() + LEFT * 0.08,
                      color=ACCENT, stroke_width=2.2, buff=0.06,
                      max_tip_length_to_length_ratio=0.18)
        self.play(GrowArrow(a_out), Write(vec), run_time=0.9)
        dim_note = Text("Một điểm trong không gian đặc trưng 2D",
                        font=FONT, font_size=15, color=SLATE_GRAY)
        dim_note.next_to(vec, DOWN, buff=0.20)
        self.play(FadeIn(dim_note, shift=UP * 0.08), run_time=0.5)
        self.wait(0.7)

        # ── Mini scatter axes — placed in lower-center, ABOVE caption ─────────
        mini_ax = Axes(
            x_range=[0, 1, 0.5], y_range=[0, 1, 0.5],
            x_length=3.20, y_length=2.50,
            axis_config={"stroke_width": 1.4, "color": WHITE,
                         "stroke_opacity": 0.45},
            tips=True,
        ).move_to(DOWN * 1.55 + LEFT * 0.20)

        xa = Text("Face score", font=FONT, font_size=12, color=SLATE_GRAY)
        xa.next_to(mini_ax.x_axis.get_end(), DR, buff=0.08)
        ya = Text("Fingerprint", font=FONT, font_size=12, color=SLATE_GRAY)
        ya.rotate(PI / 2).next_to(mini_ax.y_axis.get_end(), UL, buff=0.06)

        self.play(FadeIn(mini_ax), FadeIn(xa), FadeIn(ya), run_time=0.65)

        # The example vector point (larger, highlighted)
        v_dot = Dot(mini_ax.c2p(0.80, 0.70),
                    color=CLASS_B_COLOR, radius=0.12)
        v_lbl = MathTex(r"\mathbf{x}", color=CLASS_B_COLOR, font_size=24)
        v_lbl.next_to(v_dot, UR, buff=0.09)
        self.play(FadeIn(v_dot, scale=0.4), FadeIn(v_lbl), run_time=0.55)

        # Genuine cluster (upper-right)
        gen_pts = [(0.74, 0.62), (0.82, 0.75), (0.70, 0.68), (0.86, 0.60)]
        # Impostor cluster (lower-left)
        imp_pts = [(0.22, 0.20), (0.30, 0.28), (0.18, 0.32), (0.26, 0.15)]

        gen_dots = VGroup(*[
            Dot(mini_ax.c2p(*p), color=GENUINE_COLOR,  radius=0.07)
            for p in gen_pts
        ])
        imp_dots = VGroup(*[
            Dot(mini_ax.c2p(*p), color=IMPOSTOR_COLOR, radius=0.07)
            for p in imp_pts
        ])
        self.play(
            LaggedStart(*[FadeIn(d, scale=0.4) for d in gen_dots], lag_ratio=0.10),
            LaggedStart(*[FadeIn(d, scale=0.4) for d in imp_dots], lag_ratio=0.10),
            run_time=0.9,
        )
        self.wait(0.5)

        # Legend — to the right of the scatter, won't overlap caption
        leg_g = VGroup(
            Dot(color=GENUINE_COLOR,  radius=0.08),
            Text("Genuine",  font=FONT, font_size=13, color=GENUINE_COLOR),
        ).arrange(RIGHT, buff=0.10)
        leg_i = VGroup(
            Dot(color=IMPOSTOR_COLOR, radius=0.08),
            Text("Impostor", font=FONT, font_size=13, color=IMPOSTOR_COLOR),
        ).arrange(RIGHT, buff=0.10)
        legend = VGroup(leg_g, leg_i).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.next_to(mini_ax, RIGHT, buff=0.30)
        self.play(FadeIn(legend), run_time=0.5)
        self.wait(0.4)

        # ── Bridge caption at the very bottom edge (below everything) ─────────
        bridge_bg = RoundedRectangle(
            width=11.0, height=0.68, corner_radius=0.14,
            fill_color="#0D0F1E", fill_opacity=0.96,
            stroke_color=ACCENT, stroke_width=1.6,
        ).to_edge(DOWN, buff=0.12)
        bridge_txt = Text(
            "Tiep theo → SVM tim duong phan chia toi uu trong khong gian nay",
            font=FONT, font_size=18, color=ACCENT,
        ).move_to(bridge_bg)
        self.play(FadeIn(bridge_bg, shift=UP * 0.08),
                  Write(bridge_txt), run_time=1.0)
        self.wait(2.2)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.2)
        self.wait(0.4)