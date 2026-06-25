"""
Part 1.5 — Biometric Score Pipeline Walkthrough  (v3 — full English, zero overlap)
====================================================================================
Scene order:
  Phase 0 : Title card
  Phase 1 : Fusion-level comparison table  (4 rows, dim non-score rows)
  Phase 2 : Pipeline walkthrough  (textbook-accurate, clean layout)
  Phase 3 : Normalization side-by-side bar chart
  Phase 4 : Bridge — fusion node → 2-D scatter → teaser

Layout rules (v3):
  • ALL text in English — no Vietnamese
  • Pipeline uses TWO completely separate horizontal rows,
    joined by a single clean vertical drop arrow on the RIGHT side.
    Row 1 (top)   : Raw Input → Quality Check → Feature Extract → Matcher
    Row 2 (bottom): Raw Score → Normalize → Score Fusion → Decision
    Database cylinder sits to the RIGHT of the canvas, feeds Matcher from right.
    The vertical connector goes: Matcher.bottom → straight down → Raw Score.top
    (Raw Score is placed directly below Matcher — same x-centre.)
  • Every caption lives in a fixed bar at the very bottom (y = -3.6).
    Nothing else is drawn in that zone.
  • Bridge phase: fusion node on the LEFT half only; scatter axes on the RIGHT
    half only; caption bar at bottom — three zones never overlap.
"""

from __future__ import annotations
import numpy as np
from manim import *

# ── palette ───────────────────────────────────────────────────────────────────
BG        = "#0B0C10"
GOLD      = "#F9DC5C"
CYAN      = "#00C2D1"
GREEN     = "#2ECC71"
RED       = "#E74C3C"
PURPLE    = "#A855F7"
ORANGE    = "#F39C12"
TEAL      = "#00BFA5"
GRAY      = "#666677"
NFILL     = "#12151F"
NSTROKE   = "#3A3F55"
FONT      = "Montserrat"

# safe y-zones (Manim default frame: ±4 vertical, ±7.1 horizontal)
CAP_Y     = -3.45          # caption bar centre-y  (bottom zone)
SAFE_BOT  = CAP_Y + 0.55  # lowest y any non-caption element may use

# ── helpers ───────────────────────────────────────────────────────────────────
def nd(label: str, sub: str = "", w=2.30, h=0.88,
       sc=NSTROKE, fc=NFILL, ls=18, ss=12) -> VGroup:
    """Rounded-rect pipeline node."""
    box = RoundedRectangle(width=w, height=h, corner_radius=0.15,
                           fill_color=fc, fill_opacity=1, stroke_color=sc, stroke_width=2)
    lbl = Text(label, font=FONT, font_size=ls, color=WHITE)
    lbl.move_to(box)
    g = VGroup(box, lbl)
    if sub:
        s = Text(sub, font=FONT, font_size=ss, color=GRAY)
        s.move_to(box).shift(DOWN * h * 0.24)
        lbl.shift(UP * h * 0.16)
        g.add(s)
    return g

def harrow(a, b, col=GRAY) -> Arrow:
    return Arrow(a.get_right(), b.get_left(),
                 color=col, stroke_width=2.2, buff=0.10,
                 max_tip_length_to_length_ratio=0.18)

def varrow(a, b, col=GRAY) -> Arrow:
    return Arrow(a.get_bottom(), b.get_top(),
                 color=col, stroke_width=2.2, buff=0.10,
                 max_tip_length_to_length_ratio=0.18)


# ══════════════════════════════════════════════════════════════════════════════
class PipelineScene(Scene):

    def construct(self):
        self.camera.background_color = BG
        self._title()
        self._fusion_table()
        self._pipeline()
        self._normalization()
        self._bridge()

    # ── 0. Title ──────────────────────────────────────────────────────────────
    def _title(self):
        t1 = Text("Biometric Score Pipeline", font=FONT, weight=BOLD,
                  font_size=52, color=WHITE)
        t2 = Text("From raw input to final decision",
                  font=FONT, font_size=26, slant=ITALIC, color=GOLD)
        t2.next_to(t1, DOWN, buff=0.30)
        bar = Line(LEFT*4, RIGHT*4, color=GOLD, stroke_width=1.6)
        bar.next_to(t2, DOWN, buff=0.26)
        g = VGroup(t1, t2, bar).center()
        self.play(Write(t1), run_time=0.9)
        self.play(FadeIn(t2, shift=UP*0.1), GrowFromCenter(bar), run_time=0.75)
        self.wait(1.0)
        self.play(FadeOut(g), run_time=0.55)

    # ── 1. Fusion-level table ─────────────────────────────────────────────────
    def _fusion_table(self):
        head = Text("Why Score-Level Fusion?",
                    font=FONT, weight=BOLD, font_size=34, color=GOLD)
        head.to_edge(UP, buff=0.30)
        self.play(FadeIn(head, shift=DOWN*0.08), run_time=0.55)

        CW   = [1.80, 3.10, 4.40]
        RH_H = 0.62
        RH_D = 0.94

        def cell(txt, w, h, fc=NFILL, sc=NSTROKE, tc=WHITE, fs=15):
            box = Rectangle(width=w, height=h, fill_color=fc, fill_opacity=1,
                            stroke_color=sc, stroke_width=1.2)
            lbl = Text(txt, font=FONT, font_size=fs, color=tc)
            if lbl.width > w - 0.16:
                lbl.scale((w - 0.16) / lbl.width)
            lbl.move_to(box)
            return VGroup(box, lbl)

        hrow = VGroup(
            cell("Level",   CW[0], RH_H, fc="#181B28", tc=GRAY, fs=14),
            cell("Method",  CW[1], RH_H, fc="#181B28", tc=GRAY, fs=14),
            cell("Verdict", CW[2], RH_H, fc="#181B28", tc=GRAY, fs=14),
        ).arrange(RIGHT, buff=0)

        ROWS = [
            ("Sensor",      "Merge raw data",
             "Incompatible formats, too much noise",  GRAY,  False),
            ("Feature",     "Merge feature vectors",
             "Curse of dimensionality; mismatched dims", ORANGE, False),
            ("Score  [OK]", "Merge matching scores",
             "Simple, effective, most widely used",   GREEN, True),
            ("Decision",    "Merge Accept/Reject",
             "Rigid AND/OR labels; loses probability", ORANGE, False),
        ]

        drows = []
        for lv, meth, verd, col, star in ROWS:
            bg = "#132013" if star else NFILL
            sk = GREEN    if star else NSTROKE
            sw = 2.6      if star else 1.2
            r = VGroup(
                cell(lv,   CW[0], RH_D, fc=bg, sc=sk, tc=col,   fs=16),
                cell(meth, CW[1], RH_D, fc=bg, sc=sk, tc=WHITE,  fs=14),
                cell(verd, CW[2], RH_D, fc=bg, sc=sk, tc=col,    fs=13),
            ).arrange(RIGHT, buff=0)
            for c in r:
                c[0].set_stroke(width=sw)
            drows.append(r)

        table = VGroup(hrow, *drows).arrange(DOWN, buff=0)
        # centre table, leave room for bullets below (inside SAFE_BOT)
        table.move_to(ORIGIN + UP*0.22)

        self.play(FadeIn(hrow, shift=DOWN*0.06), run_time=0.45)
        for row in drows:
            self.play(FadeIn(row, shift=RIGHT*0.18), run_time=0.38)
            self.wait(0.10)
        self.wait(0.4)

        # dim non-score rows
        non = VGroup(drows[0], drows[1], drows[3])
        self.play(non.animate.set_opacity(0.25), run_time=0.65)

        glow = SurroundingRectangle(drows[2], color=GREEN,
                                    stroke_width=3.0, buff=0.05, corner_radius=0.04)
        tag  = Text("← Our choice", font=FONT, font_size=15, color=GREEN)
        tag.next_to(drows[2], RIGHT, buff=0.18)
        self.play(Create(glow), FadeIn(tag, shift=LEFT*0.14), run_time=0.7)
        self.wait(0.5)

        # 3 bullets — place inside SAFE_BOT zone
        blines = [
            "+ No access to raw data after matching",
            "+ Easy to combine heterogeneous matchers",
            "+ Preserves probability info (soft decision)",
        ]
        bullets = VGroup(*[
            Text(b, font=FONT, font_size=14, color=GREEN) for b in blines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        bullets.next_to(table, DOWN, buff=0.28)
        # clamp so bottom edge stays above CAP_Y
        if bullets.get_bottom()[1] < SAFE_BOT:
            bullets.shift(UP * (SAFE_BOT - bullets.get_bottom()[1]))

        self.play(LaggedStart(*[FadeIn(b, shift=RIGHT*0.20) for b in bullets],
                              lag_ratio=0.28), run_time=1.1)
        self.wait(1.5)
        self.play(FadeOut(VGroup(head, table, glow, tag, bullets)), run_time=0.8)
        self.wait(0.15)

    # ── 2. Pipeline ───────────────────────────────────────────────────────────
    def _pipeline(self):
        """
        Clean two-row layout with correct arrow directions.

        Row 1 (y=+1.6):  [RawInput]→[QualCheck]→[FeatExtract]→[Matcher]
        Row 2 (y=-0.55): [RawScore]→[Normalize]→[ScoreFusion]→[Decision]

        Matcher and RawScore share the SAME x-centre.
        A straight vertical arrow goes Matcher.bottom → RawScore.top.

        Database cylinder is placed ABOVE Matcher (not beside it).
        """
        ROW1_Y =  1.60
        ROW2_Y = -0.65
        # x-centres for 4 columns
        XS = [-5.20, -1.90,  1.40,  4.70]  # col 0..3

        head = Text("End-to-End Pipeline", font=FONT, weight=BOLD,
                    font_size=30, color=GOLD)
        head.to_edge(UP, buff=0.26)
        self.play(FadeIn(head, shift=DOWN*0.07), run_time=0.50)

        # caption bar — fixed, nothing else drawn here
        cap_bg = RoundedRectangle(
            width=13.8, height=0.62, corner_radius=0.12,
            fill_color="#0D0F18", fill_opacity=0.96,
            stroke_color=GRAY, stroke_width=0.9,
        )
        cap_bg.move_to([0, CAP_Y, 0])
        cap_txt = Text("", font=FONT, font_size=16, color=WHITE)
        cap_txt.move_to(cap_bg)
        self.play(FadeIn(cap_bg), run_time=0.25)
        cur_cap = cap_txt

        def caption(s: str, col=WHITE):
            nonlocal cur_cap
            new = Text(s, font=FONT, font_size=16, color=col)
            new.move_to(cap_bg)
            if new.width > 13.2:
                new.scale(13.2 / new.width)
            self.play(Transform(cur_cap, new), run_time=0.38)

        # ── Row 1 nodes ───────────────────────────────────────────────────────
        n_raw  = nd("Raw Input",       "Image / Audio",    w=2.18, sc=CYAN)
        n_qual = nd("Quality\nCheck",  "Sharpness / Wet",  w=2.18, sc=TEAL,  ls=17)
        n_feat = nd("Feature\nExtract","Minutiae / MFCC",  w=2.18, ls=17)
        n_mat  = nd("Matcher",         "Cosine / DTW",     w=2.18)

        for node_obj, xi in zip([n_raw, n_qual, n_feat, n_mat], XS):
            node_obj.move_to([xi, ROW1_Y, 0])

        # ── Row 2 nodes ───────────────────────────────────────────────────────
        n_scr  = nd("Raw Score",    "Different units!",  w=2.18, sc=PURPLE)
        n_norm = nd("Normalize",    "Min-Max / Z-score", w=2.18, sc=PURPLE)
        n_fuse = nd("Score Fusion", "SVM vector",        w=2.18, sc=GOLD)
        n_dec  = nd("Decision",     "Accept / Reject",   w=2.18, sc=GREEN)

        # Raw Score sits DIRECTLY below Matcher (same x = XS[3])
        # Then row 2 goes RIGHT→LEFT: score, norm, fuse, dec at XS[3],[2],[1],[0]... 
        # Actually we keep left-to-right reading order:
        # RawScore @ XS[3], Normalize @ XS[2], ScoreFusion @ XS[1], Decision @ XS[0]
        # Arrow direction: RawScore→Normalize→ScoreFusion→Decision  (right to left)
        # That looks odd. Better: keep same left-to-right column order for row 2
        # but map: col0=RawScore, col1=Normalize, col2=ScoreFusion, col3=Decision
        # and put RawScore at XS[3] (below Matcher) with a bend going LEFT.
        # Cleanest solution used by professional infographics:
        #   Row 2 left-to-right: Decision | ScoreFusion | Normalize | RawScore
        #   i.e. reverse order → read right-to-left with a "U-bend"
        # 
        # We use this snake pattern:
        #   Row1:  RawInput(0)→QualCheck(1)→FeatExtract(2)→Matcher(3)
        #                                                       ↓
        #   Row2:  Decision(0)←ScoreFusion(1)←Normalize(2)←RawScore(3)
        # Arrows in row2 point LEFT (←) which is fine visually.

        for node_obj, xi in zip([n_dec, n_fuse, n_norm, n_scr], XS):
            node_obj.move_to([xi, ROW2_Y, 0])

        # ── Database cylinder above Matcher ───────────────────────────────────
        # Cylinder sized so text has clear padding from all edges.
        # Strategy: measure labels first, then size rect around them.
        DW     = 2.40    # cylinder width
        BODY_H = 1.00    # rect body (tall enough for two padded lines)
        CAP_H  = 0.38    # ellipse cap height

        db_rect = Rectangle(
            width=DW, height=BODY_H,
            fill_color="#1C1000", fill_opacity=1.0,
            stroke_color=ORANGE, stroke_width=2.2,
        )
        db_top = Ellipse(
            width=DW, height=CAP_H,
            fill_color="#2E1E00", fill_opacity=1.0,
            stroke_color=ORANGE, stroke_width=2.2,
        )
        db_bot = Ellipse(
            width=DW, height=CAP_H,
            fill_color="#1C1000", fill_opacity=1.0,
            stroke_color=ORANGE, stroke_width=2.2,
        )
        db_top.move_to(db_rect.get_top())
        db_bot.move_to(db_rect.get_bottom())

        # Labels: font sizes chosen so max width < DW - 0.40 (0.20 pad each side)
        db_lbl = Text("Database", font=FONT, font_size=15,
                      color=ORANGE, weight=BOLD)
        db_sub = Text("Stored Templates", font=FONT, font_size=11, color=GRAY)

        # Force-clamp widths to (DW - 0.42) if overlong
        PAD_W = DW - 0.42
        if db_lbl.width > PAD_W:
            db_lbl.scale(PAD_W / db_lbl.width)
        if db_sub.width > PAD_W:
            db_sub.scale(PAD_W / db_sub.width)

        # Centre both inside rect, 0.18 unit apart vertically
        db_lbl.move_to(db_rect).shift(UP * 0.20)
        db_sub.move_to(db_rect).shift(DOWN * 0.18)

        db_cyl = VGroup(db_rect, db_bot, db_top, db_lbl, db_sub)
        db_cyl.next_to(n_mat, UP, buff=0.40)

        # modality-1 badge
        mod1 = Text("Modality 1  (e.g. Face)", font=FONT, font_size=13, color=CYAN)
        mod1.next_to(VGroup(n_raw, n_qual, n_feat), UP, buff=0.22)

        # ── Arrows ────────────────────────────────────────────────────────────
        # Row 1 (left → right)
        arr_rq = harrow(n_raw,  n_qual, CYAN)
        arr_qf = harrow(n_qual, n_feat, TEAL)
        arr_fm = harrow(n_feat, n_mat)

        # DB → Matcher (vertical down)
        arr_db = Arrow(db_cyl.get_bottom() + DOWN*0.04,
                       n_mat[0].get_top()  + UP*0.04,
                       color=ORANGE, stroke_width=2.0, buff=0,
                       max_tip_length_to_length_ratio=0.20)

        # Matcher → RawScore (vertical drop, same x-column XS[3])
        arr_drop = Arrow(n_mat[0].get_bottom() + DOWN*0.04,
                         n_scr[0].get_top()    + UP*0.04,
                         color=PURPLE, stroke_width=2.2, buff=0,
                         max_tip_length_to_length_ratio=0.20)

        # Modality-2 tag beside the drop arrow (right side, won't overlap)
        mod2 = Text("Modality 2\n(same process)", font=FONT, font_size=11, color=GRAY)
        mod2.next_to(arr_drop, RIGHT, buff=0.14)

        # Row 2 (right → left)
        arr_sn = harrow(n_scr,  n_norm, PURPLE)   # but harrow uses get_right/get_left
        # n_scr is at XS[3]=right, n_norm at XS[2] → n_scr.left to n_norm.right? No.
        # harrow goes a.right→b.left. Since n_scr(XS[3]) > n_norm(XS[2]),
        # n_scr.get_left() is the inner edge. We need arrows going LEFT.
        # Override: use Arrow directly.
        def larrow(a, b, col=GRAY):
            return Arrow(a.get_left(), b.get_right(),
                         color=col, stroke_width=2.2, buff=0.10,
                         max_tip_length_to_length_ratio=0.18)

        arr_sn = larrow(n_scr,  n_norm, PURPLE)
        arr_nf = larrow(n_norm, n_fuse, GOLD)
        arr_fd = larrow(n_fuse, n_dec,  GREEN)

        # ── Sequential reveal ─────────────────────────────────────────────────
        self.play(FadeIn(mod1), run_time=0.35)

        seq1 = [
            (n_raw,  arr_rq, CYAN,
             "Raw Input: face image, fingerprint scan, voice recording"),
            (n_qual, arr_qf, TEAL,
             "Quality Check: reject blurry images, wet fingers, extreme angles"),
            (n_feat, arr_fm, WHITE,
             "Feature Extraction: minutiae points, MFCC coefficients, eigenfaces"),
            (n_mat,  None,   WHITE,
             "Matcher needs TWO streams: fresh features AND stored template from DB"),
        ]

        for (n, arr, hi, cap) in seq1:
            caption(cap, hi)
            self.play(FadeIn(n, scale=0.88),
                      n[0].animate.set_stroke(color=hi, width=3.0),
                      run_time=0.60)
            if arr:
                self.play(Create(arr), run_time=0.45)
            self.play(n[0].animate.set_stroke(color=NSTROKE, width=2.0), run_time=0.22)
            self.wait(0.12)

        # Database
        caption("Database stores enrolled templates for comparison", ORANGE)
        self.play(FadeIn(db_cyl, shift=DOWN*0.22), run_time=0.65)
        self.play(Create(arr_db), run_time=0.50)
        self.wait(0.20)

        # Drop arrow + modality-2 tag
        caption("Raw Score: face=0.05 (distance), fingerprint=850 (similarity)", PURPLE)
        self.play(Create(arr_drop), FadeIn(mod2), run_time=0.55)

        seq2 = [
            (n_scr,  arr_sn, PURPLE,
             "Raw scores have DIFFERENT units — cannot be combined directly!"),
            (n_norm, arr_nf, PURPLE,
             "Normalization: map all scores to [0, 1] using Min-Max or Z-score"),
            (n_fuse, arr_fd, GOLD,
             "Score Fusion: combine into vector [s_face, s_finger] — SVM input"),
            (n_dec,  None,   GREEN,
             "Decision: SVM classifies vector → Accept (Genuine) or Reject"),
        ]

        for (n, arr, hi, cap) in seq2:
            caption(cap, hi)
            self.play(FadeIn(n, scale=0.88),
                      n[0].animate.set_stroke(color=hi, width=3.0),
                      run_time=0.60)
            if arr:
                self.play(Create(arr), run_time=0.45)
            self.play(n[0].animate.set_stroke(color=NSTROKE, width=2.0), run_time=0.22)
            self.wait(0.14)

        # full-pipeline pulse
        self.wait(0.4)
        all_nd = [n_raw, n_qual, n_feat, n_mat, n_scr, n_norm, n_fuse, n_dec]
        self.play(LaggedStart(*[
            n[0].animate(rate_func=there_and_back).set_stroke(color=GOLD, width=3.5)
            for n in all_nd], lag_ratio=0.14), run_time=2.0)
        self.wait(0.7)

        self.play(FadeOut(VGroup(
            head, mod1, cap_bg, cur_cap,
            n_raw, n_qual, n_feat, n_mat,
            db_cyl, arr_db,
            arr_rq, arr_qf, arr_fm,
            arr_drop, mod2,
            n_scr, n_norm, n_fuse, n_dec,
            arr_sn, arr_nf, arr_fd,
        )), run_time=0.90)
        self.wait(0.20)

    # ── 3. Normalization bar chart ────────────────────────────────────────────
    def _normalization(self):
        """
        Layout (v4 — no overlaps guaranteed):

        Vertical zones:
          y ∈ [ 1.6,  3.6] : problem / ok badges + titles
          y ∈ [-0.9,  1.6] : both axes + bars + x-labels   (CHART_Y = +0.55)
          y ∈ [-1.9, -0.9] : x-labels bottom padding
          y ∈ [-3.2, -1.9] : formula box  (completely separate zone)
          y ∈ [-3.6, -3.2] : caption / safe margin

        X-labels use TWO-line text so they fit without wrapping into formula.
        Formula box is placed at fixed y = -2.55  (between x-labels and CAP_Y).
        Charts are shifted UP so their baselines land at ~y = -0.85.
        """
        head = Text("Why Normalize Scores First?",
                    font=FONT, weight=BOLD, font_size=30, color=PURPLE)
        head.to_edge(UP, buff=0.28)
        self.play(FadeIn(head, shift=DOWN * 0.07), run_time=0.55)

        # ── geometry ─────────────────────────────────────────────────────────
        CHART_Y = 0.55      # chart centres shifted UP to leave formula room
        CHART_H = 2.60      # slightly shorter so baseline is at ~-0.75
        L_X     = -3.80
        R_X     =  3.80

        # ── LEFT: raw scores chart ────────────────────────────────────────────
        ax_l = Axes(
            x_range=[0, 4, 1], y_range=[0, 1000, 250],
            x_length=3.6, y_length=CHART_H,
            axis_config={"stroke_width": 1.5, "color": WHITE,
                         "stroke_opacity": 0.45, "include_numbers": False},
            tips=False,
        ).move_to([L_X, CHART_Y, 0])

        RAW = [
            (1, 0.05,  CYAN,   "Face\n(dist)"),
            (2, 850,   GREEN,  "Finger\n(sim)"),
            (3, 0.72,  PURPLE, "Voice\n(cos)"),
        ]
        raw_bars  = VGroup()
        raw_xlbls = VGroup()
        base_l = ax_l.c2p(0, 0)[1]
        span_l = ax_l.c2p(0, 1000)[1] - base_l

        for xi, val, col, xlbl in RAW:
            h = max(span_l * val / 1000, 0.06)
            bar = Rectangle(width=0.50, height=h,
                            fill_color=col, fill_opacity=0.85, stroke_width=0)
            bx = ax_l.c2p(xi, 0)[0]
            bar.move_to([bx, base_l + h / 2, 0])
            vt = Text(str(val), font=FONT, font_size=12, color=col)
            vt.next_to(bar, UP, buff=0.05)
            raw_bars.add(VGroup(bar, vt))
            # x-label: fixed 0.52 below baseline
            xl = Text(xlbl, font=FONT, font_size=11, color=col)
            xl.move_to([bx, base_l - 0.52, 0])
            raw_xlbls.add(xl)

        left_title = Text("Raw scores — incompatible units",
                          font=FONT, font_size=14, color=GRAY)
        left_title.next_to(ax_l, UP, buff=0.16)

        prob = Text("[!]  Finger 850  >>  Face 0.05 — cannot merge!",
                    font=FONT, font_size=13, color=RED)
        prob.next_to(left_title, UP, buff=0.14)

        # ── RIGHT: normalized chart ───────────────────────────────────────────
        ax_r = Axes(
            x_range=[0, 4, 1], y_range=[0, 1.1, 0.25],
            x_length=3.6, y_length=CHART_H,
            axis_config={"stroke_width": 1.5, "color": WHITE,
                         "stroke_opacity": 0.45, "include_numbers": False},
            tips=False,
        ).move_to([R_X, CHART_Y, 0])

        NORM = [
            (1, 0.80, CYAN,   "Face"),
            (2, 0.70, GREEN,  "Finger"),
            (3, 0.62, PURPLE, "Voice"),
        ]
        norm_bars  = VGroup()
        norm_xlbls = VGroup()
        base_r = ax_r.c2p(0, 0)[1]

        for xi, val, col, xlbl in NORM:
            h = ax_r.c2p(0, val)[1] - base_r
            bar = Rectangle(width=0.50, height=h,
                            fill_color=col, fill_opacity=0.85, stroke_width=0)
            bx = ax_r.c2p(xi, 0)[0]
            bar.move_to([bx, base_r + h / 2, 0])
            vt = Text(f"{val:.2f}", font=FONT, font_size=12, color=col)
            vt.next_to(bar, UP, buff=0.05)
            norm_bars.add(VGroup(bar, vt))
            xl = Text(xlbl, font=FONT, font_size=11, color=col)
            xl.move_to([bx, base_r - 0.42, 0])
            norm_xlbls.add(xl)

        right_title = Text("After Min-Max Normalization  [0, 1]",
                           font=FONT, font_size=14, color=PURPLE)
        right_title.next_to(ax_r, UP, buff=0.16)
        ok_badge = Text("[OK]  Same scale — fair comparison!",
                        font=FONT, font_size=13, color=GREEN)
        ok_badge.next_to(right_title, UP, buff=0.14)

        # centre transform arrow
        t_arr = Arrow(
            ax_l.get_right() + RIGHT * 0.12,
            ax_r.get_left()  + LEFT  * 0.12,
            color=PURPLE, stroke_width=2.6, buff=0.05,
            max_tip_length_to_length_ratio=0.22,
        )
        t_lbl = Text("Normalize", font=FONT, font_size=14, color=PURPLE)
        t_lbl.next_to(t_arr, UP, buff=0.10)

        # ── Formula box — pinned in its own zone below both charts ────────────
        # Lowest x-label bottom ≈ base_l - 0.52 - label_height (~0.28 for 2-line)
        # ≈  base_l - 0.80.  Formula top must be BELOW that.
        # We fix formula centre at y = -2.55, size h=1.10 → top=-2.00, bot=-3.10
        # That's well below x-labels and well above CAP_Y(-3.45).
        FORM_CY = -2.55
        f_bg = RoundedRectangle(
            width=5.8, height=1.10, corner_radius=0.16,
            fill_color="#0D0F1E", fill_opacity=0.96,
            stroke_color=PURPLE, stroke_width=1.8,
        )
        f_bg.move_to([0, FORM_CY, 0])
        formula = MathTex(
            r"s'_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}",
            color=WHITE, font_size=36,
        ).move_to(f_bg)

        # ── Animation ─────────────────────────────────────────────────────────
        self.play(FadeIn(ax_l), FadeIn(left_title), run_time=0.60)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.22) for b in raw_bars],
                        lag_ratio=0.20),
            FadeIn(raw_xlbls),
            run_time=1.0,
        )
        self.play(FadeIn(prob, shift=DOWN * 0.08), run_time=0.45)
        self.wait(0.45)

        self.play(GrowArrow(t_arr), FadeIn(t_lbl), run_time=0.60)
        self.play(FadeIn(ax_r), FadeIn(right_title), run_time=0.55)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.22) for b in norm_bars],
                        lag_ratio=0.20),
            FadeIn(norm_xlbls),
            run_time=1.0,
        )
        self.play(FadeIn(ok_badge, shift=DOWN * 0.08), run_time=0.45)
        self.wait(0.50)

        # Formula animates in from below
        self.play(FadeIn(f_bg, shift=UP * 0.15), Write(formula), run_time=1.10)
        self.wait(1.80)

        self.play(FadeOut(VGroup(
            head, ax_l, left_title, raw_bars, raw_xlbls, prob,
            ax_r, right_title, norm_bars, norm_xlbls, ok_badge,
            t_arr, t_lbl, f_bg, formula,
        )), run_time=0.90)
        self.wait(0.20)

    # ── 4. Bridge ─────────────────────────────────────────────────────────────
    def _bridge(self):
        """
        v4 layout — strict column separation, zero overlap guaranteed.

        SCREEN COLUMNS:
          LEFT  col  x ∈ [-7.1, -0.6]:
            Row A (y=+2.2)  : s1 label  →  arrives left edge of node
            Row B (y=+0.6)  : fusion node (centred x=-1.8)
            Row C (y=-1.0)  : output vector  x = [0.80, 0.70]
            Row D (y=-2.2)  : "One point in 2-D feature space"
            Each row is horizontally centred on x=-3.5 for labels,
            node at x=-1.8. Arrows are clean horizontal lines.

          RIGHT col  x ∈ [+0.6, +7.1]:
            Scatter axes centred at x=+3.8, y=+0.4
            y-label "Fingerprint" placed LEFT of y_axis with buff=0.35
            x-label "Face score" placed DR of x_axis end
            Legend: directly below axes
            NO caption bar inside the scatter area

          BOTTOM strip  y ∈ [-3.0, -3.6]:
            Single caption bar — nothing else ever enters this zone.
        """
        head = Text("Fusion Vector  →  SVM Input",
                    font=FONT, weight=BOLD, font_size=32, color=GOLD)
        head.to_edge(UP, buff=0.26)
        self.play(FadeIn(head, shift=DOWN * 0.07), run_time=0.50)

        # ════════════════════  LEFT COLUMN  ══════════════════════════════════
        # Labels shift LEFT by 0.8 unit: LBL_CX -4.40 → -5.20
        # Node/vec/note shift RIGHT by 0.6 unit: NODE_CX -1.80 → -1.20
        LBL_CX  = -5.20
        NODE_CX = -1.20
        COL_SEP = -0.60

        S1_Y    =  2.10
        S2_Y    =  0.80
        NODE_Y  =  1.45
        VEC_Y   = -0.55
        NOTE_Y  = -1.60

        # Score labels — shorter text so they fit
        s1 = Text("Face score (norm):   s₁ = 0.80",
                  font=FONT, font_size=17, color=CYAN)
        s2 = Text("Fingerprint (norm):  s₂ = 0.70",
                  font=FONT, font_size=17, color=GREEN)

        # Left-anchor so text never bleeds off screen
        s1.move_to([LBL_CX, S1_Y, 0])
        s2.move_to([LBL_CX, S2_Y, 0])

        # Clamp if too wide
        MAX_LBL_W = COL_SEP - (-7.1) - 0.3   # ~6.2 units
        for lbl in [s1, s2]:
            if lbl.width > MAX_LBL_W:
                lbl.scale(MAX_LBL_W / lbl.width)

        # Fusion node
        f_node = nd("Fusion\nNode", w=2.30, h=0.96,
                    sc=GOLD, fc="#1A1608", ls=18)
        f_node.move_to([NODE_CX, NODE_Y, 0])

        # Node left edge x
        node_lx = NODE_CX - 1.15

        # Arrow s1 → node upper-left  (goes right and slightly down)
        a1 = Arrow(
            s1.get_right() + RIGHT * 0.08,
            np.array([node_lx, NODE_Y + 0.26, 0]),
            color=CYAN, stroke_width=2.2, buff=0.05,
            max_tip_length_to_length_ratio=0.18,
        )
        # Arrow s2 → node lower-left
        a2 = Arrow(
            s2.get_right() + RIGHT * 0.08,
            np.array([node_lx, NODE_Y - 0.26, 0]),
            color=GREEN, stroke_width=2.2, buff=0.05,
            max_tip_length_to_length_ratio=0.18,
        )

        # Output vector below node — straight thick arrow, gold
        vec = MathTex(
            r"\mathbf{x} = \begin{bmatrix} 0.80 \\ 0.70 \end{bmatrix}",
            color=WHITE, font_size=40,
        )
        vec.move_to([NODE_CX, VEC_Y, 0])

        a_out = Arrow(
            f_node[0].get_bottom() + DOWN * 0.06,
            vec.get_top()          + UP   * 0.06,
            color=GOLD, stroke_width=3.2, buff=0.04,
            max_tip_length_to_length_ratio=0.16,
        )

        dim_note = Text("One point in 2-D feature space",
                        font=FONT, font_size=13, color=GRAY)
        dim_note.move_to([NODE_CX, NOTE_Y, 0])

        # ── Animate left column sequentially ──────────────────────────────────
        self.play(
            LaggedStart(
                FadeIn(s1, shift=RIGHT * 0.20),
                FadeIn(s2, shift=RIGHT * 0.20),
                lag_ratio=0.45,
            ),
            run_time=0.80,
        )
        self.play(
            FadeIn(f_node, scale=0.88),
            GrowArrow(a1),
            GrowArrow(a2),
            run_time=0.80,
        )
        self.play(GrowArrow(a_out), run_time=0.45)
        self.play(Write(vec), run_time=0.70)
        self.play(FadeIn(dim_note, shift=UP * 0.08), run_time=0.40)
        self.wait(0.50)

        # ════════════════════  RIGHT COLUMN  ═════════════════════════════════
        AX_CX = 3.80
        AX_CY = 0.40

        mini_ax = Axes(
            x_range=[0, 1, 0.25], y_range=[0, 1, 0.25],
            x_length=3.40, y_length=2.80,
            axis_config={
                "stroke_width": 1.6,
                "color": WHITE,
                "stroke_opacity": 0.55,
                "include_ticks": True,
                "tick_size": 0.055,
            },
            tips=True,
        ).move_to([AX_CX, AX_CY, 0])

        # Guarantee axes bottom stays above SAFE_BOT (legend goes below)
        ax_bot = mini_ax.get_bottom()[1]
        if ax_bot < SAFE_BOT + 0.60:   # leave 0.6 for legend
            mini_ax.shift(UP * (SAFE_BOT + 0.60 - ax_bot))

        # Axis labels
        xa = Text("Face score", font=FONT, font_size=12, color=GRAY)
        xa.next_to(mini_ax.x_axis.get_end(), DR, buff=0.08)

        # y-label: rotate then place well LEFT of the entire y-axis
        ya = Text("Fingerprint", font=FONT, font_size=12, color=GRAY)
        ya.rotate(PI / 2)
        ya.next_to(mini_ax.y_axis, LEFT, buff=0.36)

        # Subtle grid
        grid = VGroup(*[
            obj
            for gv in [0.25, 0.50, 0.75]
            for obj in [
                DashedLine(mini_ax.c2p(0, gv), mini_ax.c2p(1, gv),
                           color=WHITE, stroke_width=0.65,
                           stroke_opacity=0.13, dash_length=0.09),
                DashedLine(mini_ax.c2p(gv, 0), mini_ax.c2p(gv, 1),
                           color=WHITE, stroke_width=0.65,
                           stroke_opacity=0.13, dash_length=0.09),
            ]
        ])

        self.play(
            FadeIn(mini_ax), FadeIn(xa), FadeIn(ya), FadeIn(grid),
            run_time=0.65,
        )

        # Clusters
        GEN = [(0.76, 0.66), (0.84, 0.76), (0.71, 0.70),
               (0.88, 0.62), (0.79, 0.74)]
        IMP = [(0.20, 0.18), (0.29, 0.26), (0.16, 0.31),
               (0.25, 0.13), (0.33, 0.22)]

        gen_d = VGroup(*[Dot(mini_ax.c2p(*p), color=GREEN, radius=0.09)
                         for p in GEN])
        imp_d = VGroup(*[Dot(mini_ax.c2p(*p), color=RED,   radius=0.09)
                         for p in IMP])

        self.play(
            LaggedStart(*[FadeIn(d, scale=0.35) for d in gen_d], lag_ratio=0.08),
            LaggedStart(*[FadeIn(d, scale=0.35) for d in imp_d], lag_ratio=0.08),
            run_time=0.90,
        )

        # ── Example point  x  ─────────────────────────────────────────────────
        # Design rationale: x is a GENUINE sample being queried.
        # We show it as a larger GREEN dot (same class colour) with a GOLD ring
        # and a "x (query)" label — making it clear it belongs to the green cluster,
        # not a separate mysterious cyan blob.
        XP = mini_ax.c2p(0.80, 0.70)

        # Outer gold ring to make it stand out from the plain green dots
        ring = Circle(radius=0.18, stroke_width=2.4, stroke_color=GOLD)
        ring.set_fill(opacity=0)
        ring.move_to(XP)

        # Larger green dot (same colour = same class)
        v_dot = Dot(XP, color=GREEN, radius=0.14)

        # Label with context
        v_lbl = Text("x  (query)", font=FONT, font_size=13, color=GOLD, weight=BOLD)
        v_lbl.next_to(v_dot, UR, buff=0.12)

        # Pulse ring (gold, expands once then disappears)
        pulse = Circle(radius=0.18, stroke_width=2.2, stroke_color=GOLD)
        pulse.set_fill(opacity=0).set_stroke(opacity=0.80)
        pulse.move_to(XP)

        self.play(
            FadeIn(ring,  scale=0.4),
            FadeIn(v_dot, scale=0.4),
            FadeIn(v_lbl),
            run_time=0.55,
        )
        self.add(pulse)
        self.play(
            pulse.animate.scale(2.60).set_stroke(opacity=0),
            run_time=0.65, rate_func=linear,
        )
        self.remove(pulse)
        self.wait(0.28)

        # Legend — directly below axes, clamped above SAFE_BOT
        leg_g = VGroup(Dot(color=GREEN, radius=0.09),
                       Text("Genuine",  font=FONT, font_size=13, color=GREEN)
                       ).arrange(RIGHT, buff=0.10)
        leg_i = VGroup(Dot(color=RED,   radius=0.09),
                       Text("Impostor", font=FONT, font_size=13, color=RED)
                       ).arrange(RIGHT, buff=0.10)
        legend = VGroup(leg_g, leg_i).arrange(RIGHT, buff=0.30)
        legend.next_to(mini_ax, DOWN, buff=0.18)
        if legend.get_bottom()[1] < SAFE_BOT:
            legend.shift(UP * (SAFE_BOT - legend.get_bottom()[1]))

        self.play(FadeIn(legend), run_time=0.45)
        self.wait(0.50)

        # ════════════════════  BOTTOM CAPTION  ═══════════════════════════════
        br_bg = RoundedRectangle(
            width=13.2, height=0.65, corner_radius=0.13,
            fill_color="#0D0F1E", fill_opacity=0.97,
            stroke_color=GOLD, stroke_width=1.6,
        )
        br_bg.move_to([0, CAP_Y, 0])
        br_txt = Text(
            "Next  →  SVM finds the optimal decision boundary in this space",
            font=FONT, font_size=17, color=GOLD,
        )
        br_txt.move_to(br_bg)
        if br_txt.width > 12.8:
            br_txt.scale(12.8 / br_txt.width)

        self.play(FadeIn(br_bg, shift=UP * 0.08), Write(br_txt), run_time=0.95)
        self.wait(2.20)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.10)
        self.wait(0.35)