"""Shared vector icon factories for biometric visualization scenes.

All icons are built from primitive Manim shapes (Ellipse, Dot, Line, Arc, etc.)
so they render without external image dependencies.
"""

from manim import *
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from constants import GENUINE_COLOR, IMPOSTOR_COLOR
except ImportError:
    GENUINE_COLOR  = "#2ECC71"
    IMPOSTOR_COLOR = "#E74C3C"

_ICON_BG = "#1A1D27"


def make_genuine_icon(size: float = 0.6) -> VGroup:
    """Creates a vector face icon representing a genuine user."""
    head = Ellipse(
        width=size * 1.2, height=size * 1.5,
        stroke_color=GENUINE_COLOR, stroke_width=2,
        fill_color=_ICON_BG, fill_opacity=1,
    )
    eye_l = Dot(LEFT * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=GENUINE_COLOR)
    eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=GENUINE_COLOR)
    mouth = Arc(
        radius=size * 0.30, start_angle=-10 * DEGREES, angle=-160 * DEGREES,
        color=GENUINE_COLOR, stroke_width=2,
    ).shift(DOWN * size * 0.20)
    return VGroup(head, eye_l, eye_r, mouth)


def make_impostor_icon(size: float = 0.6) -> VGroup:
    """Creates a vector face icon representing an impostor/attacker."""
    head = Ellipse(
        width=size * 1.2, height=size * 1.5,
        stroke_color=IMPOSTOR_COLOR, stroke_width=2,
        fill_color=_ICON_BG, fill_opacity=1,
    )
    eye_l = Dot(LEFT * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=IMPOSTOR_COLOR)
    eye_r = Dot(RIGHT * size * 0.25 + UP * size * 0.20,
                radius=size * 0.06, color=IMPOSTOR_COLOR)
    mouth = Line(
        LEFT * size * 0.20, RIGHT * size * 0.30,
        color=IMPOSTOR_COLOR, stroke_width=2,
    ).shift(DOWN * size * 0.20).rotate(15 * DEGREES)
    return VGroup(head, eye_l, eye_r, mouth)


def make_noisy_icon(size: float = 0.6) -> VGroup:
    """Creates a genuine face icon overlayed with horizontal scan-line noise."""
    face = make_genuine_icon(size)
    face.set_color("#888888")
    lines = VGroup(*[
        Line(
            LEFT * size * 0.75, RIGHT * size * 0.75,
            color=WHITE, stroke_width=1.2, stroke_opacity=0.65,
        ).shift(UP * size * dy)
        for dy in [-0.20, 0.05, 0.32]
    ])
    return VGroup(face, lines)


def make_spoof_icon(size: float = 0.6) -> VGroup:
    """Creates a spoof attack icon (face photo displayed on a phone screen)."""
    phone = RoundedRectangle(
        width=size * 1.6, height=size * 2.4, corner_radius=0.10,
        stroke_color=WHITE, stroke_width=2,
        fill_color="#000000", fill_opacity=1,
    )
    screen = Rectangle(
        width=size * 1.35, height=size * 1.90,
        stroke_color="#333333", stroke_width=1,
        fill_color="#111122", fill_opacity=1,
    )
    face_on_screen = make_genuine_icon(size * 0.65)
    return VGroup(phone, screen, face_on_screen)


def make_fingerprint_icon(size: float = 0.6) -> VGroup:
    """Creates a stylised fingerprint icon using concentric arcs."""
    arcs = VGroup()
    radii = [size * r for r in [0.15, 0.25, 0.35, 0.45]]
    for i, r in enumerate(radii):
        # Alternate start angles for a realistic fingerprint whorl
        start = 30 * DEGREES if i % 2 == 0 else -30 * DEGREES
        arc = Arc(
            radius=r, start_angle=start, angle=PI + 40 * DEGREES,
            color="#AABBFF", stroke_width=1.8, stroke_opacity=0.8 - i * 0.1,
        )
        arcs.add(arc)
    # Central dot
    core = Dot(ORIGIN, radius=size * 0.04, color="#AABBFF")
    arcs.add(core)
    return arcs
