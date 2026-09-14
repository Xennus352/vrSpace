"""Holographic 3D hand rendering — Apple Vision Pro style.

Renders the user's hand from MediaPipe landmarks as a translucent,
bluish-white hologram floating in 3D space in front of the Earth.
"""

import math
from OpenGL.GL import *

# MediaPipe hand topology (bone connections between landmark indexes)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm bridge
]

# Per-joint bone thickness (1.0 at MCP/wrist, tapering to fingertips)
JOINT_WIDTH = [
    0.055, 0.05, 0.045, 0.04, 0.032,    # thumb 0-4
    0.055, 0.048, 0.042, 0.034,         # index 5-8
    0.058, 0.05, 0.043, 0.035,          # middle 9-12
    0.054, 0.046, 0.04, 0.032,          # ring 13-16
    0.05, 0.042, 0.036, 0.03,           # pinky 17-20
]

HOLO_COLOR = (0.45, 0.78, 1.0)


class HoloHandRenderer:
    def __init__(self, num_hands=2):
        # {hand_label: list of 21 (x, y, z)} normalized coords
        self._hands = {}
        self._smoothed = {}
        self._presence = {}       # per-hand fade alpha
        self.show_guide = False

    # ---- state handling ----
    def set_hands(self, hands):
        """hands: dict mapping 'Left'/'Right' -> list of 21 (x, y, z)."""
        self._hands = {k: v for k, v in hands.items() if k in ("Left", "Right")}

    def clear(self):
        self._hands = {}

    def update(self, dt):
        k = min(1.0, dt * 8.0)

        # Fade per-hand presence in/out
        for label in list(self._presence):
            if label not in self._hands:
                self._presence[label] -= k
                if self._presence[label] <= 0.0:
                    self._presence.pop(label, None)
                    self._smoothed.pop(label, None)
        for label in self._hands:
            self._presence[label] = min(1.0, self._presence.get(label, 0.0) + k)

        # Smooth landmark positions over time to remove jitter
        for label, lm in self._hands.items():
            prev = self._smoothed.get(label)
            if prev is None or len(prev) != 21:
                self._smoothed[label] = [list(p) for p in lm]
                continue
            s = 0.45
            self._smoothed[label] = [
                [p[0] + (q[0] - p[0]) * s,
                 p[1] + (q[1] - p[1]) * s,
                 p[2] + (q[2] - p[2]) * s]
                for p, q in zip(prev, lm)
            ]

    @property
    def alpha(self):
        if not self._presence:
            return 0.0
        return sum(self._presence.values()) / len(self._presence)

    # ---- coordinate mapping: normalized cam-space -> world-space ----
    def _map(self, x, y, z, aspect, scale=3.4):
        wx = (x - 0.5) * 2.0 * scale * aspect
        wy = (0.5 - y) * 2.0 * scale
        wz = -2.2 + z * 0.9   # float just in front of the Earth
        return wx, wy, wz

    # ---- rendering ----
    def render(self, aspect):
        if self.alpha <= 0.01 or not self._smoothed:
            return
        hands = self._smoothed

        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)

        # Map all landmarks to world space
        world = {}
        for label, lm in hands.items():
            mapped = [
                self._map(px, py, pz, aspect)
                for (px, py, pz) in lm
            ]
            world[label] = mapped

        # Base translucent body
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for label, pts in world.items():
            self._draw_body(pts, alpha=self.alpha * 0.16)

        # Additive holographic glow: slightly larger, additive
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for label, pts in world.items():
            self._draw_body(pts, alpha=self.alpha * 0.22, glow=True)

        # Edge lines (bright, thin) — hologram outline
        glLineWidth(2.0)
        for label, pts in world.items():
            self._draw_wire(pts, alpha=self.alpha * 0.85)

        # Bright joint dots
        glPointSize(6.0)
        for label, pts in world.items():
            self._draw_joints(pts, alpha=self.alpha * 0.9)

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    # ---- helpers ----
    def _perpendicular(self, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return 0.0, 0.0
        return -dy / length, dx / length

    def _draw_body(self, pts, alpha=0.3, glow=False):
        """Draw the solid holographic hand using billboarded quads."""
        scale = 1.35 if glow else 1.0
        base_r, g, b = HOLO_COLOR
        col = (base_r, g, b, alpha)

        for (a, bj) in HAND_CONNECTIONS:
            (ax, ay, az), (bx, by, bz) = pts[a], pts[bj]
            npx, npy = self._perpendicular(ax, ay, bx, by)
            w1 = JOINT_WIDTH[a] * scale * 0.5
            w2 = JOINT_WIDTH[bj] * scale * 0.5

            glColor4f(*col)
            glBegin(GL_QUADS)
            glVertex3f(ax + npx * w1, ay + npy * w1, az)
            glVertex3f(ax - npx * w1, ay - npy * w1, az)
            glVertex3f(bx - npx * w2, by - npy * w2, bz)
            glVertex3f(bx + npx * w2, by + npy * w2, bz)
            glEnd()

        # Palm fill (wrist + MCP spread) so the hand reads as solid
        palm = [pts[0], pts[5], pts[9], pts[13], pts[17]]
        glColor4f(*col)
        glBegin(GL_TRIANGLE_FAN)
        cx = sum(p[0] for p in palm) / len(palm)
        cy = sum(p[1] for p in palm) / len(palm)
        cz = sum(p[2] for p in palm) / len(palm)
        glVertex3f(cx, cy, cz)
        for p in palm:
            glVertex3f(p[0], p[1], p[2])
        glVertex3f(*palm[0])
        glEnd()

    def _draw_wire(self, pts, alpha=0.85):
        glColor4f(HOLO_COLOR[0], HOLO_COLOR[1], HOLO_COLOR[2], alpha)
        glBegin(GL_LINES)
        for (a, b) in HAND_CONNECTIONS:
            glVertex3f(*pts[a])
            glVertex3f(*pts[b])
        glEnd()

    def _draw_joints(self, pts, alpha=0.9):
        glColor4f(0.75, 0.9, 1.0, alpha)
        glBegin(GL_POINTS)
        for p in pts:
            glVertex3f(*p)
        glEnd()