"""Floating glass panels + shared view-space transform (Vision Pro style)."""

import math
import time
import numpy as np
import pygame
from OpenGL.GL import *

# ---------------------------------------------------------------------------
#  Shared transform: normalized camera coords  ->  view/panel world space.
#  One instance is shared by the glove and the panels so hand-grabbing stays
#  consistent when the user resizes the glove.
# ---------------------------------------------------------------------------
class ViewSpace:
    def __init__(self, scale=1.35, depth=-2.0):
        self.scale = scale
        self.depth = depth

    def to_world(self, nx, ny, nz=0.0, aspect=16.0 / 9.0):
        s = self.scale
        x = (nx - 0.5) * 2.0 * s * aspect
        y = (0.5 - ny) * 2.0 * s
        z = self.depth + nz * 0.9
        return x, y, z

    # ------------------------------------------------------------------
    #  Hand registration: the glove lands EXACTLY where the hand appears
    #  in the passthrough video. Each landmark is projected to the depth
    #  plane (ViewSpace.to_world) so the overlay tracks the real hand's
    #  position; the per-hand ``unit`` (world units per landmark unit)
    #  is derived from the wrist->middle-MCP span so the glove's mesh
    #  thickness matches each hand's actual on-screen size.
    # ------------------------------------------------------------------
    def hand_pose(self, lm, aspect=16.0 / 9.0):
        if lm is None or len(lm) < 21:
            return [self.to_world(px, py, pz, aspect) for (px, py, pz) in lm], 0.0

        p0, p9 = lm[0], lm[9]
        pts = [self.to_world(px, py, pz, aspect) for (px, py, pz) in lm]

        # landmark span in the camera frame (normalized units)
        L = math.hypot(p9[0] - p0[0], p9[1] - p0[1])
        if L < 1e-4:
            return pts, 0.0
        # ... and the same span carried into the world at this depth
        w0, w9 = pts[0], pts[9]
        span = math.hypot(w9[0] - w0[0], w9[1] - w0[1])
        if span < 1e-6:
            span = L * 2.0 * self.scale
        unit = span / L
        return pts, unit


# ---------------------------------------------------------------------------
#  Floating glass panel
# ---------------------------------------------------------------------------
class GlassPanel:
    def __init__(self, app, view, nx, ny, nz, w_norm, h_norm, title,
                 accent=(0.45, 0.95, 1.0)):
        self.view = view
        self.title = title
        self.accent = accent
        self.fill = (0.07, 0.10, 0.18, 0.62)

        cx, cy, cz = view.to_world(nx, ny, nz)
        self.center = [cx, cy, cz]           # movable world position
        self.w_norm = w_norm
        self.h_norm = h_norm

        self.highlight = 0.0                  # 0..1 focus ring
        self.grabbed = False
        self.content_lines = []
        self._tex = None
        self._dirty = True
        self.app = app

    # -- content ------------------------------------------------
    def set_content(self, lines):
        self.content_lines = list(lines)
        self._dirty = True

    def _ensure_texture(self):
        if not self._dirty:
            return
        self._dirty = False

        tw, th = 480, 300
        surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        # glass rounded-rect body
        rect = pygame.Rect(0, 0, tw, th)
        pygame.draw.rect(surf, self.fill, rect, border_radius=26)

        # accent top edge
        pygame.draw.line(surf, (*[int(c * 255) for c in self.accent], 70),
                         (40, 2), (tw - 40, 2), 2)

        # content
        pygame.font.init()
        title_font = pygame.font.Font(None, 34)
        row_font = pygame.font.Font(None, 26)

        y = 34
        t = title_font.render(self.title.upper(), True, (210, 235, 255))
        surf.blit(t, (28, 14))
        y += 4
        pygame.draw.line(surf, (255, 255, 255, 34), (28, y - 8), (tw - 28, y - 8), 1)

        for line in self.content_lines:
            pygame.draw.line(surf, (255, 255, 255, 22), (28, y - 2), (tw - 28, y - 2), 1)
            if isinstance(line, str):
                txt, col = line, (190, 205, 225, 255)
            else:
                txt, col = line
            img = row_font.render(str(txt), True, col)
            surf.blit(img, (28, y))
            y += 30

        data = pygame.image.tostring(surf, "RGBA")
        # flip vertically for OpenGL texture coords
        arr = np.frombuffer(data, dtype=np.uint8).reshape((th, tw, 4))
        arr = np.ascontiguousarray(arr[::-1])
        if self._tex is None:
            self._tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr)
        self._tex_size = (tw, th)

    # -- rendering ----------------------------------------------
    def draw(self, aspect, alpha=1.0):
        if self.app is None:
            return
        self._ensure_texture()
        cx, cy, cz = self.center
        hw = self.w_norm * self.view.scale * aspect
        hh = self.h_norm * self.view.scale

        glPushMatrix()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        # focus glow (additive) behind the panel
        if self.highlight > 0.01 or self.grabbed:
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glow = 0.05 + 0.06 * self.highlight + (0.05 if self.grabbed else 0.0)
            glColor4f(self.accent[0], self.accent[1], self.accent[2], glow * alpha)
            gx = hw * (1.03 + 0.05 * self.highlight)
            gy = hh * (1.03 + 0.05 * self.highlight)
            glBegin(GL_QUADS)
            glVertex3f(cx - gx, cy - gy, cz - 0.02)
            glVertex3f(cx + gx, cy - gy, cz - 0.02)
            glVertex3f(cx + gx, cy + gy, cz - 0.02)
            glVertex3f(cx - gx, cy + gy, cz - 0.02)
            glEnd()
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # glass body (soft radial falloff simulated with a plain fill)
        glColor4f(0.0, 0.0, 0.0, 0.28 * alpha)
        glBegin(GL_QUADS)
        glVertex3f(cx - hw, cy - hh, cz)
        glVertex3f(cx + hw, cy - hh, cz)
        glVertex3f(cx + hw, cy + hh, cz)
        glVertex3f(cx - hw, cy + hh, cz)
        glEnd()

        # content texture
        if self._tex is not None:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self._tex)
            glColor4f(1.0, 1.0, 1.0, 0.96 * alpha)
            m = 12  # inner margin
            lx = cx - hw + m
            rx = cx + hw - m
            ty = cy - hh + m
            by = cy + hh - m
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex3f(lx, ty, cz + 0.005)
            glTexCoord2f(1, 0); glVertex3f(rx, ty, cz + 0.005)
            glTexCoord2f(1, 1); glVertex3f(rx, by, cz + 0.005)
            glTexCoord2f(0, 1); glVertex3f(lx, by, cz + 0.005)
            glEnd()
            glDisable(GL_TEXTURE_2D)

        # accent border
        border = 0.45 + 0.5 * self.highlight
        if self.grabbed:
            border = 1.0
        glColor4f(self.accent[0], self.accent[1], self.accent[2], (0.5 + 0.35 * self.highlight) * alpha)
        glLineWidth(2.0 + 1.0 * self.highlight)
        glBegin(GL_LINE_LOOP)
        glVertex3f(cx - hw, cy - hh, cz + 0.006)
        glVertex3f(cx + hw, cy - hh, cz + 0.006)
        glVertex3f(cx + hw, cy + hh, cz + 0.006)
        glVertex3f(cx - hw, cy + hh, cz + 0.006)
        glEnd()

        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def contains(self, wx, wy, wz, aspect):
        hw = self.w_norm * self.view.scale * aspect
        hh = self.h_norm * self.view.scale
        cx, cy, cz = self.center
        if abs(wx - cx) > hw * 0.9:
            return False
        if abs(wy - cy) > hh * 0.9:
            return False
        return abs(wz - cz) < 1.0

    def distance_to(self, wx, wy, aspect):
        cx, cy, _ = self.center
        return ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5