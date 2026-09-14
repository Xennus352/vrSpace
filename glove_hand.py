"""Realistic 3D glove hand, rigidly registered to MediaPipe landmarks.

The overlay is anchored to the wrist (landmark 0), scaled by the
wrist->middle-MCP distance (0 -> 9), and rotated onto the palm plane so it
tracks the real hand exactly (no more floating at screen center).

Rendering:
  * depth-testing ON with depth writes so overlapping finger segments layer
    correctly from the camera's perspective,
  * lit material presets (graphite / white / skin / hologram) rather than a
    flat gray mesh,
  * an opacity knob and an optional texture image for the glove body,
  * a soft additive sheen pass + faint silhouette edges for definition.
"""

import ctypes
import math
from OpenGL.GL import *

RING_SEG = 6  # circle resolution for finger tubes

# EMA smoothing for the landmark coordinates (per frame). Alpha ~0.2-0.3
# is the sweet spot between removing jitter/drift and staying responsive.
SMOOTHING_ALPHA = 0.25

# Realistic materials (ambient/diffuse/specular/shininess)
MATERIALS = {
    "graphite": dict(ambient=[0.15, 0.15, 0.18, 1.0],
                     diffuse=[0.44, 0.44, 0.49, 1.0],
                     specular=[0.55, 0.58, 0.62, 1.0], shininess=46.0),
    "white":    dict(ambient=[0.24, 0.24, 0.26, 1.0],
                     diffuse=[0.86, 0.88, 0.92, 1.0],
                     specular=[0.9, 0.9, 0.95, 1.0], shininess=80.0),
    "skin":     dict(ambient=[0.15, 0.11, 0.08, 1.0],
                     diffuse=[0.70, 0.53, 0.40, 1.0],
                     specular=[0.35, 0.28, 0.22, 1.0], shininess=26.0),
    "hologram": dict(ambient=[0.14, 0.15, 0.17, 1.0],
                     diffuse=[0.88, 0.90, 0.94, 1.0],
                     specular=[1.0, 1.0, 1.0, 1.0], shininess=90.0),
}

# Full hand wire topology (bones between landmark indexes)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Finger joint chains (landmark indexes, MCP -> tip)
FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Joint radii (fractions of the wrist->middle-MCP distance, palm-sized)
RADII = {
    0: 0.055, 1: 0.036, 2: 0.030, 3: 0.024, 4: 0.017,
    5: 0.040, 6: 0.034, 7: 0.027, 8: 0.019,
    9: 0.041, 10: 0.034, 11: 0.027, 12: 0.019,
    13: 0.038, 14: 0.031, 15: 0.024, 16: 0.017,
    17: 0.034, 18: 0.027, 19: 0.021, 20: 0.014,
}

# Spread around palm outline for the skin
PALM_OUTLINE = [0, 17, 13, 9, 5, 2, 1, 0]

GLOVE_ACCENT = (0.92, 0.94, 1.0)


class GloveHand:
    def __init__(self, view, num_hands=2):
        self._hands = {}
        self._smoothed = {}
        self._presence = {}
        self._span = {}               # label -> world-per-unit for mesh radii
        self.view = view              # shared ViewSpace (scale/depth)
        self.show_physics = False

        # look & feel knobs — milky white, translucent by default
        self.smooth_alpha = SMOOTHING_ALPHA
        self.material = "white"
        self.opacity = 0.5            # 0..1 body opacity (translucent white)
        self._tex = None
        self._tex_on = True
        self._tex_repeat = 2.0

        self._unit_sphere = self._build_sphere(6, 4)
        self._cache_sig = {}
        self._cache_arr = {}
        self._cache_n = {}
        self._reb_at = {}

    # ------------------------------------------------------------------
    #  Unit sphere geometry (slices x stacks), reused for all joints.
    #  Vertices carry uv so a custom glove texture can wrap the mesh.
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sphere(slices=10, stacks=6):
        verts, normals, uvs, tris = [], [], [], []
        for s in range(slices + 1):
            theta = 2.0 * math.pi * s / slices
            for t in range(stacks + 1):
                phi = math.pi * t / stacks
                x = math.sin(phi) * math.cos(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.sin(theta)
                verts.append((x, y, z))
                normals.append((x, y, z))
                uvs.append((theta / (2.0 * math.pi), 1.0 - phi / math.pi))
        idx = lambda s_, t_: s_ * (stacks + 1) + t_
        for s in range(slices):
            for t in range(stacks):
                a = idx(s, t)
                b = idx(s + 1, t)
                c = idx(s + 1, t + 1)
                d = idx(s, t + 1)
                tris.extend([a, b, c, a, c, d])
        return verts, normals, uvs, tris

    # ------------------------------------------------------------------
    #  State
    # ------------------------------------------------------------------
    def set_hands(self, hands):
        self._hands = {k: v for k, v in hands.items() if k in ("Left", "Right")}

    def clear(self):
        self._hands = {}

    def update(self, dt):
        # Presence fade — fast so the glove snaps in/out with the hand.
        k = min(1.0, dt * 30.0)
        for label in list(self._presence):
            if label not in self._hands:
                self._presence[label] -= k
                if self._presence[label] <= 0.0:
                    self._presence.pop(label, None)
                    self._smoothed.pop(label, None)
                    self._span.pop(label, None)
        for label in self._hands:
            self._presence[label] = min(1.0, self._presence.get(label, 0.0) + k)

        # EMA on the landmark coordinates (alpha ~0.2-0.3) to kill jitter
        # and drift before the anchored transform is applied.
        a = self.smooth_alpha
        for label, lm in self._hands.items():
            prev = self._smoothed.get(label)
            if prev is None or len(prev) != len(lm):
                self._smoothed[label] = [list(p) for p in lm]
            else:
                self._smoothed[label] = [
                    [p[0] + (q[0] - p[0]) * a,
                     p[1] + (q[1] - p[1]) * a,
                     p[2] + (q[2] - p[2]) * a]
                    for p, q in zip(prev, lm)
                ]

    @property
    def alpha(self):
        if not self._presence:
            return 0.0
        return sum(self._presence.values()) / len(self._presence)

    # ------------------------------------------------------------------
    #  Coordinate mapping (anchored at wrist, scaled by 0->9, palm-aligned)
    # ------------------------------------------------------------------
    def _world(self, lm, aspect):
        pts, unit = self.view.hand_pose(lm, aspect)
        return pts, unit

    def _world_radius(self, r, unit):
        if unit > 0.0:
            return r * unit
        return r * 2.0 * self.view.scale

    # ------------------------------------------------------------------
    #  Texture support
    # ------------------------------------------------------------------
    def load_texture(self, source, repeat=2.0, enabled=True):
        """Load a glove-body texture from a path or (h, w, 4) numpy array."""
        import numpy as np
        if hasattr(source, "save"):
            source = source.save
        arr = np.asarray(source, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 4, axis=-1)
        if arr.shape[2] == 3:
            arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2)
        h, w = arr.shape[:2]
        arr = np.ascontiguousarray(arr[::-1])
        if self._tex is None:
            self._tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, arr)
        self._tex_repeat = repeat
        self._tex_on = enabled

    def clear_texture(self):
        self._tex_on = False

    def set_material(self, name):
        if name in MATERIALS:
            self.material = name

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------
    def render(self, aspect):
        if self.alpha <= 0.01 or not self._smoothed:
            return

        glPushMatrix()
        glEnable(GL_LIGHTING)          # _draw_background disables lighting
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glDisable(GL_TEXTURE_2D)

        a = self.alpha
        body_op = max(0.0, min(1.0, a * self.opacity))
        for label, lm in self._smoothed.items():
            pts, unit = self._world(lm, aspect)
            self._draw_hand(label, pts, body_op, unit)

        # Silhouette edges — no depth write so they never hide the body
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glLineWidth(1.6)
        for label, lm in self._smoothed.items():
            pts, _unit = self._world(lm, aspect)
            self._draw_edges(pts, a)
        glDepthMask(GL_TRUE)

        # Soft tip markers
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for label, lm in self._smoothed.items():
            pts, _unit = self._world(lm, aspect)
            self._draw_markers(pts, a)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

        glPopMatrix()

    def _draw_hand(self, label, pts, alpha, unit):
        """Flat-shaded body with depth writes ON so overlapping parts layer
        correctly. Geometry is baked to world space and emitted as ONE
        batched array draw — the fast path that keeps VR responsive on
        software GL. When the hand has barely moved the mesh is cached."""
        mat = MATERIALS.get(self.material, MATERIALS["graphite"])
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glDepthMask(GL_TRUE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # signature of the pose (quantized to ~4 mm) -> mesh reuse
        sig = tuple(int(p[0] * 256) | (int(p[1] * 256) << 10)
                    | (int(p[2] * 256) << 20) for p in pts)
        step = self._cache_n.get(label, 0)
        arr = self._cache_arr.get(label)
        up_to_date = arr is not None and self._cache_sig.get(label) == sig
        # Rebuild at most every other frame even while moving: the EMA
        # smoothing already makes adjacent poses near-identical.
        if up_to_date or step - self._reb_at.get(label, -9) < 2:
            self._cache_n[label] = step + 1
            if arr is None:
                arr = self._batch_fallback()
            self._draw_batch(mat["diffuse"][:3], alpha, arr)
            glDisable(GL_BLEND)
            return
        self._reb_at[label] = step
        self._cache_n[label] = step + 1

        self._v = []
        for chain in FINGERS.values():
            self._draw_tube(pts, chain, unit)
        for j in range(21):
            self._draw_sphere(pts[j], self._world_radius(RADII[j], unit))
        self._draw_palm(pts, unit)
        arr = (ctypes.c_float * len(self._v))(*self._v)
        self._cache_arr[label] = arr
        self._cache_sig[label] = sig
        self._draw_batch(mat["diffuse"][:3], alpha, arr)

        glDisable(GL_BLEND)

    def _batch_fallback(self):
        if self._v:
            arr = (ctypes.c_float * len(self._v))(*self._v)
            self._cache_arr[""] = arr
            return arr
        return None

    # Raw fixed-function client-array calls (PyOpenGL pointer registry is
    # unreliable in this environment; go straight to the GL library).
    _RAW = None

    @classmethod
    def _raw_gl(cls):
        if cls._RAW is None:
            from OpenGL import platform
            cls._RAW = platform.PLATFORM.GL
        return cls._RAW

    def _draw_batch(self, rgb, alpha, arr=None):
        if arr is None:
            arr = getattr(self, "_batch_arr", None)
        if arr is None or len(arr) < 9:
            return
        raw = self._raw_gl()
        raw.glEnableClientState(GL_VERTEX_ARRAY)
        raw.glVertexPointer(
            3, GL_FLOAT, 0, ctypes.cast(arr, ctypes.c_void_p))
        glColor4f(rgb[0], rgb[1], rgb[2], alpha)
        raw.glDrawArrays(GL_TRIANGLES, 0, len(arr) // 3)
        raw.glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_edges(self, pts, alpha):
        """Faint silhouette outline so the hand stays readable."""
        glColor4f(*GLOVE_ACCENT, alpha * 0.30)
        glBegin(GL_LINES)
        for (a, b) in HAND_CONNECTIONS:
            glVertex3f(*pts[a])
            glVertex3f(*pts[b])
        glEnd()
        glPointSize(2.5)
        glColor4f(1.0, 1.0, 1.0, alpha * 0.20)
        glBegin(GL_POINTS)
        for p in pts:
            glVertex3f(*p)
        glEnd()

    def _draw_markers(self, pts, alpha):
        """Subtle white glow nodes so the virtual hand stays readable."""
        glPointSize(4.0)
        glColor4f(*GLOVE_ACCENT, 0.35 * alpha)
        glBegin(GL_POINTS)
        for p in pts:
            glVertex3f(*p)
        glEnd()
        for j in (4, 8, 12, 16, 20):
            p = pts[j]
            glPushMatrix()
            glTranslatef(p[0], p[1], p[2])
            glColor4f(*GLOVE_ACCENT, 0.45 * alpha)
            self._draw_billboard(0.008)
            glPopMatrix()

    def _draw_billboard(self, size):
        s = size
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(-s, -s, 0)
        glVertex3f(s, -s, 0)
        glVertex3f(s, s, 0)
        glVertex3f(-s, s, 0)
        glEnd()

    def _draw_tube(self, pts, chain, unit=0.0):
        n = len(chain)
        centers = [pts[j] for j in chain]
        radii = [self._world_radius(RADII[j], unit) for j in chain]

        # Smooth tangent at each joint (average of adjacent segment dirs)
        dirs = []
        for i in range(n):
            if i < n - 1:
                d1 = self._sub(centers[i + 1], centers[i])
            else:
                d1 = self._sub(centers[i], centers[i - 1])
            if i > 0:
                d0 = self._sub(centers[i], centers[i - 1])
                d = self._add(d1, d0)
            else:
                d = d1
            length = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
            if length < 1e-6:
                d = (0.0, 1.0, 0.0)
            else:
                d = (d[0] / length, d[1] / length, d[2] / length)
            dirs.append(d)

        ref = (0.0, 0.0, 1.0)
        rings = []
        for i in range(n):
            d = dirs[i]
            u = (
                d[1] * ref[2] - d[2] * ref[1],
                d[2] * ref[0] - d[0] * ref[2],
                d[0] * ref[1] - d[1] * ref[0],
            )
            lu = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
            if lu < 1e-6:
                u = (1.0, 0.0, 0.0)
            else:
                u = (u[0] / lu, u[1] / lu, u[2] / lu)
            v = (d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0])
            r = radii[i]
            ring = []
            for s in range(RING_SEG):
                ang = 2.0 * math.pi * s / RING_SEG
                ca, sa = math.cos(ang), math.sin(ang)
                ring.append((
                    centers[i][0] + r * (ca * u[0] + sa * v[0]),
                    centers[i][1] + r * (ca * u[1] + sa * v[1]),
                    centers[i][2] + r * (ca * u[2] + sa * v[2]),
                ))
            rings.append(ring)

        # Quads between consecutive rings -> two triangles each
        for i in range(n - 1):
            r0, r1 = rings[i], rings[i + 1]
            for s in range(RING_SEG):
                s2 = (s + 1) % RING_SEG
                a, b = r0[s], r0[s2]
                c, d = r1[s2], r1[s]
                self._v.extend((a[0], a[1], a[2], b[0], b[1], b[2], d[0], d[1], d[2]))
                self._v.extend((a[0], a[1], a[2], d[0], d[1], d[2], c[0], c[1], c[2]))

    @staticmethod
    def _sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def _draw_sphere(self, center, radius):
        if radius <= 0.0:
            return
        verts, _normals, _uvs, tris = self._unit_sphere
        cx, cy, cz = center
        for i in range(0, len(tris), 3):
            for k in range(3):
                v = verts[tris[i + k]]
                self._v.extend((
                    cx + v[0] * radius,
                    cy + v[1] * radius,
                    cz + v[2] * radius,
                ))

    def _draw_palm(self, pts, unit=0.0):
        # Skin across wrist -> MCP spread -> thumb base
        center = (0.0, 0.0, 0.0)
        for j in PALM_OUTLINE:
            center = tuple(center[k] + pts[j][k] for k in range(3))
        center = tuple(c / len(PALM_OUTLINE) for c in center)

        # Push skin slightly behind the finger bases
        back = 0.008
        center = (center[0], center[1], center[2] + back)

        # Palm normal (approximate)
        p5 = pts[5]
        p17 = pts[17]
        p0 = pts[0]
        ux, uy, uz = self._sub(p5, p0)
        vx, vy, vz = self._sub(p17, p0)
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln > 1e-6:
            nx, ny, nz = nx / ln, ny / ln, nz / ln
        else:
            nx, ny, nz = 0.0, 0.0, 1.0

        # planar uvs for the palm skin
        xs = [pts[j][0] for j in PALM_OUTLINE]
        ys = [pts[j][1] for j in PALM_OUTLINE]
        mx, my = min(xs), min(ys)
        rx, ry = max(xs) - mx, max(ys) - my
        rx, ry = max(rx, 1e-5), max(ry, 1e-5)

        outline = PALM_OUTLINE
        self._v.extend(center)
        for q in range(len(outline)):
            p = pts[outline[q]]
            pn = pts[outline[(q + 1) % len(outline)]]
            self._v.extend(center)
            self._v.extend(p)
            self._v.extend(pn)