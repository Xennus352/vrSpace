"""JARVIS-style volumetric hologram — Iron Man 3 crime-scene table.

A floating, hand-manipulable hologram rendered in view space so it works over
the live camera passthrough (AR) and inside the VR world (HUD layer) with
no changes.

Gestures (from gesture_detector.py):
    FIST (one hand)     grab-space rotate: move the fist to spin it,
                        release to let it keep spinning (inertia)
    FIST both hands     two-hand zoom (Iron Man style: hands apart/together)
    CLICK (pinch)       grab the hologram and drag it to a new spot
    THUMBS_UP           switch hologram (crime scene / orbital atlas)
    VICTORY             reset pose (position / rotation / scale)

Look: additive cyan wireframe with hot cores, per-vertex scanlines, lamp
flicker, periodic glitch bands, emitter ring + light cone, scan sweep ring
and drifting dust motes.
"""

import ctypes
import math
import time

import numpy as np
from OpenGL.GL import *

from vr_scene import _raw_gl

# line/point class colors
_RGB = np.array([
    (0.35, 0.78, 1.00),   # 0 cyan body
    (0.80, 0.96, 1.00),   # 1 hot core
    (1.00, 0.58, 0.14),   # 2 amber accent
    (0.16, 0.38, 0.58),   # 3 dim structure
], np.float32)
_ALPHA = np.array([0.85, 0.95, 0.95, 0.50], np.float32)

C_CYAN, C_HOT, C_AMBER, C_DIM = 0, 1, 2, 3

# --------------------------------------------------------------------------
#  geometry helpers — return lists of (p0, p1, cls) or (p0,p1,p2,p3, cls)
# --------------------------------------------------------------------------
def _ring(cx, cy, cz, r, cls, seg=48):
    out = []
    for j in range(seg):
        a0 = 2.0 * math.pi * j / seg
        a1 = 2.0 * math.pi * (j + 1) / seg
        out.append(((cx + r * math.cos(a0), cy, cz + r * math.sin(a0)),
                    (cx + r * math.cos(a1), cy, cz + r * math.sin(a1)), cls))
    return out


def _box(cx, cy, cz, sx, sy, sz, cls):
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
         (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
    E = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[a], c[b], cls) for a, b in E]


def _wire_sphere(cx, cy, cz, r, cls, n_lat=2, lon=4, seg=12):
    out = []
    for i in range(1, n_lat + 1):
        phi = -math.pi / 2 + math.pi * i / (n_lat + 1)
        y, rr = math.sin(phi) * r, math.cos(phi) * r
        for j in range(seg):
            a0 = 2 * math.pi * j / seg
            a1 = 2 * math.pi * (j + 1) / seg
            out.append(((cx + rr * math.cos(a0), cy + y, cz + rr * math.sin(a0)),
                        (cx + rr * math.cos(a1), cy + y, cz + rr * math.sin(a1)), cls))
    for k in range(lon):
        a = math.pi * k / lon
        ca, sa = math.cos(a), math.sin(a)
        for j in range(seg):
            p0 = -math.pi / 2 + math.pi * j / seg
            p1 = -math.pi / 2 + math.pi * (j + 1) / seg
            out.append(((cx + r * math.cos(p0) * ca, cy + r * math.sin(p0), cz + r * math.cos(p0) * sa),
                        (cx + r * math.cos(p1) * ca, cy + r * math.sin(p1), cz + r * math.cos(p1) * sa), cls))
    return out


def _grid(half, step, cls, y=0.0):
    out = []
    n = int(round(half / step))
    for i in range(-n, n + 1):
        x = i * step
        out.append(((x, y, -half), (x, y, half), cls))
        out.append(((-half, y, x), (half, y, x), cls))
    return out


def _frame_z(z, x0, x1, y0, y1, cls):
    return [((x0, y0, z), (x1, y0, z), cls), ((x1, y0, z), (x1, y1, z), cls),
            ((x1, y1, z), (x0, y1, z), cls), ((x0, y1, z), (x0, y0, z), cls)]


def _frame_x(x, z0, z1, y0, y1, cls):
    return [((x, y0, z0), (x, y0, z1), cls), ((x, y0, z1), (x, y1, z1), cls),
            ((x, y1, z1), (x, y1, z0), cls), ((x, y1, z0), (x, y0, z0), cls)]


def _dash(p0, p1, cls, dash=0.05, gap=0.045):
    p0 = np.array(p0, np.float64)
    p1 = np.array(p1, np.float64)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return []
    u = v / L
    out, t = [], 0.0
    while t < L:
        t1 = min(L, t + dash)
        out.append((tuple(p0 + u * t), tuple(p0 + u * t1), cls))
        t = t1 + gap
    return out


def _pyramid(cx, cz, r, h, cls, y=0.0):
    b = [(cx - r, y, cz - r), (cx + r, y, cz - r),
         (cx + r, y, cz + r), (cx - r, y, cz + r)]
    apex = (cx, y + h, cz)
    out = [(b[i], b[(i + 1) % 4], cls) for i in range(4)]
    out += [(b[i], apex, cls) for i in range(4)]
    return out


def _figure_standing(cls=C_CYAN, head_cls=C_HOT):
    out = []
    out += _wire_sphere(0.0, 0.555, 0.0, 0.07, head_cls, n_lat=1, lon=3, seg=10)
    out.append(((0.0, 0.485, 0.0), (0.0, 0.505, 0.0), cls))          # neck
    out += _box(0.0, 0.35, 0.0, 0.16, 0.23, 0.085, cls)              # torso
    out += _box(0.0, 0.175, 0.0, 0.145, 0.09, 0.08, cls)             # pelvis
    for s in (1.0, -1.0):
        out.append(((s * 0.10, 0.45, 0.0), (s * 0.165, 0.335, 0.015), cls))
        out.append(((s * 0.165, 0.335, 0.015), (s * 0.175, 0.215, 0.03), cls))
    for s in (1.0, -1.0):
        out.append(((s * 0.055, 0.135, 0.0), (s * 0.06, 0.006, 0.006), cls))
    return out


def _lay_down(segs, cx, cz, lift=0.085):
    out = []
    for a, b, cls in segs:
        a2 = (a[0] + cx, a[2] + lift, -a[1] + cz)
        b2 = (b[0] + cx, b[2] + lift, -b[1] + cz)
        out.append((a2, b2, cls))
    return out


def _chalk_outline(cx, cz):
    out = []
    ex, ez, erx, erz = cx, cz + 0.11, 0.17, 0.34
    seg = 12
    for j in range(seg):
        a0 = 2 * math.pi * j / seg
        a1 = 2 * math.pi * (j + 1) / seg
        out.append(((ex + erx * math.cos(a0), 0.006, ez + erz * math.sin(a0)),
                    (ex + erx * math.cos(a1), 0.006, ez + erz * math.sin(a1)), C_HOT))
    # head circle
    out += _ring(cx, 0.006, cz - 0.555, 0.115, C_HOT, seg=12)
    return out


def _scene_crime():
    L = []
    L += _grid(1.0, 0.25, C_DIM)
    L += _ring(0.0, 0.0, 0.0, 1.04, C_HOT, seg=64)
    L += _ring(0.0, 0.0, 0.0, 0.55, C_DIM, seg=40)

    # back wall
    L += _frame_z(-1.02, -1.02, 1.02, 0.0, 1.12, C_DIM)
    L += _frame_z(-1.02, -0.86, -0.30, 0.42, 0.92, C_CYAN)
    L.append(((-0.58, 0.42, -1.02), (-0.58, 0.92, -1.02), C_CYAN))
    L.append(((-0.86, 0.67, -1.02), (-0.30, 0.67, -1.02), C_CYAN))
    L += _frame_z(-1.02, 0.42, 0.82, 0.0, 0.95, C_CYAN)
    L.append(((0.47, 0.46, -1.02), (0.51, 0.46, -1.02), C_HOT))       # door handle

    # left wall
    L += _frame_x(-1.02, -1.02, 1.02, 0.0, 1.12, C_DIM)
    L.append(((-1.02, 0.50, 0.10), (-1.02, 0.50, 0.55), C_DIM))
    L.append(((-1.02, 0.85, 0.10), (-1.02, 0.85, 0.55), C_DIM))
    L.append(((-1.02, 0.50, 0.10), (-1.02, 0.85, 0.10), C_DIM))
    L.append(((-1.02, 0.50, 0.55), (-1.02, 0.85, 0.55), C_DIM))

    # sofa (back-left)
    L += _box(-0.55, 0.10, -0.70, 0.62, 0.20, 0.28, C_CYAN)
    L += _box(-0.55, 0.30, -0.84, 0.62, 0.22, 0.08, C_CYAN)
    L += _box(-0.83, 0.17, -0.70, 0.06, 0.26, 0.28, C_CYAN)
    L += _box(-0.27, 0.17, -0.70, 0.06, 0.26, 0.28, C_CYAN)

    # coffee table (center)
    L += _box(0.18, 0.205, -0.05, 0.42, 0.03, 0.26, C_CYAN)
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            x = 0.18 + sx * 0.18
            z = -0.05 + sz * 0.10
            L.append(((x, 0.19, z), (x, 0.0, z), C_CYAN))

    # floor lamp (front-left)
    L.append(((-0.85, 0.0, 0.30), (-0.85, 0.80, 0.30), C_CYAN))
    L += _ring(-0.85, 0.80, 0.30, 0.07, C_CYAN, seg=8)

    # standing investigator
    for a, b, c in _figure_standing():
        L.append(((a[0] + 0.55, a[1], a[2] + 0.28),
                  (b[0] + 0.55, b[1], b[2] + 0.28), c))

    # lying victim + chalk outline
    L += _lay_down(_figure_standing(), -0.10, 0.42)
    L += _chalk_outline(-0.10, 0.42)

    # evidence markers
    for (mx, mz) in ((0.02, 0.28), (-0.45, 0.40), (0.30, 0.62)):
        L += _pyramid(mx, mz, 0.05, 0.09, C_AMBER)
        L += _ring(mx, 0.002, mz, 0.085, C_AMBER, seg=12)

    # ballistic trajectory
    L += _dash((0.62, 0.58, -1.0), (-0.08, 0.10, 0.05), C_AMBER)
    L += _ring(-0.08, 0.02, 0.05, 0.045, C_AMBER, seg=10)

    return {
        "name": "CASE #042 — RECON",
        "lines": L, "quads": [],
        "floor_r": 1.04,
        "top": 1.12,
        "orbiters": [],
        "scan": True,
    }


def _scene_orbit():
    L = []
    gy = 0.66

    # globe (5 lat rings incl equator, 6 longitudes, 22 segments)
    L += _wire_sphere(0.0, gy, 0.0, 0.52, C_CYAN, n_lat=5, lon=6, seg=22)

    # tilted axis rod
    s = math.sin(0.41); c = math.cos(0.41)
    L.append(((0.0, gy - 0.72 * c, -0.72 * s),
              (0.0, gy + 0.72 * c,  0.72 * s), C_DIM))

    # tilted orbit rings
    for r, tilt in ((0.74, 0.42), (0.92, -0.30)):
        st, ct = math.sin(tilt), math.cos(tilt)
        for j in range(48):
            a0 = 2 * math.pi * j / 48
            a1 = 2 * math.pi * (j + 1) / 48
            p0 = (r * math.cos(a0), gy - r * math.sin(a0) * st, r * math.sin(a0) * ct)
            p1 = (r * math.cos(a1), gy - r * math.sin(a1) * st, r * math.sin(a1) * ct)
            L.append((p0, p1, C_DIM))

    # base rings
    L += _ring(0.0, 0.0, 0.0, 0.78, C_HOT, seg=56)
    L += _ring(0.0, 0.0, 0.0, 0.97, C_DIM, seg=64)

    orbiters = [
        {"r": 0.74, "tilt": 0.42, "speed": 0.55, "phase": 0.0,
         "size": 0.028, "cls": C_HOT, "y": gy},
        {"r": 0.92, "tilt": -0.30, "speed": 0.28, "phase": 2.2,
         "size": 0.05, "cls": C_CYAN, "y": gy},
    ]

    return {
        "name": "ORBITAL ATLAS",
        "lines": L, "quads": [],
        "floor_r": 0.97,
        "top": 1.61,
        "orbiters": orbiters,
        "scan": True,
    }


def _pack(segs, quads, **meta):
    lv = np.asarray([[a, b] for a, b, _ in segs], np.float32).reshape(-1, 3)
    lc = np.asarray([[c, c] for _, _, c in segs], np.int8).reshape(-1)
    if quads:
        qv = np.asarray([[p0, p1, p2, p3] for p0, p1, p2, p3, _ in quads],
                        np.float32).reshape(-1, 3)
        qc = np.asarray([[c] * 4 for *_, c in quads], np.int8).reshape(-1)
    else:
        qv = np.zeros((0, 3), np.float32)
        qc = np.zeros((0,), np.int8)
    d = {"lines": lv, "lcls": lc, "quads": qv, "qcls": qc}
    d.update(meta)
    return d


def _build_scene(fn):
    """Wrap scene builder to pass segs/quads positionally to _pack."""
    d = fn()
    return _pack(d["lines"], d["quads"],
                 floor_r=d["floor_r"], top=d["top"],
                 orbiters=d["orbiters"], scan=d["scan"],
                 name=d["name"])


# --------------------------------------------------------------------------
#  Hologram controller + renderer
# --------------------------------------------------------------------------
class Hologram:
    SCALE_MIN = 0.26
    SCALE_MAX = 1.7

    def __init__(self, view):
        self.view = view
        self.visible = True
        self._vis = 1.0

        self.scenes = [_build_scene(_scene_crime), _build_scene(_scene_orbit)]
        self.scene_i = 0
        self._switch_to = None
        self._fade = 1.0
        self._event = None

        # pose (view-space for HUD, world-space for VR world)
        self.center = np.array([0.0, -0.14, -2.30], np.float64)   # view-space anchor
        self.world_pos = np.array([0.0, 0.0, -5.0], np.float64)    # world-space position
        self.yaw = 0.6
        self.pitch = 0.12
        self.scale = 0.62
        self._target_scale = 0.62
        self._yaw_vel = 0.0
        self._pitch_vel = 0.0
        self._resetting = False

        # interaction
        self._prev_palm = {}
        self._prev_g = {}
        self._rot_hand = None
        self._grab_hand = None
        self._grab_off = np.zeros(2, np.float64)
        self._two_hand = False
        self._th_mid0 = np.zeros(2)
        self._th_vec0 = np.zeros(2)
        self._th_dist0 = 1.0
        self._th_angle0 = 0.0
        self._th_center0 = np.zeros(3)
        self._th_yaw0 = 0.0
        self._th_scale0 = 1.0
        # locomotion (Quest 3 style: pinch + hand direction)
        self._loco_hand = None
        self._loco_start = np.zeros(2)
        self._loco_vec = np.zeros(3)   # (fwd, right, up) in world space
        self._last_palms = 0.0
        self._last_hands_state = None
        self._t = 0.0

        # fx
        self._rng = np.random.default_rng(7)
        self._next_glitch = 3.0
        self._glitch_until = -1.0
        self._glitch_bands = None

        # dust
        r = np.sqrt(self._rng.uniform(0.0, 1.0, 44)) * 0.85
        a = self._rng.uniform(0, 2 * np.pi, 44)
        self._dust = np.stack([r * np.cos(a), self._rng.uniform(0, 1, 44), r * np.sin(a)], 1).astype(np.float32)
        self._dust_spd = self._rng.uniform(0.05, 0.16, 44).astype(np.float32)

    def pop_event(self):
        e, self._event = self._event, None
        return e

    def get_locomotion(self):
        """Return (fwd, right, up) in [-1,1] for VR movement.
        Called by renderer each frame to apply gesture-based locomotion."""
        return (float(self._loco_vec[0]), float(self._loco_vec[1]), float(self._loco_vec[2]))

    # ----------------------------------------------------------------------
    #  interaction
    # ----------------------------------------------------------------------
    def handle_hands(self, hands_state, aspect):
        now = time.time()
        dt_h = min(0.2, now - self._last_palms) if self._last_palms else 0.033
        if hands_state:
            self._last_palms = now
        # Store for finger pointer rendering
        self._last_hands_state = hands_state

        palms = {}
        tips = {}
        for label in ("Left", "Right"):
            d = hands_state.get(label)
            if not d:
                continue
            lm = d.get("landmarks")
            if not lm or len(lm) < 21:
                continue
            g = d.get("gesture", "IDLE")
            pts, _ = self.view.hand_pose(lm, aspect)
            # palm center
            px = (pts[0][0] + pts[9][0]) * 0.5
            py = (pts[0][1] + pts[9][1]) * 0.5
            # index tip for precise grab
            tx, ty = pts[8][0], pts[8][1]
            palms[label] = (g, np.array([px, py], np.float64))
            tips[label] = np.array([tx, ty], np.float64)

        # edge commands (either hand)
        for label, (g, palm) in palms.items():
            prev = self._prev_g.get(label)
            if g == "VICTORY" and prev != "VICTORY":
                self._resetting = True
                self._rot_hand = None
                self._grab_hand = None
                self._two_hand = False
                self._event = f"{label}: hologram reset"
            elif g == "THUMBS_UP" and prev != "THUMBS_UP" and self._switch_to is None:
                self._switch_to = (self.scene_i + 1) % len(self.scenes)
                self._event = f"{label}: next hologram"

        # TWO-HAND MANIPULATION (both CLICK anywhere)
        both_click = ("Left" in palms and "Right" in palms
                      and palms["Left"][0] == "CLICK" and palms["Right"][0] == "CLICK")

        if both_click:
            self._resetting = False
            self._rot_hand = None
            self._grab_hand = None

            tL, tR = tips["Left"], tips["Right"]
            pL, pR = palms["Left"][1], palms["Right"][1]

            # Initialize two-hand manipulation (no proximity requirement)
            if not self._two_hand:
                self._two_hand = True
                self._th_mid0 = (pL + pR) * 0.5
                self._th_vec0 = pR - pL
                self._th_dist0 = float(np.linalg.norm(self._th_vec0))
                self._th_angle0 = math.atan2(self._th_vec0[1], self._th_vec0[0])
                self._th_center0 = self.center.copy()
                self._th_yaw0 = self.yaw
                self._th_scale0 = self.scale
            else:
                # Update transform from hand motion
                mid = (pL + pR) * 0.5
                vec = pR - pL
                dist = float(np.linalg.norm(vec))
                angle = math.atan2(vec[1], vec[0])

                # Translation: follow midpoint
                dmid = mid - self._th_mid0
                self.center[:2] = self._th_center0[:2] + dmid

                # Rotation: relative angle change
                dang = angle - self._th_angle0
                dang = (dang + math.pi) % (2 * math.pi) - math.pi
                self.yaw = self._th_yaw0 - dang * 1.5

                # Scale: relative distance change
                if self._th_dist0 > 0.02:
                    self._target_scale = float(np.clip(
                        self._th_scale0 * dist / self._th_dist0,
                        self.SCALE_MIN, self.SCALE_MAX))
        else:
            if self._two_hand:
                self._two_hand = False

        # SINGLE-HAND OPS (only when not two-hand)
        if not self._two_hand:
            # Sticky rotate with FIST
            active = None
            for label in ("Left", "Right"):
                if label in palms and palms[label][0] == "FIST" and \
                        (self._rot_hand in (None, label)) and self._grab_hand != label:
                    active = label
                    break
            if active is not None:
                self._resetting = False
                self._rot_hand = active
                prev = self._prev_palm.get(active)
                p = palms[active][1]
                if prev is not None:
                    dx = float(p[0] - prev[0])
                    dy = float(p[1] - prev[1])
                    self.yaw += dx * 2.4
                    self.pitch = float(np.clip(self.pitch - dy * 1.8, -1.1, 1.1))
                    iyaw = dx * 2.4 / max(dt_h, 1e-3)
                    ipit = -dy * 1.8 / max(dt_h, 1e-3)
                    self._yaw_vel = 0.62 * self._yaw_vel + 0.38 * iyaw
                    self._pitch_vel = 0.62 * self._pitch_vel + 0.38 * ipit
            else:
                self._rot_hand = None

            # Grab/move with single CLICK (index tip)
            if self._grab_hand is None:
                for label in ("Left", "Right"):
                    if label in palms and palms[label][0] == "CLICK":
                        t = tips[label]
                        c = self.center
                        dx = t[0] - c[0]
                        dy = t[1] - (c[1] + 0.55 * self.scale)
                        if math.hypot(dx, dy) < 0.35 + 0.9 * self.scale:
                            self._grab_hand = label
                            self._grab_off = t - c[:2]
                            self._resetting = False
                            self._event = f"{label}: hologram grabbed"
                        break
            if self._grab_hand is not None:
                g2 = palms.get(self._grab_hand)
                if g2 is not None and g2[0] == "CLICK":
                    t = tips[self._grab_hand]
                    self.center[0] = np.clip(t[0] - self._grab_off[0], -2.5, 2.5)
                    self.center[1] = np.clip(t[1] - self._grab_off[1], -1.0, 0.5)
                else:
                    self._grab_hand = None

        # LOCOMOTION (Quest 3 style: pinch + hand direction)
        # Start: CLICK on empty space (not on hologram) -> set loco hand
        # Move: while pinched, hand offset from start gives fwd/right/up
        # Release: stop
        loco_candidate = None
        for label in ("Left", "Right"):
            if label in palms and palms[label][0] == "CLICK":
                # Check if NOT grabbing hologram
                t = tips[label]
                c = self.center
                d = math.hypot(t[0] - c[0], t[1] - (c[1] + 0.55 * self.scale))
                if d >= 0.35 + 0.9 * self.scale:
                    loco_candidate = label
                    break
        if loco_candidate is not None:
            if self._loco_hand is None:
                self._loco_hand = loco_candidate
                self._loco_start = tips[loco_candidate].copy()
            elif self._loco_hand == loco_candidate:
                # Compute offset from start (normalized view coords)
                cur = tips[loco_candidate]
                dx = cur[0] - self._loco_start[0]
                dy = cur[1] - self._loco_start[1]
                # Map to world: dy -> forward/back, dx -> left/right
                # Scale sensitivity
                self._loco_vec[0] = np.clip(-dy * 3.0, -1.0, 1.0)   # forward
                self._loco_vec[1] = np.clip(-dx * 3.0, -1.0, 1.0)   # right
                # Pinch depth (z) for up/down - use thumb-index distance
                lm = hands_state[loco_candidate].get("landmarks")
                if lm and len(lm) >= 21:
                    pts, _ = self.view.hand_pose(lm, aspect)
                    pinch = math.hypot(pts[4][0] - pts[8][0], pts[4][1] - pts[8][1])
                    # Pinch tighter -> up, looser -> down (inverted)
                    self._loco_vec[2] = np.clip((0.05 - pinch) * 10.0, -1.0, 1.0)
        else:
            if self._loco_hand is not None:
                self._loco_hand = None
                self._loco_vec[:] = 0.0

        # update prev
        for label in ("Left", "Right"):
            if label in palms:
                self._prev_palm[label] = palms[label][1].copy()
                self._prev_g[label] = palms[label][0]
            else:
                self._prev_g.pop(label, None)
                self._prev_palm.pop(label, None)

        # init two-hand state
        if not hasattr(self, '_two_hand'):
            self._two_hand = False
        if not hasattr(self, '_th_mid0'):
            self._th_mid0 = np.zeros(2)
            self._th_vec0 = np.zeros(2)
            self._th_dist0 = 1.0
            self._th_angle0 = 0.0
            self._th_center0 = np.zeros(3)
            self._th_yaw0 = 0.0
            self._th_scale0 = 1.0
            if label in palms:
                self._prev_palm[label] = palms[label][1].copy()
                self._prev_g[label] = palms[label][0]
            else:
                self._prev_g.pop(label, None)
                self._prev_palm.pop(label, None)

    # ----------------------------------------------------------------------
    #  per-frame update
    # ----------------------------------------------------------------------
    def update(self, dt):
        self._t += dt
        t = self._t

        # visibility envelope
        tgt = 1.0 if self.visible else 0.0
        self._vis += (tgt - self._vis) * min(1.0, dt * 5.0)

        # scale easing
        self.scale += (self._target_scale - self.scale) * min(1.0, dt * 9.0)

        # reset animation
        if self._resetting:
            k = 1.0 - math.exp(-3.6 * dt)
            d = np.array([0.0, -0.14, -2.30], np.float64)
            self.center += (d - self.center) * k
            dw = np.array([0.0, 0.0, -5.0], np.float64)
            self.world_pos += (dw - self.world_pos) * k
            dyaw = (0.6 - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += dyaw * k
            self.pitch += (0.12 - self.pitch) * k
            self._target_scale += (0.62 - self._target_scale) * k
            if (np.linalg.norm(self.center - d) < 0.01 and abs(dyaw) < 0.01
                    and abs(self.pitch - 0.12) < 0.01
                    and abs(self._target_scale - 0.62) < 0.01
                    and np.linalg.norm(self.world_pos - dw) < 0.01):
                self._resetting = False
        else:
            if self._rot_hand is None:
                self.yaw += self._yaw_vel * dt
                self.pitch = float(np.clip(self.pitch + self._pitch_vel * dt, -1.1, 1.1))
                decay = math.exp(-2.6 * dt)
                self._yaw_vel *= decay
                self._pitch_vel *= decay
                if abs(self._yaw_vel) < 0.01:
                    self._yaw_vel = 0.0
                if abs(self._pitch_vel) < 0.01:
                    self._pitch_vel = 0.0
                # gentle turntable when idle
                if self._last_palms and (time.time() - self._last_palms) > 4.0:
                    if abs(self._yaw_vel) < 0.05:
                        self.yaw += 0.25 * dt

        # scene cross-fade
        if self._switch_to is not None:
            self._fade -= dt * 4.0
            if self._fade <= 0.0:
                self._fade = 0.0
                self.scene_i = self._switch_to
                self._switch_to = None
                self._event = f"hologram: {self.scenes[self.scene_i]['name']}"
        elif self._fade < 1.0:
            self._fade = min(1.0, self._fade + dt * 4.0)

        # glitch scheduler
        if t >= self._next_glitch:
            self._glitch_until = t + self._rng.uniform(0.09, 0.22)
            self._next_glitch = t + self._rng.uniform(3.0, 7.5)
            bands = self._rng.uniform(0.05, 1.0, 3)
            offs = self._rng.uniform(-0.09, 0.09, 3)
            self._glitch_bands = list(zip(bands, offs))

    # ----------------------------------------------------------------------
    #  rendering
    # ----------------------------------------------------------------------
    def _dynamic_locals(self, scene, t):
        """Return (seg_verts, seg_cls, seg_extra, tri_verts, tri_alpha)"""
        segs = []

    def _build_hand_pointers(self, hands_state, aspect):
        """Build dynamic, realistic finger rays with reticles and interaction feedback."""
        segs = []
        if not hands_state:
            return (np.zeros((0, 3), np.float32), np.zeros((0,), np.int8),
                    np.zeros((0,), np.float32))

        t = self._t

        for label in ("Left", "Right"):
            d = hands_state.get(label)
            if not d:
                continue
            lm = d.get("landmarks")
            if not lm or len(lm) < 21:
                continue
            g = d.get("gesture", "IDLE")
            pts, _ = self.view.hand_pose(lm, aspect)

            # Index finger: MCP (5) -> PIP (6) -> DIP (7) -> TIP (8)
            # Build smooth curve through finger joints
            finger_joints = [np.array(pts[i]) for i in (5, 6, 7, 8)]
            base = finger_joints[0]
            tip = finger_joints[-1]

            # Direction from PIP to TIP (more stable than MCP->TIP)
            dir_vec = tip - finger_joints[1]
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-4:
                continue
            dir_vec = dir_vec / norm

            # Ray length: adaptive based on gesture and distance to hologram
            if g == "CLICK":
                ray_len = 0.6
                base_color = C_AMBER
            elif g == "FIST":
                ray_len = 0.25
                base_color = C_DIM
            else:
                ray_len = 3.0
                base_color = C_HOT

            # Check intersection with hologram bounds for visual feedback
            hologram_center = np.array([self.center[0], self.center[1] + 0.55 * self.scale, self.center[2]])
            hologram_radius = (0.35 + 0.9 * self.scale) * self.scale
            # Ray-plane intersection with hologram mid-plane
            to_center = hologram_center - tip
            denom = np.dot(dir_vec, np.array([0, 1, 0]))  # vertical plane
            hit_dist = None
            if abs(denom) > 1e-4:
                d = np.dot(to_center, np.array([0, 1, 0])) / denom
                if 0 < d < ray_len:
                    hit_dist = d

            # Shorten ray if hitting hologram
            if hit_dist is not None:
                ray_len = min(ray_len, hit_dist * 1.05)
                base_color = C_AMBER  # highlight on target

            end = tip + dir_vec * ray_len

            # ---- Main beam: tapered segments with glow ----
            n_seg = 12
            for i in range(n_seg):
                u0 = i / n_seg
                u1 = (i + 1) / n_seg
                p0 = tip + dir_vec * (ray_len * u0)
                p1 = tip + dir_vec * (ray_len * u1)
                # Taper width and alpha
                w = 1.0 - u0 * 0.7
                a = 1.0 - u0 * 0.5
                # Pulsing glow
                pulse = 0.85 + 0.15 * math.sin(t * 12.0 + u0 * 4.0)
                segs.append((tuple(p0), tuple(p1), base_color, a * pulse * w))

            # ---- Reticule at end (3D crosshair) ----
            if g != "FIST":  # no reticle for fist
                cz = 0.035
                # Facing camera (billboard)
                perp1 = np.array([-dir_vec[1], dir_vec[0], 0.0])
                perp1 = perp1 / (np.linalg.norm(perp1) + 1e-6)
                perp2 = np.cross(dir_vec, perp1)
                perp2 = perp2 / (np.linalg.norm(perp2) + 1e-6)

                # Outer brackets
                for sgn in (-1, 1):
                    for axis in (perp1, perp2):
                        p0 = end + axis * (cz * sgn)
                        p1 = end + axis * (cz * sgn) + dir_vec * cz
                        segs.append((tuple(p0), tuple(p1), base_color, 1.0))

                # Center dot (small cross)
                dot_sz = 0.012
                segs.append(((end[0] - dot_sz, end[1], end[2]), (end[0] + dot_sz, end[1], end[2]), base_color, 1.0))
                segs.append(((end[0], end[1] - dot_sz, end[2]), (end[0], end[1] + dot_sz, end[2]), base_color, 1.0))
                segs.append(((end[0], end[1], end[2] - dot_sz), (end[0], end[1], end[2] + dot_sz), base_color, 1.0))

            # ---- Finger joint trail (subtle) ----
            for i in range(3):
                p0 = finger_joints[i]
                p1 = finger_joints[i + 1]
                segs.append((tuple(p0), tuple(p1), C_DIM, 0.4))

        if not segs:
            return (np.zeros((0, 3), np.float32), np.zeros((0,), np.int8),
                    np.zeros((0,), np.float32))

        sv = np.asarray([[a, b] for a, b, _, _ in segs], np.float32).reshape(-1, 3)
        sc = np.asarray([[c, c] for _, _, c, _ in segs], np.int8).reshape(-1)
        sa = np.repeat(np.asarray([v for _, _, _, v in segs], np.float32), 2)
        return sv, sc, sa

    def _dynamic_locals(self, scene, t):
        """Return (seg_verts, seg_cls, seg_extra, tri_verts, tri_alpha)"""
        segs = []
        fr = scene["floor_r"]
        top = scene["top"]

        # rotating rim arcs (emitter activity)
        for j in range(3):
            a0 = t * (0.7 + 0.22 * j) + 2.1 * j
            for k in range(8):
                u0 = a0 + 1.0 * k / 8.0
                u1 = a0 + 1.0 * (k + 1) / 8.0
                segs.append(((fr * math.cos(u0), 0.004, fr * math.sin(u0)),
                             (fr * math.cos(u1), 0.004, fr * math.sin(u1)), C_HOT, 0.9))

        # expanding pulse rings
        for i in range(2):
            f = (t * 0.4 + 0.5 * i) % 1.0
            r = fr * (0.30 + 0.72 * f)
            a_scale = (1.0 - f) * 0.7
            for k in range(36):
                a0 = 2 * math.pi * k / 36
                a1 = 2 * math.pi * (k + 1) / 36
                segs.append(((r * math.cos(a0), 0.003, r * math.sin(a0)),
                             (r * math.cos(a1), 0.003, r * math.sin(a1)), C_CYAN, a_scale))

        # rising scan ring
        if scene["scan"]:
            y = 0.02 + ((t * 0.16) % 1.0) * top * 0.96
            for k in range(40):
                a0 = 2 * math.pi * k / 40
                a1 = 2 * math.pi * (k + 1) / 40
                r = fr * 0.96
                segs.append(((r * math.cos(a0), y, r * math.sin(a0)),
                             (r * math.cos(a1), y, r * math.sin(a1)), C_HOT, 1.0))

        # orbiters + trails
        for orb in scene["orbiters"]:
            y0 = orb["y"]
            tilt = orb["tilt"]
            st, ct = math.sin(tilt), math.cos(tilt)
            a0 = orb["phase"] + orb["speed"] * t
            sz = orb["size"]
            cls = orb["cls"]
            # cross marker (3 segs)
            p = (orb["r"] * math.cos(a0), y0 - orb["r"] * math.sin(a0) * st, orb["r"] * math.sin(a0) * ct)
            for axis in ((sz, 0, 0), (0, 0, sz), (0, sz * 0.5, 0)):
                segs.append(((p[0] - axis[0], p[1] - axis[1], p[2] - axis[2]),
                             (p[0] + axis[0], p[1] + axis[1], p[2] + axis[2]), cls, 1.0))
            # trail (3 segments fading)
            for k in range(3):
                ak = a0 - (k + 1) * 0.05 * (1 if orb["speed"] > 0 else -1)
                pk = (orb["r"] * math.cos(ak), y0 - orb["r"] * math.sin(ak) * st, orb["r"] * math.sin(ak) * ct)
                nk = (orb["r"] * math.cos(ak - 0.03), y0 - orb["r"] * math.sin(ak - 0.03) * st,
                      orb["r"] * math.sin(ak - 0.03) * ct)
                segs.append((pk, nk, cls, max(0.1, 0.5 - k * 0.12)))

        # central beam (floor -> apex)
        segs.append(((0.0, -0.16, 0.0), (0.0, 0.02, 0.0), C_HOT, 0.6))

        # emitter cone (triangle fan apex -> base ring)
        apex = (0.0, -0.16, 0.0)
        tris = []
        for k in range(18):
            a0 = 2 * math.pi * k / 18
            a1 = 2 * math.pi * (k + 1) / 18
            b0 = (fr * math.cos(a0), 0.001, fr * math.sin(a0))
            b1 = (fr * math.cos(a1), 0.001, fr * math.sin(a1))
            tris.append((apex, b0, b1, 0.9))

        if not segs:
            return (np.zeros((0, 3), np.float32), np.zeros((0,), np.int8),
                    np.zeros((0,), np.float32),
                    np.zeros((0, 3), np.float32), np.zeros((0,), np.float32))

        # pack segs (per-vertex)
        sv = np.asarray([[a, b] for a, b, _, _ in segs], np.float32).reshape(-1, 3)
        sc = np.asarray([[c, c] for _, _, c, _ in segs], np.int8).reshape(-1)
        sa = np.repeat(np.asarray([v for _, _, _, v in segs], np.float32), 2)

        # pack tris
        tv = np.asarray([list(tri[:3]) for tri in tris], np.float32).reshape(-1, 3)
        ta = np.zeros(tv.shape[0], np.float32)
        for i, tri in enumerate(tris):
            ta[i * 3] = tri[3] * 0.10      # apex
            ta[i * 3 + 1] = tri[3] * 0.02  # base
            ta[i * 3 + 2] = tri[3] * 0.02

        return sv, sc, sa, tv, ta

    def _apply_glitch(self, wv, ly, cols):
        wv = wv.copy()
        cols = cols.copy()
        for band, off in self._glitch_bands:
            m = np.abs(ly - band) < 0.05
            wv[m, 0] += off
        cols[:, 0] *= 1.25
        cols[:, 1] *= 0.72
        cols[:, 2] *= 1.30
        return wv, cols

    # ----------------------------------------------------------------------
    #  world-space draw (for VR world - camera transform already applied)
    # ----------------------------------------------------------------------
    def draw_world(self, aspect):
        """Draw in world space. Call with camera transform already on MODELVIEW stack."""
        if self._vis <= 0.01:
            return

        scene = self.scenes[self.scene_i]
        t = self._t

        # breathing scale
        s = self.scale * (1.0 + 0.012 * math.sin(t * 2.1))

        # rotation matrix (yaw then pitch)
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[cy_, 0.0, sy_], [0.0, 1.0, 0.0], [-sy_, 0.0, cy_]], np.float64)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], np.float64)
        R = (Ry @ Rx).astype(np.float32)
        C = self.world_pos.astype(np.float32)

        # global alpha
        flick = 0.88 + 0.12 * math.sin(t * 61.0) * math.sin(t * 7.7)
        glitch = t < self._glitch_until
        A = self._vis * self._fade * flick
        if glitch:
            A *= float(self._rng.uniform(0.55, 1.05))

        # dynamic local geometry
        sv, sc, sa, tv, ta = self._dynamic_locals(scene, t)

        # static + dynamic lines (pointers drawn separately in view space)
        lv = np.concatenate([scene["lines"], sv], 0)
        lc = np.concatenate([scene["lcls"], sc], 0)
        le = np.concatenate([np.ones(len(scene["lcls"]), np.float32), sa], 0)

        # per-vertex scanline (object-locked)
        ys = lv[:, 1:2]
        scan = (0.80 + 0.20 * np.sin(ys * 95.0 - t * 6.0)).astype(np.float32)

        rgb = _RGB[lc]
        alp = (_ALPHA[lc].reshape(-1, 1) * scan * A * le.reshape(-1, 1))
        cols = np.concatenate([rgb, alp], 1)

        wv = lv @ R.T * s + C

        if glitch and self._glitch_bands:
            wv, cols = self._apply_glitch(wv, lv[:, 1], cols)

        # cone tris
        if tv.shape[0]:
            tw = tv @ R.T * s + C
            trgb = np.tile(np.array([0.35, 0.78, 1.00], np.float32), (tw.shape[0], 1))
            talp = (ta.reshape(-1, 1) * A * 0.05)
            tcols = np.concatenate([trgb, talp], 1)
        else:
            tw = np.zeros((0, 3), np.float32)
            tcols = np.zeros((0, 4), np.float32)

        # dust
        top = scene["top"]
        dy = (self._dust[:, 1] + t * self._dust_spd) % top
        dpos = np.concatenate([self._dust[:, 0:1], dy.reshape(-1, 1), self._dust[:, 2:3]], 1)
        dw = dpos @ R.T * s + C
        drgb = np.tile(_RGB[C_HOT], (dw.shape[0], 1))
        dalp = np.full((dw.shape[0], 1), 0.35 * A, np.float32)
        dcols = np.concatenate([drgb, dalp], 1)

        # --- draw (additive, no depth test for hologram look) ---
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        if tw.shape[0]:
            self._draw_arrays(GL_TRIANGLES, tw, tcols)

        glow = cols.copy()
        glow[:, 3] *= 0.30
        self._draw_arrays(GL_LINES, wv, glow, width=3.4)
        self._draw_arrays(GL_LINES, wv, cols, width=1.35)

        self._draw_arrays(GL_POINTS, dw, dcols, psize=3.0)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def _draw_arrays(self, mode, verts, cols, width=None, psize=None):
        n = int(verts.shape[0])
        if n == 0:
            return
        v = np.ascontiguousarray(verts, dtype=np.float32)
        c = np.ascontiguousarray(cols, dtype=np.float32)
        raw = _raw_gl()
        if width is not None:
            glLineWidth(width)
        if psize is not None:
            glPointSize(psize)
        raw.glEnableClientState(GL_VERTEX_ARRAY)
        raw.glEnableClientState(GL_COLOR_ARRAY)
        raw.glVertexPointer(3, GL_FLOAT, 0,
                            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        raw.glColorPointer(4, GL_FLOAT, 0,
                           c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        raw.glDrawArrays(mode, 0, n)
        raw.glDisableClientState(GL_COLOR_ARRAY)
        raw.glDisableClientState(GL_VERTEX_ARRAY)
        if width is not None:
            glLineWidth(1.0)
        if psize is not None:
            glPointSize(1.0)

    def _draw_smoke(self, center, s, A):
        r = s * 1.15
        seg = 24
        cx, cy, cz = center[0], center[1] + 0.3 * s, center[2] - 0.02
        # center vertex + ring
        verts = np.zeros((seg + 1, 3), np.float32)
        verts[0] = (cx, cy, cz)
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts[k + 1] = (cx + r * math.cos(a), cy, cz + r * math.sin(a))
        cols = np.zeros((seg + 1, 4), np.float32)
        cols[0] = (0.02, 0.05, 0.09, 0.30 * A)
        cols[1:] = (0.02, 0.05, 0.09, 0.0)
        raw = _raw_gl()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        raw.glEnableClientState(GL_VERTEX_ARRAY)
        raw.glEnableClientState(GL_COLOR_ARRAY)
        raw.glVertexPointer(3, GL_FLOAT, 0,
                            verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        raw.glColorPointer(4, GL_FLOAT, 0,
                           cols.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        raw.glDrawArrays(GL_TRIANGLE_FAN, 0, seg + 1)
        raw.glDisableClientState(GL_COLOR_ARRAY)
        raw.glDisableClientState(GL_VERTEX_ARRAY)

    def draw(self, aspect):
        if self._vis <= 0.01:
            return

        scene = self.scenes[self.scene_i]
        t = self._t

        # breathing scale
        s = self.scale * (1.0 + 0.012 * math.sin(t * 2.1))

        # rotation matrix (yaw then pitch)
        cy_, sy_ = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Ry = np.array([[cy_, 0.0, sy_], [0.0, 1.0, 0.0], [-sy_, 0.0, cy_]], np.float64)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], np.float64)
        R = (Ry @ Rx).astype(np.float32)
        C = self.center.astype(np.float32)

        # global alpha
        flick = 0.88 + 0.12 * math.sin(t * 61.0) * math.sin(t * 7.7)
        glitch = t < self._glitch_until
        A = self._vis * self._fade * flick
        if glitch:
            A *= float(self._rng.uniform(0.55, 1.05))

        # dynamic local geometry
        sv, sc, sa, tv, ta = self._dynamic_locals(scene, t)

        # static + dynamic lines (pointers drawn separately in view space)
        lv = np.concatenate([scene["lines"], sv], 0)
        lc = np.concatenate([scene["lcls"], sc], 0)
        le = np.concatenate([np.ones(len(scene["lcls"]), np.float32), sa], 0)

        # per-vertex scanline (object-locked)
        ys = lv[:, 1:2]
        scan = (0.80 + 0.20 * np.sin(ys * 95.0 - t * 6.0)).astype(np.float32)

        rgb = _RGB[lc]
        alp = (_ALPHA[lc].reshape(-1, 1) * scan * A * le.reshape(-1, 1))
        cols = np.concatenate([rgb, alp], 1)

        wv = lv @ R.T * s + C

        if glitch and self._glitch_bands:
            wv, cols = self._apply_glitch(wv, lv[:, 1], cols)

        # quads (walls, etc)
        qv = scene["quads"]
        qc = scene["qcls"]
        if qv.shape[0]:
            qw = qv @ R.T * s + C
            qys = qw[:, 1:2]
            qscan = (0.80 + 0.20 * np.sin(qys * 95.0 - t * 6.0)).astype(np.float32)
            qrgb = _RGB[qc]
            qalp = (_ALPHA[qc].reshape(-1, 1) * qscan * A * 0.10)
            qcols = np.concatenate([qrgb, qalp], 1)
        else:
            qw = np.zeros((0, 3), np.float32)
            qcols = np.zeros((0, 4), np.float32)

        # cone tris
        if tv.shape[0]:
            tw = tv @ R.T * s + C
            trgb = np.tile(np.array([0.35, 0.78, 1.00], np.float32), (tw.shape[0], 1))
            talp = (ta.reshape(-1, 1) * A * 0.05)
            tcols = np.concatenate([trgb, talp], 1)
        else:
            tw = np.zeros((0, 3), np.float32)
            tcols = np.zeros((0, 4), np.float32)

        # dust
        top = scene["top"]
        dy = (self._dust[:, 1] + t * self._dust_spd) % top
        dz = self._dust[:, [0, 2]]
        dpos = np.concatenate([self._dust[:, 0:1], dy.reshape(-1, 1), self._dust[:, 2:3]], 1)
        dw = dpos @ R.T * s + C
        drgb = np.tile(_RGB[C_HOT], (dw.shape[0], 1))
        dalp = np.full((dw.shape[0], 1), 0.35 * A, np.float32)
        dcols = np.concatenate([drgb, dalp], 1)

        # --- draw ---
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)

        # 1) backing smoke disc (normal blend)
        self._draw_smoke(C, s, A)

        # 2) additive for everything else
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        # quads + cone (volume hint)
        if qw.shape[0]:
            self._draw_arrays(GL_QUADS, qw, qcols)
        if tw.shape[0]:
            self._draw_arrays(GL_TRIANGLES, tw, tcols)

        # hologram lines: glow pass then core
        glow = cols.copy()
        glow[:, 3] *= 0.30
        self._draw_arrays(GL_LINES, wv, glow, width=3.4)
        self._draw_arrays(GL_LINES, wv, cols, width=1.35)

        # dust
        self._draw_arrays(GL_POINTS, dw, dcols, psize=3.0)

        # restore
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    # ----------------------------------------------------------------------
    #  View-space finger pointers (drawn in HUD pass with identity matrix)
    # ----------------------------------------------------------------------
    def draw_pointers(self, aspect):
        """Draw finger rays in view space (identity matrix) so they follow
        the hand like Meta Quest 3. Call from HUD pass with glLoadIdentity()."""
        if self._vis <= 0.01 or not self._last_hands_state:
            return

        sv, sc, sa = self._build_hand_pointers(self._last_hands_state, aspect)
        if sv.shape[0] == 0:
            return

        t = self._t
        # Per-vertex alpha with pulsing
        # Pointers don't get scanline - they're screen-aligned
        A = self._vis * self._fade
        rgb = _RGB[sc]
        # Use base alpha with subtle pulse
        pulse = 0.85 + 0.15 * np.sin(t * 12.0 + np.arange(len(sa)) * 0.1)
        alp = (_ALPHA[sc].reshape(-1, 1) * A * sa.reshape(-1, 1) * pulse.reshape(-1, 1))
        cols = np.concatenate([rgb, alp], 1)

        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        self._draw_arrays(GL_LINES, sv, cols, width=2.0)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()