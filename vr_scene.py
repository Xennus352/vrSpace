"""Full 3D VR world + a navigable player (immediate-mode GL).

The world is drawn under whatever MODELVIEW the renderer sets (player
camera + head rotation), so every object lives in world coordinates:

  * deep-space star shell (follows the player, so you can walk forever)
  * glowing grid deck that re-centers around the player
  * a family of planets: two Earth-likes, Mars, a gas giant plus a moon,
    and a bright distant sun — all self-lit from a fixed world sun dir
  * the live camera passthrough as a floating hologram window
  * a small astronaut avatar for the third-person view

VRPlayer steers first/third-person movement (see main.py for the keys).
"""

import math
import random
import ctypes

import numpy as np
from OpenGL.GL import *


_RAW = None


def _raw_gl():
    """Direct ctypes GL calls: PyOpenGL's pointer-tracker breaks on this
    llvmpipe environment ('no valid context' on client-array calls)."""
    global _RAW
    if _RAW is None:
        from OpenGL import platform
        g = platform.PLATFORM.GL
        cvt = ctypes.c_void_p
        ci = ctypes.c_int
        g.glVertexPointer.argtypes = [ci, ci, ci, cvt]
        g.glColorPointer.argtypes = [ci, ci, ci, cvt]
        g.glTexCoordPointer.argtypes = [ci, ci, ci, cvt]
        g.glDrawArrays.argtypes = [ci, ci, ci]
        g.glEnableClientState.argtypes = [ci]
        g.glDisableClientState.argtypes = [ci]
        _RAW = g
    return _RAW


# ----------------------------------------------------------------------
#  procedural planet textures
# ----------------------------------------------------------------------
def _gen_planet(w, h, build):
    """build(v, u, rng) -> (r, g, b) floats for every texel."""
    rng = np.random.default_rng(1234)
    arr = np.zeros((h, w, 3), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    v = ys / float(h - 1)          # 0 = north pole  (texture row 0)
    u = xs / float(w - 1)
    build(arr, v, u, rng)
    arr = np.clip(arr, 0.0, 1.0)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 np.ascontiguousarray((arr * 255.0).astype(np.uint8)).tobytes())
    return int(tex)


def _blobs(w, h, n, rng, rw_min=0.08, rw_max=0.26, rh_min=0.08, rh_max=0.3):
    """Sum of n gaussian blobs on a (h, w) grid (vectorised)."""
    f = np.zeros((h, w), np.float32)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    for _ in range(n):
        cx = rng.uniform(0.0, w - 1)
        cy = rng.uniform(0.0, h - 1)
        rw = rng.uniform(rw_min, rw_max) * w
        rh = rng.uniform(rh_min, rh_max) * h
        f += (np.exp(-((xs - cx) / rw) ** 2)[None, :]
              * np.exp(-((ys - cy) / rh) ** 2)[:, None])
    return f


def _make_earth(seed, cloudy=True):
    w, h = 512, 256
    def build(arr, v, u, rng):
        ocean = np.array([0.02, 0.10, 0.30])
        grass = np.array([0.16, 0.40, 0.17])
        dry = np.array([0.55, 0.45, 0.28])
        ice = np.array([0.90, 0.93, 1.00])
        arr[:] = ocean
        land = _blobs(w, h, 14, rng)
        land += _blobs(w, h, 6, rng, rw_min=0.3, rw_max=0.6, rh_min=0.3, rh_max=0.7) * 0.4
        f = np.clip((land - 0.28) / 0.5, 0.0, 1.0)
        arid = np.clip((land - 0.85) / 0.4, 0.0, 1.0)
        col = grass + (dry - grass) * arid[..., None]
        arr[:] = arr * (1.0 - f[..., None]) + col * f[..., None]
        ice_f = (np.clip((v - 0.80) / 0.12, 0.0, 1.0)
                 * (0.35 + 0.65 * f))
        ice_f += (np.clip((0.20 - v) / 0.12, 0.0, 1.0)
                  * (0.35 + 0.65 * f))
        ice_f = np.clip(ice_f, 0.0, 1.0)
        arr[:] = arr * (1.0 - ice_f[..., None]) + ice * ice_f[..., None]
        if cloudy:
            cl = _blobs(w, h, 24, rng, rw_min=0.04, rw_max=0.2, rh_min=0.03, rh_max=0.12)
            cloud = np.clip(cl / 0.9, 0.0, 1.0)
            arr[:] = arr * (1.0 - cloud[..., None]) + np.array([0.92, 0.95, 1.0]) * cloud[..., None]
        arr[:] += (rng.uniform(size=(h, w, 3)).astype(np.float32) - 0.5) * (0.015 if cloudy else 0.02)
    return _gen_planet(w, h, build)


def _make_mars():
    w, h = 512, 256
    def build(arr, v, u, rng):
        arr[:] = (0.63, 0.35, 0.20)
        dark = np.clip(_blobs(w, h, 12, rng, rw_min=0.1, rw_max=0.3, rh_min=0.08, rh_max=0.25) / 1.4, 0.0, 0.5)
        arr[:] *= (1.0 - dark[..., None] * 0.6)
        ice_f = np.clip((0.05 - v) / 0.05, 0.0, 1.0) + np.clip((v - 0.95) / 0.04, 0.0, 1.0)
        arr[:] = arr * (1.0 - np.clip(ice_f, 0, 1)[..., None]) + np.array([0.95, 0.92, 0.9]) * np.clip(ice_f, 0, 1)[..., None]
        arr[:] += (rng.uniform(size=(h, w, 3)).astype(np.float32) - 0.5) * 0.02
    return _gen_planet(w, h, build)


def _make_gas_giant():
    w, h = 512, 160
    def build(arr, v, u, rng):
        pal = np.array([
            [0.86, 0.76, 0.62], [0.66, 0.48, 0.36], [0.82, 0.70, 0.54],
            [0.50, 0.34, 0.28], [0.88, 0.80, 0.66], [0.58, 0.42, 0.34],
        ])
        n = pal.shape[0]
        t = v * (n - 1)
        i = np.clip(np.floor(t).astype(int), 0, n - 2)
        frac = (t - i)[..., None]
        grad = pal[i] + (pal[i + 1] - pal[i]) * frac
        swirl = np.sin(u * 2 * np.pi * 3.0) * 0.8
        bands = 0.5 + 0.5 * np.sin(v * 2 * np.pi * 9 + np.sin(v * 2 * np.pi * 21) * 1.3 + swirl)
        streaky = rng.uniform(size=(h, w, 3)).astype(np.float32) * 0.35
        arr[:] = grad * (0.45 + 0.55 * bands[..., None]) + streaky
        arr[:] = np.clip(arr, 0.0, 1.0)
    return _gen_planet(w, h, build)


def _make_moon():
    w, h = 256, 128
    def build(arr, v, u, rng):
        arr[:] = (0.46, 0.46, 0.50)
        craters = _blobs(w, h, 40, rng, rw_min=0.01, rw_max=0.05, rh_min=0.01, rh_max=0.05) * 1.2
        arr[:] *= (1.0 - np.clip(craters, 0, 1)[..., None] * 0.35)
        arr[:] += (rng.uniform(size=(h, w, 3)).astype(np.float32) - 0.5) * 0.02
    return _gen_planet(w, h, build)


def _make_sun():
    w, h = 128, 128
    def build(arr, v, u, rng):
        yy = np.arange(h, dtype=np.float32)
        xx = np.arange(w, dtype=np.float32)
        d2 = (xx - w / 2)[None, :] ** 2 + (yy - h / 2)[:, None] ** 2
        core = np.exp(-d2 / (0.5 * (w / 2) ** 2))
        arr[:] = np.array([1.0, 0.82, 0.45]) * core[..., None]
        arr[:] += np.array([0.55, 0.30, 0.10]) * (1.0 - core)[..., None]
    return _gen_planet(w, h, build)


# ----------------------------------------------------------------------
#  unit sphere mesh (shared): vertex == outward normal
# ----------------------------------------------------------------------
def _build_sphere(lat_seg=22, lon_seg=36):
    quads = []
    for i in range(lat_seg):
        lat0 = math.pi * (0.5 - i / float(lat_seg))
        lat1 = math.pi * (0.5 - (i + 1) / float(lat_seg))
        c0, s0 = math.cos(lat0), math.sin(lat0)
        c1, s1 = math.cos(lat1), math.sin(lat1)
        v0 = i / float(lat_seg)
        v1 = (i + 1) / float(lat_seg)
        for j in range(lon_seg):
            lon0 = 2.0 * math.pi * j / float(lon_seg)
            lon1 = 2.0 * math.pi * (j + 1) / float(lon_seg)
            q = (
                (math.cos(lon0) * c0, s0, math.sin(lon0) * c0),   # NW
                (math.cos(lon1) * c0, s0, math.sin(lon1) * c0),   # NE
                (math.cos(lon1) * c1, s1, math.sin(lon1) * c1),   # SE
                (math.cos(lon0) * c1, s1, math.sin(lon0) * c1),   # SW
                j / float(lon_seg), (j + 1) / float(lon_seg), v0, v1,
            )
            quads.append(q)
    return quads


def _rot_y(v, a):
    ca, sa = math.cos(a), math.sin(a)
    return (v[0] * ca + v[2] * sa, v[1], -v[0] * sa + v[2] * ca)


def _rot_z(v, a):
    ca, sa = math.cos(a), math.sin(a)
    return (v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca, v[2])


class _Planet:
    def __init__(self, tex, radius, pos, tilt=0.0, rot_speed=0.0,
                 atmosphere=0.0, glow=0.0, orbit=None, lat_seg=14, lon_seg=24):
        self.tex = tex
        self.radius = radius
        self.pos = pos
        self.tilt = tilt
        self.rot_speed = rot_speed
        self.atmos = atmosphere        # extra additive shell factor (0-none)
        self.glow = glow               # >0 draws an additive corona sphere
        self.orbit = orbit             # (axis, radius, speed) around self.pos

        m = _build_sphere(lat_seg, lon_seg)
        n = len(m) * 4
        xyzu = np.empty((n, 5), np.float32)
        unit = np.empty((n, 3), np.float32)
        k = 0
        for (a, b, c, d, u0, u1, v0, v1) in m:
            for p, uu, vv in ((a, u0, v0), (b, u1, v0), (c, u1, v1), (d, u0, v1)):
                xyzu[k, 0:3] = p
                xyzu[k, 3] = uu
                xyzu[k, 4] = vv
                unit[k] = p
                k += 1
        self._unit = unit
        self._n = n
        self._col = np.empty((n, 4), np.float32)
        self._vb = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vb)
        act = xyzu.copy(); act[:, 0:3] *= radius
        glBufferData(GL_ARRAY_BUFFER, act.nbytes, act, GL_STATIC_DRAW)
        self._cb = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._cb)
        glBufferData(GL_ARRAY_BUFFER, self._col.nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._shell_vb = None
        if atmosphere > 0.0 or glow > 0.0:
            self._shell_vb = int(glGenBuffers(1))
            glBindBuffer(GL_ARRAY_BUFFER, self._shell_vb)
            sh = xyzu.copy(); sh[:, 0:3] *= radius * (1.07 if atmosphere else 1.22)
            glBufferData(GL_ARRAY_BUFFER, sh.nbytes, sh, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

    def world_pos(self, t):
        if self.orbit is None:
            return self.pos
        axis, radius, speed = self.orbit
        a = speed * t
        ca, sa = math.cos(a), math.sin(a)
        ox, oy, oz = self.pos
        return (ox + axis[0] * radius * ca, oy + axis[1] * radius * sa, oz + axis[2] * radius * ca)

    def draw(self, t):
        x, y, z = self.world_pos(t)
        rot = self.rot_speed * t

        # sun direction in the planet's rotated frame (tilt then spin)
        sun = np.array(_rot_z(_rot_y((0.45, 0.28, 0.85), -rot), -self.tilt),
                       dtype=np.float32)

        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(math.degrees(rot), 0.0, 1.0, 0.0)
        glRotatef(math.degrees(self.tilt), 0.0, 0.0, 1.0)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        n = self._n
        raw = _raw_gl()

        # body: per-vertex sun shading + limb darkening (single draw call)
        sh = self._unit @ sun
        sh = np.maximum(sh, 0.0)
        lam = (0.32 + 0.68 * sh) * (0.55 + 0.45 * sh * sh)
        col = self._col
        col[:, 0] = lam; col[:, 1] = lam; col[:, 2] = lam; col[:, 3] = 1.0
        glBindBuffer(GL_ARRAY_BUFFER, self._cb)
        glBufferData(GL_ARRAY_BUFFER, col.nbytes, col, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vb)
        raw.glEnableClientState(GL_VERTEX_ARRAY)
        raw.glEnableClientState(GL_COLOR_ARRAY)
        raw.glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        raw.glVertexPointer(3, GL_FLOAT, 5 * 4, ctypes.c_void_p(0))
        raw.glTexCoordPointer(2, GL_FLOAT, 5 * 4, ctypes.c_void_p(12))
        glBindBuffer(GL_ARRAY_BUFFER, self._cb)
        raw.glColorPointer(4, GL_FLOAT, 0, ctypes.c_void_p(0))
        raw.glDrawArrays(GL_QUADS, 0, n)
        glBindBuffer(GL_ARRAY_BUFFER, self._vb)
        raw.glDisableClientState(GL_COLOR_ARRAY)

        if self._shell_vb is not None:
            shell_rgba = (0.35, 0.62, 1.0, self.atmos) if self.atmos > 0.0 \
                else (0.95, 0.78, 0.45, self.glow)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glDepthMask(GL_FALSE)
            glColor4f(*shell_rgba)
            glBindBuffer(GL_ARRAY_BUFFER, self._shell_vb)
            raw.glVertexPointer(3, GL_FLOAT, 5 * 4, ctypes.c_void_p(0))
            raw.glTexCoordPointer(2, GL_FLOAT, 5 * 4, ctypes.c_void_p(12))
            raw.glDrawArrays(GL_QUADS, 0, n)
            glDepthMask(GL_TRUE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_BLEND)
            glBindBuffer(GL_ARRAY_BUFFER, self._vb)

        raw.glDisableClientState(GL_VERTEX_ARRAY)
        raw.glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()


# ----------------------------------------------------------------------
#  player (movement + view state)
# ----------------------------------------------------------------------
class VRPlayer:
    """WASD/space flight around the world. ``facing`` is the base heading
    (radians, positive = left), combined at render time with webcam head yaw.
    Movement uses small acceleration / inertia so locomotion feels physical
    instead of instant, and a speed-scaled head bob."""

    def __init__(self):
        self.pos = np.array([0.0, 0.0, 3.0])
        self.facing = 0.0
        self.pitch = 0.0
        self.speed = 7.0
        self.vert_speed = 5.0
        self.eye_height = 1.6
        self.bob = 0.0
        self.sway = 0.0
        self.moving = False
        self._hvel = np.zeros(2, dtype=np.float64)   # (x, z) body velocity
        self._vvel = 0.0
        self._bob_t = 0.0
        self._bob_speed = 1.0

    def move(self, dt, c):
        head = c.get("head_yaw", 0.0)
        yaw = self.facing + head
        fx, fz = -math.sin(yaw), -math.cos(yaw)
        rx, rz = math.cos(yaw), -math.sin(yaw)
        vx = vz = 0.0
        if c.get("fwd"):   vx += fx; vz += fz
        if c.get("back"):  vx -= fx; vz -= fz
        if c.get("left"):  vx -= rx; vz -= rz
        if c.get("right"): vx += rx; vz += rz
        speed = self.speed * (1.7 if c.get("sprint") else 1.0)
        n = math.hypot(vx, vz)
        if n > 0.0:
            vx = vx / n * speed
            vz = vz / n * speed

        # ease velocity toward the demand (physical acceleration)
        ramp = min(1.0, dt * 8.0)
        self._hvel += (np.array([vx, vz]) - self._hvel) * ramp
        self.pos[0] += self._hvel[0] * dt
        self.pos[2] += self._hvel[1] * dt

        # vertical drift (fly up/down), clamped to the deck
        vy = (1.0 if c.get("up") else 0.0) - (1.0 if c.get("down") else 0.0)
        self._vvel += (vy * self.vert_speed - self._vvel) * min(1.0, dt * 10.0)
        self.pos[1] += self._vvel * dt
        if self.pos[1] < 0.0:
            self.pos[1] = 0.0
            self._vvel = max(0.0, self._vvel)

        # speed-scaled head bob + subtle body sway
        speed_now = float(np.hypot(*self._hvel))
        self.moving = speed_now > 0.25
        if self.moving:
            self._bob_t += dt * speed_now
            self._bob_speed = self._bob_t * 2.4
            self.bob = math.sin(self._bob_speed) * 0.045
            self.sway = math.cos(self._bob_speed) * 0.012
        else:
            self._bob_t = 0.0
            self.bob = 0.0
            self.sway = 0.0

        # smooth keyboard turning / look
        self.facing += c.get("turn", 0.0) * dt * 1.7
        self.pitch += c.get("pitch", 0.0) * dt * 1.4
        self.pitch = max(-1.4, min(1.4, self.pitch))


# ----------------------------------------------------------------------
#  VRWorld
# ----------------------------------------------------------------------
class VRWorld:
    def __init__(self, seed=7, star_count=320):
        rng = random.Random(seed)
        pts = []
        cols = []
        for _ in range(star_count):
            u = rng.uniform(-1.0, 1.0)
            theta = rng.uniform(0.0, 2.0 * math.pi)
            r = math.sqrt(max(0.0, 1.0 - u * u))
            dist = rng.uniform(40.0, 110.0)
            bright = rng.uniform(0.45, 1.3)
            pts.append((r * math.cos(theta) * dist, u * dist, r * math.sin(theta) * dist))
            cols.append((bright, bright, bright + 0.05))
        self._star_base = np.asarray(pts, dtype=np.float32)      # (N,3) world pos
        self._star_cols = np.asarray(cols, dtype=np.float32)     # (N,3)
        self._star_pos = self._star_base.copy()
        self._star_buf = self._star_cols.copy()
        self._grid_step = 1.6
        self._grid_half = 40
        self._twinkle = 0.0

        self.earth_tex = None
        self.earth2_tex = None
        self.mars_tex = None
        self.gas_tex = None
        self.moon_tex = None
        self.sun_tex = None
        self._planets = None

    def _ensure_textures(self):
        if self._planets is not None:
            return
        self.earth_tex = _make_earth(7)
        self.earth2_tex = _make_earth(99, cloudy=False)
        self.mars_tex = _make_mars()
        self.gas_tex = _make_gas_giant()
        self.moon_tex = _make_moon()
        self.sun_tex = _make_sun()

        earth_pos = (8.0, 8.0, -24.0)
        self._planets = [
            _Planet(self.sun_tex, 30.0, (34.0, 55.0, -175.0), glow=0.32),
            _Planet(self.moon_tex, 1.25, earth_pos, rot_speed=0.02, orbit=((0.0, 0.3, 1.0), 8.5, 0.45), atmosphere=0.0),
            _Planet(self.earth_tex, 6.0, earth_pos, tilt=0.40, rot_speed=0.12, atmosphere=0.16),
            _Planet(self.earth2_tex, 3.4, (-17.0, 5.5, -15.0), tilt=0.25, rot_speed=0.09, atmosphere=0.1),
            _Planet(self.mars_tex, 11.0, (-72.0, 24.0, -95.0), tilt=0.42, rot_speed=0.06, atmosphere=0.03),
            _Planet(self.gas_tex, 20.0, (85.0, 34.0, -170.0), tilt=0.14, rot_speed=0.02),
        ]

    def draw(self, live_texture=None, has_live=False, t=0.0, player_offset=(0.0, 0.0)):
        """``player_offset`` = (x, z) used to re-center the star shell and the
        grid deck under the moving player (infinite space station)."""
        self._twinkle = t
        self._ensure_textures()
        ox, oz = player_offset

        # ---- star shell (translates with the player, single batched draw) --
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        pulse = min(1.0, 0.55 + 0.45 * math.sin(t * 1.7))
        pos = self._star_pos
        pos[:] = self._star_base
        pos[:, 0] += ox
        pos[:, 1] += 2.0
        pos[:, 2] += oz
        buf = self._star_buf
        buf[:] = self._star_cols * pulse
        raw = _raw_gl()
        glPointSize(2.0)
        raw.glEnableClientState(GL_VERTEX_ARRAY)
        raw.glEnableClientState(GL_COLOR_ARRAY)
        raw.glVertexPointer(3, GL_FLOAT, 0, pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        raw.glColorPointer(3, GL_FLOAT, 0, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        raw.glDrawArrays(GL_POINTS, 0, pos.shape[0])
        raw.glDisableClientState(GL_COLOR_ARRAY)
        raw.glDisableClientState(GL_VERTEX_ARRAY)
        glPointSize(1.0)

        # distant sun glow washing over the horizon
        glColor4f(0.18, 0.14, 0.10, 0.28)
        glBegin(GL_QUADS)
        glVertex3f(-90 + ox, -2.0, -140 + oz)
        glVertex3f(90 + ox, -2.0, -140 + oz)
        glVertex3f(90 + ox, 14.0, -140 + oz)
        glVertex3f(-90 + ox, 14.0, -140 + oz)
        glEnd()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        # ---- glowing grid deck (re-centered on the player) ----------
        gx = math.floor(ox / self._grid_step + 0.5) * self._grid_step
        gz = math.floor(oz / self._grid_step + 0.5) * self._grid_step
        half = self._grid_half
        glLineWidth(1.0)
        glColor4f(0.18, 0.65, 1.0, 0.16)
        glBegin(GL_LINES)
        for i in range(-half, half + 1):
            x = gx + i * self._grid_step
            glVertex3f(x, 0.0, gz - half * self._grid_step)
            glVertex3f(x, 0.0, gz + half * self._grid_step)
        for i in range(-half, half + 1):
            z = gz + i * self._grid_step
            glVertex3f(gx - half * self._grid_step, 0.0, z)
            glVertex3f(gx + half * self._grid_step, 0.0, z)
        glEnd()

        # faint open-floor panels so the deck reads as a platform
        glColor4f(0.05, 0.2, 0.35, 0.08)
        glBegin(GL_QUADS)
        for i in range(-half + 1, half, 4):
            x0 = gx + (i - 1) * self._grid_step
            x1 = gx + (i + 1) * self._grid_step
            for j in range(-half + 1, half, 4):
                z0 = gz + (j - 1) * self._grid_step
                z1 = gz + (j + 1) * self._grid_step
                glVertex3f(x0, 0.0, z0)
                glVertex3f(x1, 0.0, z0)
                glVertex3f(x1, 0.0, z1)
                glVertex3f(x0, 0.0, z1)
        glEnd()

        # centre-line runway accent toward forward (-Z)
        glLineWidth(2.0)
        glColor4f(0.35, 0.85, 1.0, 0.30)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 2.0)
        glVertex3f(0.0, 0.0, -self._grid_half * self._grid_step)
        glEnd()
        glLineWidth(1.0)

        # ---- planets ------------------------------------------------
        for planet in self._planets:
            planet.draw(t)

        glDisable(GL_BLEND)

    # ------------------------------------------------------------------
    def draw_avatar(self, facing):
        """Small astronaut at the origin (feet at y=0). Renderer translates
        to the player position. Faces -Z; ``facing`` rotates it.
        Flat-shaded (fixed-function lighting is very slow on llvmpipe)."""
        glPushMatrix()
        glRotatef(math.degrees(facing), 0.0, 1.0, 0.0)

        suit = (0.86, 0.88, 0.90)
        dark = (0.13, 0.17, 0.23)
        cyan = (0.25, 0.90, 1.0)

        # legs
        glColor3f(*dark)
        self._box(0.0, 0.28, 0.0, 0.34, 0.56, 0.30)
        # torso
        glColor3f(*suit)
        self._box(0.0, 0.82, 0.0, 0.52, 0.56, 0.34)
        # backpack
        glColor3f(0.45, 0.48, 0.52)
        self._box(0.0, 1.0, 0.42, 0.46, 0.62, 0.18)
        # chest accent
        glColor3f(*cyan)
        self._box(0.0, 0.92, -0.18, 0.18, 0.14, 0.04)
        # shoulder pads
        glColor3f(*suit)
        self._box(0.42, 1.16, 0.02, 0.14, 0.16, 0.30)
        self._box(-0.42, 1.16, 0.02, 0.14, 0.16, 0.30)
        # arms
        glColor3f(0.72, 0.76, 0.80)
        self._box(0.52, 0.92, -0.08, 0.12, 0.56, 0.14)
        self._box(-0.52, 0.92, -0.08, 0.12, 0.56, 0.14)
        # gloves
        glColor3f(*cyan)
        self._box(0.52, 0.60, -0.16, 0.10, 0.12, 0.12)
        self._box(-0.52, 0.60, -0.16, 0.10, 0.12, 0.12)
        # helmet sphere + visor
        glColor3f(0.95, 0.97, 1.0)
        glPushMatrix()
        glTranslatef(0.0, 1.68, 0.0)
        self._sphere_quads(0.30, 10, 14)
        glPopMatrix()
        glColor3f(0.05, 0.08, 0.12)
        self._box(0.0, 1.66, -0.28, 0.32, 0.20, 0.04)

        glPopMatrix()

    @staticmethod
    def _box(cx, cy, cz, sx, sy, sz):
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2
        glBegin(GL_QUADS)
        # +X
        for p in ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)): glVertex3f(*p)
        # -X
        for p in ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)): glVertex3f(*p)
        # +Y
        for p in ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)): glVertex3f(*p)
        # -Y
        for p in ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)): glVertex3f(*p)
        # +Z
        for p in ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)): glVertex3f(*p)
        # -Z
        for p in ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)): glVertex3f(*p)
        glEnd()

    @staticmethod
    def _sphere_quads(radius, lat_seg, lon_seg):
        glBegin(GL_QUADS)
        for i in range(lat_seg):
            lat0 = math.pi * (0.5 - i / float(lat_seg))
            lat1 = math.pi * (0.5 - (i + 1) / float(lat_seg))
            c0, s0 = math.cos(lat0), math.sin(lat0)
            c1, s1 = math.cos(lat1), math.sin(lat1)
            for j in range(lon_seg):
                lon0 = 2.0 * math.pi * j / float(lon_seg)
                lon1 = 2.0 * math.pi * (j + 1) / float(lon_seg)
                for p in (
                    (math.cos(lon0) * c0, s0, math.sin(lon0) * c0),
                    (math.cos(lon1) * c0, s0, math.sin(lon1) * c0),
                    (math.cos(lon1) * c1, s1, math.sin(lon1) * c1),
                    (math.cos(lon0) * c1, s1, math.sin(lon0) * c1),
                ):
                    glVertex3f(p[0] * radius, p[1] * radius, p[2] * radius)
        glEnd()