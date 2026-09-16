"""Iron Man VR — gesture-driven flight + repulsor combat (PyOpenGL immediate mode).

Inspired by Camouflaj's Iron Man VR (Quest 2).  Pilot the suit with open
palms, fire repulsor beams by pointing and pinching, and destroy waves of
drones over a stylized city skyline.

Toggle on/off from main.py with the V key.
"""

import math
import random
import time
import ctypes

import numpy as np
from OpenGL.GL import *


# ─── colour palette ──────────────────────────────────────────────────
C_ORANGE = (1.0, 0.55, 0.12)
C_BLUE   = (0.18, 0.65, 1.0)
C_CYAN   = (0.25, 0.90, 1.0)
C_RED    = (1.0, 0.18, 0.15)
C_WHITE  = (1.0, 1.0, 1.0)
C_DARK   = (0.06, 0.08, 0.12)
C_GRID   = (0.08, 0.40, 0.55)


# ─── helper: raw GL calls (same approach as vr_scene.py) ────────────
def _raw_gl():
    from OpenGL import platform
    g = platform.PLATFORM.GL
    g.glVertexPointer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    g.glColorPointer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    g.glEnableClientState.argtypes = [ctypes.c_int]
    g.glDisableClientState.argtypes = [ctypes.c_int]
    return g


# ═════════════════════════════════════════════════════════════════════
#  Drone enemy
# ═════════════════════════════════════════════════════════════════════
class Drone:
    """Simple AI drone — orbits the player, occasionally shoots."""

    def __init__(self, kind="laser", pos=None, orbit_r=18.0, orbit_speed=0.3):
        self.kind = kind          # "laser" | "ramming" | "swarm" | "boss"
        self.pos = np.array(pos if pos else [0.0, 6.0, -20.0], np.float64)
        self.vel = np.zeros(3, np.float64)
        self.alive = True
        self.hp = 2 if kind == "laser" else (3 if kind == "boss" else 1)
        self.radius = 0.6 if kind != "boss" else 1.8
        self.orbit_r = orbit_r
        self.orbit_speed = orbit_speed
        self._phase = random.uniform(0.0, 6.28)
        self._shoot_cooldown = random.uniform(1.5, 3.5)
        self._bob = random.uniform(0.0, 6.28)
        self.flash = 0.0   # hit-flash timer

    def update(self, dt, player_pos):
        self._phase += self.orbit_speed * dt
        self._bob += 1.8 * dt
        self.flash = max(0.0, self.flash - dt * 4.0)

        # orbit target
        tx = player_pos[0] + math.cos(self._phase) * self.orbit_r
        tz = player_pos[2] + math.sin(self._phase) * self.orbit_r
        ty = 5.0 + 2.5 * math.sin(self._bob)

        if self.kind == "ramming":
            # charge straight at the player
            to_p = player_pos - self.pos
            n = np.linalg.norm(to_p)
            if n > 0.5:
                self.vel = self.vel * 0.92 + (to_p / n) * 14.0 * dt
        else:
            # smooth pursuit toward orbit point
            target = np.array([tx, ty, tz])
            self.vel += (target - self.pos) * 3.5 * dt

        self.pos += self.vel * dt

        self._shoot_cooldown -= dt
        return None  # caller checks shoot

    def wants_to_shoot(self):
        if self._shoot_cooldown > 0:
            return False
        self._shoot_cooldown = random.uniform(1.2, 3.0) if self.kind != "boss" else random.uniform(0.6, 1.2)
        return True

    def draw(self, t):
        glPushMatrix()
        glTranslatef(*self.pos)
        r = self.radius
        flash = 0.3 + 0.7 * self.flash

        if self.kind == "boss":
            glColor3f(0.9 * flash, 0.15 * flash, 0.15)
            _box(0, 0, 0, r * 2.4, r * 0.8, r * 2.4)
            # visor
            glColor3f(0.9, 0.2, 0.2)
            _box(0, 0, -r * 1.3, r * 1.6, r * 0.4, 0.15)
            # wings
            glColor3f(0.2, 0.2, 0.25)
            _box(r * 1.6, 0, 0, r * 1.2, 0.15, r * 1.2)
            _box(-r * 1.6, 0, 0, r * 1.2, 0.15, r * 1.2)
        elif self.kind == "ramming":
            glColor3f(0.85 * flash, 0.35 * flash, 0.1)
            _box(0, 0, 0, r * 1.2, r * 0.5, r * 1.8)
            # spiky nose
            glColor3f(1.0, 0.2, 0.1)
            _box(0, 0, -r * 1.2, r * 0.4, r * 0.4, r * 0.8)
        elif self.kind == "swarm":
            glColor3f(0.4 * flash, 0.85 * flash, 0.2)
            # four tiny wings
            for dx, dz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                _box(dx * r * 0.6, 0, dz * r * 0.6, r * 0.5, 0.08, r * 0.5)
            _box(0, 0, 0, r * 0.4, r * 0.3, r * 0.4)
        else:  # laser
            glColor3f(0.3 * flash, 0.5 * flash, 0.95)
            _box(0, 0, 0, r, r * 0.4, r * 1.4)
            # antenna
            glColor3f(0.6, 0.6, 0.65)
            _box(0, r * 0.5, 0, 0.04, r * 0.5, 0.04)

        # eye / sensor glow
        glow = 0.6 + 0.4 * math.sin(t * 4.0)
        glColor3f(C_RED[0] * glow, C_RED[1] * glow, C_RED[2] * glow)
        _sphere(0.18 if self.kind != "boss" else 0.5, 6, 8)

        glPopMatrix()

    def hit_test(self, point, radius):
        return np.linalg.norm(self.pos - point) < (self.radius + radius)


# ═════════════════════════════════════════════════════════════════════
#  Projectile (repulsor beam or enemy laser)
# ═════════════════════════════════════════════════════════════════════
class Projectile:
    def __init__(self, pos, vel, kind="repulsor", owner="player"):
        self.pos = np.array(pos, np.float64)
        self.vel = np.array(vel, np.float64)
        self.kind = kind          # "repulsor" | "laser" | "unibeam"
        self.owner = owner        # "player" | "enemy"
        self.alive = True
        self.age = 0.0
        self.speed = np.linalg.norm(self.vel)
        self.radius = 0.12 if kind == "repulsor" else (0.5 if kind == "unibeam" else 0.08)

    def update(self, dt):
        self.pos += self.vel * dt
        self.age += dt
        if self.age > 4.0 or abs(self.pos[1]) > 60 or abs(self.pos[0]) > 80:
            self.alive = False

    def draw(self, t):
        glPushMatrix()
        glTranslatef(*self.pos)
        glow = 0.7 + 0.3 * math.sin(t * 20.0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)

        if self.kind == "unibeam":
            glRotatef(math.degrees(math.atan2(self.vel[2], self.vel[0])), 0, 1, 0)
            glColor4f(C_BLUE[0], C_BLUE[1], C_BLUE[2], 0.85 * glow)
            _box(0, 0, 0, 0.12, 0.12, 0.6)
            # core glow
            glColor4f(1.0, 1.0, 1.0, 0.9 * glow)
            _box(0, 0, 0, 0.05, 0.05, 0.5)
        elif self.kind == "repulsor":
            glColor4f(C_ORANGE[0], C_ORANGE[1], C_ORANGE[2], 0.9 * glow)
            _sphere(self.radius * 1.5, 5, 7)
            glColor4f(1.0, 0.9, 0.6, 1.0 * glow)
            _sphere(self.radius * 0.6, 4, 5)
        else:  # enemy laser
            glColor4f(0.9, 0.15, 0.1, 0.85 * glow)
            _sphere(self.radius, 4, 5)

        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glPopMatrix()


# ═════════════════════════════════════════════════════════════════════
#  Tiny geometry helpers
# ═════════════════════════════════════════════════════════════════════
def _box(cx, cy, cz, sx, sy, sz):
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    glBegin(GL_QUADS)
    for (a, b, c, d) in (
        ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),
        ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)),
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
        ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)),
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),
    ):
        glVertex3f(*a)
        glVertex3f(*b)
        glVertex3f(*c)
        glVertex3f(*d)
    glEnd()


def _sphere(radius, lat_seg, lon_seg):
    glBegin(GL_QUADS)
    for i in range(lat_seg):
        lat0 = math.pi * (0.5 - i / float(lat_seg))
        lat1 = math.pi * (0.5 - (i + 1) / float(lat_seg))
        c0, s0 = math.cos(lat0), math.sin(lat0)
        c1, s1 = math.cos(lat1), math.sin(lat1)
        for j in range(lon_seg):
            a0 = 2.0 * math.pi * j / float(lon_seg)
            a1 = 2.0 * math.pi * (j + 1) / float(lon_seg)
            glVertex3f(radius * s0 * math.cos(a0), radius * c0, radius * s0 * math.sin(a0))
            glVertex3f(radius * s0 * math.cos(a1), radius * c0, radius * s0 * math.sin(a1))
            glVertex3f(radius * s1 * math.cos(a1), radius * c1, radius * s1 * math.sin(a1))
            glVertex3f(radius * s1 * math.cos(a0), radius * c1, radius * s1 * math.sin(a0))
    glEnd()


def _dome(radius, lat_seg, lon_seg, color_fn, center=(0.0, 0.0, 0.0)):
    """Sphere colored per-vertex via color_fn(y) with y in [-1, 1]."""
    px, py, pz = center
    glBegin(GL_QUADS)
    for i in range(lat_seg):
        lat0 = math.pi * (0.5 - i / float(lat_seg))
        lat1 = math.pi * (0.5 - (i + 1) / float(lat_seg))
        y0, y1 = math.sin(lat0), math.sin(lat1)
        r0, r1 = math.cos(lat0) * radius, math.cos(lat1) * radius
        for j in range(lon_seg):
            a0 = 2.0 * math.pi * j / float(lon_seg)
            a1 = 2.0 * math.pi * (j + 1) / float(lon_seg)
            ax0, az0 = math.cos(a0), math.sin(a0)
            ax1, az1 = math.cos(a1), math.sin(a1)
            glColor3f(*color_fn(y0))
            glVertex3f(px + r0 * ax0, py + y0 * radius, pz + r0 * az0)
            glVertex3f(px + r0 * ax1, py + y0 * radius, pz + r0 * az1)
            glColor3f(*color_fn(y1))
            glVertex3f(px + r1 * ax1, py + y1 * radius, pz + r1 * az1)
            glVertex3f(px + r1 * ax0, py + y1 * radius, pz + r1 * az0)
    glEnd()


def _dusk_color(y):
    """Dusk gradient: warm horizon band, navy overhead, near-black below."""
    if y >= 0.0:
        t = min(1.0, y * 2.4)   # warm band hugs the horizon
        return (
            0.36 - 0.33 * t,
            0.16 - 0.12 * t,
            0.07 + 0.05 * t,
        )
    t = min(1.0, -y * 2.4)
    return (
        0.36 - 0.35 * t,
        0.16 - 0.14 * t,
        0.07 - 0.04 * t,
    )


# ═════════════════════════════════════════════════════════════════════
#  City skyline (Stark Tower + buildings)
# ═════════════════════════════════════════════════════════════════════
class CitySkyline:
    def __init__(self, seed=42):
        rng = random.Random(seed)
        self._buildings = []
        for _ in range(150):
            x = rng.uniform(-130, 130)
            z = rng.uniform(-170, -12)
            h = rng.uniform(3.0, 34.0)
            w = rng.uniform(1.2, 4.2)
            d = rng.uniform(1.2, 4.2)
            grey = rng.uniform(0.12, 0.42)
            self._buildings.append((x, h, z, w, d, grey))
        # Stark Tower stands proud of the skyline
        self._stark = (0.0, 46.0, -80.0, 6.0, 6.0)

    def draw(self, t, player_pos):
        ox, _, oz = player_pos

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # ground plane (dark city floor, follows the player)
        glColor4f(0.015, 0.02, 0.035, 0.92)
        glBegin(GL_QUADS)
        glVertex3f(-140 + ox, -0.05, -180 + oz)
        glVertex3f(140 + ox, -0.05, -180 + oz)
        glVertex3f(140 + ox, -0.05, 40 + oz)
        glVertex3f(-140 + ox, -0.05, 40 + oz)
        glEnd()

        # block grid so motion through the streets reads clearly
        glColor4f(0.05, 0.07, 0.12, 0.5)
        glBegin(GL_LINES)
        for g in range(-140, 141, 20):
            glVertex3f(g + ox, -0.03, -180 + oz)
            glVertex3f(g + ox, -0.03, 40 + oz)
        for g in range(-180, 41, 20):
            glVertex3f(-140 + ox, -0.03, g + oz)
            glVertex3f(140 + ox, -0.03, g + oz)
        glEnd()

        # buildings (haze blue the further away they are)
        for bx, bh, bz, bw, bd, grey in self._buildings:
            wx, wz = bx + ox * 0.3, bz + oz * 0.3
            dist = math.sqrt((wx - ox) ** 2 + (wz - oz) ** 2)
            haze = min(0.75, dist / 260.0)
            # dusk + haze tint
            glColor3f(
                grey * 0.7 * (1.0 - haze) + 0.35 * haze,
                grey * 0.78 * (1.0 - haze) + 0.18 * haze,
                grey * 0.95 * (1.0 - haze) + 0.11 * haze,
            )
            _box(wx, bh / 2, wz, bw, bh, bd)
            # lit windows (random flicker)
            win_glow = 0.25 + 0.2 * math.sin(t * 1.7 + bx * 3.0 + bz * 2.0)
            glColor4f(0.95, 0.9, 0.55, win_glow)
            for wy in range(int(bh / 1.8)):
                _box(wx, 1.0 + wy * 1.8, wz - bd / 2 - 0.01, bw * 0.26, 0.35, 0.02)

        # Stark Tower
        sx, sh, sz, sw, sd = self._stark
        sx += ox * 0.3
        sz += oz * 0.3
        glColor3f(0.09, 0.11, 0.17)
        _box(sx, sh / 2, sz, sw, sh, sd)
        # arc reactor glow on top
        pulse = 0.6 + 0.4 * math.sin(t * 3.0)
        glPushMatrix()
        glTranslatef(sx, sh + 1.2, sz)
        glColor4f(0.2, 0.7 * pulse, 1.0, 0.85)
        _sphere(1.4, 8, 10)
        glPopMatrix()
        # antenna
        glColor3f(0.3, 0.33, 0.38)
        _box(sx, sh + 3.4, sz, 0.12, 5.0, 0.12)
        # side glow strips
        glColor4f(0.12, 0.5 * pulse, 0.9, 0.5)
        _box(sx - sw / 2 - 0.03, sh * 0.6, sz, 0.07, sh * 0.8, sd * 0.8)
        _box(sx + sw / 2 + 0.03, sh * 0.6, sz, 0.07, sh * 0.8, sd * 0.8)
        # STARK sign
        glColor4f(0.2, 0.7, 1.0, 0.85)
        _box(sx, sh * 0.93, sz - sd / 2 - 0.06, sw * 0.75, 0.7, 0.04)

        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)


# ═════════════════════════════════════════════════════════════════════
#  FRIDAY dialogue system
# ═════════════════════════════════════════════════════════════════════
class FridayVoice:
    def __init__(self, ai=None):
        self._lines = []
        self._current = None
        self._timer = 0.0
        self.ai = ai
        self._last_ai_shown = None
        self.commands = {}          # action_id -> callable returning optional speech
        self.context_fn = None      # callable() -> situation string

    def _ensure_ai(self):
        if self.ai is None:
            try:
                from friday_ai import FridayAI
                self.ai = FridayAI()
            except Exception:
                self.ai = None
        return self.ai

    def say(self, text, duration=3.0):
        self._lines.append((text, duration))

    def chat(self, user_text):
        """User spoke to FRIDAY (via mic or typed). Returns reply text."""
        ai = self._ensure_ai()
        if ai is None:
            err = "FRIDAY is offline, boss."
            self.say(err, 3.0)
            return err
        situation = self.context_fn() if self.context_fn else ""
        reply, action = ai.conversation(user_text, situation)
        if reply:
            self.say(reply, 4.0)
        if action:
            try:
                handler = self.commands.get(action)
                if handler is not None:
                    extra = handler()
                    if extra:
                        ai.speak(extra)
                        self.say(extra, 4.0)
            except Exception as exc:
                err = "Command failed."
                print("[friday] command error:", exc)
                ai.speak(err)
                self.say(err, 3.0)
        return reply

    def broadcast(self, scope, **kw):
        """Queue a line on the HUD and fire the voice assistant (if any)."""
        ai = self._ensure_ai()
        if ai is not None:
            line = ai.broadcast(scope, **kw)
        else:
            line = None
        if line:
            self.say(line, 3.0)

    def update(self, dt):
        self._timer -= dt
        # surface AI lines on the HUD as FRIDAY produces them
        if self.ai is not None and self.ai.latest \
                and self.ai.latest != self._last_ai_shown:
            self._last_ai_shown = self.ai.latest
            self._lines.insert(0, (self.ai.latest, 3.5))
        if self._timer <= 0 and self._lines:
            self._current, self._timer = self._lines.pop(0)

    @property
    def current(self):
        return self._current


# ═════════════════════════════════════════════════════════════════════
#  Main Iron Man Game
# ═════════════════════════════════════════════════════════════════════
class IronManGame:
    """Iron Man VR — the full game state, update, and draw."""

    def __init__(self):
        self.active = False
        self.t = 0.0

        # player state
        self.pos = np.array([0.0, 15.0, 30.0], np.float64)
        self.vel = np.zeros(3, np.float64)
        self.facing = 0.0   # radians
        self.pitch = 0.0
        self.health = 100.0
        self.max_health = 100.0
        self._health_was_high = True
        self.energy = 100.0
        self.max_energy = 100.0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.wave_clear_time = 0.0

        # thruster / flight
        self.thrusting = False
        self.thrust_dir = np.array([0.0, -1.0, 0.0], np.float64)
        self._boost = 0.0

        # combat
        self.drones = []
        self.projectiles = []
        self._fire_cooldown = 0.0
        self._unibeam_charge = 0.0
        self._unibeam_ready = False
        self._click_left = False
        self._click_right = False
        self._repulsor_side = 0   # alternate left/right

        # suit-up intro
        self._suit_up_t = 0.0
        self._suit_up_done = False

        # environment
        self._city = CitySkyline()
        self._friday = FridayVoice()

        # palm tracking (set externally each frame)
        self._palm_left = None   # (nx, ny, nz) normalized camera coords
        self._palm_right = None
        self._gesture_left = "IDLE"
        self._gesture_right = "IDLE"
        self._ptr_left = None    # (wx, wy, wz) world aim point
        self._ptr_right = None

    # ------------------------------------------------------------------
    #  Activation
    # ------------------------------------------------------------------
    def activate(self):
        if self.active:
            return
        self.active = True
        self._reset()
        self._friday.broadcast("boot")
        self._friday.say("Time to suit up, boss.", 3.5)
        self._friday.say("Repulsors armed.  Unibeam charging.", 3.0)

    def deactivate(self):
        self.active = False

    def _reset(self):
        self.pos[:] = [0.0, 15.0, 30.0]
        self.vel[:] = [0, 0, 0]
        self.facing = 0.0
        self.pitch = 0.0
        self.health = self.max_health
        self._health_was_high = True
        self.energy = self.max_energy
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.drones.clear()
        self.projectiles.clear()
        self._suit_up_t = 0.0
        self._suit_up_done = False
        self._unibeam_charge = 0.0
        self._unibeam_ready = False
        self._click_left = False
        self._click_right = False
        self.wave_clear_time = 0.0

    # ------------------------------------------------------------------
    #  Hand input (called from handle_hands or main loop)
    # ------------------------------------------------------------------
    def set_hands(self, left_palm, right_palm, left_gesture, right_gesture,
                  left_ptr=None, right_ptr=None):
        self._palm_left = left_palm
        self._palm_right = right_palm
        self._gesture_left = left_gesture
        self._gesture_right = right_gesture
        self._ptr_left = left_ptr
        self._ptr_right = right_ptr

    def ask_friday(self):
        """Press F: FRIDAY gives a status/banter line + a control reminder."""
        self._friday.say(
            "Controls: open palms = fly  |  pinch = repulsors  |  "
            "both pinch = unibeam  |  fist = brake", 4.0)
        self._friday.broadcast(
            "ask", score=self.score, wave=self.wave, drones=len(self.drones))

    # ------------------------------------------------------------------
    #  Wave spawner
    # ------------------------------------------------------------------
    def _spawn_wave(self):
        self.wave += 1
        count = min(3 + self.wave * 2, 14)
        kinds = ["laser", "ramming", "swarm"]
        for i in range(count):
            kind = kinds[i % len(kinds)]
            angle = random.uniform(0, 6.28)
            r = random.uniform(15, 28)
            x = self.pos[0] + math.cos(angle) * r
            z = self.pos[2] + math.sin(angle) * r - 10
            self.drones.append(Drone(kind=kind, pos=[x, 5.0, z], orbit_r=r, orbit_speed=random.uniform(0.2, 0.6)))

        # boss on odd waves
        if self.wave % 3 == 0:
            self.drones.append(Drone(kind="boss", pos=[0, 12, -30], orbit_r=22, orbit_speed=0.15))

        self._friday.say(f"Wave {self.wave}: {count} hostiles detected.", 3.0)
        self._friday.broadcast("wave_start", wave=self.wave)
        if self.wave % 3 == 0:
            self._friday.broadcast("boss")

    # ------------------------------------------------------------------
    #  Flight mechanics (Tony Stark gesture-driven)
    # ------------------------------------------------------------------
    def _flight_from_hands(self, dt):
        """Derive thrust from palm positions.

        Gesture mapping (Iron Man VR style, in normalized camera coords where
        y goes DOWN and x goes right):
          - OPEN_PALM: hands above center = lift, below = dive, sideways = strafe,
            centered = forward cruise
          - CLICK: forward dash / boost
          - FIST: brake (decelerate)
          - No hands / IDLE: gravity pulls you down
        """
        yaw = self.facing
        fwd = np.array([-math.sin(yaw), 0.0, -math.cos(yaw)], np.float64)
        rgt = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], np.float64)

        thrust = np.zeros(3, np.float64)
        thrust[1] -= 9.8   # gravity

        for gesture, palm in ((self._gesture_left, self._palm_left),
                              (self._gesture_right, self._palm_right)):
            if palm is None:
                continue
            px, py, _ = palm
            if gesture == "OPEN_PALM":
                # vertical: hands above the frame center lift you up
                lift = (0.5 - py) * 26.0
                thrust[1] += lift
                # horizontal strafe from where the palms sit in the frame
                thrust[0] += (px - 0.5) * 12.0 * math.sin(yaw)   # x strafe approx
                thrust[2] += (px - 0.5) * 12.0 * math.cos(yaw)
                # hands roughly centered => cruise forward
                center = max(0.0, 0.28 - abs(py - 0.5))
                thrust += fwd * (center * 18.0)
            elif gesture == "CLICK":
                # quick forward boost
                thrust += fwd * 14.0
            elif gesture == "FIST":
                self.vel *= max(0.0, 1.0 - dt * 6.0)

        # integrate with a touch of drift damping
        speed = np.linalg.norm(self.vel)
        if speed > 48.0:
            self.vel = self.vel / speed * 48.0

        self.vel += thrust * dt
        self.pos += self.vel * dt

        # floor + hard ceiling
        if self.pos[1] < 0.5:
            self.pos[1] = 0.5
            self.vel[1] = max(0.0, self.vel[1])
        if self.pos[1] > 55.0:
            self.pos[1] = 55.0
            self.vel[1] = min(0.0, self.vel[1])

    # ------------------------------------------------------------------
    #  Combat
    # ------------------------------------------------------------------
    def _fire_repulsor(self, aim_world):
        if self._fire_cooldown > 0 or self.energy < 5:
            return
        self._fire_cooldown = 0.12
        self.energy -= 3.0

        origin = self.pos.copy()
        origin[1] += 0.2
        direction = aim_world - origin
        n = np.linalg.norm(direction)
        if n < 0.1:
            return
        direction = direction / n
        speed = 65.0
        self.projectiles.append(Projectile(origin, direction * speed, "repulsor"))

    def _fire_unibeam(self, aim_world):
        if not self._unibeam_ready or self.energy < 30:
            return
        self._unibeam_ready = False
        self._unibeam_charge = 0.0
        self.energy -= 30.0

        origin = self.pos.copy()
        origin[1] += 0.15
        direction = aim_world - origin
        n = np.linalg.norm(direction)
        if n < 0.1:
            return
        direction = direction / n
        self.projectiles.append(Projectile(origin, direction * 90.0, "unibeam"))
        self._friday.say("Unibeam fired.", 1.5)

    # ------------------------------------------------------------------
    #  Main update
    # ------------------------------------------------------------------
    def update(self, dt, hands_state=None, view=None, aspect=1.0):
        if not self.active:
            return

        self.t += dt

        # suit-up animation
        if not self._suit_up_done:
            self._suit_up_t += dt
            if self._suit_up_t > 3.0:
                self._suit_up_done = True
                self._friday.broadcast("suit_up")
            return  # don't update game logic during suit-up

        if self.game_over:
            return

        # palm positions (normalized → world using view)
        left_world = self._world_from_palm(self._palm_left, view, aspect) if self._palm_left is not None else None
        right_world = self._world_from_palm(self._palm_right, view, aspect) if self._palm_right is not None else None

        # choose aim: use whichever pointer hit, or default forward
        aim = self.pos + np.array([-math.sin(self.facing), 0.0, -math.cos(self.facing)]) * 15.0
        if left_world is not None and self._ptr_left is not None:
            aim = left_world
        elif right_world is not None and self._ptr_right is not None:
            aim = right_world

        # flight
        self._flight_from_hands(dt)

        # pinch press state (used for edge-trigger firing + auto-aim)
        l_click = self._gesture_left == "CLICK"
        r_click = self._gesture_right == "CLICK"

        # face aim direction (yaw only)
        dx = aim[0] - self.pos[0]
        dz = aim[2] - self.pos[2]
        if abs(dx) > 0.01 or abs(dz) > 0.01:
            target_yaw = math.atan2(-dx, -dz)
            diff = (target_yaw - self.facing + math.pi) % (2 * math.pi) - math.pi
            self.facing += diff * min(1.0, dt * 4.0)

        # auto-aim: only nudge toward aim while actively targeting (a pinch is held)
        if l_click or r_click:
            self.pos[0] += (aim[0] - self.pos[0]) * dt * 0.5
            self.pos[2] += (aim[2] - self.pos[2]) * dt * 0.5

        # fire repulsor on pinch press edge (tap to fire, hold to boost)
        l_edge = l_click and not self._click_left
        r_edge = r_click and not self._click_right
        self._click_left = l_click
        self._click_right = r_click

        if l_edge and left_world is not None:
            self._fire_repulsor(left_world)
        if r_edge and right_world is not None:
            self._fire_repulsor(right_world)

        # unibeam: both hands pinch, triggered on the press edge
        if (l_edge and r_click and right_world is not None):
            mid = (left_world + right_world) * 0.5 if left_world is not None \
                else right_world
            self._fire_unibeam(mid)
        if (r_edge and l_click and left_world is not None):
            mid = (right_world + left_world) * 0.5 if right_world is not None \
                else left_world
            self._fire_unibeam(mid)

        self._fire_cooldown = max(0.0, self._fire_cooldown - dt)

        # unibeam charge
        if not self._unibeam_ready:
            self._unibeam_charge += dt * 10.0
            if self._unibeam_charge >= 100.0:
                self._unibeam_ready = True
                self._unibeam_charge = 100.0
                self._friday.broadcast("unibeam_ready")

        # energy regen
        self.energy = min(self.max_energy, self.energy + dt * 8.0)

        # suit-critical FRIDAY warning (threshold watcher)
        if self.health <= 30.0 and self._health_was_high:
            self._health_was_high = False
            self._friday.broadcast("suit_critical")
        elif self.health > 50.0:
            self._health_was_high = True

        # wave management
        if not self.drones:
            self.wave_clear_time += dt
            if self.wave_clear_time > 2.0:
                self._spawn_wave()
                self.wave_clear_time = 0.0
        else:
            self.wave_clear_time = 0.0

        # update drones
        for drone in self.drones:
            drone.update(dt, self.pos)
            if drone.wants_to_shoot():
                to_player = self.pos - drone.pos
                n = np.linalg.norm(to_player)
                if n > 0.1:
                    vel = to_player / n * 22.0
                    self.projectiles.append(Projectile(drone.pos.copy(), vel, "laser", "enemy"))

        # player projectiles: move + swept collision vs drones (below)
        # (enemy projectiles are updated inside their own collision pass)
        for p in self.projectiles:
            if p.owner == "enemy":
                p.update(dt)

        # collision: player projectiles vs drones (swept to avoid tunneling)
        for p in self.projectiles:
            if not p.alive or p.owner != "player":
                continue
            prev = np.array(p.pos, np.float64)
            p.update(dt)
            seg = p.pos - prev
            steps = max(1, int(np.linalg.norm(seg) / 0.5))
            hit = False
            for i in range(steps):
                sample = prev + seg * ((i + 1) / steps)
                for drone in self.drones:
                    if not drone.alive:
                        continue
                    if drone.hit_test(sample, p.radius):
                        p.alive = False
                        drone.hp -= 1
                        drone.flash = 1.0
                        if drone.hp <= 0:
                            drone.alive = False
                            self.score += 100 if drone.kind != "boss" else 500
                            self._friday.say(f"Target neutralized. Score: {self.score}", 1.8)
                        hit = True
                        break
                if hit:
                    break

        # collision: enemy projectiles vs player
        for p in self.projectiles:
            if not p.alive or p.owner != "enemy":
                continue
            if np.linalg.norm(p.pos - self.pos) < 1.0:
                p.alive = False
                dmg = 12 if p.kind == "laser" else 5
                self.health -= dmg
                if self.health <= 0:
                    self.health = 0
                    self.game_over = True
                    self._friday.broadcast("game_over", score=self.score)
                    break

        # collision: ramming drones vs player
        for drone in self.drones:
            if not drone.alive or drone.kind != "ramming":
                continue
            if drone.hit_test(self.pos, 1.0):
                self.health -= 15
                drone.alive = False
                self.score += 50
                if self.health <= 0:
                    self.health = 0
                    self.game_over = True
                    self._friday.broadcast("game_over", score=self.score)

        # cleanup
        self.drones = [d for d in self.drones if d.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]

        # FRIDAY voice
        self._friday.update(dt)

    def _world_from_palm(self, palm_norm, view, aspect):
        """Convert normalized palm coords to a world-space aim point.

        The palm point is camera-space (as if the camera sat at the world
        origin facing -Z); rotate it into world space using the current
        facing yaw and add the player eye position.
        """
        if view is None or palm_norm is None:
            return None
        nx, ny, nz = palm_norm
        wx, wy, wz = view.to_world(nx, ny, 0.0, aspect)

        yaw = self.facing
        cos = math.cos(yaw)
        sin = math.sin(yaw)
        rx = wx * cos + wz * sin
        rz = -wx * sin + wz * cos
        return np.array([self.pos[0] + rx, self.pos[1] + wy,
                         self.pos[2] + rz], np.float64)

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------
    def draw(self, aspect, t=None):
        if not self.active:
            return
        t = t or self.t

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)

        # dusk sky dome + sun glow (behind everything)
        self._draw_atmosphere(t)

        # city skyline
        self._city.draw(t, self.pos)

        # drones
        for drone in self.drones:
            drone.draw(t)

        # projectiles
        for p in self.projectiles:
            p.draw(t)

        # thruster glow under palms
        self._draw_thrusters(t)

        # arc reactor on chest (always visible)
        self._draw_arc_reactor(t)

        glEnable(GL_LIGHTING)

    def _draw_atmosphere(self, t):
        """Dusk sky dome around the eye + a glowing sun on the horizon."""
        p = self.pos
        ey = self.pos[1] + 0.1
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        _dome(280.0, 20, 28, _dusk_color, center=(p[0], ey, p[2]))

        # sun glow (additive) hanging just over the skyline
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        sx, sy, sz = 130.0, 70.0, -270.0
        pulse = 0.75 + 0.25 * math.sin(t * 0.4)
        for rr, a in ((70.0, 0.08 * pulse), (36.0, 0.16 * pulse), (12.0, 0.5 * pulse)):
            glPushMatrix()
            glTranslatef(sx, sy, sz)
            glColor4f(1.0, 0.5, 0.25, a)
            _sphere(rr, 8, 10)
            glPopMatrix()

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

    def _draw_thrusters(self, t):
        """Draw thruster glow at active palm positions (open palm / pinch)."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)

        pulse = 0.6 + 0.4 * math.sin(t * 12.0)
        for gesture, palm in ((self._gesture_left, self._palm_left),
                              (self._gesture_right, self._palm_right)):
            if palm is None or gesture not in ("OPEN_PALM", "CLICK"):
                continue
            # approximate world position in front of the player
            hx = (palm[0] - 0.5) * 2.0
            hy = -(palm[1] - 0.5) * 2.0
            px = self.pos[0] - math.sin(self.facing) * 2.0 + math.cos(self.facing) * hx
            py = self.pos[1] + hy * 0.5
            pz = self.pos[2] - math.cos(self.facing) * 2.0 - math.sin(self.facing) * hx

            glPushMatrix()
            glTranslatef(px, py - 0.5, pz)
            glColor4f(C_ORANGE[0], C_ORANGE[1], C_ORANGE[2], 0.4 * pulse)
            _sphere(0.35, 5, 6)
            # flame trail
            glColor4f(1.0, 0.7, 0.2, 0.25 * pulse)
            _box(0, -0.6, 0, 0.25, 1.0, 0.25)
            glPopMatrix()

        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)

    def _draw_arc_reactor(self, t):
        """Small arc-reactor glow at the player's chest (visible from inside)."""
        pulse = 0.7 + 0.3 * math.sin(t * 4.0)
        glPushMatrix()
        glTranslatef(0.0, -0.05, -0.3)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)
        glColor4f(C_BLUE[0] * pulse, C_BLUE[1] * pulse, C_BLUE[2] * pulse, 0.7)
        _sphere(0.06, 5, 6)
        glDepthMask(GL_TRUE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_BLEND)
        glPopMatrix()

    # ------------------------------------------------------------------
    #  HUD (2D overlay — call after GL frame, before pygame flip)
    # ------------------------------------------------------------------
    def draw_hud(self, screen):
        """Draw Iron Man HUD on a pygame surface (called after GL render)."""
        if not self.active:
            return

        import pygame

        w, h = screen.get_size()
        font = pygame.font.Font(None, 30)
        font_sm = pygame.font.Font(None, 22)
        font_lg = pygame.font.Font(None, 48)

        # ── top-left: suit status ──
        # health bar
        hx, hy, hw, hh = 20, 20, 200, 14
        self._draw_bar(screen, hx, hy, hw, hh, self.health / self.max_health,
                       (255, 80, 60), (60, 20, 15), "HEALTH", font_sm)
        # energy bar
        self._draw_bar(screen, hx, hy + 24, hw, hh, self.energy / self.max_energy,
                       (0, 200, 255), (10, 30, 45), "ENERGY", font_sm)
        # unibeam charge
        self._draw_bar(screen, hx, hy + 48, hw, hh, self._unibeam_charge / 100.0,
                       (180, 120, 255) if self._unibeam_ready else (100, 80, 130),
                       (30, 20, 40), "UNIBEAM", font_sm,
                       ready=self._unibeam_ready)

        # ── top-right: score + wave ──
        score_txt = font.render(f"SCORE  {self.score}", True, (0, 220, 255))
        screen.blit(score_txt, (w - 220, 20))
        wave_txt = font.render(f"WAVE  {self.wave}", True, (180, 200, 220))
        screen.blit(wave_txt, (w - 220, 48))

        # ── center: targeting reticle ──
        cx, cy = w // 2, h // 2
        r = 28
        ret_color = (0, 200, 255) if self._unibeam_ready else (0, 180, 230)
        pygame.draw.circle(screen, ret_color, (cx, cy), r, 2)
        pygame.draw.circle(screen, ret_color, (cx, cy), r + 12, 1)
        # crosshair lines
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pygame.draw.line(screen, ret_color, (cx, cy), (cx + dx * (r + 6), cy + dy * (r + 6)), 1)
        # lock indicator on nearest drone
        if self.drones:
            nearest = min(self.drones, key=lambda d: np.linalg.norm(d.pos - self.pos))
            dist = np.linalg.norm(nearest.pos - self.pos)
            if dist < 30:
                lock_txt = font_sm.render(f"LOCK  {dist:.0f}m", True, (255, 100, 80))
                screen.blit(lock_txt, (cx - 30, cy + r + 18))

        # ── bottom-center: FRIDAY dialogue ──
        friday_line = self._friday.current
        if friday_line:
            txt = font.render(friday_line, True, (100, 200, 255))
            tw = txt.get_width()
            screen.blit(txt, (w // 2 - tw // 2, h - 74))

        # ── bottom-left: gesture hint strip ──
        hints = font_sm.render(
            "OPEN PALMS = fly   PINCH = repulsors   BOTH PINCH = unibeam   "
            "FIST = brake   F = ask FRIDAY", True, (80, 150, 200))
        screen.blit(hints, (16, h - 26))

        # ── FRIDAY voice status ──
        ai = getattr(self._friday, "ai", None)
        if ai is not None:
            status = "FRIDAY: voice online" if ai.available else "FRIDAY: text only"
            st = font_sm.render(status, True,
                                (120, 220, 140) if ai.available else (200, 170, 90))
            screen.blit(st, (16, h - 48))

        # ── suit-up overlay ──
        if not self._suit_up_done:
            alpha = int(255 * max(0.0, 1.0 - self._suit_up_t / 3.0))
            if alpha > 0:
                suit_txt = font_lg.render("SUITING UP...", True, (0, 200, 255))
                stw = suit_txt.get_width()
                screen.blit(suit_txt, (w // 2 - stw // 2, h // 2 - 30))

        # ── game over ──
        if self.game_over:
            go_txt = font_lg.render("SUIT DOWN", True, (255, 80, 60))
            gt = font.render("Press V to restart", True, (180, 200, 220))
            screen.blit(go_txt, (w // 2 - go_txt.get_width() // 2, h // 2 - 50))
            screen.blit(gt, (w // 2 - gt.get_width() // 2, h // 2 + 20))

    @staticmethod
    def _draw_bar(surface, x, y, w, h, fraction, fg, bg, label, font, ready=False):
        import pygame
        frac = max(0.0, min(1.0, fraction))
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=3)
        pygame.draw.rect(surface, fg, (x, y, int(w * frac), h), border_radius=3)
        border_color = (200, 220, 240) if ready else (80, 90, 100)
        pygame.draw.rect(surface, border_color, (x, y, w, h), 1, border_radius=3)
        txt = font.render(label, True, (200, 220, 240))
        surface.blit(txt, (x + 4, y - 1))
