"""Phone gyroscope -> head look + movement control.

The phone page streams raw ``deviceorientation`` euler angles (alpha, beta,
gamma, in degrees) over HTTPS to the PC. This module turns them into the same
yaw/pitch/roll interface the webcam face tracker drives
(renderer.set_head_pose), plus a throttle-style walk command.

THE FIXED-LOOK MAPPING (why not naive alpha->yaw / beta->pitch):
``DeviceOrientationEvent`` euler angles live in the device's physical frame
and stay there no matter how the phone is gripped, but the most comfortable
grip for this game is landscape, which phones report as
``screen.orientation.angle`` = 90/270. Naively swapping axes on that made the
look flip for landscape users, and ``alpha`` on Android is
compass/magnetometer based and wanders - so the view shook and rotated by
itself while the phone was perfectly still.

Robust device-space interpretation, in any grip:
  * head-YAW   = the compass swivel ``alpha``. ``alpha`` IS the rotation of
    the phone around the world vertical, i.e. turning left/right. Always.
  * head-PITCH = the elevation of the screen normal (device +Z) above the
    horizon. Tilt the phone up/down in ANY grip and this changes; it is the
    true "how high am I looking" angle.
  * head-ROLL  = a cosmetic bank, derived from the same tilt as walking.
  * WALK       = the gamma tilt relative to the recentered reference
    (tilt to walk, tilt farther to sprint).

STABILITY vs the previous version's complaints:
  * A forward reference (set at recenter or first sample) anchors yaw+pitch,
    and while the phone is judged to be still it slowly RE-ANCHORS toward the
    current reading - so compass wander decays to zero instead of
    accumulating into continuous view motion.
  * Look deltas get a deadzone + EMA smoothing, so micro sensor jitter is
    never shown.
"""

import math
import time

import numpy as np

# Sign/gain tuning per phone brand. If turning the phone left moves the view
# right, flip YAW_SIGN (or PITCH_SIGN for up/down mirroring).
YAW_SIGN = -1.0
PITCH_SIGN = -1.0
ROLL_SIGN = 1.0

YAW_GAIN = 1.0
PITCH_GAIN = 1.0
ROLL_GAIN = 0.25

MAX_YAW_DEG = 170.0     # clamp look so the world cannot over-rotate
MAX_PITCH_DEG = 70.0
MAX_ROLL_DEG = 20.0

RAW_TAU = 0.18          # EMA tau (s) for the raw sensor feed
SMOOTH_TAU = 0.10       # EMA tau (s) for the displayed look deltas
LOOK_DEADZONE_DEG = 0.6  # ignore smaller look deltas (jitter kill)

# Drift compensation: while the phone is near the forward reference, slowly
# move the reference toward the current reading so compass wander decays to
# zero instead of turning the view by itself.
DRIFT_TAU = 6.0         # seconds for full re-anchoring of a still phone
STILL_YAW_DEG = 3.5     # |yaw delta| under this => phone judged 'still'
STILL_PITCH_DEG = 2.5

# Movement (gamma roll tilt = walking throttle)
THROTTLE_ON_DEG = 14.0       # start walking past this tilt
THROTTLE_SPRINT_DEG = 34.0   # full tilt = sprint
THROTTLE_DEADZONE = 6.0      # below this tilt we hold still
TILT_BASE_DRIFT = 0.02       # per-second re-anchor of walking tilt to 0

STALE_AFTER = 2.0            # seconds without an update before auto-fallback


def _norm(a):
    """Wrap to -180..180 (deg)."""
    return (a + 180.0) % 360.0 - 180.0


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _device_to_world(a_deg, b_deg, g_deg):
    """Device-orientation matrix (device frame -> world frame), following the
    spec: rotate by alpha about world Z (compass), then beta about the device
    X axis, then gamma about the device Y axis."""
    a = math.radians(a_deg)
    b = math.radians(b_deg)
    g = math.radians(g_deg)
    return _rz(a) @ _rx(b) @ _ry(g)


def _pitch_elevation(M):
    """Elevation (degrees) of the screen normal above the horizon - grip
    independent look up/down."""
    zs = M @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.asin(max(-1.0, min(1.0, float(zs[2])))))


def _soft_deadzone(d, dz):
    """Shrink a delta toward 0 by dz, clipping small values to 0."""
    if abs(d) <= dz:
        return 0.0
    return (abs(d) - dz) * (1.0 if d >= 0 else -1.0)


class GyroController:
    def __init__(self):
        now = time.time()
        self.active = False
        self._got_data = False
        self._seed_baseline = False
        self._base = {"heading": 0.0, "pitch": 0.0, "tilt": 0.0}
        self._sm = {"heading": 0.0, "pitch": 0.0, "tilt": 0.0,
                    "dy": 0.0, "dp": 0.0, "dt": 0.0}
        self._last_update = now
        self._last_time = now
        self.pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        self.move_state = (False, False, False)
        self.now_deg = {}

    # ------------------------------------------------------------------
    def toggle(self, state=None):
        """Enable/disable gyro control. state=None flips it."""
        if state is None:
            self.active = not self.active
        else:
            self.active = bool(state)
        if self.active:
            self.recenter()
        else:
            self.move_state = (False, False, False)
        return self.active

    def recenter(self):
        """Make the phone's CURRENT orientation the 'forward' reference."""
        self._seed_baseline = True

    def stale(self):
        return self.active and (time.time() - self._last_update) > STALE_AFTER

    # ------------------------------------------------------------------
    def update(self, alpha, beta, gamma, angle=0):
        """Feed one deviceorientation event (degrees).

        angle (screen.orientation.angle) is accepted for compatibility with
        the phone page but deliberately ignored - the look mapping is grip
        independent.
        """
        now = time.time()
        self._last_update = now
        heading = float(alpha)           # compass swivel = head yaw
        pitch = _pitch_elevation(
            _device_to_world(alpha, beta, gamma))  # screen-normal elevation
        tilt = float(gamma)               # walking throttle source

        # First real sensor sample: seed reference + smoothing together, so
        # the starting pose is 'forward', not a huge rotation.
        if not self._got_data:
            self._got_data = True
            self._last_time = now
            self._sm = {"heading": heading, "pitch": pitch, "tilt": tilt,
                        "dy": 0.0, "dp": 0.0, "dt": 0.0}
            self._base = {"heading": heading, "pitch": pitch, "tilt": tilt}
            self._seed_baseline = False
            self.pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            self.move_state = (False, False, False)
            return

        dt = max(0.001, min(0.2, now - self._last_time))
        self._last_time = now

        # Smooth the raw feed (heading wraps at +/-180).
        a = 1.0 - math.exp(-dt / RAW_TAU)
        self._sm["pitch"] += a * (pitch - self._sm["pitch"])
        self._sm["tilt"] += a * (tilt - self._sm["tilt"])
        hdiff = _norm(heading - self._sm["heading"])
        self._sm["heading"] = _norm(self._sm["heading"] + a * hdiff)

        # A recenter() re-anchors on the next (already-smoothed) sample.
        if self._seed_baseline:
            self._seed_baseline = False
            self._base["heading"] = self._sm["heading"]
            self._base["pitch"] = self._sm["pitch"]
            self._base["tilt"] = self._sm["tilt"]

        # Raw look deltas vs the forward reference (degrees).
        dy = _norm(self._sm["heading"] - self._base["heading"])
        dp = self._sm["pitch"] - self._base["pitch"]
        dt_t = self._sm["tilt"] - self._base["tilt"]

        # Drift compensation: still phone => absorb sensor wander into the
        # reference so it never accumulates into view motion.
        if abs(dy) < STILL_YAW_DEG and abs(dp) < STILL_PITCH_DEG:
            k = dt / DRIFT_TAU
            self._base["heading"] += k * dy
            self._base["pitch"] += k * dp
            self._base["tilt"] += dt_t * TILT_BASE_DRIFT * dt
            dy = _norm(self._sm["heading"] - self._base["heading"])
            dp = self._sm["pitch"] - self._base["pitch"]
            dt_t = self._sm["tilt"] - self._base["tilt"]

        # Deadzone + EMA smoothing on the displayed deltas.
        dy = _soft_deadzone(dy, LOOK_DEADZONE_DEG)
        dp = _soft_deadzone(dp, LOOK_DEADZONE_DEG)
        dt_t = _soft_deadzone(dt_t, LOOK_DEADZONE_DEG)
        b = 1.0 - math.exp(-dt / SMOOTH_TAU)
        self._sm["dy"] += (dy - self._sm["dy"]) * b
        self._sm["dp"] += (dp - self._sm["dp"]) * b
        self._sm["dt"] += (dt_t - self._sm["dt"]) * b

        yaw = math.radians(YAW_SIGN * self._clamp(self._sm["dy"],
                                                  MAX_YAW_DEG) * YAW_GAIN)
        pitch_r = math.radians(PITCH_SIGN * self._clamp(self._sm["dp"],
                                                        MAX_PITCH_DEG)
                               * PITCH_GAIN)
        roll = math.radians(ROLL_SIGN * self._clamp(self._sm["dt"],
                                                    MAX_ROLL_DEG)
                            * ROLL_GAIN)

        # Walking throttle from the same tilt (relative to the fixed
        # reference); the look deadzone already swallowed the small stuff.
        t = self._sm["tilt"] - self._base["tilt"]
        fwd = back = sprint = False
        if abs(t) > THROTTLE_DEADZONE:
            if t > THROTTLE_ON_DEG:
                fwd = True
                sprint = t > THROTTLE_SPRINT_DEG
            elif t < -THROTTLE_ON_DEG:
                back = True
                sprint = t < -THROTTLE_SPRINT_DEG
        self.move_state = (fwd, back, sprint)

        self.pose = {"yaw": yaw, "pitch": pitch_r, "roll": roll}
        self.now_deg = {"yaw": dy, "pitch": dp, "roll": dt_t, "tilt": t,
                        "heading": self._sm["heading"],
                        "pitch_raw": self._sm["pitch"]}

    # ------------------------------------------------------------------
    def head(self):
        """(yaw, pitch, roll) in radians for renderer.set_head_pose."""
        return (self.pose["yaw"], self.pose["pitch"], self.pose["roll"])

    def move(self):
        """(forward, back, sprint) booleans for the player controller."""
        return self.move_state

    @staticmethod
    def _clamp(v, mx):
        return max(-mx, min(mx, v))

    def reset(self):
        self.__init__()