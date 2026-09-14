"""Camera passthrough scene with a realistic glove hand overlay.

The camera frame fills the screen (no 3D Earth/stars); the tracked hand
is drawn on top as a realistic lit 3D glove using PyOpenGL immediate
mode.
"""

import math
import numpy as np
import cv2
from OpenGL.GL import *


def _perspective(fov_y_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    d = near - far
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / d, 2 * far * near / d],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    ).T  # column-major for OpenGL


class PassthroughRenderer:
    def __init__(self):
        self.viewport = None
        self._bg_texture = None
        self._bg_size = None
        self._init_done = False
        from glass_panel import ViewSpace
        from glove_hand import GloveHand
        from vr_scene import VRWorld, VRPlayer
        from hologram import Hologram
        self.view = ViewSpace()
        self._glove = GloveHand(self.view)
        self.holo = Hologram(self.view)
        self._world = VRWorld()
        self.player = VRPlayer()

        # ---- Stereo headset (VR) state -------------------------------
        self.vr_enabled = True
        self.stereo = False    # True = side-by-side lens split; False = one full frame
        self._yaw = 0.0      # radians; positive = head turned left
        self._pitch = 0.0    # radians; positive = looking up
        self._roll = 0.0     # radians; head tilt counter-rotation
        self._ipd = 0.032    # half-distance per eye in world units

    @property
    def glove(self):
        return self._glove

    # ------------------------------------------------------------------
    #  VR mode
    # ------------------------------------------------------------------
    def set_vr(self, enabled):
        self.vr_enabled = bool(enabled)
        return self.vr_enabled

    def set_head_pose(self, yaw, pitch, roll):
        """yaw/pitch in radians (see vr_head.HeadTracker for sign convention)."""
        self._yaw = float(yaw)
        self._pitch = float(pitch)
        self._roll = float(roll)

    def _set_projection(self, aspect, fov_deg=60.0, far=60.0):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixf(_perspective(fov_deg, max(0.1, aspect), 0.1, far))
        glMatrixMode(GL_MODELVIEW)

    def init_gl(self):
        if self._init_done:
            return
        self._init_done = True

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Warm key light + cool rim fill for a realistic, readable glove body
        glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 4.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.28, 0.27, 0.26, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 0.96, 0.90, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.75, 0.75, 0.75, 1.0])

        glLightfv(GL_LIGHT1, GL_POSITION, [-2.0, -1.0, -3.0, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.30, 0.34, 0.42, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.25, 0.28, 0.34, 1.0])

        self._set_projection(16.0 / 9.0)

    def set_viewport(self, w, h):
        if h <= 0:
            h = 1
        self.viewport = (w, h)
        glViewport(0, 0, w, h)
        self._set_projection(float(w) / float(h))

    # ------------------------------------------------------------------
    #  Background (camera passthrough)
    # ------------------------------------------------------------------
    def set_background(self, frame_bgr):
        """frame_bgr: OpenCV BGR frame. Uploaded as a fullscreen texture."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb[::-1])  # flip so texture has top at row 0

        data = rgb.tobytes()
        if self._bg_texture is None:
            self._bg_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._bg_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
            self._bg_size = (w, h)
        else:
            glBindTexture(GL_TEXTURE_2D, self._bg_texture)
            if self._bg_size == (w, h):
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, data)
            else:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
                self._bg_size = (w, h)

    def _draw_background(self):
        if self._bg_texture is None:
            glClearColor(0.05, 0.05, 0.08, 1.0)
            return
        glClearColor(0.0, 0.0, 0.0, 1.0)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_LIGHTING)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self._bg_texture)
        glColor3f(1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, -1.0, -1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, -1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, 1.0, -1.0)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    # ------------------------------------------------------------------
    #  Glove hands
    # ------------------------------------------------------------------
    def set_hands(self, hands):
        self._glove.set_hands(hands)

    def update(self, dt, ctrl=None):
        self._glove.update(dt)
        self.holo.update(dt)
        # Get gesture locomotion and merge with keyboard
        if ctrl is not None:
            gfwd, gright, gup = self.holo.get_locomotion()
            if abs(gfwd) > 0.05 or abs(gright) > 0.05 or abs(gup) > 0.05:
                ctrl = dict(ctrl)
                ctrl["fwd"] = ctrl.get("fwd", False) or gfwd > 0.1
                ctrl["back"] = ctrl.get("back", False) or gfwd < -0.1
                ctrl["right"] = ctrl.get("right", False) or gright > 0.1
                ctrl["left"] = ctrl.get("left", False) or gright < -0.1
                ctrl["up"] = ctrl.get("up", False) or gup > 0.1
                ctrl["down"] = ctrl.get("down", False) or gup < -0.1
            self.player.move(dt, ctrl)

    # ------------------------------------------------------------------
    #  Main render
    # ------------------------------------------------------------------
    def render(self, dt=0.0):
        self._world_t = getattr(self, "_world_t", 0.0) + dt
        if self.stereo:
            self._render_vr()
        else:
            self._render_vr_single()

    # ------------------------------------------------------------------
    #  VR cameras (first person)
    # ------------------------------------------------------------------
    def _apply_camera_fp(self, ipd=0.0):
        """First-person: player eye position + head rotation (+ eye offset)."""
        yaw = self.player.facing + self._yaw
        pitch = self.player.pitch + self._pitch
        px, py, pz = self.player.pos
        ey = self.player.eye_height + self.player.bob
        glLoadIdentity()
        glTranslatef(ipd, 0.0, 0.0)
        glRotatef(-math.degrees(pitch), 1.0, 0.0, 0.0)
        glRotatef(-math.degrees(yaw), 0.0, 1.0, 0.0)
        glRotatef(math.degrees(self._roll), 0.0, 0.0, 1.0)
        glTranslatef(-px, -(py + ey), -pz)

    @staticmethod
    def _look_at(eye, target, roll_deg=0.0):
        f = target - eye
        flen = np.linalg.norm(f)
        if flen < 1e-6:
            f = np.array([0.0, 0.0, -1.0])
        else:
            f = f / flen
        up = np.array([0.0, 1.0, 0.0])
        s = np.cross(f, up)
        slen = np.linalg.norm(s)
        if slen < 1e-6:
            s = np.array([1.0, 0.0, 0.0])
        else:
            s = s / slen
        u = np.cross(s, f)
        # column-major buffer: view matrix M = [s u -f | -dot(s,e) -dot(u,e) dot(f,e)]
        glLoadIdentity()
        glMultMatrixf([
            s[0], s[1], s[2], 0.0,
            u[0], u[1], u[2], 0.0,
            -f[0], -f[1], -f[2], 0.0,
            -np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1.0,
        ])
        glRotatef(roll_deg, 0.0, 0.0, 1.0)

    def _world_draw(self, aspect):
        px, _, pz = self.player.pos
        self._world.draw(
            live_texture=self._bg_texture,
            has_live=self._bg_texture is not None,
            t=self._world_t,
            player_offset=(float(px), float(pz)),
        )
        # draw hologram in world space (camera transform already applied)
        self.holo.draw_world(aspect)

    def _hud_draw(self, aspect, with_glove=True):
        """Render the HUD (glove + finger pointers) in view space."""
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        if with_glove:
            self._glove.render(aspect)
        # Finger pointers in view space (attached to hand like Quest 3)
        self.holo.draw_pointers(aspect)
        glPopMatrix()

    def _render_vr_single(self):
        """One full-frame immersive VR view (no lens split)."""
        w, h = self.viewport or (1280, 720)
        aspect = float(w) / max(1.0, float(h))

        glClearColor(0.004, 0.008, 0.02, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._set_projection(aspect, 76.0, far=300.0)

        self._apply_camera_fp()

        self._world_draw(aspect)
        self._hud_draw(aspect, with_glove=True)
        self._draw_vr_mask(w, h, single=True)

    # ------------------------------------------------------------------
    #  Stereo headset render: one pass per eye + lens field mask
    # ------------------------------------------------------------------
    def _render_vr(self):
        w, h = self.viewport or (1280, 720)
        half_w = max(1, w // 2)
        eye_aspect = half_w / float(h)
        vr_fov = 76.0  # wider vertical FOV => immersive headset feel

        # Color cleared ONCE for the whole frame; each eye only refreshes
        # its own depth buffer (scissored to its half of the screen).
        glClearColor(0.004, 0.008, 0.02, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for side in ("L", "R"):
            x0 = 0 if side == "L" else half_w
            glViewport(x0, 0, half_w, h)
            glEnable(GL_SCISSOR_TEST)
            glScissor(x0, 0, half_w, h)
            glClear(GL_DEPTH_BUFFER_BIT)
            glDisable(GL_SCISSOR_TEST)
            self._set_projection(eye_aspect, vr_fov, far=300.0)

            self._apply_camera_fp(self._ipd if side == "L" else -self._ipd)

            self._world_draw(eye_aspect)
            self._hud_draw(eye_aspect, with_glove=True)

        # Restore the full viewport for the headset rubber / divider overlay.
        glViewport(0, 0, w, h)
        self._set_projection(float(w) / max(1.0, float(h)))
        self._draw_vr_mask(w, h)

    def _draw_vr_mask(self, w, h, single=False):
        """Headset framing: soft full-frame vignette (single) or two lens
        fields with a phone-screen divider (stereo split)."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if single:
            # subtle cinematic vignette so the edges of the world fall away
            v = 0.10
            glColor4f(0.03, 0.04, 0.08, 0.45)
            glBegin(GL_QUADS)
            glVertex3f(-1.0, 1.0 - v, -1.0); glVertex3f(1.0, 1.0 - v, -1.0)
            glVertex3f(1.0, 1.0, -1.0);      glVertex3f(-1.0, 1.0, -1.0)
            glVertex3f(-1.0, -1.0, -1.0);    glVertex3f(1.0, -1.0, -1.0)
            glVertex3f(1.0, -1.0 + v, -1.0); glVertex3f(-1.0, -1.0 + v, -1.0)
            glVertex3f(-1.0 + v, -1.0, -1.0); glVertex3f(-1.0, -1.0, -1.0)
            glVertex3f(-1.0, 1.0, -1.0);     glVertex3f(-1.0 + v, 1.0, -1.0)
            glVertex3f(1.0 - v, -1.0, -1.0); glVertex3f(1.0, -1.0, -1.0)
            glVertex3f(1.0, 1.0, -1.0);      glVertex3f(1.0 - v, 1.0, -1.0)
            glEnd()
        else:
            # horizontal separator (phone screen division between the lenses)
            glColor4f(0.05, 0.06, 0.10, 0.85)
            glBegin(GL_QUADS)
            glVertex3f(-0.004, -1.0, -1.0)
            glVertex3f(0.004, -1.0, -1.0)
            glVertex3f(0.004, 1.0, -1.0)
            glVertex3f(-0.004, 1.0, -1.0)
            glEnd()

            # lens border frame (left half and right half share the divider line)
            b = 0.055
            for x0, x1 in ((-1.0, 0.0), (0.0, 1.0)):
                glColor4f(0.04, 0.05, 0.09, 0.70)
                glBegin(GL_QUADS)
                # top band
                glVertex3f(x0, 1.0 - b, -1.0); glVertex3f(x1, 1.0 - b, -1.0)
                glVertex3f(x1, 1.0, -1.0);     glVertex3f(x0, 1.0, -1.0)
                # bottom band
                glVertex3f(x0, -1.0, -1.0);    glVertex3f(x1, -1.0, -1.0)
                glVertex3f(x1, -1.0 + b, -1.0); glVertex3f(x0, -1.0 + b, -1.0)
                # outer side band
                if x0 < 0.0:
                    glVertex3f(x0, -1.0, -1.0);     glVertex3f(x0 + b, -1.0, -1.0)
                    glVertex3f(x0 + b, 1.0, -1.0);  glVertex3f(x0, 1.0, -1.0)
                else:
                    glVertex3f(x1 - b, -1.0, -1.0); glVertex3f(x1, -1.0, -1.0)
                    glVertex3f(x1, 1.0, -1.0);      glVertex3f(x1 - b, 1.0, -1.0)
                glEnd()

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()