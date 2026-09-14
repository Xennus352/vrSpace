"""Webcam head tracking for the stereo headset view (Cardboard-style VR).

Uses MediaPipe FaceMesh, run in the camera thread. A single face's key
landmarks are turned into smoothed yaw / pitch / roll angles:

  * yaw   - positive when the head turns LEFT  (nose right of face center)
  * pitch - positive when looking UP           (nose above face center)
  * roll  - head tilt; the renderer counter-rotates to hold the horizon

``recenter()`` re-zeroes the current head orientation so any frame the
user is currently in becomes "forward". When no face is detected the last
pose is kept (the view does not jump back to zero).
"""

import math
import time

import cv2
import mediapipe as mp


class HeadTracker:
    def __init__(self, yaw_gain=1.0, pitch_gain=1.2, max_tilt=0.85):
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.max_tilt = max_tilt  # radians, 1.0 = +/-57 deg

        self._smooth = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        self._base = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        self._base_set = False
        self._last_run = 0.0
        self._last_time = time.time()
        self.pose = {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "detected": False,
        }

    # ------------------------------------------------------------------
    def recenter(self):
        """Make the current head orientation the 'forward' reference."""
        self._base = {
            "yaw": self._smooth["yaw"],
            "pitch": self._smooth["pitch"],
            "roll": self._smooth["roll"],
        }
        self._base_set = True
        self.pose["detected"] = True

    # ------------------------------------------------------------------
    def process(self, frame_bgr):
        now = time.time()
        dt = max(0.001, now - self._last_time)
        self._last_time = now

        # Throttle the expensive FaceMesh pass to ~30 Hz; between runs we
        # simply re-emit the last pose so the render loop never stalls.
        if now - self._last_run >= 0.033 and frame_bgr is not None and frame_bgr.size > 0:
            self._last_run = now
            self._run(frame_bgr, dt)
        return self.pose

    # ------------------------------------------------------------------
    def _run(self, frame_bgr, dt):
        h, w = frame_bgr.shape[:2]
        process_w = 320
        process_h = max(1, int((process_w * h) / w))
        small = cv2.resize(frame_bgr, (process_w, process_h),
                           interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face.process(rgb)

        if not results.multi_face_landmarks:
            self.pose["detected"] = False
            return

        lm = results.multi_face_landmarks[0].landmark
        left_cheek = lm[234]
        right_cheek = lm[454]
        nose = lm[4]
        forehead = lm[10]
        chin = lm[152]

        # Face width / height in normalized units (guard against tiny faces).
        face_w = max(1e-4, math.hypot(right_cheek.x - left_cheek.x,
                                      right_cheek.y - left_cheek.y))
        face_h = max(1e-4, math.hypot(forehead.x - chin.x,
                                      forehead.y - chin.y))
        center_x = (left_cheek.x + right_cheek.x) / 2.0
        center_y = (forehead.y + chin.y) / 2.0

        # Head turned left -> nose sits right of center on the flattened face.
        yaw = self._clamp((nose.x - center_x) / face_w) * self.yaw_gain
        # Head pitched up -> nose tip sits above the face center line.
        pitch = self._clamp((center_y - nose.y) / face_h) * self.pitch_gain
        # Counter-roll needs the true sign of the tilt.
        roll = math.atan2(right_cheek.y - left_cheek.y,
                          right_cheek.x - left_cheek.x)

        target = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll * 0.6,
        }
        # Soft EMA so micro-jitter of the nose/cheeks is not felt in the view.
        # alpha ~ dt*10 keeps response snappy (less perceived lag).
        alpha = min(1.0, dt * 10.0)
        for key, value in target.items():
            self._smooth[key] += (value - self._smooth[key]) * alpha

        if not self._base_set:
            self.recenter()
            self._smooth = dict(self._base)  # start from identity

        # Re-emit stored angles as net motion vs. the recenter reference.
        self.pose["yaw"] = float(self._clamp(self._smooth["yaw"] - self._base["yaw"]))
        self.pose["pitch"] = float(self._clamp(self._smooth["pitch"] - self._base["pitch"]))
        self.pose["roll"] = float(self._smooth["roll"] - self._base["roll"])
        self.pose["detected"] = True

    @staticmethod
    def _clamp(v):
        return max(-1.0, min(1.0, v * 1.6)) * 0.85