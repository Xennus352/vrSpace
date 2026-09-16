"""MediaPipe hand gesture detection - high accuracy mode for XR interaction."""

import math
import cv2
import numpy as np
import mediapipe as mp


class GestureDetector:
    def __init__(self, swap_handedness=False):
        self.mp_hands = mp.solutions.hands
        # Higher accuracy: model_complexity=1, balanced thresholds
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.swap_handedness = swap_handedness

        self.current_gesture = {"Left": "UNKNOWN", "Right": "UNKNOWN"}
        self.candidate_gesture = {"Left": None, "Right": None}
        self.candidate_count = {"Left": 0, "Right": 0}
        self.min_confirm_frames = 2
        self.min_hand_confidence = 0.65
        # EMA smoothing for landmarks - stable, no velocity prediction to avoid flashing
        self.landmark_alpha = 0.3
        self._smoothed_landmarks = {"Left": None, "Right": None}

        self._hand_first_seen_time = None
        self._gesture_activation_seconds = 0.6
        self._prev_pinch = {"Left": None, "Right": None}
        self._smooth_xy = {"Left": (0.5, 0.5), "Right": (0.5, 0.5)}
        self._smooth_alpha = 0.35
        self._xy_deadzone = 0.004

        # Frame preprocessing
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def finger_up(self, tip, pip):
        return tip.y < pip.y

    def _f_ext(self, landmarks, mcp_i, pip_i, tip_i):
        """Direction-agnostic extension check: tip is far from MCP compared to
        PIP (finger out straight in ANY direction, not just up)."""
        mcp, pip, tip = landmarks[mcp_i], landmarks[pip_i], landmarks[tip_i]
        ext = math.hypot(tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z)
        curl = math.hypot(pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z)
        return ext > curl * 1.15

    def finger_extended(self, landmarks):
        """Return True if index finger is extended in ANY direction (not curled)."""
        return self._f_ext(landmarks, 5, 6, 8)

    def pointing_direction(self, landmarks):
        """Determine the dominant 2D direction the index finger points.
        Returns 'UP', 'DOWN', 'LEFT', 'RIGHT', or None.
        Uses MCP(5) -> TIP(8) vector in image space."""
        mcp = landmarks[5]
        tip = landmarks[8]
        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        adx, ady = abs(dx), abs(dy)
        if adx < 0.015 and ady < 0.015:
            return None
        if ady > adx:
            return "DOWN" if dy > 0 else "UP"
        return "RIGHT" if dx > 0 else "LEFT"

    def normalize_handedness(self, label):
        if not self.swap_handedness:
            return label
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return label

    def detect_gesture(self, landmarks, handedness):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        # Direction-aware finger extension (works at ANY hand angle)
        index_ext = self.finger_extended(landmarks)
        middle_ext = self._f_ext(landmarks, 9, 10, 12)
        ring_ext = self._f_ext(landmarks, 13, 14, 16)
        pinky_ext = self._f_ext(landmarks, 17, 18, 20)

        if handedness == "Right":
            thumb_up = thumb_tip.x > thumb_ip.x
        else:
            thumb_up = thumb_tip.x < thumb_ip.x

        pinch_distance = self.distance(index_tip, thumb_tip)

        if thumb_up and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "THUMBS_UP"
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return "OPEN_PALM"
        if pinch_distance < 0.05:
            return "CLICK"
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "VICTORY"
        if index_ext and middle_ext and ring_ext and not pinky_ext:
            return "THREE_FINGER_CLICK"
        if middle_ext and not index_ext and not ring_ext and not pinky_ext:
            return "RIGHT_CLICK_G"
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            direction = self.pointing_direction(landmarks)
            return f"POINTING_{direction}" if direction else "POINTING_UP"
        if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "FIST"

        return "UNKNOWN"

    def _is_triangle_pair(self, left_landmarks, right_landmarks):
        left_index_tip = left_landmarks[8]
        right_index_tip = right_landmarks[8]
        left_thumb_tip = left_landmarks[4]
        right_thumb_tip = right_landmarks[4]
        left_wrist = left_landmarks[0]
        right_wrist = right_landmarks[0]

        index_distance = self.distance(left_index_tip, right_index_tip)
        thumb_distance = self.distance(left_thumb_tip, right_thumb_tip)

        index_avg_y = (left_index_tip.y + right_index_tip.y) / 2.0
        thumb_avg_y = (left_thumb_tip.y + right_thumb_tip.y) / 2.0
        vertical_gap = abs(index_avg_y - thumb_avg_y)
        wrists_aligned = abs(left_wrist.y - right_wrist.y) < 0.12

        if index_distance < 0.16 and thumb_distance < 0.16:
            if vertical_gap >= 0.03:
                if index_avg_y < thumb_avg_y:
                    return "TRIANGLE"
                return "REVERSE_TRIANGLE"
            if wrists_aligned and vertical_gap < 0.03:
                return "REVERSE_TRIANGLE"

        if (
            thumb_distance < 0.12
            and 0.12 <= index_distance <= 0.45
            and index_avg_y + 0.02 < thumb_avg_y
            and wrists_aligned
        ):
            return "TRIANGLE"

        if (
            index_distance < 0.12
            and 0.12 <= thumb_distance <= 0.45
            and thumb_avg_y + 0.02 < index_avg_y
            and wrists_aligned
        ):
            return "REVERSE_TRIANGLE"

        return None

    def stable_output(self, hand_label, gesture):
        current = self.current_gesture.get(hand_label, "UNKNOWN")

        if gesture == current:
            self.candidate_gesture[hand_label] = None
            self.candidate_count[hand_label] = 0
            return current

        if self.candidate_gesture.get(hand_label) == gesture:
            self.candidate_count[hand_label] += 1
        else:
            self.candidate_gesture[hand_label] = gesture
            self.candidate_count[hand_label] = 1

        if self.candidate_count[hand_label] >= self.min_confirm_frames:
            self.current_gesture[hand_label] = gesture
            self.candidate_gesture[hand_label] = None
            self.candidate_count[hand_label] = 0

        return self.current_gesture[hand_label]

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        process_w = 480
        process_h = max(1, int((process_w * h) / w))
        small = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_LINEAR)

        # CLAHE on L channel for lighting normalization
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        results = self.hands.process(frame_rgb)

        output = []
        landmarks_by_label = {}

        if results.multi_hand_landmarks:
            import time
            now = time.time()

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                score = float(handedness.classification[0].score)
                if score < self.min_hand_confidence:
                    continue
                label = self.normalize_handedness(handedness.classification[0].label)
                landmarks = hand_landmarks.landmark
                prev = self._smoothed_landmarks.get(label)

                # Simple stable EMA smoothing (no velocity prediction to avoid flashing)
                if prev is None or len(prev) != len(landmarks):
                    smoothed = [(lm.x, lm.y, lm.z) for lm in landmarks]
                else:
                    a = self.landmark_alpha
                    smoothed = []
                    for idx, lm in enumerate(landmarks):
                        px, py, pz = prev[idx]
                        sx = px + (lm.x - px) * a
                        sy = py + (lm.y - py) * a
                        sz = pz + (lm.z - pz) * a
                        smoothed.append((sx, sy, sz))
                self._smoothed_landmarks[label] = smoothed

                # Write smoothed back to landmarks for gesture detection
                for idx, lm in enumerate(landmarks):
                    sx, sy, sz = smoothed[idx]
                    lm.x, lm.y, lm.z = sx, sy, sz

                landmarks_by_label[label] = landmarks
                gesture = self.detect_gesture(landmarks, label)
                stable = self.stable_output(label, gesture)

                index_tip = landmarks[8]
                thumb_tip = landmarks[4]
                wrist = landmarks[0]
                middle_mcp = landmarks[9]
                pinch_distance = self.distance(index_tip, thumb_tip)
                palm_x = (wrist.x + middle_mcp.x) / 2.0
                palm_y = (wrist.y + middle_mcp.y) / 2.0

                raw_x = float(index_tip.x)
                raw_y = float(index_tip.y)
                prev_x, prev_y = self._smooth_xy.get(label, (0.5, 0.5))
                dx = raw_x - prev_x
                dy = raw_y - prev_y
                if abs(dx) < self._xy_deadzone:
                    dx = 0.0
                if abs(dy) < self._xy_deadzone:
                    dy = 0.0
                x = prev_x + dx * self._smooth_alpha
                y = prev_y + dy * self._smooth_alpha
                self._smooth_xy[label] = (x, y)

                prev_pinch = self._prev_pinch.get(label)
                pinch_delta = 0.0
                scale_action = None
                if pinch_distance is not None:
                    if prev_pinch is not None:
                        pinch_delta = pinch_distance - prev_pinch
                        if pinch_delta > 0.008:
                            scale_action = "ENLARGE"
                        elif pinch_delta < -0.008:
                            scale_action = "REDUCE"
                    self._prev_pinch[label] = pinch_distance

                output.append({
                    "hand": label,
                    "gesture": stable,
                    "x": x,
                    "y": y,
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in landmarks],
                    "palm_x": float(palm_x),
                    "palm_y": float(palm_y),
                    "wrist_x": float(wrist.x),
                    "wrist_y": float(wrist.y),
                    "middle_mcp_x": float(middle_mcp.x),
                    "middle_mcp_y": float(middle_mcp.y),
                    "dx": float(dx),
                    "dy": float(dy),
                    "pinch": float(pinch_delta),
                    "is_click": bool(stable == "CLICK"),
                    "scale_delta": float(abs(pinch_delta)),
                    "scale_action": scale_action,
                })

            left_lm = landmarks_by_label.get("Left")
            right_lm = landmarks_by_label.get("Right")
            if left_lm is not None and right_lm is not None:
                pair_gesture = self._is_triangle_pair(left_lm, right_lm)
                if pair_gesture is not None:
                    self.current_gesture["Left"] = pair_gesture
                    self.current_gesture["Right"] = pair_gesture
                    self.candidate_gesture["Left"] = None
                    self.candidate_gesture["Right"] = None
                    self.candidate_count["Left"] = 0
                    self.candidate_count["Right"] = 0
                    for item in output:
                        item["gesture"] = pair_gesture

        if output:
            if self._hand_first_seen_time is None:
                import time
                self._hand_first_seen_time = now
            gesture_ready = (now - self._hand_first_seen_time) >= self._gesture_activation_seconds
        else:
            self._hand_first_seen_time = None
            gesture_ready = False

        if not gesture_ready and output:
            for item in output:
                item["gesture"] = "WARMUP"

        return frame, output
