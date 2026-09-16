import cv2
import mediapipe as mp
import math

class StableGestureSystem:
    def __init__(self, swap_handedness=False):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.swap_handedness = swap_handedness
        self.current_gesture = {"Left": "UNKNOWN", "Right": "UNKNOWN"}
        self.candidate_gesture = {"Left": None, "Right": None}
        self.candidate_count = {"Left": 0, "Right": 0}
        self.min_confirm_frames = 2
        self.min_hand_confidence = 0.6
        self.landmark_alpha = 0.35
        self._smoothed_landmarks = {"Left": None, "Right": None}

    # ------------------------
    # Utility
    # ------------------------
    def distance(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def finger_up(self, tip, pip):
        return tip.y < pip.y

    def finger_down(self, tip, pip):
        return tip.y > pip.y

    def _f_ext(self, landmarks, mcp_i, pip_i, tip_i):
        mcp, pip, tip = landmarks[mcp_i], landmarks[pip_i], landmarks[tip_i]
        ext = math.hypot(tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z)
        curl = math.hypot(pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z)
        return ext > curl * 1.15

    def finger_extended(self, landmarks):
        mcp = landmarks[5]
        pip = landmarks[6]
        tip = landmarks[8]
        ext_len = math.hypot(tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z)
        curl_len = math.hypot(pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z)
        return ext_len > curl_len * 1.15

    def pointing_direction(self, landmarks):
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

    # ------------------------
    # Gesture Detection
    # ------------------------
    def detect_gesture(self, landmarks, handedness):
        thumb_tip = landmarks[4]
        thumb_ip  = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        # Direction-aware extension (works at ANY hand angle)
        index_ext = self.finger_extended(landmarks)
        middle_ext = self._f_ext(landmarks, 9, 10, 12)
        ring_ext = self._f_ext(landmarks, 13, 14, 16)
        pinky_ext = self._f_ext(landmarks, 17, 18, 20)

        # Better thumb logic (depends on hand side)
        if handedness == "Right":
            thumb_up = thumb_tip.x > thumb_ip.x
        else:
            thumb_up = thumb_tip.x < thumb_ip.x

        pinch_distance = self.distance(index_tip, thumb_tip)

        # ---- PRIORITY ORDER ----
        # 1) THUMBS UP (must come before fist to avoid false fist detection)
        if thumb_up and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return "THUMBS_UP"

        # 2) OPEN PALM
        if index_ext and middle_ext and ring_ext and pinky_ext:
            return "OPEN_PALM"

        # 3) PINCH (Click)
        if pinch_distance < 0.05:
            return "CLICK"

        # 4) VICTORY (index + middle up)
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return "VICTORY"

        # 5) THREE_FINGER_CLICK (index + middle + ring up)
        if index_ext and middle_ext and ring_ext and not pinky_ext:
            return "THREE_FINGER_CLICK"

        # 6) RIGHT_CLICK_G (middle finger up only)
        if middle_ext and not index_ext and not ring_ext and not pinky_ext:
            return "RIGHT_CLICK_G"

        # 7) POINTING — index extended in ANY direction (up/down/left/right)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            direction = self.pointing_direction(landmarks)
            return f"POINTING_{direction}" if direction else "POINTING_UP"

        # 8) FIST
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

        # Closed triangle: both index tips and thumb tips are close.
        if index_distance < 0.16 and thumb_distance < 0.16:
            if vertical_gap >= 0.03:
                if index_avg_y < thumb_avg_y:
                    return "TRIANGLE"
                return "REVERSE_TRIANGLE"
            # Flat/horizontal compact shape is usually your reverse sign.
            if wrists_aligned and vertical_gap < 0.03:
                return "REVERSE_TRIANGLE"

        # Open reverse-triangle: thumbs close (bottom apex), indexes apart (top base).
        if (
            thumb_distance < 0.12
            and 0.12 <= index_distance <= 0.45
            and index_avg_y + 0.02 < thumb_avg_y
            and wrists_aligned
        ):
            return "TRIANGLE"

        # Open normal reverse-triangle: indexes close (bottom apex), thumbs apart (top base).
        if (
            index_distance < 0.12
            and 0.12 <= thumb_distance <= 0.45
            and thumb_avg_y + 0.02 < index_avg_y
            and wrists_aligned
        ):
            return "REVERSE_TRIANGLE"

        return None

    # ------------------------
    # Stable Output
    # ------------------------
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

    # ------------------------
    # Process Frame
    # ------------------------
    def process(self, frame):
        h, w = frame.shape[:2]
        process_w = 480
        process_h = max(1, int((process_w * h) / w))
        small = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        output = []

        if results.multi_hand_landmarks:
            landmarks_by_label = {}
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                score = float(handedness.classification[0].score)
                if score < self.min_hand_confidence:
                    continue
                label = self.normalize_handedness(handedness.classification[0].label)
                landmarks = hand_landmarks.landmark
                prev = self._smoothed_landmarks.get(label)
                if prev is None or len(prev) != len(landmarks):
                    smoothed = [(lm.x, lm.y) for lm in landmarks]
                else:
                    a = self.landmark_alpha
                    smoothed = []
                    for idx, lm in enumerate(landmarks):
                        px, py = prev[idx]
                        smoothed.append(
                            (px + (lm.x - px) * a, py + (lm.y - py) * a)
                        )
                self._smoothed_landmarks[label] = smoothed

                # Use smoothed coordinates in-place to reduce jitter without extra objects.
                for idx, lm in enumerate(landmarks):
                    sx, sy = smoothed[idx]
                    lm.x = sx
                    lm.y = sy
                landmarks_by_label[label] = landmarks
                gesture = self.detect_gesture(
                    landmarks,
                    label
                )

                stable = self.stable_output(label, gesture)
                index_tip = landmarks[8]
                thumb_tip = landmarks[4]
                wrist = landmarks[0]
                middle_mcp = landmarks[9]
                pinch_distance = self.distance(index_tip, thumb_tip)
                palm_x = (wrist.x + middle_mcp.x) / 2.0
                palm_y = (wrist.y + middle_mcp.y) / 2.0

                output.append({
                    "hand": label,
                    "gesture": stable,
                    "x": float(index_tip.x),
                    "y": float(index_tip.y),
                    "wrist_x": float(wrist.x),
                    "wrist_y": float(wrist.y),
                    "middle_mcp_x": float(middle_mcp.x),
                    "middle_mcp_y": float(middle_mcp.y),
                    "palm_x": float(palm_x),
                    "palm_y": float(palm_y),
                    "pinch_distance": float(pinch_distance),
                })

                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

            left_landmarks = landmarks_by_label.get("Left")
            right_landmarks = landmarks_by_label.get("Right")
            if left_landmarks is not None and right_landmarks is not None:
                pair_gesture = self._is_triangle_pair(left_landmarks, right_landmarks)
                if pair_gesture is not None:
                    self.current_gesture["Left"] = pair_gesture
                    self.current_gesture["Right"] = pair_gesture
                    self.candidate_gesture["Left"] = None
                    self.candidate_gesture["Right"] = None
                    self.candidate_count["Left"] = 0
                    self.candidate_count["Right"] = 0
                    for item in output:
                        if item.get("hand") in {"Left", "Right"}:
                            item["gesture"] = pair_gesture

        return frame, output
