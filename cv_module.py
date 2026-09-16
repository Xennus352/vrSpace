import asyncio
import json
import queue
import threading
import time

import cv2
import pyttsx3
import websockets
from hand_gesture import StableGestureSystem
from main_voice_assistance import VoiceAssistantService

MIRROR_CAMERA = True
SWAP_HANDEDNESS = False
TARGET_FPS = 22


class SpeechWorker:
    """Threaded offline TTS so speech does not block video processing."""

    def __init__(self, rate=180):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self._queue = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()

    def speak_async(self, text):
        self._queue.put(text)

    def stop(self):
        self._running = False
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class GestureWebSocketServer:
    def __init__(self, payload_getter, send_fps=TARGET_FPS):
        self._payload_getter = payload_getter
        self._clients = set()
        self._send_interval = 1.0 / max(1.0, float(send_fps))

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                if data.get("type") == "ping":
                    await websocket.send(
                        json.dumps({"type": "pong", "ts": int(time.time() * 1000)})
                    )
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)

    async def _broadcast_once(self):
        if not self._clients:
            return
        payload = self._payload_getter()
        messages = payload if isinstance(payload, list) else [payload]
        stale = []
        for ws in tuple(self._clients):
            try:
                for message in messages:
                    await ws.send(json.dumps(message))
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)

    async def broadcast_loop(self, host="0.0.0.0", port=8765):
        async with websockets.serve(self._handler, host, port):
            while True:
                await self._broadcast_once()
                await asyncio.sleep(self._send_interval)


class CameraServer:
    GESTURE_MAP = {
        "FIST": "CLOSED_FIST",
        "OPEN_PALM": "OPEN_PALM",
        "CLICK": "PINCH",
        "VICTORY": "VICTORY",
        "THREE_FINGER_CLICK": "MIDDLE_CLICK_G",
        "POINTING_UP": "POINTING_UP",
        "POINTING_DOWN": "POINTING_DOWN",
        "POINTING_LEFT": "POINTING_LEFT",
        "POINTING_RIGHT": "POINTING_RIGHT",
        "RIGHT_CLICK_G": "RIGHT_CLICK_G",
        "THUMBS_UP": "THUMBS_UP",
        "TRIANGLE": "TRIANGLE",
        "REVERSE_TRIANGLE": "REVERSE_TRIANGLE",
        "UNKNOWN": "IDLE",
        None: "IDLE",
    }

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.gesture_system = StableGestureSystem(swap_handedness=SWAP_HANDEDNESS)
        self.server = GestureWebSocketServer(self.get_payload, send_fps=TARGET_FPS)
        self._lock = threading.Lock()
        self._running = True
        self._latest_frame = None
        self._latest_payload = self._blank_payload()
        self._prev_pinch = {"Left": None, "Right": None}
        self._smooth_xy = {"Left": (0.5, 0.5), "Right": (0.5, 0.5)}
        self._smooth_alpha = 0.45
        self._xy_deadzone = 0.005
        self._ai_active = False
        self._speech = SpeechWorker()
        self._voice_assistant = VoiceAssistantService()
        self._pair_candidate = None
        self._pair_candidate_count = 0
        self._pair_hold_frames = 4
        self._last_voice_time = 0.0
        self._voice_cooldown_seconds = 1.0
        self._hand_first_seen_time = None
        self._gesture_activation_seconds = 1.2
        self._frame_interval = 1.0 / float(TARGET_FPS)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _blank_hand(self):
        return {
            "gesture": "NO_HAND",
            "x": 0.5,
            "y": 0.5,
            "palm_x": 0.5,
            "palm_y": 0.5,
            "wrist_x": 0.5,
            "wrist_y": 0.5,
            "middle_mcp_x": 0.5,
            "middle_mcp_y": 0.5,
            "dx": 0.0,
            "dy": 0.0,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "pinch": 0.0,
            "is_click": False,
            "scale_delta": 0.0,
            "scale_action": None,
        }

    def _blank_payload(self):
        return {"left": self._blank_hand(), "right": self._blank_hand()}

    def _capture_loop(self):
        next_tick = time.perf_counter()
        while self._running:
            now_tick = time.perf_counter()
            wait_s = next_tick - now_tick
            if wait_s > 0:
                time.sleep(min(wait_s, 0.01))
                continue
            next_tick = now_tick + self._frame_interval

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            # Mirror for selfie-style interaction so right hand feels like right hand on screen.
            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            frame, raw_gestures = self.gesture_system.process(frame)
            now = time.time()
            if raw_gestures:
                if self._hand_first_seen_time is None:
                    self._hand_first_seen_time = now
                gesture_ready = (now - self._hand_first_seen_time) >= self._gesture_activation_seconds
            else:
                self._hand_first_seen_time = None
                gesture_ready = False

            gestures = raw_gestures if gesture_ready else []
            self._handle_voice_cues(gestures)
            payload = self._to_payload(gestures)

            y = 32
            for g in gestures:
                text = f"{g.get('hand', '?')}: {g.get('gesture', 'UNKNOWN')}"
                cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                y += 30
            if raw_gestures and not gesture_ready and self._hand_first_seen_time is not None:
                remaining = max(0.0, self._gesture_activation_seconds - (now - self._hand_first_seen_time))
                cv2.putText(
                    frame,
                    f"Gesture warmup: {remaining:.1f}s",
                    (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 215, 255),
                    2,
                )

            with self._lock:
                self._latest_frame = frame
                self._latest_payload = payload

    def _handle_voice_cues(self, gestures):
        left_gesture = None
        right_gesture = None

        for g in gestures:
            hand = g.get("hand")
            if hand == "Left":
                left_gesture = g.get("gesture")
            elif hand == "Right":
                right_gesture = g.get("gesture")

        pair_gesture = None
        if left_gesture == right_gesture and left_gesture in {"TRIANGLE", "REVERSE_TRIANGLE"}:
            pair_gesture = left_gesture

        # Reset counter only when gesture changes
        if pair_gesture != self._pair_candidate:
            self._pair_candidate = pair_gesture
            self._pair_candidate_count = 1
        elif pair_gesture is not None:
            self._pair_candidate_count += 1

        # No gesture → do nothing
        if pair_gesture is None:
            return

        # Wait until gesture held for required frames
        if self._pair_candidate_count < self._pair_hold_frames:
            return

        now = time.time()
        if now - self._last_voice_time < self._voice_cooldown_seconds:
            return

        # ACTIVATE
        if pair_gesture == "TRIANGLE" and not self._ai_active:
            self._ai_active = True
            self._last_voice_time = now
            self._voice_assistant.start()
            #self._speech.speak_async("AI assistance is activated")

        # DEACTIVATE
        elif pair_gesture == "REVERSE_TRIANGLE" and self._ai_active:
            self._ai_active = False
            self._last_voice_time = now
            self._voice_assistant.stop()
            self._speech.speak_async("AI assistance is deactivated")

    def _to_payload(self, gestures):
        payload = self._blank_payload()

        for g in gestures:
            hand_name = g.get("hand")
            side = "left" if hand_name == "Left" else "right" if hand_name == "Right" else None
            if side is None:
                continue

            mapped = self.GESTURE_MAP.get(g.get("gesture"), "IDLE")
            raw_x = float(g.get("x", 0.5))
            raw_y = float(g.get("y", 0.5))
            prev_x, prev_y = self._smooth_xy.get(hand_name, (0.5, 0.5))
            alpha = self._smooth_alpha
            dx = raw_x - prev_x
            dy = raw_y - prev_y
            if abs(dx) < self._xy_deadzone:
                dx = 0.0
            if abs(dy) < self._xy_deadzone:
                dy = 0.0
            x = prev_x + dx * alpha
            y = prev_y + dy * alpha
            self._smooth_xy[hand_name] = (x, y)
            delta_x = max(-1.0, min(1.0, dx / 0.5))
            delta_y = max(-1.0, min(1.0, dy / 0.5))
            pinch = g.get("pinch_distance")
            prev = self._prev_pinch.get(hand_name)
            delta = 0.0
            scale_action = None
            raw_gesture = g.get("gesture")

            if pinch is not None:
                pinch = float(pinch)
                if prev is not None:
                    delta = pinch - prev
                    if delta > 0.008:
                        scale_action = "ENLARGE"
                        if mapped == "IDLE":
                            mapped = "STRETCH"
                    elif delta < -0.008:
                        scale_action = "REDUCE"
                        if mapped == "IDLE":
                            mapped = "PINCH"
                self._prev_pinch[hand_name] = pinch

            payload[side] = {
                "gesture": mapped,
                "x": x,
                "y": y,
                "palm_x": float(g.get("palm_x", raw_x)),
                "palm_y": float(g.get("palm_y", raw_y)),
                "wrist_x": float(g.get("wrist_x", raw_x)),
                "wrist_y": float(g.get("wrist_y", raw_y)),
                "middle_mcp_x": float(g.get("middle_mcp_x", raw_x)),
                "middle_mcp_y": float(g.get("middle_mcp_y", raw_y)),
                "dx": float(delta_x),
                "dy": float(delta_y),
                "delta_x": float(delta_x),
                "delta_y": float(delta_y),
                "pinch": float(delta),
                "is_click": bool(raw_gesture == "CLICK"),
                "scale_delta": float(abs(delta)),
                "scale_action": scale_action,
            }

        return payload

    def read_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_payload(self):
        messages = []
        if self._ai_active and self._voice_assistant is not None:
            voice_action = self._voice_assistant.pop_pending_action()
            if voice_action is not None:
                messages.append(voice_action)
        with self._lock:
            messages.append(self._latest_payload)
        return messages if len(messages) > 1 else messages[0]

    def close(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self._voice_assistant is not None:
            self._voice_assistant.stop()
        if self._speech is not None:
            self._speech.stop()

    def __del__(self):
        self.close()
