"""Vision Pro-style hand tracking passthrough.

Pure Python (PyOpenGL + pygame + OpenCV + MediaPipe). The camera thread runs
only the cheap webcam head tracker; the expensive hand/gesture inference runs
on its own decoupled thread so the GL render loop — and head tracking — are
never delayed by it.
"""

import os
import threading
import time

import cv2

# Pygame/OpenGL window
import pygame

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
CAPTURE_FPS = 30


def run():
    from renderer import PassthroughRenderer
    from gesture_detector import GestureDetector
    from vr_head import HeadTracker

    # Open camera first (fail fast with a friendly message)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open camera. Is it connected and free?")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = GestureDetector()
    head_tracker = HeadTracker()

    # Pygame / OpenGL init
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
    )
    pygame.display.set_caption("Vision Glove — VR Hologram")

    renderer = PassthroughRenderer()
    renderer.init_gl()
    renderer.set_viewport(WINDOW_WIDTH, WINDOW_HEIGHT)

    # --- Producer / consumer state ---
    result_lock = threading.Lock()
    latest = {"frame": None, "raw": None, "gestures": [], "head": None}

    running = True
    clock = pygame.time.Clock()

    # --- Camera thread (head tracking; never does the heavy hand pass) ---
    capture_running = True

    def camera_loop():
        while capture_running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            frame = cv2.flip(frame, 1)
            try:
                head_pose = head_tracker.process(frame)
            except Exception:
                head_pose = None
            with result_lock:
                latest["frame"] = frame
                latest["raw"] = frame
                latest["head"] = head_pose

    # --- Hand/gesture thread (decoupled from capture so it can't add lag) ---
    def hand_loop():
        while capture_running:
            with result_lock:
                f = latest["raw"]
            if f is None:
                time.sleep(0.005)
                continue
            try:
                _, gestures = detector.process_frame(f)
            except Exception:
                gestures = []
            with result_lock:
                latest["gestures"] = gestures

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    hand_thread = threading.Thread(target=hand_loop, daemon=True)
    cam_thread.start()
    hand_thread.start()

    print("Hologram controls: FIST rotate | two FISTS zoom | PINCH grab/move | "
          "THUMBS-UP next scene | VICTORY reset | H toggle on/off")
    print("VR controls: WASD move | Space/Ctrl up/down | Arrows look | Shift sprint | B stereo | R recenter")

    last_gesture = {"left": None, "right": None}
    gesture_events = {"last_event": None, "last_event_time": 0.0, "left": {}, "right": {}}
    event_cooldown = 0.0

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        renderer.view.scale = min(3.0, renderer.view.scale + 0.15)
                    elif event.key == pygame.K_MINUS:
                        renderer.view.scale = max(0.5, renderer.view.scale - 0.15)
                    elif event.key == pygame.K_m:
                        mats = ["graphite", "white", "skin", "hologram"]
                        cur = mats.index(renderer.glove.material)
                        renderer.glove.set_material(mats[(cur + 1) % len(mats)])
                        print("material:", renderer.glove.material)
                    elif event.key == pygame.K_o:
                        renderer.glove.opacity = 0.25 if renderer.glove.opacity != 0.25 else 0.85
                        print("opacity:", renderer.glove.opacity)
                    elif event.key == pygame.K_t:
                        renderer.glove._tex_on = not renderer.glove._tex_on
                        print("texture:", renderer.glove._tex_on)
                    elif event.key == pygame.K_b:
                        renderer.stereo = not renderer.stereo
                        print("VR lens mode:", "stereo split" if renderer.stereo else "single frame")
                    elif event.key == pygame.K_r:
                        head_tracker.recenter()
                        print("head view recentered")
                    elif event.key == pygame.K_h:
                        renderer.holo.visible = not renderer.holo.visible
                        print("hologram:", "on" if renderer.holo.visible else "off")
                elif event.type == pygame.VIDEORESIZE:
                    w, h = event.w, event.h
                    pygame.display.set_mode(
                        (w, h),
                        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
                    )
                    renderer.set_viewport(w, h)

            dt = clock.tick(TARGET_FPS) / 1000.0
            now = time.time()

            # ---- Grab the freshest paired result (no blocking) ----
            frame_to_process = None
            gestures = []
            head_pose = None
            with result_lock:
                if latest["frame"] is not None:
                    frame_to_process = latest["frame"]
                    gestures = latest["gestures"]
                    head_pose = latest["head"]
                    latest["frame"] = None
                    latest["head"] = None

            if frame_to_process is not None:
                if head_pose is not None:
                    renderer.set_head_pose(
                        head_pose["yaw"], head_pose["pitch"], head_pose["roll"]
                    )
                renderer.set_background(frame_to_process)

                hand_landmarks = {}
                for g in gestures:
                    if "landmarks" in g:
                        hand_landmarks[g["hand"]] = g["landmarks"]
                    if g["gesture"] == "WARMUP":
                        continue
                    last_gesture[g["hand"].lower()] = g
                    gesture_events[g["hand"].lower()] = g
                renderer.set_hands(hand_landmarks)

                hands_state = {}
                for label, lm in hand_landmarks.items():
                    prev = last_gesture.get(label.lower()) or {}
                    hands_state[label] = {
                        "gesture": prev.get("gesture", "IDLE"),
                        "landmarks": lm,
                    }
                aspect = renderer.viewport[0] / max(1.0, renderer.viewport[1]) \
                    if renderer.viewport else 16.0 / 9.0
                renderer.holo.handle_hands(hands_state, aspect)
                ev = renderer.holo.pop_event()
                if ev:
                    gesture_events["last_event"] = ev
                    gesture_events["last_event_time"] = now

                if gestures and now - event_cooldown > 1.5:
                    top = max(gestures, key=lambda g: abs(g.get("dy", 0)) + abs(g.get("dx", 0)))
                    gesture_events["last_event"] = f"{top['hand']}: {top['gesture']}"
                    gesture_events["last_event_time"] = now
                    event_cooldown = now

            # ---- Update & render (never blocked by MediaPipe) ----
            keys = pygame.key.get_pressed()
            ctrl = {
                "fwd": keys[pygame.K_w],
                "back": keys[pygame.K_s],
                "left": keys[pygame.K_a],
                "right": keys[pygame.K_d],
                "up": keys[pygame.K_SPACE],
                "down": keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL],
                "sprint": keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT],
                "turn": int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
                "pitch": int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN]),
                "head_yaw": renderer._yaw,
            }
            renderer.update(dt, ctrl)
            renderer.render(dt)
            pygame.display.flip()

        print("Shutting down...")
    finally:
        capture_running = False
        if cap is not None and cap.isOpened():
            cap.release()
        pygame.quit()


def main():
    run()


if __name__ == "__main__":
    main()