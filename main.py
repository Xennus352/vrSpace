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
import numpy as np
from OpenGL.GL import (
    GL_BACK,
    GL_RGB,
    GL_UNSIGNED_BYTE,
    glReadBuffer,
    glReadPixels,
)

# Pygame/OpenGL window
import pygame

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

CAMERA_INDEX = 0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
CAPTURE_FPS = 30
STREAM_FPS = 20  # MJPEG rate pushed to the phone view


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

    # ---- Phone view streaming + phone gyro control (MJPEG over Wi-Fi) ----
    streamer = None
    gyro = None
    try:
        from gyro_controller import GyroController
        from stream_server import FrameStreamer
        gyro = GyroController()
        streamer = FrameStreamer(target_fps=STREAM_FPS, gyro=gyro)
        streamer.start()
        print(f"Phone view + gyro: {streamer.url()}  (same Wi-Fi; P to reprint)")
        print("  G = gyro on/off | R = recenter (gyro or face) | gyro "
              "buttons on the phone page")
    except Exception as exc:
        print(f"[stream] phone view disabled: {exc}")

    # --- Producer / consumer state ---
    result_lock = threading.Lock()
    latest = {"frame": None, "raw": None, "gestures": [], "head": None}

    # ---- FRIDAY voice control (talk to her, she runs the system) ----
    _MATERIALS = ["graphite", "white", "skin", "hologram"]

    def _friday_chat(friday, text):
        print("You:", text)
        try:
            friday.chat(text)
        except Exception as exc:
            print("[friday] chat error:", exc)

    mic = None
    try:
        from friday_mic import VoiceListener

        def _stream_url():
            if streamer is not None:
                return streamer.url()
            return None

        def _scene_name():
            holo = renderer.holo
            return holo.scenes[holo.scene_i].get("name", "scene")

        friday = renderer.iron_man._friday
        friday.context_fn = lambda: (
            f"Iron Man mode {'active' if renderer.iron_man.active else 'idle'} | "
            f"score {renderer.iron_man.score} | wave {renderer.iron_man.wave} | "
            f"{len(renderer.iron_man.drones)} drones | "
            f"health {renderer.iron_man.health:.0f} | "
            f"energy {renderer.iron_man.energy:.0f} | "
            f"hologram {_scene_name()}"
        )

        def _recenter_view():
            if gyro is not None and gyro.active:
                gyro.recenter()
                return "Gyro view recentered."
            head_tracker.recenter()
            return "View recentered."

        friday.commands = {
            "ironman:on": lambda: (
                renderer.iron_man.activate(), "Suit up. Repulsors online.")[1],
            "ironman:off": lambda: (
                renderer.iron_man.deactivate(), "Suit powered down.")[1],
            "holo:toggle": lambda: (
                setattr(renderer.holo, "visible",
                        not renderer.holo.visible),
                f"Hologram {'on' if renderer.holo.visible else 'off'}.")[1],
            "scene:next": lambda: (
                renderer.holo.next_scene(),
                f"Now showing {_scene_name()}.")[1],
            "scene:prev": lambda: (
                renderer.holo.prev_scene(),
                f"Now showing {_scene_name()}.")[1],
            "materials:next": lambda: (
                renderer.glove.set_material(_MATERIALS[
                    (_MATERIALS.index(renderer.glove.material) + 1)
                    % len(_MATERIALS)]),
                f"Glove material {renderer.glove.material}.")[1],
            "glove:toggle": lambda: (
                setattr(renderer.glove, "opacity",
                        0.25 if renderer.glove.opacity != 0.25 else 0.85),
                "Glove opacity toggled.")[1],
            "recenter": lambda: _recenter_view(),
            "stream:url": lambda: (
                f"Phone view is at {_stream_url()}." if _stream_url()
                else "Phone view streaming is off."),
            "status": lambda: (
                f"Iron Man {'active' if renderer.iron_man.active else 'idle'}."
                f" Score {renderer.iron_man.score}, wave {renderer.iron_man.wave}, "
                f"{len(renderer.iron_man.drones)} drones on screen, "
                f"health {renderer.iron_man.health:.0f}, "
                f"energy {renderer.iron_man.energy:.0f}."),
            "help": lambda: (
                "I can run the suit, toggle the hologram, change scenes, "
                "cycle glove materials, recenter the view, and report status. "
                "Just tell me what you need, boss."),
        }

        fai = friday._ensure_ai()
        if fai is not None:
            mic = VoiceListener(
                transcriber=fai.transcribe,
                on_text=lambda text: _friday_chat(friday, text),
            )
            if mic.sd_ok:
                mic.start()
                print("FRIDAY voice: L = push to talk (mic)")
            else:
                mic = None
    except Exception as exc:
        print("[friday] voice control disabled:", exc)
        mic = None

    # --- Producer / consumer state (cont) ---

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

    print("Vision Pro gloves: POINT to aim | PINCH grab/drag/drop |")
    print("  PINCH both = zoom + rotate | FIST rotate |")
    print("  THUMBS-UP next scene | VICTORY reset | H toggle hologram")
    print("  F = ask FRIDAY AI | V = Iron Man mode on/off")
    print("VR controls: WASD move | Space/Ctrl up/down | Arrows look | Shift sprint | B stereo | R recenter")
    print("Phone gyro: G toggle | R recenter | tilt phone to walk | hold landscape")

    last_gesture = {"left": None, "right": None}
    gesture_events = {"last_event": None, "last_event_time": 0.0, "left": {}, "right": {}}
    event_cooldown = 0.0
    last_f_press = 0.0
    stream_last = 0.0

    hint_font = pygame.font.SysFont("freesansbold,dejavusans", 26)
    hud_font = pygame.font.SysFont("freesansbold,dejavusans", 30)

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
                        if gyro is not None and gyro.active:
                            gyro.recenter()
                            print("gyro view recentered")
                        else:
                            head_tracker.recenter()
                            print("head view recentered")
                    elif event.key == pygame.K_g:
                        if gyro is not None:
                            gyro.toggle()
                            print("phone gyro:", "on" if gyro.active else "off")
                    elif event.key == pygame.K_h:
                        renderer.holo.visible = not renderer.holo.visible
                        print("hologram:", "on" if renderer.holo.visible else "off")
                    elif event.key == pygame.K_f:
                        if time.time() - last_f_press > 0.4:
                            last_f_press = time.time()
                            renderer.iron_man.ask_friday()
                            print("FRIDAY: status report requested")
                    elif event.key == pygame.K_v:
                        if renderer.iron_man.active:
                            renderer.iron_man.deactivate()
                            print("Iron Man mode off")
                        else:
                            renderer.iron_man.activate()
                            print("Iron Man mode on - suit up!")
                    elif event.key == pygame.K_p:
                        if streamer is not None:
                            print("Phone view + gyro:", streamer.url())
                        else:
                            print("Phone view streaming is disabled")
                    elif event.key == pygame.K_l:
                        if mic is not None:
                            if mic.armed:
                                mic.disarm()
                                print("mic: listening off")
                            else:
                                mic.arm()
                                friday.say("Listening, boss.", 2.0)
                                print("mic: listening (push to talk)")
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

            # ---- Head pose: phone gyro (when active) else webcam face ----
            if gyro is not None and gyro.active:
                if not gyro.stale():
                    renderer.set_head_pose(*gyro.head())
                else:
                    # Phone lost contact -> fall back to the face tracker.
                    if head_pose is not None:
                        renderer.set_head_pose(
                            head_pose["yaw"], head_pose["pitch"],
                            head_pose["roll"])
            elif frame_to_process is not None and head_pose is not None:
                renderer.set_head_pose(
                    head_pose["yaw"], head_pose["pitch"], head_pose["roll"]
                )

            if frame_to_process is not None:
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
                # Feed the Iron Man flight/combat controller with palm positions
                # (normalized camera coords) + current gestures.
                palms = {}
                for label, lm in hand_landmarks.items():
                    if len(lm) < 9:
                        continue
                    palms[label] = {
                        "gesture": hands_state[label]["gesture"],
                        "palm": [(lm[0][0] + lm[9][0]) * 0.5,
                                 (lm[0][1] + lm[9][1]) * 0.5,
                                 (lm[0][2] + lm[9][2]) * 0.5],
                    }
                renderer.iron_man.set_hands(
                    palms.get("Left", {}).get("palm"),
                    palms.get("Right", {}).get("palm"),
                    palms.get("Left", {}).get("gesture", "IDLE"),
                    palms.get("Right", {}).get("gesture", "IDLE"),
                )
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
            if gyro is not None and gyro.active and not gyro.stale():
                gf, gb, gs = gyro.move()
                ctrl["fwd"] = ctrl["fwd"] or gf
                ctrl["back"] = ctrl["back"] or gb
                ctrl["sprint"] = ctrl["sprint"] or gs
            renderer.update(dt, ctrl)
            renderer.render(dt)

            # ---- Push the composed GL view to the phone (capped rate, never
            # fails the render loop). pygame.image.tostring returns a black
            # buffer on OpenGL/X11, so read the GL back buffer directly.
            if streamer is not None and now - stream_last >= 1.0 / STREAM_FPS:
                stream_last = now
                try:
                    w, h = renderer.viewport
                    buf = np.zeros((h, w, 3), np.uint8)
                    glReadBuffer(GL_BACK)
                    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE,
                                 buf.ctypes.data)
                    streamer.publish(np.ascontiguousarray(buf[::-1]))
                except Exception:
                    pass

            screen = pygame.display.get_surface()
            if renderer.iron_man.active:
                renderer.iron_man.draw_hud(screen)
            else:
                # subtle hint so the mode is discoverable
                hint = hint_font.render(
                    "V = Iron Man mode | F = ask FRIDAY AI | L = talk to FRIDAY",
                    True, (130, 200, 255))
                hint.set_alpha(170)
                screen.blit(hint, (16, screen.get_height() - 34))
            pygame.display.flip()

        print("Shutting down...")
    finally:
        capture_running = False
        if mic is not None:
            mic.close()
        if streamer is not None:
            streamer.close()
        if cap is not None and cap.isOpened():
            cap.release()
        pygame.quit()


def main():
    run()


if __name__ == "__main__":
    main()