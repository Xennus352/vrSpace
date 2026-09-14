"""Interactive HUD / UI system rendered with pygame fonts (drawn after GL frame)."""

import time
import pygame


COLOR_DIM = (80, 90, 100)
COLOR_ACCENT = (0, 200, 255)
COLOR_GREEN = (80, 255, 120)
COLOR_RED = (255, 90, 80)
COLOR_WHITE = (230, 240, 250)
COLOR_GOLD = (255, 200, 80)


class UIOverlay:
    def __init__(self):
        pygame.font.init()
        self.font_big = pygame.font.Font(None, 42)
        self.font_title = pygame.font.Font(None, 52)
        self.font_hud = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 22)
        self.clock = pygame.time.Clock()
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self._last_fps_update = time.time()

    def update(self, dt):
        self.frame_count += 1
        now = time.time()
        if now - self._last_fps_update >= 0.5:
            self.fps = self.frame_count / max(0.001, now - self._last_fps_update)
            self.frame_count = 0
            self._last_fps_update = now

    def render(self, screen, gesture_info):
        """Draw all overlay UI on top of the passthrough scene."""
        w, h = screen.get_size()

        # Top-left title + FPS (small, unobtrusive)
        self._draw_panel(screen, (12, 12, 240, 34), accent=False)
        fps_text = self.font_hud.render(f"Vision Glove   FPS {self.fps:.0f}", True, COLOR_WHITE)
        screen.blit(fps_text, (22, 18))

        # Bottom-left gesture readout
        self._draw_panel(screen, (12, h - 150, 300, 138))
        screen.blit(self.font_hud.render("HAND TRACKING", True, COLOR_ACCENT), (22, h - 140))
        for idx, (hand, g) in enumerate(gedict_text(gesture_info)):
            color = COLOR_GREEN if g not in ("—", "WARMUP") else COLOR_DIM
            txt = self.font_hud.render(f"{hand}: {g}", True, color)
            screen.blit(txt, (22, h - 112 + idx * 26))

        # Bottom-right control legend
        legend = [
            "Pinch + drag   grab/move a panel",
            "Release pinch  drop it in place",
            "Point finger    focus panel",
            "V VR world | B single/stereo | R recenter",
            "F third person | WASD move | Arrows look",
            "Space/Ctrl up/down | Shift fast",
            "M material | O opacity | T texture",
            "+/- size | Esc / Q quit",
        ]
        self._draw_panel(screen, (w - 320, h - 226, 308, 214))
        screen.blit(self.font_hud.render("CONTROLS", True, COLOR_ACCENT), (w - 310, h - 216))
        for idx, line in enumerate(legend):
            screen.blit(self.font_small.render(line, True, COLOR_WHITE), (w - 310, h - 186 + idx * 23))

        # Status toast for recent events
        if gesture_info.get("last_event") and time.time() - gesture_info["last_event_time"] < 2.5:
            msg = gesture_info["last_event"]
            tw = self.font_hud.size(msg)[0]
            box_x = w // 2 - tw // 2 - 20
            self._draw_panel(screen, (box_x, 70, tw + 40, 36))
            screen.blit(self.font_hud.render(msg, True, COLOR_GOLD), (box_x + 20, 76))

    @staticmethod
    def _draw_panel(screen, rect, accent=True):
        surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        surf.fill((12, 16, 28, 150))
        pygame.draw.rect(surf, COLOR_ACCENT if accent else (70, 80, 95), surf.get_rect(), 1, border_radius=6)
        screen.blit(surf, (rect[0], rect[1]))


def gedict_text(gesture_info):
    """Return a deterministic hand -> gesture list for the readout panel."""
    left = gesture_info.get("left", {})
    right = gesture_info.get("right", {})
    return [("Left", left.get("gesture", "—") if left else "—"),
            ("Right", right.get("gesture", "—") if right else "—")]