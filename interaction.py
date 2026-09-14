"""Vision Pro style interactions: grab/drag/drop floating panels, point-to-focus.

State machine:
  * a pinch (CLICK gesture) near a panel grabs it
  * while pinched, the panel follows the pinching hand's palm in 3D
  * releasing the pinch drops the panel in place
  * an extended index finger (POINTING_UP) focuses the nearest panel
"""

import time
from glass_panel import GlassPanel

GRAB_RADIUS_PAD = 1.15  # factor over panel half-size for grab hit-test


class VisionInteraction:
    def __init__(self, view):
        self.view = view
        self.panels = [
            GlassPanel(self, view, 0.50, 0.50, -0.10, 0.34, 0.30, "Clock",
                       accent=(0.45, 0.95, 1.0)),
            GlassPanel(self, view, 0.50, 0.12, -0.05, 0.30, 0.22, "Status",
                       accent=(0.30, 1.0, 0.72)),
        ]
        self._pinch_state = {"Left": False, "Right": False}
        self._grabbed = None      # (hand_label, panel, offset_x, offset_y)
        self._palm_world = {}     # hand_label -> (x, y, z)
        self._tip_world = {}      # hand_label -> index tip world pos
        self._gesture = {}
        self._t_prev = 0.0

    # ------------------------------------------------------------------
    #  Per-frame update
    # ------------------------------------------------------------------
    def update(self, hands_state, aspect, dt):
        now = time.time()
        self._palm_world.clear()
        self._tip_world.clear()

        hands = {}
        for label, data in hands_state.items():
            lm = data.get("landmarks")
            gesture = data.get("gesture", "IDLE")
            if not lm or len(lm) < 21:
                continue
            hands[label] = (gesture, lm)
            # Use the SAME anchored/rotated/scaled transform as the glove so
            # grab & focus hit-tests line up with the rendered hand.
            pts, _unit = self.view.hand_pose(lm, aspect)
            self._palm_world[label] = (
                (pts[0][0] + pts[9][0]) / 2.0,
                (pts[0][1] + pts[9][1]) / 2.0,
                (pts[0][2] + pts[9][2]) / 2.0,
            )
            self._tip_world[label] = pts[8]

        # Resolve highlight (point-to-focus) before grab moves panels
        self._update_focus(hands, aspect)

        for label, (gesture, _lm) in hands.items():
            pinching = (gesture == "CLICK")
            prev = self._pinch_state.get(label, False)
            started = pinching and not prev
            released = (not pinching) and prev
            self._pinch_state[label] = pinching

            if self._grabbed and self._grabbed[0] == label:
                if released:
                    panel = self._grabbed[1]
                    panel.grabbed = False
                    self._grabbed = None
                    continue
                if pinching:
                    self._move_grabbed(label, aspect)
                    continue

            if started:
                palm = self._palm_world.get(label)
                if palm is None:
                    continue
                panel = self._nearest_panel(palm, aspect)
                if panel is not None:
                    panel.grabbed = True
                    self._grabbed = (label, panel, palm[0] - panel.center[0],
                                     palm[1] - panel.center[1])

        # Throttled live content
        if now - self._t_prev >= 0.5:
            self._t_prev = now
            self._refresh_content(hands_state)

    def _update_focus(self, hands, aspect):
        # reset highlights, keep grabbed panel lit
        for p in self.panels:
            p.highlight *= 0.6

        for label, (gesture, _lm) in hands.items():
            if gesture in ("POINTING_UP", "POINTING", "LEFT_POINTING", "RIGHT_POINTING"):
                tip = self._tip_world.get(label)
                if tip is None:
                    continue
                near = self._nearest_panel(tip, aspect)
                if near is not None:
                    near.highlight = min(1.0, near.highlight + 0.5)
            palm = self._palm_world.get(label)
            if palm is not None and self._grabbed is None:
                near = self._nearest_panel(palm, aspect)
                if near is not None:
                    near.highlight = min(1.0, near.highlight + 0.3)

    def _move_grabbed(self, label, aspect):
        palm = self._palm_world.get(label)
        if palm is None:
            return
        _hand, panel, ox, oy = self._grabbed
        panel.center[0] = palm[0] - ox
        panel.center[1] = palm[1] - oy
        # clamp into view so a panel can't be flung off screen
        limit_x = self.view.scale * aspect * 1.9
        limit_y = self.view.scale * 1.9
        panel.center[0] = max(-limit_x, min(limit_x, panel.center[0]))
        panel.center[1] = max(-limit_y, min(limit_y, panel.center[1]))

    def _nearest_panel(self, world, aspect):
        best, best_d = None, 1e9
        for p in self.panels:
            if p.grabbed:
                continue
            d = p.distance_to(world[0], world[1], aspect)
            hw = p.w_norm * self.view.scale * aspect * GRAB_RADIUS_PAD
            hh = p.h_norm * self.view.scale * GRAB_RADIUS_PAD
            if d < best_d and d < max(hw, hh):
                best, best_d = p, d
        return best

    # ------------------------------------------------------------------
    #  Panel content
    # ------------------------------------------------------------------
    def _refresh_content(self, hands_state):
        from datetime import datetime
        now_t = datetime.now().strftime("%H:%M:%S")
        clock_lines = [
            (now_t, (255, 235, 255, 255)),
            (datetime.now().strftime("%Y-%m-%d  %A"), (150, 200, 240, 255)),
            f"Panels {self.panels_len_lit()}",
        ]
        self.panels[0].set_content(clock_lines)

        lhs = hands_state.get("Left", {})
        rhs = hands_state.get("Right", {})
        status_lines = [
            f"L: {lhs.get('gesture', '—')}",
            f"R: {rhs.get('gesture', '—')}",
            ("Grab a panel, drag it, release" if not any(
                p.grabbed for p in self.panels) else "Panel grabbed — drag to move",
             (255, 200, 120, 255)),
        ]
        self.panels[1].set_content(status_lines)

    def panels_len_lit(self):
        return len(self.panels)

    def draw(self, aspect):
        # painter's order by z (farther first) for correct overlap
        for p in sorted(self.panels, key=lambda p: p.center[2], reverse=True):
            if not p.grabbed:
                p.draw(aspect)
        for p in self.panels:
            if p.grabbed:
                p.draw(aspect)