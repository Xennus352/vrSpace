# Upgrade: World-Locked Persisted Item Placement (ArUco "Space Anchor")

> Status: implemented as the anchor layer of the flat passthrough mode.

## Goal

Place glowing 3D items at exact physical spots on the desk, lock them to the desk
in real-world coordinates, and have them reappear **exactly where they were**
after the webcam looks away and back — and after app restarts.

This is the single-lens-webcam stand-in for Meta Quest 3 "Space Setup"
anchoring: not whole-room SLAM, but an *mm-exact, desk-anchored* alternative
that requires no phone app, no depth camera, and no C++ SLAM build.

## Principle

A printed **ArUco marker** taped flat on the desk is the fixed metric world
origin. Knowing the real marker size gives exact real-world scale, so 3D anchor
points stored in marker coordinates survive:

- camera rotation away and back (pose is re-recovered by marker re-detection),
- app restarts (anchors serialize to `anchors.json`),
- camera distance changes (scale is metric, not estimate).

## Hardware prep

1. Print a 5–7 cm ArUco marker. A generator is provided:
   `python -m placement` writes `aruco_marker.png` (use the printed size).
2. Tape the marker flat on the desk. Optional: place 2–3 markers far apart and
   the engine uses the largest visible one per frame (wider usable area).
3. Measure the printed marker's side in centimeters and set it in the app
   (`[` / `]` keys) or in `anchors.json` (`marker_size_m`).

## Camera model (one-time)

Webcam intrinsics are approximated from the vertical FOV:

- `fy = (H/2) / tan(fov_deg/2)`, `fx = fy`, principal point at image center.
- `set_fov_deg()` rebuilds the intrinsics; the same FOV drives the OpenGL
  projection so item size/placement match the passthrough exactly.
- If you have full-calibration intrinsics, drop them into `intrinsics.json`
  (`{"K": [...9 floats], "dist": [...], "fov_deg": ..., "size": [W,H]}`) and they
  take precedence.

Coordinate conventions:

- Item positions are 3D points in **marker frame** (meters; the marker plane
  is `z = 0`, i.e. the desk surface where the marker lies flat).
- Camera pose comes from `cv2.solvePnP` (OpenCV camera space: +Z forward, image
  y DOWN). Points are pushed into OpenGL camera space with
  `p_gl = (x, -y, -z)` to match the identity-view flat pass (GL looks down −Z,
  y UP).

## Modules

### `ar_anchor.py` (new) — the anchor engine

- `ArucoAnchor(marker_size_m, fov_deg, dictionary=DICT_4X4_50)`
  - `detect(bgr)` → picks the largest detected marker, derives pose via
    `cv2.solvePnP` (`SOLVEPNP_IPPE_SQUARE` where available), caches `R, t`.
    Returns `False` when the marker is out of view (**tracking lost**).
  - `project_to_gl(p_marker)` → GL-space position or `None` if behind camera.
  - `pixel_to_marker(u, v)` → desk point in marker frame from an image pixel
    (ray ∩ marker plane). **This is the "place it exactly there" primitive.**
  - FOV / marker-size / frame-size setters rebuild intrinsics cheaply.
- `generate_marker_png(path, size_px, marker_id)` — printable marker.

### `placement.py` (new) — items + interaction

- `PlacedObject{pos (marker meters), style: box|sphere, color, size (meters)}`
- `ArPlacement(anchor)` — owner of the item list:
  - `update_pose(frame)` → runs `anchor.detect`.
  - `ingest(hands_state, img_w, img_h)` → pinch (CLICK) gesture handling:
    - while pinching: ghost preview at the desk point under the fingertip
      (thumb/index midpoint), drawn translucent;
    - on pinch release: commit a `PlacedObject` at that exact desk point;
    - pinch near a placed item (hit-test on its projected screen pos) → re-place.
  - `place_at_window(mx, my, win_w, win_h)` and `undo()` / `clear()` — used by
    mouse fallback.
  - `draw()` → renders items + ghost under the current projection (webcam FOV):
    pulsing hologram body (box or sphere), additive glow halo behind it, and a
    base ring at the desk contact point. Fades out while tracking is lost.
  - `save(path)` / `load(path)` → `anchors.json` persistence (marker size, FOV,
    every item). Loaded on startup, saved on quit/place/clear.

### `main.py` (extend)

- HD capture 1280×720 (was 640×360) for a crisp passthrough.
- `A` toggles anchor mode (forces flat passthrough) and creates/removes the
  `ArPlacement`.
- Per frame in anchor mode: `ar_placement.update_pose(frame)` then
  `ar_placement.ingest(hands_state, ...)`.
- Mouse fallback in anchor mode: left-click = place at cursor, right-click =
  undo.
- Keys: `[`/`]` marker size ±1 cm, `;`/`'` webcam FOV ±2°, `Backspace` undo,
  `C` clear, `A` toggle; saves `anchors.json` on exit.

### `renderer.py` (extend)

- `anchor_placement` (+ `anchor_mode`) on the renderer.
- In `_render_flat`, when anchor mode is active: set projection to the webcam
  FOV, draw `anchor_placement.draw()`, restore the 60° projection, then draw the
  glove/panels as before.

### `ui_overlay.py` (extend)

- HUD line (top-right, anchor mode only):
  `TRACKING OK/… · ITEMS N · MARKER 6cm · FOV 60°`.

## Controls (anchor mode)

| Input | Action |
|---|---|
| `A` | toggle anchor mode on/off |
| Pinch hold (CLICK) | ghost preview at exact desk point |
| Pinch release | place an item exactly there |
| Left click | place at cursor |
| Right click / `Backspace` | undo last |
| `C` | clear all items |
| `[` / `]` | marker size ±1 cm |
| `;` / `'` | webcam FOV ±2° |

## The demo it delivers

1. Tape marker → run → `A` → aim at desk → pinch/click to place items.
2. Turn the webcam fully away → items fade (tracking lost).
3. Turn back to the desk → items **reappear instantly at the same physical
   spots** (mm-exact).
4. Quit the app, relaunch, `A`, aim at desk → items restore from
   `anchors.json`.

## Limits vs Quest 3

- Anchoring is **desk-local**: the marker must be in view. Walk the marker out
  of frame and items hide (they never move) until it is re-locked.
- No whole-room mesh, no markerless 6DoF walking.
- Within the marker's field of view the anchor is *more* stable than Quest
  (known metric scale, no SLAM drift), but coverage is smaller.

## Future upgrade paths (documented, out of scope)

- **Markerless keyframe re-localization** — ORB features of the desk itself act
  as the anchor (no paper marker), trading robustness for zero setup.
- **ORB-SLAM3** — full monocular 6DoF with in-session map persistence; heavy
  C++ build and CPU cost.
- **Phone ARCore relay** — Android app streams pose + planes + cloud anchors;
  true room-anchoring with cross-session Cloud Anchors, at the cost of an app
  build and network latency.