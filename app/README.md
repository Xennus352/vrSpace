# VrSpace Android headset display

The app displays desktop vrSpace stereo video and sends native Android fused
head orientation back to the same server. The USB webcam stays on the laptop
for hand tracking. The phone camera is not used.

This app requires the matching desktop vrSpace streaming server. The desktop
implementation and its scene assets are maintained separately from this app
folder. The server must support the token-authenticated `/stream`, `/pose`,
`/control` and `/status` endpoints; the legacy phone-gyro server is incompatible.

1. Install `VrSpace-headtracking.apk` (1.0.2, build 3). The original `VrSpace.apk`
   does not include the native head-orientation bridge.
2. Connect the laptop to the phone hotspot, or put both on the same Wi-Fi.
3. From the repository root run:

   ```sh
   uv run main.py --phone --stereo --ipd 64 --camera /dev/video3
   ```

   `/dev/video3` was the USB webcam found on this laptop. Prefer its stable
   `/dev/v4l/by-id/...-video-index0` path if indices change.
4. Keep vrSpace running. Paste its complete printed URL, including `?token=...`,
   into the Android app. There is no second server URL to configure.
5. Confirm **Head tracking connected**, face forward and tap **Recenter head**.
   The eye/menu button hides app controls; the page has a separate Hide controls
   button. Head movement rotates the view in yaw, pitch and roll without steering
   the ship. F1 or the phone's help button restores full game instructions.
6. Use **Fit inside headset** to reduce the image size until the edges are visible
   through your holder. Both eyes stay centered separately. The desktop uses a
   wider 105-degree horizontal view; add `--vr-fov 110` if you need more coverage.

To download the updated APK directly, open the display URL in the phone browser
and tap **Download updated VrSpace Android app**, install it, then use the app.
The server serves the `VrSpace-headtracking.apk` file in this folder.

On connection loss the last head orientation is held, the status disconnects,
and held phone movement keys expire. Recenter again if yaw drifts. Tracking
provides orientation only; lens correction, positional tracking and camera
passthrough are not implemented. Actual phone latency and optical alignment
still require hardware testing.

For connection problems, use the laptop's Wi-Fi IPv4 address and `http://`,
keep vrSpace open, and allow TCP 8080 from the phone through the laptop firewall.
`0.0.0.0` is the listening address, not the destination address.

Run `flutter analyze`, `flutter test`, and `flutter build apk --release` here.
The build is `build/app/outputs/flutter-apk/app-release.apk`; copy it to
`VrSpace-headtracking.apk` to update the server's download. The Android project
retains its existing debug signing key for local release builds.
