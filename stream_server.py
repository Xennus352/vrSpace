"""Live MJPEG view of the app + phone gyro control, streamed over Wi-Fi.

The render loop publishes the finished display surface through
``FrameStreamer.publish``; a background stdlib HTTP(S) server (no Flask)
serves it to any browser on the local network at ``GET /`` (HTML page) and
``GET /stream.mjpg`` (multipart/x-mixed-replace JPEG stream).

HTTPS is required: browsers only expose ``DeviceOrientationEvent`` (gyro)
in secure contexts, so a self-signed certificate is generated on first run
and cached (one-time browser warning). The phone page streams raw gyro
euler angles to ``POST /gyro``; ``GyroController`` turns them into
head-look + movement for the app.
"""

import http.server
import json
import os
import socket
import socketserver
import ssl
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Vision Glove - phone controller</title>
<style>
  html, body { margin:0; padding:0; height:100%; background:#000; color:#cfe9ff;
               font-family: ui-sans-serif, system-ui, sans-serif; overflow:hidden; }
  #view { position:fixed; inset:0; width:100vw; height:100vh; object-fit:contain; }
  .ui { position:fixed; z-index:10; left:50%; transform:translateX(-50%);
        bottom:12px; display:flex; gap:10px; }
  .ui button { background:rgba(0,120,200,.75); color:#fff; border:1px solid #4dd6ff;
               border-radius:14px; padding:14px 16px; font-size:18px; min-width:120px; }
  .ui button:active { background:rgba(0,160,240,.9); }
  #badge { position:fixed; z-index:10; top:10px; right:10px; font-size:14px;
           padding:6px 10px; border-radius:10px; background:rgba(0,0,0,.6);
           border:1px solid #4dd6ff; }
  #mode { position:fixed; z-index:10; top:10px; left:10px; font-size:14px;
          padding:6px 10px; border-radius:10px; background:rgba(0,0,0,.6); }
</style>
</head>
<body>
<img id="view" src="/stream.mjpg" alt="Loading view...">
<div id="mode">GYRO OFF</div>
<div id="badge">connecting...</div>
<div class="ui">
  <button id="gyro">Enable gyro</button>
  <button id="recenter">Recenter</button>
</div>
<script>
(function(){
  var stream = document.getElementById('view');
  var btn = document.getElementById('gyro');
  var mode = document.getElementById('mode');
  var badge = document.getElementById('badge');
  var rate = 1000/30;              // send gyro at ~30 Hz
  var last = 0, posting = false;
  var enabled = false;

  function send(cmd){
    fetch('/gyro', {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify(cmd)}).catch(function(){});
  }

  function onOrientation(e){
    var now = Date.now();
    if (now - last < rate) return;
    last = now;
    if (posting) return;
    posting = true;
    var angle = (window.screen && screen.orientation) ? screen.orientation.angle : 0;
    fetch('/gyro', {method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({
                      alpha: e.alpha, beta: e.beta, gamma: e.gamma,
                      angle: angle, t: now
                    })}).then(function(){ posting = false; if (badge) badge.textContent = 'gyro live'; }
                           ).catch(function(){ posting = false; });
  }

  function startGyro(){
    var h = function(e){ onOrientation(e); };
    window.addEventListener('deviceorientation', h);
    window.addEventListener('deviceorientationabsolute', h);
  }

  btn.addEventListener('click', function(){
    if (enabled) { send({cmd:'toggle'}); enabled=false; return; }
    var p = Promise.resolve();
    if (window.DeviceOrientationEvent && DeviceOrientationEvent.requestPermission){
      p = DeviceOrientationEvent.requestPermission();
    }
    p.then(function(state){
      if (state && state !== 'granted') { badge.textContent = 'gyro denied'; return; }
      startGyro();
      send({cmd:'on'});
      enabled = true;
      btn.textContent = 'Gyro OFF';
    }).catch(function(){ badge.textContent = 'gyro permission error'; });
  });

  document.getElementById('recenter').addEventListener('click', function(){ send({cmd:'recenter'}); });

  // keep the button in sync with the PC side (G key there)
  setInterval(function(){
    fetch('/gyro/state').then(function(r){ return r.json(); }).then(function(s){
      enabled = !!s.active;
      btn.textContent = enabled ? 'Gyro OFF' : (enabled ? 'Gyro OFF' : 'Enable gyro');
      mode.textContent = enabled ? 'GYRO ON' : 'GYRO OFF';
    }).catch(function(){});
  }, 2000);
})();
</script>
</body>
</html>
"""

_BOUNDARY = "frame"


def _ensure_self_signed_cert(root_dir="~/.config/spacex"):
    """Generate (once) and return (cert_path, key_path) for HTTPS."""
    d = Path(root_dir).expanduser()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None, None
    cert = d / "gyro_https.crt"
    key = d / "gyro_https.key"
    if cert.exists() and key.exists():
        return str(cert), str(key)
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", str(key), "-out", str(cert), "-days", "3650",
             "-subj", "/CN=spacex-phone-control", "-sha256"],
            check=True, capture_output=True, timeout=30)
        return str(cert), str(key)
    except Exception as exc:
        print(f"[stream] could not generate HTTPS cert: {exc}")
        return None, None


class _StreamHandler(http.server.BaseHTTPRequestHandler):
    server_version = "VisionGloveStream/1.0"
    streamer = None  # set on the handler class by the streamer

    def log_message(self, *_args):  # quiet
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path in ("/stream.mjpg", "/stream.mjpeg"):
            self._serve_stream()
        elif path == "/gyro/state":
            self._json({"active": bool(self.streamer.gyro
                                       and self.streamer.gyro.active),
                        "ts": int(time.time() * 1000)})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except Exception:
            data = {}
        gyro = self.streamer.gyro
        if gyro is not None:
            try:
                if "cmd" in data:
                    cmd = data.get("cmd")
                    if cmd == "recenter":
                        gyro.recenter()
                    elif cmd == "toggle":
                        gyro.toggle()
                    elif cmd == "on":
                        gyro.toggle(True)
                    elif cmd == "off":
                        gyro.toggle(False)
                else:
                    gyro.update(
                        float(data.get("alpha", 0.0) or 0.0),
                        float(data.get("beta", 0.0) or 0.0),
                        float(data.get("gamma", 0.0) or 0.0),
                        float(data.get("angle", 0.0) or 0.0),
                    )
            except Exception:
                pass
        self._json({"ok": True})

    # ------------------------------------------------------------------
    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        for chunk in self.streamer._mjpeg_stream():
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                break


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FrameStreamer:
    """Shared single-frame slot + MJPEG/gyro HTTP(S) server in a daemon thread."""

    def __init__(self, port=None, jpeg_quality=72, max_width=800, target_fps=20,
                 gyro=None):
        self.port = int(port or os.environ.get("STREAM_PORT", 8090))
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self.target_fps = target_fps
        self.gyro = gyro          # optional GyroController
        self._lock = threading.Lock()
        self._data = None
        self._thread = None
        self._server = None
        self._https = True

    # -- producer side (called from the render loop) --
    def publish(self, rgb):
        """Store the latest fully-composed RGB frame (HxWx3 numpy array)."""
        with self._lock:
            self._data = rgb

    def _snapshot(self):
        with self._lock:
            return self._data

    # -- consumer side (called by a stream connection thread) --
    def _encode_jpeg(self):
        frame = self._snapshot()
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape[1] > self.max_width:
            scale = self.max_width / float(frame.shape[1])
            frame = cv2.resize(frame,
                               (int(frame.shape[1] * scale),
                                int(frame.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)],
        )
        return jpg.tobytes() if ok else None

    def _mjpeg_stream(self):
        interval = 1.0 / max(1.0, float(self.target_fps))
        boundary = _BOUNDARY
        while True:
            start = time.perf_counter()
            jpg = self._encode_jpeg()
            if jpg is not None:
                yield (f"\r\n--{boundary}\r\n".encode()
                       + b"Content-Type: image/jpeg\r\n"
                       + b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                       + jpg)
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, interval - elapsed))

    # -- server --
    def start(self):
        if self._thread is not None:
            return
        _StreamHandler.streamer = self
        try:
            self._server = _ThreadingHTTPServer(("0.0.0.0", self.port),
                                                _StreamHandler)
        except OSError as exc:
            print(f"[stream] phone view server failed to bind port "
                  f"{self.port}: {exc}")
            return

        cert, key = _ensure_self_signed_cert()
        if cert and key:
            try:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert, key)
                self._server.socket = ctx.wrap_socket(
                    self._server.socket, server_side=True)
            except Exception as exc:
                print(f"[stream] HTTPS unavailable, falling back to plain "
                      f"HTTP (no phone gyro): {exc}")
                self._https = False
        else:
            self._https = False
            print("[stream] HTTPS certificate unavailable, serving plain HTTP "
                  "(no phone gyro).")

        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
            name="mjpeg-stream-server")
        self._thread.start()

    def url(self):
        scheme = "https" if self._https else "http"
        return f"{scheme}://{lan_ip()}:{self.port}/"

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            if self._thread is not None:
                self._thread.join(timeout=1.0)
                self._thread = None


def lan_ip():
    """Best-effort LAN IP of this machine (falls back to 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        pass
    finally:
        s.close()
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"