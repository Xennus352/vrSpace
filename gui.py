from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtGui import QImage, QPixmap
import sys

class AppGUI(QWidget):
    def __init__(self, camera_server, web_url):
        super().__init__()
        self.camera_server = camera_server
        self.setWindowTitle("GlobeMan - Gesture Controller")
        self.resize(1280, 720)

        # 1. Setup Main Layout (Full Screen WebView)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0) # Remove padding

        self.webview = QWebEngineView()
        self.webview.setUrl(QUrl(web_url))
        self.webview.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.webview.loadFinished.connect(lambda _ok: self.webview.setFocus())
        QTimer.singleShot(300, self.webview.setFocus)
        self.layout.addWidget(self.webview)

        # 2. Setup Floating Camera Overlay
        self.cam_label = QLabel(self)
        self.cam_label.setFixedSize(320, 180) # Modern 16:9 ratio
        self.cam_label.move(20, 520) # Bottom-left
        
        # Apply the "Tech" Styling
        self.cam_label.setStyleSheet("""
            QLabel {
                border: 2px solid #00E5FF; 
                border-radius: 12px; 
                background-color: rgba(0, 0, 0, 180);
            }
        """)

        # 3. Setup Status Indicator (Feedback for the user)
        self.status_label = QLabel("Ready to explore...", self)
        self.status_label.setFixedSize(200, 40)
        self.status_label.move(20, 470)
        self.status_label.setStyleSheet("""
            background-color: rgba(0, 229, 255, 40);
            color: #00E5FF;
            font-weight: bold;
            border-radius: 5px;
            padding-left: 10px;
        """)

        # Timer to update camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(50)

        self.show()

    def update_camera(self):
        frame = self.camera_server.read_frame()
        if frame is not None:
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            
            # Scale to fill the label (KeepAspectRatioByExpanding)
            pixmap = QPixmap.fromImage(img).scaled(
                self.cam_label.width(),
                self.cam_label.height(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding, # This fills the box
                Qt.TransformationMode.FastTransformation
            )

            # Optional: Crop the center so it doesn't "spill out" of the label
            self.cam_label.setPixmap(pixmap.copy(
                (pixmap.width() - self.cam_label.width()) // 2,
                (pixmap.height() - self.cam_label.height()) // 2,
                self.cam_label.width(),
                self.cam_label.height()
            ))

def run_gui(camera_server, web_url):
    app = QApplication(sys.argv)
    window = AppGUI(camera_server, web_url)
    sys.exit(app.exec())
