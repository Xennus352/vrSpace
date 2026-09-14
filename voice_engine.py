import pyttsx3
from threading import Thread

class VoiceEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Optional: adjust voice rate, volume
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

    def speak(self, text):
        """Blocking speak"""
        self.engine.say(text)
        self.engine.runAndWait()

    def speak_async(self, text):
        """Non-blocking speak"""
        t = Thread(target=self.speak, args=(text,), daemon=True)
        t.start()
