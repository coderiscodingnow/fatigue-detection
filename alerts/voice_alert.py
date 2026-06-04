# alerts/voice_alert.py
import pyttsx3
import threading
import time

class VoiceAlert:
    def __init__(self, threshold=0.65, cooldown=30):
        self.threshold = threshold
        self.cooldown = cooldown  # seconds between alerts
        self._last_alert = 0
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', 150)
        self._engine.setProperty('volume', 1.0)
        self._lock = threading.Lock()

    def _speak(self, message):
        with self._lock:
            self._engine.say(message)
            self._engine.runAndWait()

    def check_and_alert(self, alert_score):
        now = time.time()
        if alert_score >= self.threshold:
            if now - self._last_alert >= self.cooldown:
                self._last_alert = now
                t = threading.Thread(target=self._speak,
                                     args=("Driver alert, please rest",),
                                     daemon=True)
                t.start()
                return True
        return False