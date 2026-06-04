import cv2
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO
import os

WEBCAM_INDEX = 0
MODEL_PATH = "YOLO-Drowsiness-Detection/runs/train/exp/weights/best.pt"
PERCLOS_WINDOW = 60
ALERT_WEIGHTS = {"yolo": 0.50, "perclos": 0.35, "head": 0.15}

class VisionPipeline:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            print(f"Loading fine-tuned model from {MODEL_PATH}")
            self.model = YOLO(MODEL_PATH)
            self.use_finetuned = True
        else:
            print("Fine-tuned model not found — using generic YOLOv8n")
            self.model = YOLO("yolov8n.pt")
            self.use_finetuned = False

        self.perclos_buffer = deque(maxlen=PERCLOS_WINDOW)
        self.latest = {
            "drowsy_confidence": 0.0,
            "awake_confidence":  0.0,
            "perclos":           0.0,
            "head_state":        "unknown",
            "alert_score":       0.0,
            "label":             "unknown",
            "ts":                int(time.time()),
        }
        self._lock   = threading.Lock()
        self._running = False
        self._thread  = None

    def _head_state(self, x1, y1, x2, y2, frame_w, frame_h):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        if h == 0:
            return "unknown"
        ratio = w / h
        if cy > frame_h * 0.65:
            return "nodding"
        if ratio < 0.6:
            return "tilted"
        return "upright"

    def _process_frame(self, frame):
        h, w = frame.shape[:2]
        results = self.model(frame, verbose=False)[0]

        drowsy_conf = 0.0
        awake_conf  = 0.0
        head_state  = "unknown"

        if results.boxes and len(results.boxes):
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if self.use_finetuned:
                    cls_name = self.model.names[cls_id].lower()
                    if "drowsy" in cls_name and conf > drowsy_conf:
                        drowsy_conf = conf
                        head_state  = self._head_state(x1, y1, x2, y2, w, h)
                    elif "awake" in cls_name and conf > awake_conf:
                        awake_conf = conf
                        if head_state == "unknown":
                            head_state = self._head_state(x1, y1, x2, y2, w, h)
                else:
                    if conf > drowsy_conf:
                        drowsy_conf = conf * 0.3
                        head_state  = self._head_state(x1, y1, x2, y2, w, h)

        self.perclos_buffer.append(1 if drowsy_conf > 0.5 else 0)
        perclos = sum(self.perclos_buffer) / len(self.perclos_buffer) if self.perclos_buffer else 0.0

        head_score = {"nodding": 1.0, "tilted": 0.6, "upright": 0.0, "unknown": 0.0}.get(head_state, 0.0)
        alert_score = (
            ALERT_WEIGHTS["yolo"]    * drowsy_conf +
            ALERT_WEIGHTS["perclos"] * perclos +
            ALERT_WEIGHTS["head"]    * head_score
        )

        if drowsy_conf > 0.5:
            label = "drowsy"
        elif awake_conf > 0.5:
            label = "awake"
        else:
            label = "uncertain"

        return {
            "drowsy_confidence": round(drowsy_conf, 3),
            "awake_confidence":  round(awake_conf,  3),
            "perclos":           round(perclos,      3),
            "head_state":        head_state,
            "alert_score":       round(alert_score,  3),
            "label":             label,
            "ts":                int(time.time()),
        }

    def _draw_hud(self, frame, data):
        h, w = frame.shape[:2]
        color = (0, 0, 255) if data["alert_score"] > 0.65 else (0, 200, 0)

        cv2.rectangle(frame, (10, 10), (340, 175), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (340, 175), color, 2)

        lines = [
            f"Label       : {data['label'].upper()}",
            f"Drowsy conf : {data['drowsy_confidence']:.2f}",
            f"Awake conf  : {data['awake_confidence']:.2f}",
            f"PERCLOS     : {data['perclos']:.2f}",
            f"Head state  : {data['head_state']}",
            f"Alert score : {data['alert_score']:.2f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (18, 38 + i * 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        bar_x, bar_y, bar_w = 10, 185, int((w - 20) * data["alert_score"])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + w - 20, bar_y + 12), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), color, -1)
        return frame

    def _run(self):
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print(f"ERROR: Cannot open webcam index {WEBCAM_INDEX}")
            return

        print("Webcam opened. Press Q to quit.")
        last_print = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Empty frame — retrying")
                time.sleep(0.05)
                continue

            data = self._process_frame(frame)

            with self._lock:
                self.latest = data

            frame = self._draw_hud(frame, data)
            cv2.imshow("Fatigue Detection", frame)

            if time.time() - last_print >= 1.0:
                print(data)
                last_print = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

if __name__ == "__main__":
    pipeline = VisionPipeline()
    pipeline.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        pipeline.stop()