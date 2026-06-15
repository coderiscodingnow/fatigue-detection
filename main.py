"""
main.py — Full fatigue detection pipeline
ESP32 sensors (via Firebase) + YOLO webcam + Weather + LightGBM + RL bandit + Voice alert
"""

import time
import json
import os
import threading
import requests
import datetime

from vision.yolo_stream     import VisionPipeline
from fusion.feature_builder import build_feature_vector
from fusion.train_model     import load_model, predict_fatigue
from rl.bandit              import AlertEngine, FirebaseAdapter
from alerts.voice_alert     import speak_alert
from firebase.db_client     import write_sensor, log_alert, get_db
from weather.weather_client import get_weather
from config import (
    DRIVER_ID,
    FIREBASE_READ_URL,
    FIREBASE_URL,
    FIREBASE_SECRET,
    POLL_INTERVAL_SEC,
    WEATHER_CITY,
    MODEL_PATH,
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(__file__)
STATUS_PATH          = os.path.join(BASE_DIR, "dashboard", "live_status.json")
STATUS_HISTORY_LIMIT = 90

# Ensure dashboard folder exists
os.makedirs(os.path.join(BASE_DIR, "dashboard"), exist_ok=True)

# ── Shared sensor state ───────────────────────────────────────
sensor_data   = {}
sensor_lock   = threading.Lock()
session_start = time.time()
last_ts       = None

# ── Firebase polling thread ───────────────────────────────────
def poll_firebase():
    global sensor_data, last_ts
    print(f"[Firebase] Polling Firebase via Admin SDK every {POLL_INTERVAL_SEC}s")
    while True:
        try:
            fb_db = get_db()
            if fb_db is not None:
                payload = fb_db.reference("/drowsiness/latest").get()
            else:
                response = requests.get(FIREBASE_READ_URL, timeout=5)
                if response.status_code == 200:
                    payload = response.json()
                else:
                    payload = None
                    print(f"[Firebase] HTTP {response.status_code}")

            if payload is None:
                print("[Firebase] No data at path yet. Waiting for ESP32...")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            if not isinstance(payload, dict):
                print(f"[Firebase] Warning: Payload is not a dict: {payload}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            current_ts = payload.get("ts")
            if current_ts != last_ts:
                last_ts = current_ts
                with sensor_lock:
                    sensor_data = payload
                write_sensor(DRIVER_ID, payload)
                print(f"[Firebase] New reading → "
                      f"jerk={payload.get('jerk_rms')}  "
                      f"dist={payload.get('distance_cm')}cm  "
                      f"ts={current_ts}")
        except Exception as e:
            print(f"[Firebase] Error in poll thread: {e}")
        time.sleep(POLL_INTERVAL_SEC)

threading.Thread(target=poll_firebase, daemon=True).start()

# ── Vision pipeline ───────────────────────────────────────────
pipeline = VisionPipeline()
pipeline.start()
print("[Vision] Webcam pipeline started.")

# ── Load LightGBM fusion model ────────────────────────────────
model = load_model(MODEL_PATH)
print("[Fusion] Model loaded.")

# ── RL bandit ─────────────────────────────────────────────────
db     = FirebaseAdapter("serviceAccountKey.json", FIREBASE_URL)
engine = AlertEngine(driver_id=DRIVER_ID, db=db)
engine.start_session()
print(f"[RL] Session started for {DRIVER_ID}")

# ── Weather ───────────────────────────────────────────────────
weather_cache      = get_weather()
last_weather_fetch = time.time()

# ── Alert cooldown ────────────────────────────────────────────
last_alert_time = 0
ALERT_COOLDOWN  = 30
latest_alert_msg = "No alert recorded"

print("\n=== System running. Press Ctrl+C to stop. ===\n")

# ── Write live_status.json for dashboard ─────────────────────
def write_live_status(payload: dict):
    """
    Writes latest values + rolling history to dashboard/live_status.json.
    Dashboard reads this file every second to update all panels.
    """
    history = []
    try:
        with open(STATUS_PATH, "r") as f:
            previous = json.load(f)
            history  = previous.get("history", [])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        history = []

    # Build history point — only numeric values
    history_point = {"time": payload["time"]}
    for key, value in payload.items():
        if key in ("driver_id", "time", "weather", "head_state",
                   "history", "latest_alert", "weather_city"):
            continue
        if isinstance(value, (int, float)):
            history_point[key] = value
    history.append(history_point)
    payload["history"] = history[-STATUS_HISTORY_LIMIT:]

    # Atomic write — prevents dashboard reading a half-written file
    temp_path = f"{STATUS_PATH}.tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        print(f"[Dashboard] Error writing temp status file: {e}")
        return

    for attempt in range(5):
        try:
            os.replace(temp_path, STATUS_PATH)
            break
        except PermissionError:
            time.sleep(0.05)
        except OSError as e:
            print(f"[Dashboard] Error replacing status file: {e}")
            break
    else:
        print("[Dashboard] Warning: Could not update live_status.json due to Windows file lock.")

# ── Alert message builder ─────────────────────────────────────
def build_alert_message(fusion_score, yolo_conf, perclos, weather, hour) -> str:
    if fusion_score > 0.80:
        severity = "You are severely drowsy"
    elif fusion_score > 0.60:
        severity = "You are showing signs of fatigue"
    else:
        severity = "Please stay alert"

    time_msg    = ""
    weather_msg = ""

    if 0 <= hour <= 5:
        time_msg = "It is very late at night"
    elif 13 <= hour <= 15:
        time_msg = "Post-lunch drowsiness is common at this hour"

    if any(w in weather for w in ["rain", "fog", "mist"]):
        weather_msg = f"Visibility is reduced due to {weather}"

    parts = [severity]
    if time_msg:    parts.append(time_msg)
    if weather_msg: parts.append(weather_msg)
    parts.append("Please pull over and take a rest")
    return ". ".join(parts) + "."

# ── Main loop ─────────────────────────────────────────────────
try:
    while True:
        # 1. Refresh weather every 10 minutes
        if time.time() - last_weather_fetch > 600:
            weather_cache      = get_weather()
            last_weather_fetch = time.time()

        # 2. Get latest vision data
        vision = pipeline.latest

        # 3. Get latest sensor data from Firebase polling thread
        with sensor_lock:
            sensors = dict(sensor_data)

        if not sensors:
            print("[Main] Waiting for first ESP32 reading from Firebase...")
            time.sleep(1)
            continue

        # 4. Extract sensor values
        jerk_rms    = float(sensors.get("jerk_rms",      0.0) or 0.0)
        posture_dev = float(sensors.get("posture_dev_cm", 0.0) or 0.0)
        distance_cm = float(sensors.get("distance_cm",    0.0) or 0.0)
        hour        = int(sensors.get("hour", datetime.datetime.now().hour))
        session_min = (time.time() - session_start) / 60

        # 5. Build feature vector
        features = build_feature_vector(
            yolo_conf   = vision["drowsy_confidence"],
            perclos     = vision["perclos"],
            head_state  = vision["head_state"],
            jerk_rms    = jerk_rms,
            posture_dev = posture_dev,
            hour        = hour,
            weather     = weather_cache["condition"],
            session_min = session_min,
            distance_cm = distance_cm,
        )

        # 6. LightGBM fusion score
        fusion_score = predict_fatigue(model, features)

        # 7. RL bandit
        should_alert = engine.evaluate(
            yolo_conf   = vision["drowsy_confidence"],
            jerk_rms    = jerk_rms,
            posture_dev = posture_dev,
            hour        = hour,
            weather     = weather_cache["condition"],
            session_min = session_min,
            distance_cm = distance_cm,
        )

        # 8. Print live status
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now_str}] "
              f"YOLO={vision['drowsy_confidence']:.2f}  "
              f"PERCLOS={vision['perclos']:.2f}  "
              f"Jerk={jerk_rms:.2f}  "
              f"Posture={posture_dev:.1f}cm  "
              f"Dist={distance_cm:.1f}cm  "
              f"Fusion={fusion_score:.2f}  "
              f"Weather={weather_cache['condition']}")

        # 9. Write live_status.json — dashboard reads this
        write_live_status({
            "driver_id":       DRIVER_ID,
            "time":            now_str,
            "timestamp":       time.time(),
            "yolo_confidence": vision["drowsy_confidence"],
            "perclos":         vision["perclos"],
            "head_state":      vision["head_state"],
            "jerk_rms":        jerk_rms,           # from Firebase/ESP32
            "posture_dev_cm":  posture_dev,         # from Firebase/ESP32
            "distance_cm":     distance_cm,         # from Firebase/ESP32
            "fusion_score":    fusion_score,
            "weather":         weather_cache["condition"],
            "weather_score":   weather_cache.get("score"),
            "weather_city":    WEATHER_CITY,
            "session_min":     session_min,
            "hour":            hour,
            "latest_alert":    latest_alert_msg,
        })

        # 10. Voice alert with cooldown
        if should_alert and (time.time() - last_alert_time > ALERT_COOLDOWN):
            last_alert_time  = time.time()
            latest_alert_msg = build_alert_message(
                fusion_score = fusion_score,
                yolo_conf    = vision["drowsy_confidence"],
                perclos      = vision["perclos"],
                weather      = weather_cache["condition"],
                hour         = hour,
            )
            print(f"\n ALERT: {latest_alert_msg}\n")
            speak_alert(latest_alert_msg)
            log_alert(DRIVER_ID, engine.session_id, {
                "fusion_score": fusion_score,
                "message":      latest_alert_msg,
                "weather":      weather_cache["condition"],
                "hour":         hour,
            })
            engine.record_feedback("ack")

        time.sleep(1)

except KeyboardInterrupt:
    engine.end_session()
    pipeline.stop()
    print("\nSystem stopped cleanly.")