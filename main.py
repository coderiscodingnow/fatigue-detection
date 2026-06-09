"""
main.py — Full fatigue detection pipeline
ESP32 sensors (via Firebase) + YOLO webcam + Weather + LightGBM + RL bandit + Voice alert

MQTT removed. ESP32 now pushes to Firebase Realtime DB.
main.py polls Firebase every 2 seconds instead of subscribing to MQTT.
"""

import time
import json
import os
import threading
import requests                             # replaces paho.mqtt

from vision.yolo_stream       import VisionPipeline
from fusion.feature_builder   import build_feature_vector
from fusion.train_model       import load_model, predict_fatigue
from rl.bandit                import AlertEngine, FirebaseAdapter
from alerts.voice_alert       import speak_alert
from firebase.db_client       import write_sensor, log_alert
from weather.weather_client   import get_weather
from config                   import (
    DRIVER_ID,
    FIREBASE_READ_URL,          # replaces BROKER, MQTT_PORT, MQTT_TOPIC
    FIREBASE_URL,
    FIREBASE_SECRET,
    POLL_INTERVAL_SEC,
    WEATHER_CITY,
)

# ── Shared sensor state ───────────────────────────────────────
sensor_data   = {}
sensor_lock   = threading.Lock()
session_start = time.time()
last_ts       = None           # tracks last seen timestamp to skip stale data

# ── Firebase polling thread (replaces MQTT loop_forever) ──────
def poll_firebase():
    """
    Runs in background thread.
    Every POLL_INTERVAL_SEC seconds, fetches /drowsiness/latest from Firebase
    and updates sensor_data — exactly what on_mqtt_message used to do.
    """
    global sensor_data, last_ts
    print(f"[Firebase] Polling {FIREBASE_READ_URL} every {POLL_INTERVAL_SEC}s")

    while True:
        try:
            response = requests.get(FIREBASE_READ_URL, timeout=5)
            if response.status_code == 200:
                payload = response.json()

                if payload is None:
                    print("[Firebase] No data at path yet. Waiting for ESP32...")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                current_ts = payload.get("ts")

                # Only update if ESP32 has written new data
                if current_ts != last_ts:
                    last_ts = current_ts
                    with sensor_lock:
                        sensor_data = payload
                    # Write to Firebase log (same as before — write_sensor was
                    # called inside on_mqtt_message previously)
                    write_sensor(DRIVER_ID, payload)
                    print(f"[Firebase] New reading: ts={current_ts}")
            else:
                print(f"[Firebase] HTTP {response.status_code} — check URL/secret")

        except requests.exceptions.RequestException as e:
            print(f"[Firebase] Network error: {e}")

        time.sleep(POLL_INTERVAL_SEC)

# Start Firebase polling in background (replaces mqtt_client.loop_forever thread)
threading.Thread(target=poll_firebase, daemon=True).start()

# ── Vision pipeline ───────────────────────────────────────────
pipeline = VisionPipeline()
pipeline.start()
print("[Vision] Webcam pipeline started.")

# ── Load LightGBM fusion model ────────────────────────────────
model = load_model("fusion/fusion_model.pkl")
print("[Fusion] Model loaded.")

# ── RL bandit ─────────────────────────────────────────────────
db     = FirebaseAdapter("serviceAccountKey.json",
                         "https://fatigue-detection-26427-default-rtdb.asia-southeast1.firebasedatabase.app")
engine = AlertEngine(driver_id=DRIVER_ID, db=db)
engine.start_session()
print(f"[RL] Session started for {DRIVER_ID}")

# ── Weather — fetch once every 10 minutes ─────────────────────
weather_cache      = get_weather()
last_weather_fetch = time.time()

# ── Alert cooldown — don't spam alerts ────────────────────────
last_alert_time  = 0
ALERT_COOLDOWN   = 30   # seconds
STATUS_PATH      = os.path.join(os.path.dirname(__file__), "dashboard", "live_status.json")
STATUS_HISTORY_LIMIT = 90

print("\n=== System running. Press Ctrl+C to stop. ===\n")

def write_live_status(payload):
    history = []
    try:
        with open(STATUS_PATH, "r") as f:
            previous = json.load(f)
            history = previous.get("history", [])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        history = []

    history_point = {"time": payload["time"]}
    for key, value in payload.items():
        if key in ("driver_id", "time", "weather", "head_state", "history"):
            continue
        if isinstance(value, (int, float)):
            history_point[key] = value
    history.append(history_point)

    payload["history"] = history[-STATUS_HISTORY_LIMIT:]
    temp_path = f"{STATUS_PATH}.tmp"
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, STATUS_PATH)

# ── Alert message builder ─────────────────────────────────────
def build_alert_message(fusion_score, yolo_conf, perclos, weather, hour) -> str:
    if fusion_score > 0.80:
        severity = "You are severely drowsy"
    elif fusion_score > 0.60:
        severity = "You are showing signs of fatigue"
    else:
        severity = "Please stay alert"

    if 0 <= hour <= 5:
        time_msg = "It is very late at night"
    elif 13 <= hour <= 15:
        time_msg = "Post-lunch drowsiness is common at this hour"
    else:
        time_msg = ""

    if "rain" in weather or "fog" in weather or "mist" in weather:
        weather_msg = f"Visibility is reduced due to {weather}"
    else:
        weather_msg = ""

    parts = [severity]
    if time_msg:    parts.append(time_msg)
    if weather_msg: parts.append(weather_msg)
    parts.append("Please pull over and take a rest")

    return ". ".join(parts) + "."

# ── Main loop ─────────────────────────────────────────────────
try:
    while True:
        loop_start = time.time()

        # 1. Refresh weather every 10 minutes
        if time.time() - last_weather_fetch > 600:
            weather_cache      = get_weather()
            last_weather_fetch = time.time()

        # 2. Get latest vision data
        vision = pipeline.latest

        # 3. Get latest sensor data (written by Firebase polling thread above)
        with sensor_lock:
            sensors = dict(sensor_data)

        # If Firebase hasn't delivered any data yet, skip this cycle
        if not sensors:
            print("[Main] Waiting for first ESP32 reading from Firebase...")
            time.sleep(1)
            continue

        jerk_rms    = sensors.get("jerk_rms",      0.0)
        posture_dev = sensors.get("posture_dev_cm", 0.0)
        hour        = sensors.get("hour",
                                  __import__("datetime").datetime.now().hour)
        session_min = (time.time() - session_start) / 60

        # 4. Build feature vector — UNCHANGED
        features = build_feature_vector(
            yolo_conf    = vision["drowsy_confidence"],
            perclos      = vision["perclos"],
            head_state   = vision["head_state"],
            jerk_rms     = jerk_rms,
            posture_dev  = posture_dev,
            hour         = hour,
            weather      = weather_cache["condition"],
            session_min  = session_min,
        )

        # 5. LightGBM fusion score — UNCHANGED
        fusion_score = predict_fatigue(model, features)

        # 6. RL bandit — UNCHANGED
        should_alert = engine.evaluate(
            yolo_conf    = vision["drowsy_confidence"],
            jerk_rms     = jerk_rms,
            posture_dev  = posture_dev,
            hour         = hour,
            weather      = weather_cache["condition"],
            session_min  = session_min,
        )

        # 7. Print live status — UNCHANGED
        print(f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] "
              f"YOLO={vision['drowsy_confidence']:.2f}  "
              f"PERCLOS={vision['perclos']:.2f}  "
              f"Jerk={jerk_rms:.2f}  "
              f"Posture={posture_dev:.1f}cm  "
              f"Fusion={fusion_score:.2f}  "
              f"Weather={weather_cache['condition']}")

        write_live_status({
            "driver_id": DRIVER_ID,
            "time": __import__("datetime").datetime.now().strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "yolo_confidence": vision["drowsy_confidence"],
            "perclos": vision["perclos"],
            "head_state": vision["head_state"],
            "jerk_rms": jerk_rms,
            "posture_dev_cm": posture_dev,
            "fusion_score": fusion_score,
            "weather": weather_cache["condition"],
            "weather_score": weather_cache.get("score"),
            "weather_city": WEATHER_CITY,
            "session_min": session_min,
            "distance_cm": sensors.get("distance_cm"),
            "hour": hour,
        })

        # 8. Trigger voice alert with cooldown — UNCHANGED
        if should_alert and (time.time() - last_alert_time > ALERT_COOLDOWN):
            last_alert_time = time.time()

            msg = build_alert_message(
                fusion_score = fusion_score,
                yolo_conf    = vision["drowsy_confidence"],
                perclos      = vision["perclos"],
                weather      = weather_cache["condition"],
                hour         = hour,
            )

            print(f"\n ALERT: {msg}\n")
            speak_alert(msg)

            log_alert(DRIVER_ID, engine.session_id, {
                "fusion_score": fusion_score,
                "message":      msg,
                "weather":      weather_cache["condition"],
                "hour":         hour,
            })

            engine.record_feedback("ack")

        time.sleep(1)

except KeyboardInterrupt:
    engine.end_session()
    pipeline.stop()
    print("\nSystem stopped cleanly.")
