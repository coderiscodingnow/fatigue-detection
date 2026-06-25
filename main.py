"""
main.py — Full fatigue detection pipeline
ESP32 sensors (via Firebase) + YOLO webcam + DHT22 cabin sensor + LightGBM + RL bandit + Voice alert
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
from config import (
    DRIVER_ID,
    FIREBASE_READ_URL,
    FIREBASE_URL,
    FIREBASE_SECRET,
    POLL_INTERVAL_SEC,
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

# NOTE: weather API removed — cabin_temp/humidity now come from the DHT22
# on the ESP32, already included in every /drowsiness/latest payload, so
# they're pulled out of `sensors` each loop iteration below (no separate
# polling thread or refresh timer needed).

# ── Alert cooldown ────────────────────────────────────────────
last_alert_time = 0
ALERT_COOLDOWN  = 30
latest_alert_msg = "No alert recorded"

# ── ESP32 alert-case config ────────────────────────────────────
FB_ALERT_PATH = "/alerts/current_case"
BUZZER_WAIT_SECONDS = 2.0  # must match the ESP32's BUZZER_MS exactly

# Tune these against your sensor logs / model behavior
POSTURE_DEV_THRESHOLD_CM     = 12.0
JERK_RMS_THRESHOLD           = 2.5
FUSION_SCORE_ALERT_THRESHOLD = 0.4

ALERT_VOICE_MESSAGES = {
    1: "Your posture is incorrect.",
    2: "Wake up! Drowsiness detected.",
    3: "Drive smoothly.",
}

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
        if key in ("driver_id", "time", "head_state",
                   "history", "latest_alert", "cabin_condition"):
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

# ── Cabin condition helpers (DHT22, replaces old weather API) ─
def classify_cabin_condition(cabin_temp: float, humidity: float) -> str:
    """
    Simple heuristic label from the in-cabin DHT22 reading.
    A hot/humid cabin is a real drowsiness contributor, so this
    plugs into build_feature_vector/engine.evaluate the same way
    the old outdoor weather string used to.
    Tune the thresholds to your climate if these don't feel right.
    """
    if cabin_temp < 0 or humidity < 0:
        return "unknown"
    if cabin_temp >= 32:
        return "hot"
    if humidity >= 70:
        return "humid"
    return "normal"

def cabin_risk_score(cabin_temp: float, humidity: float) -> float:
    """0-1 heuristic score for the dashboard, replacing weather_cache['score']."""
    if cabin_temp < 0 or humidity < 0:
        return 0.0
    score = 0.0
    score += 0.5 if cabin_temp >= 32 else 0.25 if cabin_temp >= 28 else 0.0
    score += 0.5 if humidity >= 70 else 0.25 if humidity >= 55 else 0.0
    return min(score, 1.0)


# ── ESP32 alert-case logic ──────────────────────────────────────
def determine_alert_case(fusion_score, jerk_rms, posture_dev):
    """
    Maps the current readings onto one of the three ESP32 alert cases.
    Priority: drowsiness > sudden jerk > bad posture, since fatigue is the
    most safety-critical signal. Reorder if your project treats them
    differently. Returns (case, voice_message) or (None, None).
    """
    if fusion_score < FUSION_SCORE_ALERT_THRESHOLD:
        return 2, ALERT_VOICE_MESSAGES[2]
    if jerk_rms > JERK_RMS_THRESHOLD:
        return 3, ALERT_VOICE_MESSAGES[3]
    if posture_dev > POSTURE_DEV_THRESHOLD_CM:
        return 1, ALERT_VOICE_MESSAGES[1]
    return None, None

def write_alert_case_to_firebase(case: int):
    """Writes the alert case + a fresh alert_id to /alerts/current_case.
    The ESP32 dedups on alert_id (not case), so the same case can fire
    twice in a row and still be picked up as a new event."""
    payload = {"case": case, "alert_id": int(time.time() * 1000)}
    try:
        fb_db = get_db()
        if fb_db is not None:
            fb_db.reference(FB_ALERT_PATH).set(payload)
        else:
            requests.put(f"{FIREBASE_URL.rstrip('/')}{FB_ALERT_PATH}.json",
                         json=payload, timeout=5)
    except Exception as e:
        print(f"[Firebase] alert_case write failed: {e}")

def handle_alert(case: int, voice_message: str):
    """Fire-and-forget: writes the case to Firebase immediately, then waits
    BUZZER_WAIT_SECONDS (so the ESP32 buzzer finishes first) before playing
    the local voice alert. Runs on a daemon thread so this never blocks
    the vision loop."""
    def _worker():
        write_alert_case_to_firebase(case)
        time.sleep(BUZZER_WAIT_SECONDS)
        speak_alert(voice_message)
    threading.Thread(target=_worker, daemon=True).start()


try:
    while True:
        # 1. Get latest vision data
        vision = pipeline.latest

        # 2. Get latest sensor data from Firebase polling thread
        with sensor_lock:
            sensors = dict(sensor_data)

        if not sensors:
            print("[Main] Waiting for first ESP32 reading from Firebase...")
            time.sleep(1)
            continue

        # 3. Extract sensor values
        jerk_rms    = float(sensors.get("jerk_rms",      0.0) or 0.0)
        posture_dev = float(sensors.get("posture_dev_cm", 0.0) or 0.0)
        distance_cm = float(sensors.get("distance_cm",    0.0) or 0.0)
        cabin_temp  = float(sensors.get("cabin_temp",    -1.0) or -1.0)
        humidity    = float(sensors.get("humidity",      -1.0) or -1.0)
        hour        = int(sensors.get("hour", datetime.datetime.now().hour))
        session_min = (time.time() - session_start) / 60

        cabin_condition = classify_cabin_condition(cabin_temp, humidity)

        # 4. Build feature vector
        features = build_feature_vector(
            yolo_conf   = vision["drowsy_confidence"],
            perclos     = vision["perclos"],
            head_state  = vision["head_state"],
            jerk_rms    = jerk_rms,
            posture_dev = posture_dev,
            hour        = hour,
            weather     = cabin_condition,
            session_min = session_min,
            distance_cm = distance_cm,
        )

        # 5. LightGBM fusion score
        fusion_score = predict_fatigue(model, features)

        # 6. RL bandit
        should_alert = engine.evaluate(
            yolo_conf   = vision["drowsy_confidence"],
            jerk_rms    = jerk_rms,
            posture_dev = posture_dev,
            hour        = hour,
            weather     = cabin_condition,
            session_min = session_min,
            distance_cm = distance_cm,
        )

        # 7. Print live status
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now_str}] "
              f"YOLO={vision['drowsy_confidence']:.2f}  "
              f"PERCLOS={vision['perclos']:.2f}  "
              f"Jerk={jerk_rms:.2f}  "
              f"Posture={posture_dev:.1f}cm  "
              f"Dist={distance_cm:.1f}cm  "
              f"Fusion={fusion_score:.2f}  "
              f"Cabin={cabin_temp:.1f}C/{humidity:.0f}%")

        # 8. Write live_status.json — dashboard reads this
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
            "cabin_temp":      cabin_temp,          # from Firebase/ESP32 (DHT22)
            "humidity":        humidity,            # from Firebase/ESP32 (DHT22)
            "cabin_condition": cabin_condition,
            "cabin_risk_score": cabin_risk_score(cabin_temp, humidity),
            "session_min":     session_min,
            "hour":            hour,
            "latest_alert":    latest_alert_msg,
        })

        # 9. Voice alert with cooldown — now routed through ESP32 alert cases
        if should_alert and (time.time() - last_alert_time > ALERT_COOLDOWN):
            last_alert_time = time.time()

            alert_case, voice_message = determine_alert_case(fusion_score, jerk_rms, posture_dev)

            # If the RL bandit triggered but no specific threshold was crossed, default to Fatigue
            if alert_case is None:
                alert_case = 2
                voice_message = ALERT_VOICE_MESSAGES[2]

            latest_alert_msg = voice_message
            print(f"\n ALERT (case {alert_case}): {latest_alert_msg}\n")

            # Async: writes /alerts/current_case now, waits 2s for the
            # ESP32 buzzer to finish, then speaks locally — non-blocking.
            handle_alert(alert_case, voice_message)

            log_alert(DRIVER_ID, engine.session_id, {
                "fusion_score":     fusion_score,
                "alert_case":       alert_case,
                "message":          latest_alert_msg,
                "cabin_condition":  cabin_condition,
                "hour":             hour,
            })
            engine.record_feedback("ack")

        time.sleep(0.2)

except KeyboardInterrupt:
    engine.end_session()
    pipeline.stop()
    print("\nSystem stopped cleanly.")