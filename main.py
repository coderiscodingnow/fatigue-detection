"""
main.py — Fatigue detection pipeline (corrected per your conditions)

Current logic:
  1. Drowsiness is determined ONLY by PERCLOS continuity.
  2. Ultrasonic distance_cm is used for head/posture-position alert.
  3. The old posture deviation signal is not used.
  4. Distance/object alert path is removed completely.
  5. Fusion score is not used for alert decisions.
  6. RL alert decision/evaluation is removed.
  7. pipeline.latest is guarded before use.
  8. Missing/invalid Firebase timestamp and hour values are handled safely.
"""

import time
import json
import os
import threading
import requests
import datetime
import collections

from vision.yolo_stream import VisionPipeline
from rl.bandit          import AlertEngine, FirebaseAdapter
from alerts.voice_alert import speak_alert
from firebase.db_client import write_sensor, log_alert, get_db
from config import (
    DRIVER_ID,
    FIREBASE_READ_URL,
    FIREBASE_URL,
    POLL_INTERVAL_SEC,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(__file__)
STATUS_PATH          = os.path.join(BASE_DIR, "dashboard", "live_status.json")
STATUS_HISTORY_LIMIT = 90
os.makedirs(os.path.join(BASE_DIR, "dashboard"), exist_ok=True)

# ── Thresholds ─────────────────────────────────────────────────────────────────
# Drowsiness is based ONLY on PERCLOS.
PERCLOS_THRESHOLD       = 0.35   # Eyes closed ratio threshold
DROWSY_VIOLATION_FRAMES = 6      # ~1.2 s of sustained closure @ 5 Hz

# Ultrasonic sensor is used for head/posture position.
HEAD_DISTANCE_THRESHOLD_CM = 3.5  # Trigger posture alert if distance_cm > this
POSTURE_VIOLATION_FRAMES   = 10   # ~2 s of continuous bad head/posture position @ 5 Hz

# Jerk detection
JERK_SPIKE_THRESHOLD      = 0.8   # g-units
JERK_SPIKE_CONFIRM_FRAMES = 10    # spikes must appear in N of last WINDOW frames
JERK_SPIKE_WINDOW         = 10    # rolling window size for spike counting

JERK_DEBUG          = True
BUZZER_WAIT_SECONDS = 2.0         # ESP32 buzzer duration before voice alert

# Normal-state guard — if ALL signals stay inside these bounds, no alert fires.
NORMAL_HEAD_DISTANCE_SAFE_CM = 3.5
NORMAL_JERK_MAX              = 0.8

ALERT_COOLDOWN = {
    "drowsy":  45,
    "posture": 20,
    "jerk":    25,
}

# ── Shared sensor state ────────────────────────────────────────────────────────
_sensor_data  = {}
_sensor_lock  = threading.Lock()
_sensor_event = threading.Event()

session_start = time.time()
_last_payload_signature = None


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def get_sensors() -> dict:
    with _sensor_lock:
        return dict(_sensor_data)


def set_sensors(payload: dict):
    with _sensor_lock:
        _sensor_data.clear()
        _sensor_data.update(payload)
    _sensor_event.set()


# ── Firebase polling thread ────────────────────────────────────────────────────
def poll_firebase():
    global _last_payload_signature
    print(f"[Firebase] Polling every {POLL_INTERVAL_SEC}s")

    while True:
        try:
            fb_db = get_db()
            if fb_db is not None:
                payload = fb_db.reference("/drowsiness/latest").get()
            else:
                resp = requests.get(FIREBASE_READ_URL, timeout=5)
                payload = resp.json() if resp.status_code == 200 else None

            if not isinstance(payload, dict):
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Prefer ts/timestamp if available. If not available, compare the full payload.
            current_ts = payload.get("ts") or payload.get("timestamp")
            if current_ts is not None:
                payload_signature = str(current_ts)
            else:
                payload_signature = json.dumps(payload, sort_keys=True, default=str)

            if payload_signature != _last_payload_signature:
                _last_payload_signature = payload_signature
                set_sensors(payload)
                write_sensor(DRIVER_ID, payload)
                print(
                    f"[Firebase] New → jerk={payload.get('jerk_rms')}  "
                    f"dist={payload.get('distance_cm')}cm  "
                    f"ts={current_ts if current_ts is not None else 'no-ts'}"
                )

        except Exception as e:
            print(f"[Firebase] Poll error: {e}")

        time.sleep(POLL_INTERVAL_SEC)


threading.Thread(target=poll_firebase, daemon=True).start()

# ── Vision pipeline ────────────────────────────────────────────────────────────
pipeline = VisionPipeline()
pipeline.start()
print("[Vision] Webcam pipeline started.")

# ── RL/Firebase session support ────────────────────────────────────────────────
# Note: RL decision/evaluation is intentionally removed. This engine is kept only
# because the existing project uses engine.session_id for logging/session tracking.
db     = FirebaseAdapter("serviceAccountKey.json", FIREBASE_URL)
engine = AlertEngine(driver_id=DRIVER_ID, db=db)
engine.start_session()
print(f"[Session] Started for {DRIVER_ID}")

# ── Bulb / LED state ───────────────────────────────────────────────────────────
bulb_state = {"red": False, "yellow": False, "green": False}
bulb_lock  = threading.Lock()


def set_bulb(color: str, on: bool):
    with bulb_lock:
        bulb_state[color] = on


def get_bulb_snapshot() -> dict:
    with bulb_lock:
        return dict(bulb_state)


# ── Per-type alert cooldown tracker ───────────────────────────────────────────
_last_alert_times: dict[str, float] = {k: 0.0 for k in ALERT_COOLDOWN}


def can_alert(alert_type: str) -> bool:
    return time.time() - _last_alert_times.get(alert_type, 0.0) > ALERT_COOLDOWN[alert_type]


def mark_alerted(alert_type: str):
    _last_alert_times[alert_type] = time.time()


# ── Alert voice messages ───────────────────────────────────────────────────────
ALERT_VOICE = {
    "drowsy":  "Wake up! Drowsiness detected.",
    "posture": "Your head position is incorrect.",
    "jerk":    "Drive smoothly.",
}

ALERT_CASE = {
    "drowsy":  2,
    "posture": 1,
    "jerk":    3,
}

latest_alert_msg = "No alert recorded"

# ── Firebase alert-case writer ─────────────────────────────────────────────────
FB_ALERT_PATH = "/alerts/current_case"


def write_alert_case_to_firebase(case: int):
    payload = {"case": case, "alert_id": int(time.time() * 1000)}
    try:
        fb_db = get_db()
        if fb_db is not None:
            fb_db.reference(FB_ALERT_PATH).set(payload)
        else:
            requests.put(
                f"{FIREBASE_URL.rstrip('/')}{FB_ALERT_PATH}.json",
                json=payload,
                timeout=5,
            )
    except Exception as e:
        print(f"[Firebase] alert_case write failed: {e}")


# ── Alert dispatch (async, non-blocking) ──────────────────────────────────────
_alert_queue: collections.deque = collections.deque()
_alert_event = threading.Event()


def _alert_worker():
    global latest_alert_msg
    while True:
        _alert_event.wait()
        _alert_event.clear()

        while _alert_queue:
            alert_type, perclos_score, hour, cabin_condition = _alert_queue.popleft()
            case      = ALERT_CASE[alert_type]
            voice_msg = ALERT_VOICE[alert_type]
            latest_alert_msg = voice_msg

            print(f"\n  ALERT [{alert_type.upper()}] (ESP32 case {case}): {voice_msg}\n")
            write_alert_case_to_firebase(case)
            time.sleep(BUZZER_WAIT_SECONDS)
            speak_alert(voice_msg)

            log_alert(DRIVER_ID, engine.session_id, {
                "alert_type":      alert_type,
                "perclos":         perclos_score,
                "alert_case":      case,
                "message":         voice_msg,
                "cabin_condition": cabin_condition,
                "hour":            hour,
            })

            # Kept only if your existing AlertEngine expects feedback for the session.
            # It does NOT decide whether an alert should fire.
            engine.record_feedback("ack")


threading.Thread(target=_alert_worker, daemon=True).start()


def enqueue_alert(alert_type: str, perclos_score: float, hour: int, cabin_condition: str):
    if can_alert(alert_type):
        mark_alerted(alert_type)
        _alert_queue.append((alert_type, perclos_score, hour, cabin_condition))
        _alert_event.set()


# ── Jerk detector ──────────────────────────────────────────────────────────────
_jerk_spike_window: collections.deque = collections.deque(maxlen=JERK_SPIKE_WINDOW)


def update_jerk_detector(jerk_rms: float) -> bool:
    """
    Returns True when there is continuous high jerk_rms.
    Fires when at least JERK_SPIKE_CONFIRM_FRAMES of the last
    JERK_SPIKE_WINDOW readings exceed the threshold.
    """
    _jerk_spike_window.append(jerk_rms >= JERK_SPIKE_THRESHOLD)
    is_continuous_jerk = (
        len(_jerk_spike_window) == JERK_SPIKE_WINDOW and
        sum(_jerk_spike_window) >= JERK_SPIKE_CONFIRM_FRAMES
    )

    if JERK_DEBUG:
        print(
            f"    [JerkDebug] jerk_rms={jerk_rms:.2f} "
            f"(spike={jerk_rms >= JERK_SPIKE_THRESHOLD}, "
            f"window_hits={sum(_jerk_spike_window)}/{JERK_SPIKE_WINDOW}, "
            f"alert_triggered={is_continuous_jerk})"
        )

    return is_continuous_jerk


# ── Cabin helpers ──────────────────────────────────────────────────────────────
def classify_cabin_condition(cabin_temp: float, humidity: float) -> str:
    if cabin_temp < 0 or humidity < 0:
        return "unknown"
    if cabin_temp >= 32:
        return "hot"
    if humidity >= 70:
        return "humid"
    return "normal"


def cabin_risk_score(cabin_temp: float, humidity: float) -> float:
    if cabin_temp < 0 or humidity < 0:
        return 0.0
    score  = 0.5 if cabin_temp >= 32 else 0.25 if cabin_temp >= 28 else 0.0
    score += 0.5 if humidity >= 70 else 0.25 if humidity >= 55 else 0.0
    return min(score, 1.0)


# ── live_status.json writer ────────────────────────────────────────────────────
def write_live_status(payload: dict):
    history = []
    try:
        with open(STATUS_PATH, "r") as f:
            previous = json.load(f)
            history = previous.get("history", [])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        history = []

    history_point = {"time": payload["time"]}
    for key, value in payload.items():
        if key in (
            "driver_id",
            "time",
            "head_state",
            "history",
            "latest_alert",
            "cabin_condition",
            "bulbs",
        ):
            continue
        if isinstance(value, (int, float)):
            history_point[key] = value

    history.append(history_point)
    payload["history"] = history[-STATUS_HISTORY_LIMIT:]

    temp_path = f"{STATUS_PATH}.tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as e:
        print(f"[Dashboard] Write error: {e}")
        return

    for _ in range(5):
        try:
            os.replace(temp_path, STATUS_PATH)
            break
        except PermissionError:
            time.sleep(0.05)
        except OSError as e:
            print(f"[Dashboard] Replace error: {e}")
            break


# ══════════════════════════════════════════════════════════════════════════════
# Continuity trackers
# ══════════════════════════════════════════════════════════════════════════════
consecutive_posture_violations = 0
consecutive_drowsy_frames      = 0

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== System running. Press Ctrl+C to stop. ===\n")

try:
    while True:
        _sensor_event.wait(timeout=0.1)
        _sensor_event.clear()

        # 1. Vision guard
        vision = pipeline.latest
        if not isinstance(vision, dict):
            print("[Main] Waiting for first vision frame...")
            continue

        yolo_confidence = safe_float(vision.get("drowsy_confidence"), 0.0)
        perclos         = safe_float(vision.get("perclos"), 0.0)
        head_state      = vision.get("head_state") or "unknown"

        # 2. Sensors snapshot
        sensors = get_sensors()
        if not sensors:
            print("[Main] Waiting for first ESP32 reading...")
            continue

        # 3. Extract sensor values safely
        jerk_rms    = safe_float(sensors.get("jerk_rms"), 0.0)
        distance_cm = safe_float(sensors.get("distance_cm"), 0.0)
        cabin_temp  = safe_float(sensors.get("cabin_temp"), -1.0)
        humidity    = safe_float(sensors.get("humidity"), -1.0)
        ax          = safe_float(sensors.get("ax"), 0.0)
        ay          = safe_float(sensors.get("ay"), 0.0)
        az          = safe_float(sensors.get("az"), 0.0)
        hour        = safe_int(sensors.get("hour"), datetime.datetime.now().hour)
        session_min = (time.time() - session_start) / 60

        cabin_condition = classify_cabin_condition(cabin_temp, humidity)

        # ── Drowsiness continuity ────────────────────────────────────────────
        # Drowsiness is determined ONLY by PERCLOS.
        if perclos > PERCLOS_THRESHOLD:
            consecutive_drowsy_frames += 1
        else:
            consecutive_drowsy_frames = 0

        is_drowsy = consecutive_drowsy_frames >= DROWSY_VIOLATION_FRAMES

        # ── Head/posture-position continuity using ultrasonic distance ───────
        # Only ultrasonic distance_cm is used here.
        if distance_cm > HEAD_DISTANCE_THRESHOLD_CM:
            consecutive_posture_violations += 1
        else:
            consecutive_posture_violations = 0

        is_bad_posture = consecutive_posture_violations >= POSTURE_VIOLATION_FRAMES

        # ── Jerk continuity ──────────────────────────────────────────────────
        continuous_jerk    = update_jerk_detector(jerk_rms)
        is_continuous_jerk = continuous_jerk

        # ── Normal-state guard ────────────────────────────────────────────────
        everything_normal = (
            distance_cm <= NORMAL_HEAD_DISTANCE_SAFE_CM and
            jerk_rms    <= NORMAL_JERK_MAX and
            not continuous_jerk and
            not is_drowsy and
            not is_bad_posture
        )

        # ── Overlap resolution / suppression ─────────────────────────────────
        # Intentional suppression: only one highest-priority alert fires.
        # Priority: drowsy > posture/head-position > jerk.
        active_drowsy  = is_drowsy
        active_posture = is_bad_posture and not active_drowsy
        active_jerk    = is_continuous_jerk and not active_drowsy and not active_posture

        is_drowsy          = active_drowsy
        is_bad_posture     = active_posture
        is_continuous_jerk = active_jerk

        # ── Bulb state ────────────────────────────────────────────────────────
        set_bulb("red",    is_drowsy)
        set_bulb("yellow", is_bad_posture)
        set_bulb("green",  is_continuous_jerk)

        # ── Enqueue alerts ────────────────────────────────────────────────────
        if is_drowsy:
            enqueue_alert("drowsy", perclos, hour, cabin_condition)

        if is_bad_posture:
            enqueue_alert("posture", perclos, hour, cabin_condition)

        if is_continuous_jerk:
            enqueue_alert("jerk", perclos, hour, cabin_condition)

        # ── Console status ────────────────────────────────────────────────────
        now_str   = datetime.datetime.now().strftime("%H:%M:%S")
        bulbs     = get_bulb_snapshot()
        jerk_hits = sum(_jerk_spike_window)
        bulb_str  = (
            f"RED={'ON' if bulbs['red'] else '--'}  "
            f"YEL={'ON' if bulbs['yellow'] else '--'}  "
            f"GRN={'ON' if bulbs['green'] else '--'}  "
            f"{'[NORMAL]' if everything_normal else ''}"
        )

        print(
            f"[{now_str}] "
            f"YOLO={yolo_confidence:.2f}  "
            f"PERCLOS={perclos:.2f}"
            f"(frames={consecutive_drowsy_frames}/{DROWSY_VIOLATION_FRAMES})  "
            f"Jerk={jerk_rms:.2f}(spikes={jerk_hits}/{JERK_SPIKE_WINDOW})  "
            f"HeadDist={distance_cm:.1f}cm"
            f"(frames={consecutive_posture_violations}/{POSTURE_VIOLATION_FRAMES})  "
            f"Cabin={cabin_temp:.1f}°C/{humidity:.0f}%  "
            f"{bulb_str}"
        )

        # ── Dashboard JSON ────────────────────────────────────────────────────
        write_live_status({
            "driver_id":                  DRIVER_ID,
            "time":                       now_str,
            "timestamp":                  time.time(),
            "yolo_confidence":            yolo_confidence,
            "perclos":                    perclos,
            "head_state":                 head_state,
            "jerk_rms":                   jerk_rms,
            "distance_cm":                distance_cm,
            "cabin_temp":                 cabin_temp,
            "humidity":                   humidity,
            "cabin_condition":            cabin_condition,
            "cabin_risk_score":           cabin_risk_score(cabin_temp, humidity),
            "session_min":                session_min,
            "hour":                       hour,
            "ax":                         ax,
            "ay":                         ay,
            "az":                         az,
            "consecutive_drowsy_frames":  consecutive_drowsy_frames,
            "consecutive_posture_frames": consecutive_posture_violations,
            "latest_alert":               latest_alert_msg,
            "bulbs":                      get_bulb_snapshot(),
        })

except KeyboardInterrupt:
    engine.end_session()
    pipeline.stop()
    print("\nSystem stopped cleanly.")
