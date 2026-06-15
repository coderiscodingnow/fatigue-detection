# config.py — Driver Fatigue Detection (cleaned up, no duplicate keys)

DRIVER_ID             = "driver_001"
WEBCAM_INDEX          = 0
ALERT_SCORE_THRESHOLD = 0.65

# ── Firebase ──────────────────────────────────────────────────
FIREBASE_URL          = "https://fatigue-detection-26427-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_SECRET       = "4g6DYXF1PyHL8BuTLEPMbdBCtb9gacd7PXMrkWJo"
SERVICE_ACCOUNT       = "serviceAccountKey.json"
FIREBASE_PATH         = "/drowsiness/latest.json"
FIREBASE_READ_URL     = f"{FIREBASE_URL}{FIREBASE_PATH}?auth={FIREBASE_SECRET}"

# ── Polling ───────────────────────────────────────────────────
POLL_INTERVAL_SEC     = 2

# ── Model paths ───────────────────────────────────────────────
MODEL_PATH            = "fusion/fusion_model.pkl"
YOLO_MODEL_PATH       = "runs/detect/train-4/weights/best.pt"

# ── Thresholds ────────────────────────────────────────────────
NEUTRAL_DISTANCE_CM   = 28.0

# ── Weather ───────────────────────────────────────────────────
WEATHER_API_KEY       = "9bedc23df039b6877fdbf241053797eb"
WEATHER_CITY          = "Bangalore"

# ── Mock sensor fallback (used when ESP32 data is unavailable) ─
MOCK_SENSOR = {
    "jerk_rms":       1.2,
    "distance_cm":    28.5,
    "posture_dev_cm": 3.1,
    "hour":           14,
    "weather":        "rain",
    "session_min":    45,
}