DRIVER_ID             = "driver_001"
MQTT_BROKER           = "10.103.43.84"
BROKER                = MQTT_BROKER
MQTT_PORT             = 1883
MQTT_TOPIC            = "driver/driver_001/sensors"
FIREBASE_URL          = "https://fatigue-detection-26427-default-rtdb.asia-southeast1.firebasedatabase.app"          # fill in after Phase 2
SERVICE_ACCOUNT       = "serviceAccountKey.json"
WEBCAM_INDEX          = 0
MODEL_PATH            = "runs/detect/train-4/weights/best.pt"
ALERT_SCORE_THRESHOLD = 0.65
MOCK_SENSOR = {
    "jerk_rms": 1.2, "distance_cm": 28.5,
    "posture_dev_cm": 3.1, "hour": 14,
    "weather": "rain", "session_min": 45,
}
WEATHER_API_KEY = "9bedc23df039b6877fdbf241053797eb"
WEATHER_CITY    = "Bangalore"   # your city
FIREBASE_URL    = "https://fatigue-detection-26427-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_SECRET = "4g6DYXF1PyHL8BuTLEPMbdBCtb9gacd7PXMrkWJo"
 
# Path where the ESP32 writes sensor data
FIREBASE_PATH   = "/drowsiness/latest.json"
 
# Full URL used by requests.get()
FIREBASE_READ_URL = f"{FIREBASE_URL}{FIREBASE_PATH}?auth={FIREBASE_SECRET}"
 
# ── Polling interval ──────────────────────────────────────────
# How often (seconds) the Python model polls Firebase for new data
POLL_INTERVAL_SEC = 2
 
# ── Driver / session ──────────────────────────────────────────
DRIVER_ID = "driver_001"
 
# ── Drowsiness thresholds (tune to your model) ────────────────
NEUTRAL_DISTANCE_CM = 28.0   # expected head distance when alert
 
# ── Model path ────────────────────────────────────────────────
MODEL_PATH = "fusion/fusion_model.pkl"   # adjust to your actual path
 