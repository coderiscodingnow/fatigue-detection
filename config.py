DRIVER_ID             = "driver_001"
MQTT_BROKER           = "10.103.43.32"
MQTT_PORT             = 1883
MQTT_TOPIC            = f"driver/{DRIVER_ID}/sensors"
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