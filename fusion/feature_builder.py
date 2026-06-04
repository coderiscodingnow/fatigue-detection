import numpy as np
from config import MOCK_SENSOR

def build_features(vision_data, sensor_data=None):
    if sensor_data is None:
        sensor_data = MOCK_SENSOR

    return {
        "drowsy_conf":    vision_data.get("drowsy_confidence", 0.0),
        "perclos":        vision_data.get("perclos", 0.0),
        "head_nodding":   1 if vision_data.get("head_state") == "nodding" else 0,
        "head_tilted":    1 if vision_data.get("head_state") == "tilted"  else 0,
        "alert_score":    vision_data.get("alert_score", 0.0),
        "jerk_rms":       sensor_data.get("jerk_rms", 0.0),
        "distance_cm":    sensor_data.get("distance_cm", 0.0),
        "posture_dev_cm": sensor_data.get("posture_dev_cm", 0.0),
        "hour":           sensor_data.get("hour", 12),
        "session_min":    sensor_data.get("session_min", 0),
        "weather_rain":   1 if sensor_data.get("weather") == "rain" else 0,
    }

def features_to_array(features):
    keys = [
        "drowsy_conf", "perclos", "head_nodding", "head_tilted",
        "alert_score", "jerk_rms", "distance_cm", "posture_dev_cm",
        "hour", "session_min", "weather_rain",
    ]
    return np.array([features[k] for k in keys], dtype=np.float32)