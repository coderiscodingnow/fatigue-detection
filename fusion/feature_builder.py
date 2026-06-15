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

def build_feature_vector(yolo_conf, perclos, head_state, jerk_rms, posture_dev, hour, weather, session_min, distance_cm):
    # Calculate alert_score
    head_score = {"nodding": 1.0, "tilted": 0.6, "upright": 0.0, "unknown": 0.0}.get(head_state, 0.0)
    alert_score = 0.50 * yolo_conf + 0.35 * perclos + 0.15 * head_score
    
    # Increase alert_score when distance_cm is more than 3.5 cm
    if distance_cm > 3.5:
        alert_score = min(1.0, alert_score + 0.20)

    # Construct vision and sensor dicts to pass to build_features
    vision_data = {
        "drowsy_confidence": yolo_conf,
        "perclos": perclos,
        "head_state": head_state,
        "alert_score": alert_score,
    }
    
    sensor_data = {
        "jerk_rms": jerk_rms,
        "posture_dev_cm": posture_dev,
        "hour": hour,
        "weather": weather,
        "session_min": session_min,
        "distance_cm": distance_cm,
    }
    
    features = build_features(vision_data, sensor_data)
    return features_to_array(features)