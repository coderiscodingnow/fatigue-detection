import numpy as np
import lightgbm as lgb
import pickle
import random
from fusion.feature_builder import build_features, features_to_array

def generate_mock_sample(label):
    if label == 1:  # drowsy
        # Support drowsiness due to vision, posture/distance, or both
        drowsy_type = random.choice(["posture", "vision", "both"])
        if drowsy_type == "posture":
            drowsy_conf = random.uniform(0.0, 0.3)
            perclos = random.uniform(0.0, 0.15)
            head_state = "upright"
            distance_cm = random.uniform(5.0, 50.0)
        elif drowsy_type == "vision":
            drowsy_conf = random.uniform(0.5, 0.95)
            perclos = random.uniform(0.3, 0.8)
            head_state = random.choice(["nodding", "tilted"])
            distance_cm = random.choice([random.uniform(0.0, 3.5), -1.0])
        else:
            drowsy_conf = random.uniform(0.5, 0.95)
            perclos = random.uniform(0.3, 0.8)
            head_state = random.choice(["nodding", "tilted"])
            distance_cm = random.uniform(5.0, 50.0)

        head_score = {"nodding": 1.0, "tilted": 0.6}.get(head_state, 0.0)
        alert_score = 0.50 * drowsy_conf + 0.35 * perclos + 0.15 * head_score
        if distance_cm > 3.5:
            alert_score = min(1.0, alert_score + 0.20)

        vision = {"drowsy_confidence": drowsy_conf,
                  "perclos": perclos,
                  "head_state": head_state,
                  "alert_score": alert_score}
        sensor = {"jerk_rms": random.uniform(0.1, 0.8),
                  "distance_cm": distance_cm,
                  "posture_dev_cm": random.uniform(5, 12),
                  "hour": random.choice([1,2,3,4,14,15]),
                  "session_min": random.randint(60, 180),
                  "weather": random.choice(["rain", "fog"])}
    else:  # awake
        drowsy_conf = random.uniform(0.0, 0.3)
        perclos = random.uniform(0.0, 0.15)
        head_state = "upright"
        head_score = 0.0
        distance_cm = random.choice([random.uniform(0.0, 3.5), -1.0])
        alert_score = 0.50 * drowsy_conf + 0.35 * perclos + 0.15 * head_score
        if distance_cm > 3.5:
            alert_score = min(1.0, alert_score + 0.20)

        vision = {"drowsy_confidence": drowsy_conf,
                  "perclos": perclos,
                  "head_state": head_state,
                  "alert_score": alert_score}
        sensor = {"jerk_rms": random.uniform(1.0, 3.0),
                  "distance_cm": distance_cm,
                  "posture_dev_cm": random.uniform(0, 3),
                  "hour": random.randint(8, 12),
                  "session_min": random.randint(0, 30),
                  "weather": "clear"}
    return features_to_array(build_features(vision, sensor))

if __name__ == "__main__":
    print("Generating mock training data...")
    X, y = [], []
    for _ in range(500):
        X.append(generate_mock_sample(1))
        y.append(1)
        X.append(generate_mock_sample(0))
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    with open("fusion/fusion_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to fusion/fusion_model.pkl")
    print(f"Training accuracy: {model.score(X, y):.2f}")

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict_fatigue(model, features):
    probs = model.predict_proba(features.reshape(1, -1))
    return float(probs[0][1])