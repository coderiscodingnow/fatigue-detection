import numpy as np
import lightgbm as lgb
import pickle
import random
from fusion.feature_builder import build_features, features_to_array

def generate_mock_sample(label):
    if label == 1:  # drowsy
        vision = {"drowsy_confidence": random.uniform(0.5, 0.95),
                  "perclos": random.uniform(0.3, 0.8),
                  "head_state": random.choice(["nodding", "tilted"]),
                  "alert_score": random.uniform(0.5, 0.9)}
        sensor = {"jerk_rms": random.uniform(0.1, 0.8),
                  "distance_cm": random.uniform(10, 20),
                  "posture_dev_cm": random.uniform(5, 12),
                  "hour": random.choice([1,2,3,4,14,15]),
                  "session_min": random.randint(60, 180),
                  "weather": random.choice(["rain", "fog"])}
    else:  # awake
        vision = {"drowsy_confidence": random.uniform(0.0, 0.3),
                  "perclos": random.uniform(0.0, 0.15),
                  "head_state": "upright",
                  "alert_score": random.uniform(0.0, 0.3)}
        sensor = {"jerk_rms": random.uniform(1.0, 3.0),
                  "distance_cm": random.uniform(25, 50),
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