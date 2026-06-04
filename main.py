import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import pickle
import threading
from alerts.voice_alert import VoiceAlert
from vision.yolo_stream import VisionPipeline
from fusion.feature_builder import build_features, features_to_array
from firebase.db_client import write_sensor, log_alert, read_rl_state, write_rl_state
from rl.bandit import choose_action, update_reward
from config import DRIVER_ID, MOCK_SENSOR, ALERT_SCORE_THRESHOLD

SESSION_ID = f"session_{int(time.time())}"

with open("fusion/fusion_model.pkl", "rb") as f:
    fusion_model = pickle.load(f)

def run():
    print("Starting fatigue detection system...")
    pipeline = VisionPipeline()
    pipeline.start()
    time.sleep(2)

    voice = VoiceAlert(threshold=0.65, cooldown=30)  # ← ADDED HERE

    print(f"Session: {SESSION_ID} | Driver: {DRIVER_ID}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            vision = pipeline.latest
            sensor = MOCK_SENSOR

            features = build_features(vision, sensor)
            X = features_to_array(features).reshape(1, -1)
            fusion_score = float(fusion_model.predict_proba(X)[0][1])

            payload = {**sensor, **vision, "fusion_score": round(fusion_score, 3)}
            write_sensor(DRIVER_ID, payload)

            if fusion_score >= ALERT_SCORE_THRESHOLD:
                action = choose_action(DRIVER_ID)
                alert = {
                    "action":       action,
                    "fusion_score": round(fusion_score, 3),
                    "vision":       vision,
                    "ts":           int(time.time()),
                }
                alert_key = log_alert(DRIVER_ID, SESSION_ID, alert)
                reward = 1.0 if fusion_score > 0.75 else 0.5
                update_reward(action, reward, DRIVER_ID)
                voice.check_and_alert(fusion_score)  # ← ADDED HERE
                print(f"ALERT [{action}] score={fusion_score:.2f} key={alert_key}")
            else:
                print(f"OK    fusion={fusion_score:.2f} perclos={vision['perclos']:.2f} head={vision['head_state']}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
        pipeline.stop()

if __name__ == "__main__":
    run()