import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import json
import time
import firebase_admin
from firebase_admin import credentials
from firebase.db_client import read_rl_state, write_rl_state
from config import DRIVER_ID

class FirebaseAdapter:
    def __init__(self, service_account_path, database_url):
        self.service_account_path = service_account_path
        self.database_url = database_url
        if os.path.exists(service_account_path):
            try:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            except ValueError:
                # Already initialized by firebase.db_client or previous call
                pass
        else:
            print(f"[FirebaseAdapter] Warning: '{service_account_path}' not found. Running in mock firebase mode.")

class AlertEngine:
    def __init__(self, driver_id, db):
        self.driver_id = driver_id
        self.db = db
        self.session_id = None
        self.last_action = "voice_alert"

    def start_session(self):
        self.session_id = f"session_{int(time.time())}"
        get_state(self.driver_id)

    def evaluate(self, yolo_conf, jerk_rms, posture_dev, hour, weather, session_min, distance_cm=0.0):
        # Alert if yolo confidence is high, posture deviation is high, or head is far from ultrasonic sensor
        is_dangerous_distance = distance_cm > 3.5
        is_fatigued = (yolo_conf > 0.4) or (posture_dev > 6.0) or is_dangerous_distance
        if is_fatigued:
            self.last_action = choose_action(self.driver_id)
            return True
        return False

    def record_feedback(self, feedback):
        reward = 1.0 if feedback == "ack" else 0.0
        update_reward(self.last_action, reward, self.driver_id)

    def end_session(self):
        pass


ACTIONS = ["voice_alert", "vibration_alert", "visual_alert"]
EPSILON = 0.15

def get_state(driver_id=DRIVER_ID):
    state = read_rl_state(driver_id)
    if state is None:
        state = {a: {"count": 0, "reward_sum": 0.0} for a in ACTIONS}
        write_rl_state(driver_id, state)
    return state

def choose_action(driver_id=DRIVER_ID):
    state = get_state(driver_id)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    best = max(ACTIONS, key=lambda a:
        state[a]["reward_sum"] / state[a]["count"]
        if state[a]["count"] > 0 else 0.0)
    return best

def update_reward(action, reward, driver_id=DRIVER_ID):
    state = get_state(driver_id)
    state[action]["count"]      += 1
    state[action]["reward_sum"] += reward
    write_rl_state(driver_id, state)
    return state

def simulate(rounds=20):
    print(f"Simulating {rounds} rounds of bandit...\n")
    for i in range(rounds):
        action = choose_action()
        reward = random.choice([1.0, 1.0, 0.0])
        state  = update_reward(action, reward)
        q_vals = {a: round(state[a]["reward_sum"] / state[a]["count"], 2)
                  if state[a]["count"] > 0 else 0.0 for a in ACTIONS}
        print(f"Round {i+1:2d} | action={action:15s} | reward={reward} | Q={q_vals}")

if __name__ == "__main__":
    simulate()