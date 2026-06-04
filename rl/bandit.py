import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import json
from firebase.db_client import read_rl_state, write_rl_state
from config import DRIVER_ID

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