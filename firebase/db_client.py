import firebase_admin
from firebase_admin import credentials, db
from config import FIREBASE_URL, SERVICE_ACCOUNT

_initialized = False

def get_db():
    global _initialized
    if not _initialized:
        cred = credentials.Certificate(SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        _initialized = True
    return db

def write_sensor(driver_id, payload):
    get_db().reference(f"drivers/{driver_id}/latest_sensor").update(payload)

def read_rl_state(driver_id):
    return get_db().reference(f"drivers/{driver_id}/rl_state").get()

def write_rl_state(driver_id, state):
    get_db().reference(f"drivers/{driver_id}/rl_state").set(state)

def log_alert(driver_id, session_id, alert):
    ref = get_db().reference(f"drivers/{driver_id}/sessions/{session_id}/alerts")
    return ref.push(alert).key

def update_alert_feedback(driver_id, session_id, alert_key, feedback, reward):
    get_db().reference(
        f"drivers/{driver_id}/sessions/{session_id}/alerts/{alert_key}"
    ).update({"feedback": feedback, "reward": reward})