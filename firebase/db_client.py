import os
import json
import time
import firebase_admin
from firebase_admin import credentials, db
from config import FIREBASE_URL, SERVICE_ACCOUNT

_initialized = False
_mock_mode = False
_mock_db = {}

def get_db():
    global _initialized, _mock_mode, _mock_db
    if not _initialized:
        if not os.path.exists(SERVICE_ACCOUNT):
            print(f"[Firebase] Warning: '{SERVICE_ACCOUNT}' not found. Running in mock database mode.")
            _mock_mode = True
            if os.path.exists("mock_db.json"):
                try:
                    with open("mock_db.json", "r") as f:
                        _mock_db = json.load(f)
                except Exception:
                    _mock_db = {}
            else:
                _mock_db = {}
            _initialized = True
            return None
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
        except Exception as e:
            print(f"[Firebase] Error initializing firebase admin: {e}. Falling back to mock database mode.")
            _mock_mode = True
            _mock_db = {}
        _initialized = True
    return db

def _save_mock_db():
    try:
        with open("mock_db.json", "w") as f:
            json.dump(_mock_db, f, indent=2)
    except Exception:
        pass

def write_sensor(driver_id, payload):
    get_db()
    if _mock_mode:
        if "drivers" not in _mock_db: _mock_db["drivers"] = {}
        if driver_id not in _mock_db["drivers"]: _mock_db["drivers"][driver_id] = {}
        if "latest_sensor" not in _mock_db["drivers"][driver_id]:
            _mock_db["drivers"][driver_id]["latest_sensor"] = {}
        _mock_db["drivers"][driver_id]["latest_sensor"].update(payload)
        _save_mock_db()
    else:
        db.reference(f"drivers/{driver_id}/latest_sensor").update(payload)

def read_rl_state(driver_id):
    get_db()
    if _mock_mode:
        try:
            return _mock_db["drivers"][driver_id]["rl_state"]
        except KeyError:
            return None
    else:
        return db.reference(f"drivers/{driver_id}/rl_state").get()

def write_rl_state(driver_id, state):
    get_db()
    if _mock_mode:
        if "drivers" not in _mock_db: _mock_db["drivers"] = {}
        if driver_id not in _mock_db["drivers"]: _mock_db["drivers"][driver_id] = {}
        _mock_db["drivers"][driver_id]["rl_state"] = state
        _save_mock_db()
    else:
        db.reference(f"drivers/{driver_id}/rl_state").set(state)

def log_alert(driver_id, session_id, alert):
    get_db()
    if _mock_mode:
        if "drivers" not in _mock_db: _mock_db["drivers"] = {}
        if driver_id not in _mock_db["drivers"]: _mock_db["drivers"][driver_id] = {}
        if "sessions" not in _mock_db["drivers"][driver_id]: _mock_db["drivers"][driver_id]["sessions"] = {}
        if session_id not in _mock_db["drivers"][driver_id]["sessions"]:
            _mock_db["drivers"][driver_id]["sessions"][session_id] = {"alerts": {}}
        alerts_dict = _mock_db["drivers"][driver_id]["sessions"][session_id]["alerts"]
        alert_key = f"alert_{int(time.time() * 1000)}"
        alerts_dict[alert_key] = alert
        _save_mock_db()
        return alert_key
    else:
        ref = db.reference(f"drivers/{driver_id}/sessions/{session_id}/alerts")
        return ref.push(alert).key

def update_alert_feedback(driver_id, session_id, alert_key, feedback, reward):
    get_db()
    if _mock_mode:
        try:
            _mock_db["drivers"][driver_id]["sessions"][session_id]["alerts"][alert_key].update({
                "feedback": feedback,
                "reward": reward
            })
            _save_mock_db()
        except KeyError:
            pass
    else:
        db.reference(
            f"drivers/{driver_id}/sessions/{session_id}/alerts/{alert_key}"
        ).update({"feedback": feedback, "reward": reward})