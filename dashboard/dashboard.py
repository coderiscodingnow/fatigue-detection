import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time
import random
from firebase.db_client import get_db
from config import DRIVER_ID, MOCK_SENSOR

st.set_page_config(page_title="Fatigue Monitor", layout="wide")
st.title("Driver Fatigue Detection — Live Dashboard")

driver_id = DRIVER_ID

def get_latest(driver_id):
    try:
        data = get_db().reference(f"drivers/{driver_id}/latest_sensor").get()
        return data or {}
    except:
        return {}

def mock_vision():
    return {
        "drowsy_confidence": round(random.uniform(0.0, 0.6), 3),
        "perclos":           round(random.uniform(0.0, 0.4), 3),
        "head_state":        random.choice(["upright", "nodding", "tilted"]),
        "alert_score":       round(random.uniform(0.0, 0.7), 3),
        "label":             random.choice(["awake", "uncertain", "drowsy"]),
    }

placeholder = st.empty()

while True:
    vision = mock_vision()
    sensor = get_latest(driver_id) or MOCK_SENSOR

    score = vision["alert_score"]
    color = "🔴" if score > 0.65 else "🟡" if score > 0.4 else "🟢"

    with placeholder.container():
        st.subheader(f"{color} Alert score: {score:.2f}  |  Status: {vision['label'].upper()}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Drowsy confidence", f"{vision['drowsy_confidence']:.2f}")
        col1.metric("PERCLOS",           f"{vision['perclos']:.2f}")
        col2.metric("Head state",        vision["head_state"])
        col2.metric("Session minutes",   sensor.get("session_min", "—"))
        col3.metric("Jerk RMS",          sensor.get("jerk_rms", "—"))
        col3.metric("Distance cm",       sensor.get("distance_cm", "—"))

        st.progress(min(score, 1.0))
        st.caption(f"Driver: {driver_id}  |  Last updated: {time.strftime('%H:%M:%S')}")

    time.sleep(2)