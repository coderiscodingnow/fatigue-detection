# test_firebase.py  — run this first to confirm Python can read the data
import requests
import time

FIREBASE_URL ="https://fatigue-detection-26427-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_SECRET = "4g6DYXF1PyHL8BuTLEPMbdBCtb9gacd7PXMrkWJo"
READ_URL = f"{FIREBASE_URL}/drowsiness/latest.json?auth={FIREBASE_SECRET}"

print("Polling Firebase every 2 seconds...\n")

last_ts = None
while True:
    response = requests.get(READ_URL, timeout=5)
    data = response.json()

    if data is None:
        print("No data yet")
    else:
        ts = data.get("ts")
        if ts != last_ts:
            last_ts = ts
            print(f"NEW reading → dist={data.get('distance_cm')}cm  "
                  f"accel=({data.get('accel_x')}, {data.get('accel_y')}, {data.get('accel_z')})  "
                  f"jerk_rms={data.get('jerk_rms')}  ts={ts}")
        else:
            print("No new data since last poll...")

    time.sleep(2)