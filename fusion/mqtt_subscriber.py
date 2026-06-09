import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import threading
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

class SensorSubscriber:
    def __init__(self):
        self.latest = None
        self._lock  = threading.Lock()
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        print(f"MQTT connected (rc={rc}), subscribing to {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            with self._lock:
                self.latest = payload
        except Exception as e:
            print(f"MQTT parse error: {e}")

    def start(self):
        self._client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()

    def get(self):
        with self._lock:
            return self.latest