# Driver Monitoring System (DMS) - Project Implementation Document

This document provides a comprehensive technical overview of the implementation, architecture, datasets, models, database schemas, and hardware communication protocols utilized in the **Driver Monitoring System**.

---

## 1. System Architecture

The system consists of three core components working in tandem to monitor, fuse, evaluate, and alert when driver fatigue or drowsiness is detected:

```mermaid
graph TD
    ESP32[ESP32 Microcontroller] -->|Wi-Fi / HTTPS| Firebase[(Firebase Realtime DB)]
    Webcam[Driver Webcam] -->|Video Stream| VisionPipeline[YOLO v8 Vision Pipeline]
    
    subgraph Python Backend (main.py)
        Firebase -->|Polls Admin SDK| SensorState[Sensor Data Broker]
        VisionPipeline -->|Extracts Vision Metrics| Features[Feature Vector Builder]
        SensorState --> Features
        WeatherAPI[OpenWeather API] -->|Weather Context| Features
        
        Features -->|11-Feature Array| LightGBM[LightGBM Fusion Classifier]
        Features -->|Raw Features| RLBandit[Multi-Armed Bandit Alert Engine]
        
        LightGBM -->|Fusion Score| LiveStatus[live_status.json]
        RLBandit -->|Alert Decision| LiveStatus
        RLBandit -->|Triggers Action| AlertSystems[Voice / Audio HUD Alert]
    end

    LiveStatus -->|Read Every 1s| Streamlit[Streamlit Night HUD Dashboard]
```

---

## 2. Feature Definitions & Vector Extraction

A total of **11 features** are extracted or computed in real-time. These features capture computer vision indicators, physical sensor metrics from the vehicle/driver, temporal/circadian variables, and environmental factors.

| Feature Name | Source | Description | Range / Units | Type |
| :--- | :--- | :--- | :--- | :--- |
| `drowsy_conf` | YOLO Vision | Confidence score of the driver being classified as "drowsy". | `0.0` - `1.0` | Float |
| `perclos` | YOLO Vision | Percentage of Eye Closure computed over a rolling buffer of 60 frames. | `0.0` - `1.0` | Float |
| `head_nodding` | YOLO Vision | Binary flag indicating if the driver's head position indicates nodding (center-y $> 65\%$ of height). | `0` or `1` | Binary |
| `head_tilted` | YOLO Vision | Binary flag indicating if the driver's head is tilted (aspect ratio of bounding box $< 0.6$). | `0` or `1` | Binary |
| `alert_score` | Heuristic Fusion | Heuristic blend of YOLO, PERCLOS, and Head position, penalized by head distance. | `0.0` - `1.0` | Float |
| `jerk_rms` | ESP32 (Firebase) | Root Mean Square acceleration jerk (erratic steering wheel movements). | `0.0` - `10.0` (m/s³) | Float |
| `distance_cm` | ESP32 (Firebase) | Ultrasonic distance from sensor near the head. $> 3.5$ cm is dangerous (head slumped/far). | `-1.0` to `100.0` cm | Float |
| `posture_dev_cm`| ESP32 (Firebase) | Measured lateral head posture deviation. | `0.0` - `50.0` cm | Float |
| `hour` | System / Firebase| Hour of the day (circadian cycle influence). | `0` - `23` | Integer |
| `session_min` | System | Continuous duration of the current driving session. | `0` - `500` min | Float |
| `weather_rain` | OpenWeather API | Binary indicator of rain, mist, fog, or thunderstorm. | `0` or `1` | Binary |

---

## 3. Models & Machine Learning Architectures

### A. YOLOv8 Drowsiness Detection Model
The front-facing camera captures frames processed by a custom fine-tuned **YOLOv8 nano (YOLOv8n)** object detection model.
- **Base Architecture**: YOLOv8n (Lightweight, ideal for real-time edge CPU/GPU execution).
- **Classes**: `drowsy`, `awake`.
- **Training Parameters**:
  - **Epochs**: 20 epochs
  - **Optimizer**: Automatic selection (AdamW / SGD)
  - **Image Size**: 640x640 pixels
  - **Device**: CPU / CUDA
  - **Initial Learning Rate ($lr_0$)**: $0.01$
  - **Final Learning Rate ($lrf$)**: $0.01$

#### Confusion Matrix & Metrics (train-4)
The validation metrics after 20 epochs represent high detection capabilities:

| Metric Name | Value | Percentage |
| :--- | :--- | :--- |
| **Precision** | `0.95585` | 95.59% |
| **Recall (Sensitivity)** | `0.97003` | 97.00% |
| **mAP50** | `0.98070` | 98.07% |
| **mAP50-95** | `0.93703` | 93.70% |

- **YOLO Heuristic Integration ($H_{alert}$)**:
  $$H_{alert} = 0.50 \cdot C_{drowsy\_yolo} + 0.35 \cdot PERCLOS + 0.15 \cdot H_{head}$$
  - Where $H_{head} \in \{1.0 \text{ (nodding)}, 0.6 \text{ (tilted)}, 0.0 \text{ (upright)}\}$.
  - If the head distance $d > 3.5\text{ cm}$, a penalty of $+0.20$ is added directly to $H_{alert}$ (capped at 1.0) to prioritize mechanical physical cues.

---

### B. LightGBM Sensor Fusion Classifier
A LightGBM tree classifier fuses multi-modal signals (vision + physical vehicle sensors + weather + driver time-on-road) to generate a final probability score.
- **Model**: `LGBMClassifier`
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 4
  - `random_state`: 42
- **Training Logic**: Trained on a dataset mapping complex edge conditions. Specifically, the generator labels data as drowsy (Class 1) if:
  1. The vision model outputs high drowsiness and eye closure.
  2. The posture sensor registers head distance $d > 3.5\text{ cm}$ (simulating slumping/slouching away from the headrest sensor), even if visual features are normal.
  3. Continuous driving time is high combined with late-night hours.
- **Training Accuracy**: ~100% on the synthesized edge-case scenario distribution, enabling robust generalization to complex sensor failures.

---

### C. Multi-Armed Bandit Alert Selection Engine (Reinforcement Learning)
A reinforcement learning layer optimizes which alert method is most effective at waking the driver or getting their attention.
- **Algorithm**: $\epsilon$-Greedy Bandit ($\epsilon = 0.15$)
- **Action Space**: `["voice_alert", "vibration_alert", "visual_alert"]`
- **Reward Policy ($R$)**:
  - $1.0$ (Acknowledge) if the driver hits "ack" or resumes neutral posture/distance $\le 3.5\text{ cm}$.
  - $0.0$ if the driver ignores the alert.
- **Updating Rule**:
  $$Q(a) \leftarrow \frac{R_{sum}(a)}{C(a)}$$
  - Where $C(a)$ is the select count of action $a$ and $R_{sum}(a)$ is the total reward accumulated.
  - $15\%$ of the time the engine **explores** to capture shifts in driver responsiveness.
  - $85\%$ of the time the engine **exploits** the alert method showing the highest historical Q-value.

---

## 4. Database Setup & Realtime Schema

The database is built on **Firebase Realtime Database** to enable zero-latency synchronicity between the physical vehicle components (ESP32) and the Python monitoring system.

### Database JSON Tree

```json
{
  "drowsiness": {
    "latest": {
      "distance_cm": 2.8,
      "jerk_rms": 1.24,
      "posture_dev_cm": 3.1,
      "ts": 1783459281,
      "hour": 14
    }
  },
  "drivers": {
    "driver_001": {
      "latest_sensor": {
        "distance_cm": 2.8,
        "jerk_rms": 1.24,
        "posture_dev_cm": 3.1,
        "ts": 1783459281
      },
      "rl_state": {
        "voice_alert": {
          "count": 14,
          "reward_sum": 12.0
        },
        "vibration_alert": {
          "count": 5,
          "reward_sum": 3.0
        },
        "visual_alert": {
          "count": 3,
          "reward_sum": 0.0
        }
      },
      "sessions": {
        "session_1783459000": {
          "alerts": {
            "alert_1783459100": {
              "fusion_score": 0.72,
              "message": "You are showing signs of fatigue. Please pull over and take a rest.",
              "weather": "moderate rain",
              "hour": 14,
              "feedback": "ack",
              "reward": 1.0
            }
          }
        }
      }
    }
  }
}
```

---

## 5. ESP32 Hardware Integration & Communication

The hardware layer runs on an **ESP32 microcontroller** which collects real-time physical readings from the cabin sensors and pushes them to Firebase.

### ESP32 Setup & Wiring
1. **Ultrasonic Sensor (HC-SR04)**: Placed near the headrest/head level of the driver.
   - **VCC** $\rightarrow$ ESP32 5V
   - **GND** $\rightarrow$ ESP32 GND
   - **Trig Pin** $\rightarrow$ ESP32 GPIO 5
   - **Echo Pin** $\rightarrow$ ESP32 GPIO 18
2. **Accelerometer/Gyroscope (MPU6050)**: Mounted on the vehicle steering column.
   - **VCC** $\rightarrow$ ESP32 3.3V
   - **GND** $\rightarrow$ ESP32 GND
   - **SCL Pin** $\rightarrow$ ESP32 GPIO 22
   - **SDA Pin** $\rightarrow$ ESP32 GPIO 21

### Communication Protocols & Upload Cycle
- **Connection Mode**: Local Wi-Fi router or mobile hotspot connection using the standard `WiFi.h` library.
- **Protocol**: HTTPS PATCH requests or Firebase Arduino Client library (`FirebaseESP32.h`).
- **Upload Period**: Uploads every $2$ seconds (matching Python polling interval).
- **ESP32 Pseudocode**:
  ```cpp
  #include <WiFi.h>
  #include <FirebaseESP32.h>
  #include <Adafruit_MPU6050.h>
  #include <Adafruit_Sensor.h>

  #define FIREBASE_HOST "fatigue-detection-26427-default-rtdb.asia-southeast1.firebasedatabase.app"
  #define FIREBASE_AUTH "4g6DYXF1PyHL8BuTLEPMbdBCtb9gacd7PXMrkWJo"
  #define WIFI_SSID "Your_WiFi_Name"
  #define WIFI_PASSWORD "Your_WiFi_Password"

  #define TRIG_PIN 5
  #define ECHO_PIN 18

  FirebaseData firebaseData;
  Adafruit_MPU6050 mpu;

  float last_ax = 0, last_ay = 0, last_az = 0;

  void setup() {
    Serial.begin(115200);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    // Initialize WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) { delay(500); }

    // Initialize MPU6050
    mpu.begin();

    // Connect to Firebase
    Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);
    Firebase.reconnectWiFi(true);
  }

  float getDistance() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long duration = pulseIn(ECHO_PIN, HIGH);
    float distance = duration * 0.034 / 2;
    return distance; // Returns distance in cm
  }

  void loop() {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Compute Jerk (Rate of change of acceleration)
    float jerk_x = (a.acceleration.x - last_ax) / 2.0;
    float jerk_y = (a.acceleration.y - last_ay) / 2.0;
    float jerk_z = (a.acceleration.z - last_az) / 2.0;
    float jerk_rms = sqrt(jerk_x*jerk_x + jerk_y*jerk_y + jerk_z*jerk_z);

    last_ax = a.acceleration.x;
    last_ay = a.acceleration.y;
    last_az = a.acceleration.z;

    float distance = getDistance();
    float posture_dev = abs(distance - 3.0); // Baseline distance is around 3.0 cm

    // Build JSON Payload
    FirebaseJson json;
    json.set("distance_cm", distance);
    json.set("jerk_rms", jerk_rms);
    json.set("posture_dev_cm", posture_dev);
    json.set("ts", (int)(millis() / 1000));
    json.set("hour", 14); // Or fetch via NTP client

    // Upload to Firebase
    if (Firebase.updateNode(firebaseData, "/drowsiness/latest", json)) {
      Serial.println("Data pushed successfully");
    } else {
      Serial.println(firebaseData.errorReason());
    }

    delay(2000); // 2 second cycle
  }
  ```
