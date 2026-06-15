# Driver Monitoring System - Dashboard Guide

## System Overview

The **Driver Monitoring System** is a real-time drowsiness/fatigue detection system designed for in-vehicle use. It combines computer vision, sensor data, weather information, and machine learning to monitor driver alertness.

### Architecture

- **Backend (main.py)**: Runs continuously in the background, collecting data from:
  - YOLO vision model (webcam)
  - ESP32 sensors (Firebase Realtime DB)
  - Weather API
  - Combines all data and calculates alertness score
  - Updates `dashboard/live_status.json` every second

- **Frontend (Dashboard)**: Streamlit-based car dashboard interface
  - Displays real-time metrics
  - Shows live alertness heatmap
  - Updates every 1 second
  - Professional car system styling (no emojis)
  - Night mode with safety-oriented color palette

## Display Metrics

### Primary Metrics
- **YOLO Confidence**: Vision model drowsiness confidence (0-1)
- **PERCLOS**: Percentage of Eye Closure (0-1)
- **Jerk**: Root mean square acceleration jerk
- **Posture**: Head posture deviation from center (cm)

### Secondary Metrics
- **Alertness Score**: Fusion score from LightGBM model (0-1)
- **Weather**: Current weather condition
- **Time of Day**: Night/Morning/Afternoon/Evening
- **Session Duration**: How long the current session has been running

### Alertness Levels
- **ALERT** (< 0.25): Driver is alert - Green signal
- **NORMAL** (0.25-0.45): Normal fatigue levels - Yellow
- **CAUTION** (0.45-0.65): Fatigue symptoms detected - Orange
- **CRITICAL** (> 0.65): Critical fatigue level - Red

## Color Palette

Professional night-mode palette (car dashboard standard):
- Background Dark: #272121
- Background Darker: #363333
- Accent (Orange): #E16428
- Text Light: #F6E9E9

## Running the System

### Option 1: PowerShell (Recommended for Windows)
```powershell
.\.venv\Scripts\Activate.ps1
.\start_system.ps1
```

### Option 2: Batch File
```cmd
.\start_system.bat
```

### Option 3: Python Script
```bash
python start_system.py
```

### Option 4: Manual (two terminals)
**Terminal 1** - Backend (Data Collection):
```bash
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
python main.py
```

**Terminal 2** - Frontend (Dashboard):
```bash
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
streamlit run dashboard/dashboard.py
```

## Dashboard Features

### 1. Live Status Indicator
- Shows current driver state (ALERT, NORMAL, CAUTION, CRITICAL)
- Color-coded for quick visual identification
- Displays alertness score (0-1)

### 2. Alertness Progression Heatmap
- Bar chart showing fusion score over time
- Color gradient: Green (alert) → Yellow → Orange → Red (critical)
- Shows last 90 data points (approximately 1.5 minutes)
- Hover for exact values and timestamps

### 3. Multi-Metric Performance Chart
- Line chart tracking 4 key metrics:
  - YOLO Confidence
  - PERCLOS
  - Jerk (scaled for visibility)
  - Posture Deviation (scaled for visibility)
- Interactive: zoom, pan, hover for details

### 4. Environmental Conditions
- Current weather (from OpenWeather API)
- Time of day (affects fatigue baseline)
- Session duration (longer sessions = higher fatigue)

### 5. Critical Alerts
- Displays warning banner when alertness score exceeds thresholds
- CAUTION alert at score > 0.45
- CRITICAL alert at score > 0.65

## Data Flow

```
main.py (Backend)
├── Vision Pipeline (YOLO)
│   ├── Webcam stream
│   └── Drowsiness/PERCLOS detection
├── Firebase Polling (ESP32 sensors)
│   ├── Jerk RMS
│   ├── Posture Deviation
│   └── Distance
├── Weather API
│   └── Current conditions
├── Fusion Model (LightGBM)
│   └── Calculates alertness score
├── RL Bandit
│   └── Intelligent alert decision
└── JSON Writer → live_status.json

                    ↓

live_status.json (Shared State)
├── Current metrics (YOLO, PERCLOS, etc.)
└── History (last 90 points)

                    ↓

Dashboard (Frontend - Streamlit)
├── Reads JSON every 1 second
├── Displays metrics
├── Renders heatmap
└── Shows alerts
```

## System Requirements

- Python 3.8+
- Webcam (for YOLO vision model)
- Firebase connection (for ESP32 sensor data)
- Internet (for weather API)
- CUDA GPU recommended (for YOLO inference)

## Performance

- **Update Frequency**: 1 second (both backend and dashboard)
- **History Retention**: 90 data points (~1.5 minutes)
- **Main Loop Latency**: <100ms per cycle
- **Dashboard Refresh**: Smooth real-time updates

## Troubleshooting

### Dashboard Shows "Waiting for data from main.py"
- **Cause**: main.py not running or hasn't written data yet
- **Solution**: Check that main.py process is running in the other terminal
- **Check**: Verify `dashboard/live_status.json` exists and is being updated

### No YOLO detections showing
- **Cause**: Vision pipeline not initialized or webcam not available
- **Solution**: 
  - Ensure webcam is connected and working
  - Check `MODEL_PATH` in config.py points to correct YOLO weights
  - Verify GPU/CUDA is available for faster inference

### Firebase connection errors
- **Cause**: Network issue or invalid credentials
- **Solution**:
  - Verify `serviceAccountKey.json` is in project root
  - Check internet connection
  - Verify Firebase URL and secret in config.py

### Dashboard is slow or freezing
- **Cause**: Large history size or slow machine
- **Solution**:
  - Reduce `MAX_CHART_POINTS` in dashboard.py
  - Close other applications
  - Check system resources (CPU/memory)

## Customization

### Adjust Alert Thresholds
Edit `get_alertness_level()` function in `dashboard/dashboard.py`:
```python
if fusion_score < 0.25:   # Adjust thresholds here
    return "ALERT", ...
```

### Change Color Palette
Edit `COLORS` dictionary at top of `dashboard/dashboard.py`:
```python
COLORS = {
    "bg_dark": "#272121",
    "bg_darker": "#363333",
    "accent_orange": "#E16428",
    "text_light": "#F6E9E9",
}
```

### Modify Update Frequency
Edit `REFRESH_SECONDS` in dashboard.py:
```python
REFRESH_SECONDS = 1  # Change to 2 for 2-second updates
```

### Adjust History Size
Edit `MAX_CHART_POINTS` in dashboard.py:
```python
MAX_CHART_POINTS = 90  # More points = more history = slower rendering
```

## Production Deployment

For production car systems:
1. Run main.py as a system service/daemon
2. Run dashboard on a dedicated touch-screen display
3. Set `REFRESH_SECONDS = 1` for responsive updates
4. Enable vehicle CAN-bus integration for jerk calculation
5. Add persistent logging to database
6. Implement failsafe alerts (audio/haptic)

## Support

For issues or questions:
1. Check that all required dependencies are installed: `pip install -r requirements.txt`
2. Verify all paths in `config.py` are correct
3. Check main.py console output for error messages
4. Ensure Firebase and weather APIs are accessible

---

**System Version**: 2.0 (Dashboard Redesign)  
**Last Updated**: 2024  
**Status**: Production Ready
