import json
import os
import sys
import time
from datetime import datetime
from html import escape

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DRIVER_ID, FIREBASE_READ_URL, MOCK_SENSOR, WEATHER_CITY


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(BASE_DIR, "dashboard", "live_status.json")
MOCK_DB_PATH = os.path.join(BASE_DIR, "mock_db.json")
REFRESH_SECONDS = 2
MAX_CHART_POINTS = 90

COLORS = {
    "navy": "#111844",
    "indigo": "#4B5694",
    "cream": "#EAE0CF",
    "steel": "#7288AE",
    "panel": "#151B3D",
    "panel_soft": "#202856",
    "line_a": "#EAE0CF",
    "line_b": "#7288AE",
    "line_c": "#B9C5DD",
    "accent": "#D96F4B",
}

PRIMARY_METRICS = [
    ("YOLO Score", "yolo_confidence", "Vision model drowsy confidence", "YS"),
    ("Time", "time", "Latest main.py reading", "TM"),
    ("Weather", "weather", "OpenWeather condition", "WX"),
    ("Timestamp", "timestamp", "Raw live status timestamp", "TS"),
]

NUMERIC_GRAPH_FIELDS = [
    ("yolo_confidence", "YOLO Score"),
    ("perclos", "PERCLOS"),
    ("fusion_score", "Fusion Score"),
    ("jerk_rms", "Jerk RMS"),
    ("posture_dev_cm", "Posture Deviation"),
]

st.set_page_config(page_title="Fatigue Monitor", layout="wide", initial_sidebar_state="expanded")


def inject_styles():
    st.markdown(
        f"""
        <style>
            :root {{
                --navy: {COLORS["navy"]};
                --indigo: {COLORS["indigo"]};
                --cream: {COLORS["cream"]};
                --steel: {COLORS["steel"]};
                --panel: {COLORS["panel"]};
                --panel-soft: {COLORS["panel_soft"]};
                --accent: {COLORS["accent"]};
                --muted: rgba(234, 224, 207, 0.68);
            }}

            .stApp {{
                background: radial-gradient(circle at top left, rgba(75,86,148,0.20), transparent 34%),
                            linear-gradient(135deg, #070913 0%, #0A0D1B 48%, #111844 100%);
                color: var(--cream);
            }}

            header[data-testid="stHeader"] {{
                background: transparent;
            }}

            [data-testid="block-container"] {{
                max-width: 1480px;
                padding: 1rem 1.25rem 2rem;
            }}

            [data-testid="stSidebar"] {{
                background: #070913;
                border-right: 1px solid rgba(234, 224, 207, 0.10);
            }}

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
                color: var(--cream);
            }}

            .account-card {{
                align-items: center;
                border-bottom: 1px solid rgba(234, 224, 207, 0.12);
                display: flex;
                gap: 0.75rem;
                padding: 0.55rem 0 1.05rem;
            }}

            .account-avatar {{
                align-items: center;
                background: var(--accent);
                border: 1px solid rgba(234, 224, 207, 0.42);
                border-radius: 8px;
                color: white;
                display: flex;
                font-size: 1.15rem;
                font-weight: 850;
                height: 48px;
                justify-content: center;
                width: 48px;
            }}

            .account-name {{
                color: var(--cream);
                font-size: 1.2rem;
                font-weight: 850;
                line-height: 1.15;
            }}

            .account-mail, .muted {{
                color: var(--muted);
                font-size: 0.78rem;
            }}

            .side-label {{
                color: rgba(234, 224, 207, 0.46);
                font-size: 0.72rem;
                font-weight: 760;
                letter-spacing: 0.08em;
                margin: 1.25rem 0 0.52rem;
                text-transform: uppercase;
            }}

            .nav-item, .nav-item-active {{
                border: 1px solid rgba(234, 224, 207, 0.08);
                border-radius: 8px;
                color: rgba(234, 224, 207, 0.78);
                font-size: 0.91rem;
                font-weight: 720;
                margin-bottom: 0.48rem;
                padding: 0.72rem 0.78rem;
            }}

            .nav-item-active {{
                background: rgba(75, 86, 148, 0.32);
                border-color: var(--accent);
                color: var(--cream);
                box-shadow: inset 3px 0 0 var(--accent);
            }}

            .topbar {{
                align-items: center;
                display: flex;
                gap: 1rem;
                justify-content: space-between;
                margin-bottom: 1rem;
            }}

            .page-title {{
                color: var(--cream);
                font-size: 1.55rem;
                font-weight: 850;
                letter-spacing: 0;
                margin: 0;
            }}

            .page-subtitle {{
                color: var(--muted);
                font-size: 0.86rem;
                margin-top: 0.25rem;
            }}

            .status-chip {{
                background: rgba(234, 224, 207, 0.08);
                border: 1px solid rgba(234, 224, 207, 0.13);
                border-radius: 8px;
                color: var(--cream);
                display: inline-block;
                font-size: 0.78rem;
                font-weight: 780;
                margin-left: 0.5rem;
                padding: 0.62rem 0.78rem;
                white-space: nowrap;
            }}

            .metric-card, .panel, .alert-box {{
                background: linear-gradient(180deg, rgba(21,27,61,0.98), rgba(12,17,38,0.98));
                border: 1px solid rgba(234, 224, 207, 0.08);
                border-radius: 8px;
                box-shadow: 0 20px 42px rgba(0,0,0,0.24);
            }}

            .metric-card {{
                min-height: 146px;
                padding: 1rem;
            }}

            .metric-head {{
                align-items: center;
                display: flex;
                gap: 0.72rem;
                margin-bottom: 1rem;
            }}

            .metric-icon {{
                align-items: center;
                background: rgba(217, 111, 75, 0.16);
                border: 1px solid rgba(217, 111, 75, 0.40);
                border-radius: 8px;
                color: var(--cream);
                display: flex;
                font-size: 0.8rem;
                font-weight: 850;
                height: 38px;
                justify-content: center;
                width: 38px;
            }}

            .metric-label {{
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 760;
                letter-spacing: 0.05em;
                line-height: 1.22;
                text-transform: uppercase;
            }}

            .metric-value {{
                color: var(--cream);
                font-size: 1.85rem;
                font-weight: 850;
                line-height: 1.08;
                overflow-wrap: anywhere;
            }}

            .metric-foot {{
                color: rgba(234, 224, 207, 0.54);
                font-size: 0.76rem;
                margin-top: 0.58rem;
            }}

            .panel {{
                padding: 1rem;
            }}

            .panel-title {{
                color: var(--cream);
                font-size: 1rem;
                font-weight: 820;
                margin-bottom: 0.25rem;
            }}

            .panel-caption {{
                color: var(--muted);
                font-size: 0.8rem;
                margin-bottom: 0.85rem;
            }}

            .alert-box {{
                min-height: 156px;
                padding: 1rem;
            }}

            .alert-label {{
                color: var(--muted);
                font-size: 0.76rem;
                font-weight: 780;
                letter-spacing: 0.07em;
                text-transform: uppercase;
            }}

            .alert-value {{
                color: var(--cream);
                font-size: 1.1rem;
                font-weight: 820;
                line-height: 1.35;
                margin-top: 0.8rem;
            }}

            .field-grid {{
                display: grid;
                gap: 0.55rem;
                grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            }}

            .field-tile {{
                background: rgba(32, 40, 86, 0.58);
                border: 1px solid rgba(234, 224, 207, 0.08);
                border-radius: 8px;
                min-height: 76px;
                padding: 0.78rem;
            }}

            .field-name {{
                color: rgba(234, 224, 207, 0.54);
                font-size: 0.7rem;
                font-weight: 780;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }}

            .field-value {{
                color: var(--cream);
                font-size: 1rem;
                font-weight: 760;
                margin-top: 0.38rem;
                overflow-wrap: anywhere;
            }}

            .stAltairChart {{
                background: transparent;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.45rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                background: rgba(234, 224, 207, 0.07);
                border: 1px solid rgba(234, 224, 207, 0.10);
                border-radius: 8px;
                color: rgba(234, 224, 207, 0.72);
                font-size: 0.82rem;
                font-weight: 760;
                height: 38px;
                padding: 0 0.8rem;
            }}

            .stTabs [aria-selected="true"] {{
                background: rgba(217, 111, 75, 0.18);
                border-color: rgba(217, 111, 75, 0.55);
                color: var(--cream);
            }}

            @media (max-width: 760px) {{
                [data-testid="block-container"] {{
                    padding: 0.75rem 0.75rem 1.5rem;
                }}
                .topbar {{
                    align-items: flex-start;
                    flex-direction: column;
                }}
                .status-chip {{
                    margin: 0 0.35rem 0.35rem 0;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_text(value, fallback="Waiting"):
    if value is None or value == "":
        return fallback
    return escape(str(value))


def value_text(value, digits=2, suffix=""):
    if value is None or value == "":
        return "Waiting"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    if isinstance(value, int):
        return f"{value}{suffix}"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def read_json_file(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def read_live_sensor():
    try:
        response = requests.get(FIREBASE_READ_URL, timeout=4)
        if response.status_code == 200:
            return response.json() or {}
    except requests.exceptions.RequestException:
        return {}
    return {}


def read_latest_alert():
    mock_data = read_json_file(MOCK_DB_PATH)
    sessions = mock_data.get("drivers", {}).get(DRIVER_ID, {}).get("sessions", {})
    newest_key = ""
    newest_alert = {}
    for session in sessions.values():
        alerts = session.get("alerts", {}) if isinstance(session, dict) else {}
        for alert_key, alert in alerts.items():
            marker = str(alert_key)
            if marker > newest_key:
                newest_key = marker
                newest_alert = alert if isinstance(alert, dict) else {}
    return newest_alert


def build_status():
    status = read_json_file(STATUS_PATH)
    live_sensor = read_live_sensor()
    latest_alert = read_latest_alert()

    if not status:
        status = {
            "driver_id": DRIVER_ID,
            "jerk_rms": live_sensor.get("jerk_rms", MOCK_SENSOR.get("jerk_rms")),
            "posture_dev_cm": live_sensor.get("posture_dev_cm", MOCK_SENSOR.get("posture_dev_cm")),
            "distance_cm": live_sensor.get("distance_cm", MOCK_SENSOR.get("distance_cm")),
            "hour": live_sensor.get("hour", MOCK_SENSOR.get("hour")),
            "weather": latest_alert.get("weather", MOCK_SENSOR.get("weather")),
            "weather_city": WEATHER_CITY,
            "fusion_score": latest_alert.get("fusion_score"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "history": [],
        }

    for key in ("jerk_rms", "posture_dev_cm", "distance_cm", "hour"):
        if status.get(key) in (None, "") and live_sensor.get(key) not in (None, ""):
            status[key] = live_sensor.get(key)

    status.setdefault("weather_city", WEATHER_CITY)
    if status.get("weather") in (None, "") and latest_alert.get("weather"):
        status["weather"] = latest_alert.get("weather")
    if status.get("fusion_score") in (None, "") and latest_alert.get("fusion_score") is not None:
        status["fusion_score"] = latest_alert.get("fusion_score")

    status["latest_alert"] = latest_alert.get("message", "No alert recorded")
    return status


def render_sidebar(status):
    st.sidebar.markdown(
        f"""
        <div class="account-card">
            <div class="account-avatar">D</div>
            <div>
                <div class="account-name">EliteDrive</div>
                <div class="account-mail">{safe_text(status.get("driver_id"), DRIVER_ID)}</div>
            </div>
        </div>
        <div class="side-label">Menu</div>
        <div class="nav-item-active">Dashboard</div>
        <div class="nav-item">Live Metrics</div>
        <div class="nav-item">Weather</div>
        <div class="nav-item">Alerts</div>
        <div class="nav-item">Sensor Feed</div>
        <div class="side-label">Support</div>
        <div class="nav-item">Settings</div>
        <div class="nav-item">Support & Ticket</div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label, value, foot, icon, digits=2, suffix=""):
    shown = value_text(value, digits=digits, suffix=suffix) if isinstance(value, (int, float)) else safe_text(value)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-head">
                <div class="metric-icon">{escape(icon)}</div>
                <div class="metric-label">{escape(label)}</div>
            </div>
            <div class="metric-value">{shown}</div>
            <div class="metric-foot">{escape(foot)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_label(fusion_score):
    try:
        score = float(fusion_score)
    except (TypeError, ValueError):
        return "Waiting for main.py"
    if score >= 0.8:
        return "Critical fatigue risk"
    if score >= 0.6:
        return "Fatigue warning"
    return "Monitoring"


def chart_data(history):
    rows = []
    for index, point in enumerate(history[-MAX_CHART_POINTS:]):
        label = point.get("time") or str(index + 1)
        for key, metric in NUMERIC_GRAPH_FIELDS:
            value = point.get(key)
            if value is None and key == "yolo_confidence":
                value = point.get("confidence_score")
            if value is None and key == "posture_dev_cm":
                value = point.get("position_value")
            try:
                rows.append({"time": label, "metric": metric, "value": float(value)})
            except (TypeError, ValueError):
                continue
    return pd.DataFrame(rows)


def render_chart(history):
    data = chart_data(history)
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Live Value Trends</div>
            <div class="panel-caption">Line graph of numeric JSON values from main.py. No bar chart is used.</div>
        """,
        unsafe_allow_html=True,
    )

    if data.empty:
        st.markdown(
            '<div class="panel-caption">Run main.py to start collecting graph values.</div></div>',
            unsafe_allow_html=True,
        )
        return

    chart = (
        alt.Chart(data)
        .mark_line(point=alt.OverlayMarkDef(size=42), strokeWidth=2.6)
        .encode(
            x=alt.X("time:N", title="Time", axis=alt.Axis(labelAngle=-35, labelColor=COLORS["cream"], titleColor=COLORS["cream"])),
            y=alt.Y("value:Q", title="Value", axis=alt.Axis(labelColor=COLORS["cream"], titleColor=COLORS["cream"])),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(range=[COLORS["accent"], COLORS["cream"], COLORS["steel"], COLORS["line_c"], COLORS["indigo"]]),
                legend=alt.Legend(labelColor=COLORS["cream"], titleColor=COLORS["cream"], orient="top"),
            ),
            tooltip=["time:N", "metric:N", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=405)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="rgba(234,224,207,0.08)", domainColor="rgba(234,224,207,0.18)")
        .configure(background="#0F142D")
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_weather_panel(status):
    condition = status.get("weather") or "Waiting"
    score = status.get("weather_score")
    city = status.get("weather_city") or WEATHER_CITY
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Weather API</div>
            <div class="panel-caption">Live OpenWeather data used by the fatigue model.</div>
            <div class="field-grid">
                <div class="field-tile">
                    <div class="field-name">City</div>
                    <div class="field-value">{safe_text(city)}</div>
                </div>
                <div class="field-tile">
                    <div class="field-name">Condition</div>
                    <div class="field-value">{safe_text(condition)}</div>
                </div>
                <div class="field-tile">
                    <div class="field-name">Severity</div>
                    <div class="field-value">{value_text(score, digits=2)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_tile(label, value, icon, suffix="", digits=2):
    shown = value_text(value, digits=digits, suffix=suffix) if isinstance(value, (int, float)) else safe_text(value)
    st.markdown(
        f"""
        <div class="field-tile">
            <div class="field-name">{escape(icon)} | {escape(label)}</div>
            <div class="field-value">{shown}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_data_tabs(status):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Live Data</div>
            <div class="panel-caption">Tabbed values update every {REFRESH_SECONDS} seconds from main.py.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    vision_tab, motion_tab, weather_tab = st.tabs(["Vision", "Motion", "Weather"])

    with vision_tab:
        col1, col2 = st.columns(2)
        with col1:
            render_live_tile("YOLO Score", status.get("yolo_confidence"), "YS")
        with col2:
            render_live_tile("PERCLOS", status.get("perclos"), "PC")

    with motion_tab:
        col1, col2 = st.columns(2)
        with col1:
            render_live_tile("Posture Dev CM", status.get("posture_dev_cm"), "PV", suffix=" cm")
        with col2:
            render_live_tile("Jerk RMS", status.get("jerk_rms"), "JR")

    with weather_tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            render_live_tile("Weather", status.get("weather"), "WX")
        with col2:
            render_live_tile("City", status.get("weather_city") or WEATHER_CITY, "CT")
        with col3:
            render_live_tile("Weather Severity", status.get("weather_score"), "WS")


def render_right_tabs(status):
    alert_tab, driver_tab, weather_tab = st.tabs(["Alert", "Driver", "Weather"])

    with alert_tab:
        st.markdown(
            f"""
            <div class="alert-box">
                <div class="alert-label">Latest Alert</div>
                <div class="alert-value">{safe_text(status.get("latest_alert"), "No alert recorded")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        render_metric("Fusion Score", status.get("fusion_score"), "Current fatigue prediction", "FS")

    with driver_tab:
        render_metric("Head State", status.get("head_state"), "Vision pipeline state", "HS")
        render_metric("PERCLOS", status.get("perclos"), "Eye closure ratio", "PC")
        render_metric("Jerk RMS", status.get("jerk_rms"), "Movement variation", "JR")

    with weather_tab:
        render_weather_panel(status)


def render_dashboard():
    status = build_status()
    render_sidebar(status)

    fusion = status.get("fusion_score")
    now = datetime.now().strftime("%H:%M:%S")
    source_time = status.get("time") or now

    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <h1 class="page-title">Fatigue Detection Dashboard</h1>
                <div class="page-subtitle">Driver {safe_text(status.get("driver_id"), DRIVER_ID)} | Live refresh every {REFRESH_SECONDS}s</div>
            </div>
            <div>
                <span class="status-chip">{escape(status_label(fusion))}</span>
                <span class="status-chip">JSON {safe_text(source_time)}</span>
                <span class="status-chip">UI {now}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    for column, (label, key, foot, icon) in zip(metric_cols, PRIMARY_METRICS):
        with column:
            render_metric(label, status.get(key), foot, icon)

    left, right = st.columns([2.15, 1])
    with left:
        render_chart(status.get("history", []))

    with right:
        render_right_tabs(status)

    st.write("")
    render_live_data_tabs(status)


inject_styles()
render_dashboard()
time.sleep(REFRESH_SECONDS)
st.rerun()
