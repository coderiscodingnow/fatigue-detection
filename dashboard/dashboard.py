"""
Driver Monitoring System Dashboard
Ultra-premium dark mode glassmorphism dashboard.
"""

import json
import os
import sys
import time
import html
from datetime import datetime
import pandas as pd
import altair as alt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DRIVER_ID

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(BASE_DIR, "dashboard", "live_status.json")

# User Palette
COLORS = {
    "bg_dark":      "#021326",  # Darker shade of Steel Azure for gradient
    "steel_azure":  "#084887",
    "lavender":     "#909CC2",
    "tiger_orange": "#F58A07",
    "sandy_brown":  "#F9AB55",
    "ghost_white":  "#F7F5FB",
    "panel_bg":     "rgba(8, 72, 135, 0.25)",
    "panel_border": "rgba(144, 156, 194, 0.15)",
}

st.set_page_config(
    page_title="Driver Monitoring HUD",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        /* Hide Streamlit default elements */
        header[data-testid="stHeader"] {{ display: none !important; }}
        footer {{ display: none !important; }}
        .stDeployButton {{ display: none !important; }}

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"] {{
            background: linear-gradient(135deg, {COLORS["bg_dark"]} 0%, {COLORS["steel_azure"]} 100%) !important;
            background-attachment: fixed !important;
            color: {COLORS["ghost_white"]} !important;
            font-family: 'Outfit', sans-serif;
            overflow-x: hidden;
        }}
        
        [data-testid="block-container"] {{
            max-width: 100% !important;
            padding: 20px 40px !important;
        }}

        /* ── Animations ── */
        @keyframes pulse-glow {{
            0% {{ box-shadow: 0 0 10px rgba(245, 138, 7, 0.2); }}
            50% {{ box-shadow: 0 0 35px rgba(245, 138, 7, 0.8); }}
            100% {{ box-shadow: 0 0 10px rgba(245, 138, 7, 0.2); }}
        }}
        @keyframes pulse-ring {{
            0% {{ filter: drop-shadow(0 0 8px rgba(245, 138, 7, 0.4)); }}
            50% {{ filter: drop-shadow(0 0 25px rgba(245, 138, 7, 0.9)); }}
            100% {{ filter: drop-shadow(0 0 8px rgba(245, 138, 7, 0.4)); }}
        }}
        @keyframes shimmer {{
            0% {{ background-position: 200% center; }}
            100% {{ background-position: -200% center; }}
        }}
        @keyframes float-in {{
            0% {{ transform: translateY(20px); opacity: 0; }}
            100% {{ transform: translateY(0); opacity: 1; }}
        }}
        @keyframes slide-in {{
            0% {{ transform: translateX(-20px); opacity: 0; }}
            100% {{ transform: translateX(0); opacity: 1; }}
        }}

        /* ── Top Nav / Header ── */
        .glass-nav {{
            background: {COLORS["panel_bg"]};
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid {COLORS["panel_border"]};
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            border-radius: 16px;
            animation: float-in 0.4s ease-out forwards;
        }}
        .nav-logo {{ font-size: 1.8rem; font-weight: 800; color: {COLORS["ghost_white"]}; letter-spacing: 2px; }}
        .nav-logo span {{ color: {COLORS["tiger_orange"]}; }}
        .nav-items {{ display: flex; gap: 30px; }}
        .nav-item {{ color: {COLORS["lavender"]}; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }}
        .nav-item.active {{ color: {COLORS["tiger_orange"]}; border-bottom: 2px solid {COLORS["tiger_orange"]}; padding-bottom: 4px; }}

        /* ── Glass Cards ── */
        .glass-card {{
            background: {COLORS["panel_bg"]};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {COLORS["panel_border"]};
            border-radius: 16px;
            padding: 24px;
            height: 100%;
            display: flex;
            flex-direction: column;
            animation: float-in 0.6s ease-out forwards;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            border-color: rgba(245, 138, 7, 0.3);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(144, 156, 194, 0.1);
            padding-bottom: 12px;
        }}
        .card-title {{
            font-size: 0.9rem;
            font-weight: 600;
            color: {COLORS["lavender"]};
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }}
        .card-icon {{ color: {COLORS["lavender"]}; font-size: 1.2rem; }}

        /* ── Stat Displays ── */
        .big-stat {{
            font-size: 3.5rem;
            font-weight: 800;
            color: {COLORS["ghost_white"]};
            line-height: 1;
            margin: 10px 0;
            text-shadow: 0 0 20px rgba(247, 245, 251, 0.3);
        }}
        .stat-sub {{ font-size: 0.85rem; color: {COLORS["sandy_brown"]}; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
        
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }}
        .stat-item {{ display: flex; flex-direction: column; }}
        .si-label {{ color: {COLORS["lavender"]}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
        .si-val {{ color: {COLORS["ghost_white"]}; font-size: 1.6rem; font-weight: 800; margin-top: 4px; }}
        .si-val span {{ color: {COLORS["tiger_orange"]}; font-size: 0.9rem; margin-left: 4px; }}

        /* ── Progress Bars ── */
        .prog-container {{ margin-bottom: 22px; }}
        .prog-header {{ display: flex; justify-content: space-between; margin-bottom: 8px; }}
        .prog-label {{ color: {COLORS["ghost_white"]}; font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; }}
        .prog-val {{ color: {COLORS["sandy_brown"]}; font-size: 0.85rem; font-weight: 800; }}
        .prog-track {{ width: 100%; height: 12px; background: rgba(144, 156, 194, 0.1); border-radius: 6px; overflow: hidden; }}
        .prog-fill {{ height: 100%; border-radius: 6px; transition: width 0.5s ease; box-shadow: 0 0 12px rgba(245, 138, 7, 0.6); animation: shimmer 3s infinite linear; background-size: 200% auto !important; }}

        /* ── Timeline ── */
        .tl-item {{ display: flex; align-items: flex-start; gap: 15px; margin-bottom: 18px; animation: slide-in 0.4s ease-out forwards; }}
        .tl-icon {{ width: 12px; height: 12px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }}
        .tl-icon.crit {{ background: {COLORS["tiger_orange"]}; box-shadow: 0 0 10px {COLORS["tiger_orange"]}; }}
        .tl-icon.warn {{ background: {COLORS["sandy_brown"]}; box-shadow: 0 0 10px {COLORS["sandy_brown"]}; }}
        .tl-icon.ok   {{ background: {COLORS["lavender"]}; }}
        .tl-content {{ flex-grow: 1; border-bottom: 1px solid rgba(144, 156, 194, 0.1); padding-bottom: 12px; }}
        .tl-title {{ color: {COLORS["ghost_white"]}; font-size: 0.9rem; font-weight: 600; }}
        .tl-desc {{ color: {COLORS["lavender"]}; font-size: 0.8rem; margin-top: 4px; }}
        .tl-time {{ color: {COLORS["sandy_brown"]}; font-size: 0.75rem; font-weight: 600; margin-top: 4px; display: block; }}

        /* ── Gauge / Circular ── */
        .circ-wrap {{ position: relative; width: 100%; display: flex; justify-content: center; align-items: center; margin: 20px 0; }}
        .circ-svg {{ transform: rotate(-90deg); animation: pulse-ring 2s infinite ease-in-out; }}
        .circ-bg {{ fill: none; stroke: rgba(144, 156, 194, 0.1); stroke-width: 8; }}
        .circ-fill {{ fill: none; stroke: {COLORS["tiger_orange"]}; stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset 0.5s ease; }}
        .circ-content {{ position: absolute; text-align: center; }}
        .circ-val {{ font-size: 2.8rem; font-weight: 800; color: {COLORS["ghost_white"]}; line-height: 1; }}
        .circ-label {{ font-size: 0.8rem; color: {COLORS["sandy_brown"]}; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }}

        /* ── Environment Metrics Grid ── */
        .env-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }}
        .env-item {{ display: flex; align-items: center; gap: 10px; background: rgba(0,0,0,0.2); padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(144, 156, 194, 0.05); }}
        .env-icon {{ width: 32px; height: 32px; border-radius: 50%; background: rgba(245, 138, 7, 0.1); display: flex; align-items: center; justify-content: center; color: {COLORS["tiger_orange"]}; font-size: 1rem; font-weight: 800; flex-shrink: 0; }}
        .env-details {{ display: flex; flex-direction: column; overflow: hidden; }}
        .env-val {{ color: {COLORS["ghost_white"]}; font-size: 1.1rem; font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .env-label {{ color: {COLORS["lavender"]}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
def sf(v, d=0.0):
    try:    return float(v)
    except: return d

def st_(v, d="N/A"):
    if v is None or v == "": v = d
    return html.escape(str(v))

def data_age(ts):
    if not ts: return "N/A"
    s = max(0.0, time.time() - sf(ts, time.time()))
    return f"{s:.0f}s" if s < 60 else f"{s/60:.1f}m"

def load_status():
    try:
        if os.path.exists(STATUS_PATH):
            with open(STATUS_PATH) as f:
                return json.load(f)
    except Exception: pass
    return None


# ─────────────────────────────────────────────────────────────
#  Components
# ─────────────────────────────────────────────────────────────
def render_progress_bar(label, value, max_val, unit="", color_grad=True):
    pct = min(100, max(0, (value / max_val) * 100))
    bg = f"linear-gradient(90deg, {COLORS['sandy_brown']}, {COLORS['tiger_orange']}, {COLORS['sandy_brown']})" if color_grad else f"linear-gradient(90deg, {COLORS['steel_azure']}, {COLORS['lavender']}, {COLORS['steel_azure']})"
    return f"""<div class="prog-container">
<div class="prog-header">
<div class="prog-label">{label}</div>
<div class="prog-val">{value:.1f}{unit}</div>
</div>
<div class="prog-track">
<div class="prog-fill" style="width: {pct}%; background: {bg};"></div>
</div>
</div>"""

def render_circular_gauge(value, label, max_val=100):
    pct = min(100, max(0, (value / max_val) * 100))
    da  = 440  # Circumference for r=70
    off = da - da * (pct / 100)
    
    return f"""<div class="circ-wrap">
<svg class="circ-svg" width="160" height="160" viewBox="0 0 160 160">
<circle cx="80" cy="80" r="70" class="circ-bg"/>
<circle cx="80" cy="80" r="70" class="circ-fill" style="stroke-dasharray:{da}; stroke-dashoffset:{off}; stroke:{COLORS['tiger_orange']};"/>
</svg>
<div class="circ-content">
<div class="circ-val">{value:.0f}</div>
<div class="circ-label">{label}</div>
</div>
</div>"""

def render_timeline(history):
    rows = [p for p in history if p.get("fusion_score") is not None][-5:]
    if not rows:
        return f'<div style="color:{COLORS["lavender"]}; padding: 20px;">No events recorded.</div>'
    
    out = "<div>"
    for p in reversed(rows):
        fs = sf(p.get("fusion_score", 0))
        t = st_(p.get("time", "–"))
        
        if fs >= 0.65:
            cls = "crit"
            title = "CRITICAL ALERT"
        elif fs >= 0.45:
            cls = "warn"
            title = "CAUTION WARNING"
        else:
            cls = "ok"
            title = "SYSTEM NOMINAL"
            
        yolo = sf(p.get("yolo_confidence", 0))
        posture = sf(p.get("posture_dev_cm", 0))
        
        desc = f"YOLO: {yolo:.2f} | Posture: {posture:.1f}cm"
        
        out += f"""
        <div class="tl-item">
            <div class="tl-icon {cls}"></div>
            <div class="tl-content">
                <div class="tl-title">{title}</div>
                <div class="tl-desc">{desc}</div>
                <div class="tl-time">⏱ {t}</div>
            </div>
        </div>
        """
    out += "</div>"
    return out


# ─────────────────────────────────────────────────────────────
#  Main Dashboard Layout
# ─────────────────────────────────────────────────────────────
def display_dashboard():
    inject_styles()
    status = load_status()

    if status is None:
        st.markdown(f"""
        <div style="height: 100vh; display: flex; align-items: center; justify-content: center; flex-direction: column;">
          <div style="font-size: 2.5rem; font-weight: 800; color: {COLORS['tiger_orange']}; animation: pulse-glow 2s infinite; letter-spacing: 2px;">
            INITIALIZING SYSTEM...
          </div>
          <div style="color: {COLORS['lavender']}; margin-top: 15px; font-weight: 600; letter-spacing: 1px;">
            Awaiting telemetry uplink from vehicle
          </div>
        </div>""", unsafe_allow_html=True)
        time.sleep(1); st.rerun(); return

    # Extract Live Data
    history     = status.get("history", [])
    fusion      = sf(status.get("fusion_score", 0))
    alertness   = max(0.0, min(1.0, 1.0 - fusion)) * 100
    yolo        = sf(status.get("yolo_confidence", 0))
    perclos     = sf(status.get("perclos", 0)) * 100
    jerk        = sf(status.get("jerk_rms", 0))
    posture     = sf(status.get("posture_dev_cm", 0))
    distance    = sf(status.get("distance_cm", 0))
    
    humidity    = sf(status.get("humidity", 45.2))
    temperature = sf(status.get("temperature", 22.8))
    rtc_time    = st_(status.get("rtc", datetime.now().strftime("%H:%M:%S")))
    sess_min    = sf(status.get("session_min", 0))
    age         = data_age(status.get("timestamp"))

    # Top Navigation Bar
    st.markdown(f"""
    <div class="glass-nav">
        <div class="nav-logo">HA<span>-</span>NET <span>HUD</span></div>
        <div class="nav-items">
            <div class="nav-item active">Live Telemetry</div>
        </div>
        <div style="color: {COLORS['ghost_white']}; font-weight: 800;">
            {rtc_time} &nbsp;|&nbsp; <span style="color: {COLORS['tiger_orange']};">{st_(DRIVER_ID)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Grid Layout
    col1, col2, col3 = st.columns([1, 2, 1], gap="large")

    # ── Left Column ──
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">Driver State</div>
                <div class="card-icon">👁</div>
            </div>
            {render_circular_gauge(alertness, "ALERTNESS %")}
            <div style="text-align: center; margin-top: 10px;">
                <div style="color: {COLORS['tiger_orange']}; font-weight: 800; font-size: 1.2rem;">
                    {"CRITICAL" if fusion >= 0.65 else ("WARNING" if fusion >= 0.45 else "NOMINAL")}
                </div>
                <div style="color: {COLORS['lavender']}; font-size: 0.8rem; margin-top: 5px;">
                    Data Age: {age}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 25px;">
            <div class="card-header">
                <div class="card-title">Recent Events</div>
                <div class="card-icon">⚡</div>
            </div>
            {render_timeline(history)}
        </div>
        """, unsafe_allow_html=True)

    # ── Center Column ──
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">Primary Telemetry</div>
                <div class="card-icon">∿</div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                <div>
                    <div class="stat-sub">Fusion Risk Score</div>
                    <div class="big-stat" style="color: {COLORS['tiger_orange'] if fusion > 0.45 else COLORS['ghost_white']}">{fusion:.3f}</div>
                </div>
                <div class="stat-grid" style="width: 50%;">
                    <div class="stat-item">
                        <div class="si-label">Sess. Duration</div>
                        <div class="si-val">{sess_min:.1f}<span>m</span></div>
                    </div>
                    <div class="stat-item">
                        <div class="si-label">Data Pts</div>
                        <div class="si-val">{len(history)}<span>pts</span></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Main Area Chart
        pts = history[-60:]
        if pts:
            df = pd.DataFrame([{"Time": p.get("time",""), "Risk": sf(p.get("fusion_score",0)), "i": i} for i, p in enumerate(pts)])
            chart = alt.Chart(df).mark_area(
                line={"color": COLORS["tiger_orange"], "strokeWidth": 3},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color=COLORS["tiger_orange"], offset=0),
                        alt.GradientStop(color="rgba(245, 138, 7, 0)", offset=1)
                    ], x1=1, y1=0, x2=1, y2=1
                )
            ).encode(
                x=alt.X("i:Q", axis=None),
                y=alt.Y("Risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(gridColor=COLORS["panel_border"], labelColor=COLORS["lavender"], title="")),
                tooltip=["Time:N", "Risk:Q"]
            ).properties(height=420).configure_view(strokeOpacity=0).configure(background="transparent")
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.markdown(f'<div style="height: 420px; display:flex; align-items:center; justify-content:center; color:{COLORS["lavender"]};">Awaiting Chart Data...</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Right Column ──
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-header">
                <div class="card-title">Risk Factors</div>
                <div class="card-icon">📊</div>
            </div>
            {render_progress_bar("YOLO Confidence", yolo * 100, 100, "%")}
            {render_progress_bar("Eye Closure (PERCLOS)", perclos, 100, "%")}
            {render_progress_bar("Posture Deviation", posture, 100, "cm")}
            {render_progress_bar("Head Distance", distance, 150, "cm", False)}
            {render_progress_bar("Motion Jerk RMS", jerk * 100, 100, "", False)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="margin-top: 25px;">
            <div class="card-header">
                <div class="card-title">Cabin Context</div>
                <div class="card-icon">🌡</div>
            </div>
            <div class="env-grid">
                <div class="env-item">
                    <div class="env-icon">🌡</div>
                    <div class="env-details">
                        <div class="env-val">{temperature:.1f}°</div>
                        <div class="env-label">Temp</div>
                    </div>
                </div>
                <div class="env-item">
                    <div class="env-icon">💧</div>
                    <div class="env-details">
                        <div class="env-val">{humidity:.1f}%</div>
                        <div class="env-label">Humid</div>
                    </div>
                </div>
            </div>
            <div class="env-grid" style="grid-template-columns: 1fr;">
                <div class="env-item" style="justify-content: center; background: rgba(245, 138, 7, 0.05); border-color: rgba(245, 138, 7, 0.2);">
                    <div style="text-align: center;">
                        <div class="env-val" style="color: {COLORS['sandy_brown']}; font-size: 1.5rem;">{rtc_time}</div>
                        <div class="env-label" style="color: {COLORS['tiger_orange']};">System RTC</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    display_dashboard()
