"""
Driver Monitoring System Dashboard
Night mode car dashboard — tabbed layout with live heatmap, sensor values, and alert timeline.
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

from config import DRIVER_ID, WEATHER_CITY

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(BASE_DIR, "dashboard", "live_status.json")

COLORS = {
    "bg":           "#272121",
    "panel":        "#363333",
    "accent":       "#E16428",
    "text":         "#F6E9E9",
    "muted":        "rgba(246,233,233,0.6)",
    "muted_border": "rgba(246,233,233,0.1)",
    "success":      "#4CAF82",
    "warning":      "#E1A028",
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
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"] {{
            background-color: {COLORS["bg"]} !important;
            color: {COLORS["text"]} !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        [data-testid="block-container"] {{
            max-width: 100% !important;
            padding: 20px 28px !important;
            background-color: {COLORS["bg"]};
        }}

        /* ── Tab bar ── */
        [data-testid="stTabs"] > div:first-child {{
            border-bottom: 1px solid {COLORS["muted_border"]};
            gap: 4px;
            margin-bottom: 20px;
        }}
        button[data-baseweb="tab"] {{
            background: transparent !important;
            color: {COLORS["muted"]} !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            letter-spacing: 1.2px !important;
            text-transform: uppercase !important;
            padding: 10px 20px !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 0 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {COLORS["text"]} !important;
            border-bottom: 2px solid {COLORS["accent"]} !important;
        }}
        button[data-baseweb="tab"]:hover {{
            color: {COLORS["text"]} !important;
            background: rgba(246,233,233,0.04) !important;
        }}
        [data-testid="stTabsContent"] {{ padding-top: 0 !important; }}

        /* ── HUD header ── */
        .hud-header {{
            background: {COLORS["panel"]};
            border: 1px solid {COLORS["muted_border"]};
            border-radius: 8px;
            padding: 14px 22px;
            margin-bottom: 22px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .hud-title  {{ font-size:1.35rem; font-weight:800; letter-spacing:2px; text-transform:uppercase; }}
        .hud-sub    {{ font-size:0.72rem; color:{COLORS["muted"]}; letter-spacing:1px; text-transform:uppercase; margin-top:3px; }}
        .status-pill {{
            border: 2px solid {COLORS["muted_border"]};
            border-radius: 4px;
            padding: 7px 15px;
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: {COLORS["text"]};
        }}
        .status-pill.warn    {{ border-color:{COLORS["warning"]}; color:{COLORS["warning"]}; }}
        .status-pill.crit    {{ border-color:{COLORS["accent"]};  color:{COLORS["accent"]};  }}

        /* ── Cards ── */
        .card {{
            background: {COLORS["panel"]};
            border: 1px solid {COLORS["muted_border"]};
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 18px;
        }}
        .card-title {{
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 1.5px;
            color: {COLORS["muted"]};
            text-transform: uppercase;
            margin-bottom: 14px;
        }}

        /* ── Metric grid ── */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 14px;
            margin-bottom: 18px;
        }}
        .metric-card {{
            background: {COLORS["panel"]};
            border: 1px solid {COLORS["muted_border"]};
            border-radius: 8px;
            padding: 14px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content:''; position:absolute; top:0; left:0;
            width:4px; height:100%;
            background: rgba(246,233,233,0.15);
        }}
        .metric-card.lit::before {{ background:{COLORS["accent"]}; }}
        .m-label {{ font-size:0.66rem; font-weight:700; letter-spacing:1px; color:{COLORS["muted"]}; text-transform:uppercase; }}
        .m-val   {{ font-size:1.6rem; font-weight:800; color:{COLORS["text"]}; margin:7px 0; font-variant-numeric:tabular-nums; }}
        .m-sub   {{ font-size:0.62rem; color:{COLORS["muted"]}; }}

        /* ── Sensor value grid ── */
        .sv-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }}
        .sv-card {{
            border: 1px solid {COLORS["muted_border"]};
            border-radius: 8px;
            padding: 13px 15px;
            background: rgba(39,33,33,0.4);
        }}
        .sv-card.lit {{ border-color:rgba(225,100,40,0.55); box-shadow:inset 3px 0 0 {COLORS["accent"]}; }}
        .sv-label  {{ font-size:0.63rem; font-weight:700; letter-spacing:1px; color:{COLORS["muted"]}; text-transform:uppercase; white-space:nowrap; }}
        .sv-val    {{ margin-top:7px; font-size:1.1rem; font-weight:800; color:{COLORS["text"]}; font-variant-numeric:tabular-nums; }}
        .sv-unit   {{ color:{COLORS["muted"]}; font-size:0.68rem; margin-left:3px; }}

        /* ── Gauge ── */
        .gauge-wrap {{ display:flex; justify-content:center; padding:8px 0 4px; }}
        .gauge-inner {{ position:relative; width:160px; height:160px; display:flex; align-items:center; justify-content:center; }}
        .gauge-svg  {{ transform:rotate(-90deg); }}
        .gauge-bg   {{ fill:none; stroke:rgba(246,233,233,0.05); stroke-width:10; }}
        .gauge-arc  {{ fill:none; stroke-width:10; stroke-linecap:round; transition:stroke-dashoffset .5s ease; }}
        .gauge-overlay {{ position:absolute; display:flex; flex-direction:column; align-items:center; justify-content:center; }}
        .gauge-pct  {{ font-size:2rem; font-weight:800; color:{COLORS["text"]}; line-height:1; }}
        .gauge-lbl  {{ font-size:0.6rem; font-weight:700; letter-spacing:1px; color:{COLORS["muted"]}; margin-top:5px; text-transform:uppercase; }}

        /* ── Heatmap ── */
        .hm-title {{ font-size:0.7rem; font-weight:700; letter-spacing:1.5px; color:{COLORS["muted"]}; text-transform:uppercase; margin-bottom:12px; }}
        .hm-grid  {{ display:grid; grid-template-columns:repeat(12,1fr); gap:6px; }}
        .hm-cell  {{
            aspect-ratio:1; border-radius:4px;
            border:1px solid rgba(246,233,233,0.05);
            transition:transform .15s, box-shadow .15s;
        }}
        .hm-cell:hover {{ transform:scale(1.2); z-index:10; border-color:{COLORS["accent"]}; }}

        /* ── Warning banner ── */
        .warn-banner {{
            background: linear-gradient(90deg,rgba(225,100,40,0.14) 0%,rgba(39,33,33,0) 100%);
            border-left: 4px solid {COLORS["accent"]};
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 20px;
        }}
        .warn-title {{ font-size:0.82rem; font-weight:800; color:{COLORS["accent"]}; letter-spacing:1.5px; text-transform:uppercase; }}
        .warn-desc  {{ font-size:0.78rem; color:{COLORS["text"]}; margin-top:4px; }}

        /* ── Alert timeline ── */
        .timeline {{ display:flex; flex-direction:column; gap:10px; }}
        .tl-row {{
            display:flex; align-items:flex-start; gap:14px;
            background:{COLORS["panel"]}; border:1px solid {COLORS["muted_border"]};
            border-radius:8px; padding:12px 16px;
        }}
        .tl-row.crit {{ border-left:3px solid {COLORS["accent"]}; }}
        .tl-row.warn {{ border-left:3px solid {COLORS["warning"]}; }}
        .tl-row.ok   {{ border-left:3px solid {COLORS["success"]}; }}
        .tl-time  {{ font-size:0.72rem; font-weight:700; font-family:monospace; color:{COLORS["muted"]}; white-space:nowrap; margin-top:2px; }}
        .tl-score {{ font-size:1.1rem; font-weight:800; color:{COLORS["text"]}; min-width:52px; }}
        .tl-msg   {{ font-size:0.78rem; color:{COLORS["muted"]}; margin-top:3px; line-height:1.45; }}

        /* ── Responsive ── */
        @media (max-width:1100px) {{
            .metric-grid {{ grid-template-columns:repeat(3,1fr); }}
            .sv-grid     {{ grid-template-columns:repeat(2,1fr); }}
        }}
        @media (max-width:640px) {{
            .metric-grid, .sv-grid {{ grid-template-columns:1fr; }}
            [data-testid="block-container"] {{ padding:12px !important; }}
        }}
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
#  Component renderers
# ─────────────────────────────────────────────────────────────
def render_gauge(alertness):
    pct = int(alertness * 100)
    da  = 314.16
    off = da - da * (pct / 100)
    lbl = "NORMAL" if pct >= 75 else ("WARNING" if pct >= 55 else "CRITICAL")
    clr = COLORS["success"] if pct >= 75 else (COLORS["warning"] if pct >= 55 else COLORS["accent"])
    return f"""
    <div class="gauge-wrap"><div class="gauge-inner">
      <svg class="gauge-svg" width="160" height="160" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="50" class="gauge-bg"/>
        <circle cx="60" cy="60" r="50" class="gauge-arc"
          style="stroke-dasharray:{da};stroke-dashoffset:{off};stroke:{clr};"/>
      </svg>
      <div class="gauge-overlay">
        <div class="gauge-pct" style="color:{clr};">{pct}%</div>
        <div class="gauge-lbl">{lbl}</div>
      </div>
    </div></div>"""


def render_heatmap(history, n=60):
    pts = history[-n:]
    vals = []
    for p in pts:
        try:    vals.append(max(0.0, min(1.0, 1.0 - float(p.get("fusion_score") or 0))))
        except: vals.append(1.0)
    vals = [1.0] * (n - len(vals)) + vals

    html_out = '<div class="hm-title">ALERTNESS HEATMAP — LAST 60 SECONDS</div><div class="hm-grid">'
    for a in vals:
        d = 1.0 - a
        r = int(54  + d * (225 - 54))
        g = int(51  + d * (100 - 51))
        b = int(51  + d * (40  - 51))
        glow = (f'box-shadow:0 0 6px rgba(225,100,40,{min(0.8,d)});border-color:rgba(225,100,40,0.8);'
                if d >= 0.45 else "")
        html_out += f'<div class="hm-cell" style="background:rgb({r},{g},{b});{glow}" title="Alertness:{a*100:.0f}%"></div>'
    html_out += "</div>"
    return html_out


def render_sv(label, val, unit="", lit=False):
    cls = "sv-card lit" if lit else "sv-card"
    return f'<div class="{cls}"><div class="sv-label">{st_(label)}</div><div class="sv-val">{st_(val)}<span class="sv-unit">{st_(unit,"")}</span></div></div>'


def render_alert_timeline(history):
    """Build alert timeline from history — shows every second with a fusion score + severity."""
    rows = [p for p in history if p.get("fusion_score") is not None][-40:]
    if not rows:
        return f'<div style="color:{COLORS["muted"]};text-align:center;padding:40px;">No history yet.</div>'

    html_out = '<div class="timeline">'
    for p in reversed(rows):
        fs  = sf(p.get("fusion_score", 0))
        t   = st_(p.get("time", "–"))
        if fs >= 0.65:
            sev, cls = "CRITICAL", "crit"
        elif fs >= 0.45:
            sev, cls = "CAUTION",  "warn"
        else:
            sev, cls = "ALERT",    "ok"

        jerk = sf(p.get("jerk_rms", 0))
        post = sf(p.get("posture_dev_cm", 0))
        yolo = sf(p.get("yolo_confidence", 0))

        html_out += f"""
        <div class="tl-row {cls}">
          <div class="tl-time">{t}</div>
          <div>
            <div class="tl-score" style="color:{'#E16428' if cls=='crit' else ('#E1A028' if cls=='warn' else '#4CAF82')};">{fs:.3f} <span style="font-size:0.65rem;font-weight:700;letter-spacing:1px;">{sev}</span></div>
            <div class="tl-msg">YOLO {yolo:.2f} &nbsp;|&nbsp; Jerk {jerk:.2f} &nbsp;|&nbsp; Posture {post:.1f} cm</div>
          </div>
        </div>"""
    html_out += "</div>"
    return html_out


def make_trend_chart(history, max_pts=90):
    pts = history[-max_pts:]
    if not pts: return None
    rows = []
    for i, p in enumerate(pts):
        t = p.get("time", str(i))
        for metric, key in [
            ("Fusion Score",     "fusion_score"),
            ("YOLO Confidence",  "yolo_confidence"),
            ("PERCLOS",          "perclos"),
            ("Jerk RMS",         "jerk_rms"),
            ("Posture Dev (cm)", "posture_dev_cm"),
        ]:
            rows.append({"i": i, "Time": t, "Metric": metric, "Value": sf(p.get(key, 0))})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df).mark_line(strokeWidth=2).encode(
            x=alt.X("i:Q", title="Time →", axis=alt.Axis(labels=False, grid=False)),
            y=alt.Y("Value:Q", title=""),
            color=alt.Color("Metric:N",
                scale=alt.Scale(
                    domain=["Fusion Score","YOLO Confidence","PERCLOS","Jerk RMS","Posture Dev (cm)"],
                    range=["#E16428","#F6E9E9","rgba(246,233,233,0.45)","rgba(246,233,233,0.65)","rgba(225,100,40,0.5)"],
                ),
                legend=alt.Legend(labelColor="#F6E9E9", titleColor="#F6E9E9", orient="top", title=None),
            ),
            tooltip=["Time:N","Metric:N", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="rgba(246,233,233,0.05)", domainColor="rgba(246,233,233,0.1)",
                        labelColor="#F6E9E9", titleColor="#F6E9E9")
        .configure(background="#363333")
    )


# ─────────────────────────────────────────────────────────────
#  Main dashboard
# ─────────────────────────────────────────────────────────────
def display_dashboard():
    inject_styles()
    status = load_status()

    if status is None:
        st.markdown(f"""
        <div style="text-align:center;padding:120px 20px;">
          <div style="color:{COLORS['accent']};font-size:1.3rem;font-weight:800;letter-spacing:2px;">
            INITIALIZING COCKPIT HUD...
          </div>
          <div style="color:{COLORS['text']};opacity:.55;margin-top:10px;font-size:.85rem;letter-spacing:1px;">
            Waiting for main.py to write live_status.json…
          </div>
        </div>""", unsafe_allow_html=True)
        time.sleep(1); st.rerun(); return

    # ── Extract values ───────────────────────────────────────
    history     = status.get("history", [])
    fusion      = sf(status.get("fusion_score", 0))
    alertness   = max(0.0, min(1.0, 1.0 - fusion))
    yolo        = sf(status.get("yolo_confidence", 0))
    perclos     = sf(status.get("perclos", 0))
    jerk        = sf(status.get("jerk_rms", 0))
    posture     = sf(status.get("posture_dev_cm", 0))
    distance    = sf(status.get("distance_cm", 0))
    w_score     = sf(status.get("weather_score", 0))
    sess_min    = sf(status.get("session_min", 0))
    hour        = int(sf(status.get("hour", datetime.now().hour)))
    w_cond      = st_(status.get("weather", "N/A"))
    w_city      = st_(status.get("weather_city", WEATHER_CITY))
    head        = st_(status.get("head_state", "unknown")).upper()
    age         = data_age(status.get("timestamp"))
    t_str       = st_(status.get("time", datetime.now().strftime("%H:%M:%S")))
    latest_alert= st_(status.get("latest_alert", ""))

    sev_label = "NOMINAL"
    sev_class = ""
    if fusion >= 0.65:   sev_label, sev_class = "CRITICAL", "crit"
    elif fusion >= 0.45: sev_label, sev_class = "CAUTION",  "warn"

    # ── HUD header (always visible) ──────────────────────────
    st.markdown(f"""
    <div class="hud-header">
      <div>
        <div class="hud-title">DRIVER COCKPIT HUD</div>
        <div class="hud-sub">VEHICLE: {st_(DRIVER_ID)} &nbsp;|&nbsp; TELEMETRY ACTIVE &nbsp;|&nbsp; DATA AGE: {age}</div>
      </div>
      <div style="display:flex;gap:14px;align-items:center;">
        <div style="font-family:monospace;font-size:1.2rem;font-weight:700;letter-spacing:1px;">{t_str}</div>
        <div class="status-pill {sev_class}">{sev_label}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Warning banner
    if fusion >= 0.45 and latest_alert and latest_alert not in ("No alert recorded", "N/A"):
        st.markdown(f"""
        <div class="warn-banner">
          <div class="warn-title">DRIVER ALERT</div>
          <div class="warn-desc">{latest_alert}</div>
        </div>""", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab_overview, tab_sensors, tab_heatmap, tab_timeline, tab_trends = st.tabs([
        "OVERVIEW", "SENSOR VALUES", "ALERTNESS HEATMAP", "ALERT TIMELINE", "TRENDS"
    ])

    # ══════════════════════════════════════════════════════════
    #  TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tab_overview:
        left, right = st.columns([1, 2.2])

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">ALERTNESS GAUGE</div>', unsafe_allow_html=True)
            st.markdown(render_gauge(alertness), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(render_heatmap(history), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="card-title">LIVE TELEMETRY</div>
            <div class="metric-grid">
              <div class="metric-card lit">
                <div class="m-label">YOLO CONFIDENCE</div>
                <div class="m-val">{yolo:.3f}</div>
                <div class="m-sub">Vision model</div>
              </div>
              <div class="metric-card lit">
                <div class="m-label">PERCLOS</div>
                <div class="m-val">{perclos*100:.1f}%</div>
                <div class="m-sub">Eye closure</div>
              </div>
              <div class="metric-card lit">
                <div class="m-label">JERK RMS</div>
                <div class="m-val">{jerk:.2f}</div>
                <div class="m-sub">Motion variation</div>
              </div>
              <div class="metric-card lit">
                <div class="m-label">POSTURE DEV</div>
                <div class="m-val">{posture:.1f}<span style="font-size:1rem"> cm</span></div>
                <div class="m-sub">Head deviation</div>
              </div>
              <div class="metric-card">
                <div class="m-label">WEATHER</div>
                <div class="m-val" style="font-size:1.05rem;margin:10px 0 14px;">{w_cond}</div>
                <div class="m-sub">{w_city}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">TELEMETRY TRENDS</div>', unsafe_allow_html=True)
            chart = make_trend_chart(history)
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.markdown(f'<div style="color:{COLORS["muted"]};text-align:center;padding:50px;">Awaiting data…</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 2 — SENSOR VALUES
    # ══════════════════════════════════════════════════════════
    with tab_sensors:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ESP32 SENSOR READINGS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sv-grid">
          {render_sv("Distance",      f"{distance:.1f}",  "cm",  lit=True)}
          {render_sv("Jerk RMS",      f"{jerk:.3f}",      "",    lit=True)}
          {render_sv("Posture Dev",   f"{posture:.1f}",   "cm",  lit=True)}
          {render_sv("Head State",    head,               "",    lit=True)}
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">VISION + FUSION</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sv-grid">
          {render_sv("YOLO Confidence", f"{yolo:.4f}",      "",    lit=True)}
          {render_sv("PERCLOS",         f"{perclos*100:.1f}","%" , lit=True)}
          {render_sv("Fusion Score",    f"{fusion:.4f}",    "",    lit=fusion>=0.45)}
          {render_sv("Alertness",       f"{alertness*100:.1f}", "%", lit=False)}
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">CONTEXT</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sv-grid">
          {render_sv("Weather",       w_cond,                 "",     lit=w_score>=0.5)}
          {render_sv("Weather Score", f"{w_score:.2f}",       "",     lit=w_score>=0.5)}
          {render_sv("Session",       f"{sess_min:.1f}",      "min")}
          {render_sv("Hour",          f"{hour:02d}:00",       "")}
          {render_sv("Data Age",      age,                    "")}
          {render_sv("History Pts",   str(len(history)),      "")}
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 3 — ALERTNESS HEATMAP
    # ══════════════════════════════════════════════════════════
    with tab_heatmap:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ALERTNESS STATE GAUGE</div>', unsafe_allow_html=True)
        st.markdown(render_gauge(alertness), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(render_heatmap(history, n=60), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;gap:20px;margin-top:14px;align-items:center;">
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#363333;border:1px solid rgba(246,233,233,.2);"></div>
            <span style="font-size:0.7rem;color:{COLORS['muted']};">ALERT</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:14px;height:14px;border-radius:3px;background:rgb(139,75,45);"></div>
            <span style="font-size:0.7rem;color:{COLORS['muted']};">CAUTION</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:14px;height:14px;border-radius:3px;background:#E16428;"></div>
            <span style="font-size:0.7rem;color:{COLORS['muted']};">DROWSY</span>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Full heatmap of all history
        if len(history) > 60:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(render_heatmap(history, n=len(history)), unsafe_allow_html=True)
            st.markdown('<div class="hm-title" style="margin-top:10px;">FULL SESSION HISTORY</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 4 — ALERT TIMELINE
    # ══════════════════════════════════════════════════════════
    with tab_timeline:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ALERT TIMELINE — LATEST FIRST</div>', unsafe_allow_html=True)
        st.markdown(render_alert_timeline(history), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  TAB 5 — TRENDS
    # ══════════════════════════════════════════════════════════
    with tab_trends:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ALL METRICS — LIVE TREND</div>', unsafe_allow_html=True)
        chart = make_trend_chart(history, max_pts=90)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.markdown(f'<div style="color:{COLORS["muted"]};text-align:center;padding:50px;">Awaiting data…</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Fusion score only chart
        pts = history[-90:]
        if pts:
            df_f = pd.DataFrame([
                {"i": i, "Time": p.get("time",""), "Fusion Score": sf(p.get("fusion_score",0))}
                for i, p in enumerate(pts)
            ])
            fusion_chart = (
                alt.Chart(df_f).mark_area(
                    line={"color": "#E16428", "strokeWidth": 2},
                    color=alt.Gradient(
                        gradient="linear", stops=[
                            alt.GradientStop(color="rgba(225,100,40,0.35)", offset=0),
                            alt.GradientStop(color="rgba(225,100,40,0)",    offset=1),
                        ],
                        x1=1, x2=1, y1=1, y2=0,
                    )
                ).encode(
                    x=alt.X("i:Q", title="Time →", axis=alt.Axis(labels=False, grid=False)),
                    y=alt.Y("Fusion Score:Q", scale=alt.Scale(domain=[0,1])),
                    tooltip=["Time:N", alt.Tooltip("Fusion Score:Q", format=".3f")],
                )
                .properties(height=180, title=alt.TitleParams("FUSION SCORE OVER TIME", color="#F6E9E9", fontSize=11))
                .configure_view(strokeOpacity=0)
                .configure_axis(gridColor="rgba(246,233,233,0.05)", domainColor="rgba(246,233,233,0.1)",
                                labelColor="#F6E9E9", titleColor="#F6E9E9")
                .configure(background="#363333")
            )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.altair_chart(fusion_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    display_dashboard()