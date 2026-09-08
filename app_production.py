import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import io
import datetime
import folium
import requests
from streamlit_folium import st_folium
from sklearn.neighbors import BallTree
from streamlit_geolocation import streamlit_geolocation

def _dedent(html: str) -> str:
    """Collapse to a single line. Streamlit's markdown parser processes
    multi-line strings line-by-line even inside raw HTML blocks, so an
    indented line becomes a code block and a CSS rule starting with
    '* {' gets reinterpreted as a bullet list. Joining everything onto
    one line sidesteps all of that (CSS/HTML are whitespace-insensitive
    here, so this is visually and functionally identical)."""
    return " ".join(line.strip() for line in html.split("\n") if line.strip() != "")


# ─────────────────────────────────────────────────────────────
# BACKEND — UNCHANGED (Ball Tree / k-NN / dataset / distance logic)
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_current_location():
    try:
        res = requests.get('http://ip-api.com/json', timeout=5)
        data = res.json()
        if data.get('status') == 'success':
            return data['lat'], data['lon']
    except Exception:
        pass
    return 9.9312, 76.2673

st.set_page_config(
    page_title="ResQ | Emergency Services Intelligence Platform",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

FA_CDN = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"

# ─────────────────────────────────────────────────────────────
# ENTERPRISE DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────
def inject_global_css():
    st.markdown(_dedent(f"""
    <link rel="stylesheet" href="{FA_CDN}">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
        --navy-950: #0b1220;
        --navy-900: #0f172a;
        --navy-800: #16213a;
        --navy-700: #1e293b;
        --blue-700: #1d4ed8;
        --blue-600: #2563eb;
        --blue-500: #3b82f6;
        --gray-50:  #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-500: #55606e;
        --gray-600: #3d4653;
        --emerald-600: #059669;
        --emerald-50: #ecfdf5;
        --amber-600: #d97706;
        --amber-50: #fffbeb;
        --red-600: #dc2626;
        --red-50: #fef2f2;
        --card-shadow: 0 1px 2px rgba(15,23,42,.04), 0 1px 3px rgba(15,23,42,.06);
        --card-shadow-hover: 0 10px 28px rgba(37,99,235,.12), 0 2px 6px rgba(15,23,42,.06);
        --accent-grad: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    }}

    * {{ font-family: 'Inter', -apple-system, sans-serif; }}

    [data-testid="stAppViewContainer"] {{ background: var(--gray-50) !important; }}
    section.main > div {{ padding-top: 1rem; }}
    .block-container {{ padding-top: 1.2rem; max-width: 1360px; }}

    @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform: translateY(0); }} }}
    .fade-in {{ animation: fadeIn .45s ease-out both; }}

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {{
        background: var(--navy-950) !important;
        border-right: 1px solid rgba(255,255,255,.06);
    }}
    [data-testid="stSidebar"] * {{ color: var(--gray-300) !important; }}
    [data-testid="stSidebar"] label {{ font-size: .82rem; }}

    .brand {{
        display:flex; align-items:center; gap:12px;
        padding: 4px 0 18px 0; border-bottom: 1px solid rgba(255,255,255,.08);
        margin-bottom: 16px;
    }}
    .brand-mark {{
        width: 38px; height: 38px; border-radius: 12px;
        background: var(--accent-grad);
        display:flex; align-items:center; justify-content:center;
        color:#fff !important; font-size: 1.05rem; flex-shrink:0;
        box-shadow: 0 4px 12px rgba(124,58,237,.35);
    }}
    .brand-name {{ font-size: 1.05rem; font-weight: 700; color:#fff !important; line-height:1.1; }}
    .brand-sub {{ font-size: .65rem; color: var(--gray-500) !important; letter-spacing:.06em; text-transform:uppercase; }}

    .sidebar-section-label {{
        font-size: .66rem; font-weight: 600; letter-spacing:.1em; text-transform:uppercase;
        color: var(--gray-500) !important; margin: 14px 0 6px 2px;
    }}

    [data-testid="stSidebar"] .stRadio > div:first-child {{ display: none !important; }}
    [data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {{ display: none !important; }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{ gap: 2px !important; display:flex; flex-direction:column; }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
        display:flex !important; align-items:center;
        padding: 9px 14px !important; border-radius: 999px;
        font-size: .86rem !important; font-weight: 500;
        transition: background .2s, color .2s, transform .15s; cursor:pointer; margin:0 !important;
        border-left: 3px solid transparent;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,.06); transform: translateX(2px); }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {{
        background: rgba(59,130,246,.15) !important; color: #fff !important; font-weight: 600;
        border-left-color: var(--blue-500);
    }}
    [data-testid="stSidebar"] .stRadio input[type="radio"] {{ display: none !important; }}

    .status-panel {{
        background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07);
        border-radius: 14px; padding: 12px 14px; margin-top: 10px;
    }}
    .status-row {{ display:flex; align-items:center; gap:8px; font-size:.72rem; padding:3px 0; }}
    .status-dot {{ width:7px; height:7px; border-radius:50%; flex-shrink:0; }}
    .status-dot.ok {{ background: var(--emerald-600); box-shadow: 0 0 0 3px rgba(5,150,105,.18); }}
    .status-dot.info {{ background: var(--blue-500); box-shadow: 0 0 0 3px rgba(59,130,246,.18); }}
    .status-dot.warn {{ background: var(--amber-600); box-shadow: 0 0 0 3px rgba(217,119,6,.18); }}

    .sidebar-footer {{
        font-size: .62rem; color: var(--gray-500) !important; margin-top: 18px;
        border-top: 1px solid rgba(255,255,255,.07); padding-top: 10px; line-height:1.6;
    }}

    /* ---------- HEADER ---------- */
    .app-header {{
        display:flex; align-items:center; justify-content:space-between;
        padding: 4px 0 16px 0; flex-wrap: wrap; gap: 10px;
    }}
    .app-header-left {{ display:flex; align-items:center; gap:14px; }}
    .app-header-icon {{
        width: 46px; height: 46px; border-radius: 14px;
        background: var(--accent-grad);
        display:flex; align-items:center; justify-content:center;
        color:#fff; font-size:1.2rem; flex-shrink:0;
        box-shadow: 0 4px 14px rgba(124,58,237,.3);
        animation: iconPop .5s cubic-bezier(.34,1.56,.64,1) both;
    }}
    @keyframes iconPop {{ from {{ transform: scale(.6) rotate(-8deg); opacity:0; }} to {{ transform: scale(1) rotate(0); opacity:1; }} }}
    .app-header h1 {{
        font-size: 1.35rem; font-weight: 700; color: var(--navy-900);
        margin: 0; line-height:1.25;
    }}
    .app-header p {{
        font-size: .8rem; color: var(--gray-600) !important; margin: 2px 0 0 0;
    }}
    .header-badge {{
        display:flex; align-items:center; gap:6px;
        font-size: .72rem; font-weight: 600; color: var(--emerald-600);
        background: var(--emerald-50); border: 1px solid rgba(5,150,105,.2);
        padding: 6px 12px; border-radius: 999px;
    }}
    .header-badge .dot {{ width:6px; height:6px; border-radius:50%; background: var(--emerald-600); animation: blink 2s infinite; }}
    @keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.25; }} }}

    /* ---------- CARDS ---------- */
    .kpi-card, .info-card, .meta-card, .result-card, .step-card {{
        background: #fff; border: 1px solid var(--gray-200); border-radius: 18px;
        transition: box-shadow .25s cubic-bezier(.34,1.56,.64,1), transform .25s cubic-bezier(.34,1.56,.64,1), border-color .2s;
    }}
    .kpi-card:hover, .meta-card:hover, .result-card:hover, .step-card:hover {{
        box-shadow: var(--card-shadow-hover); transform: translateY(-3px) scale(1.012); border-color: var(--blue-500);
    }}

    [data-testid="stMetric"] {{
        background:#fff; border:1px solid var(--gray-200); border-radius:16px;
        padding: 14px 18px; box-shadow: var(--card-shadow);
        transition: box-shadow .25s cubic-bezier(.34,1.56,.64,1), transform .25s cubic-bezier(.34,1.56,.64,1);
    }}
    [data-testid="stMetric"]:hover {{ box-shadow: var(--card-shadow-hover); transform: translateY(-3px) scale(1.02); }}
    [data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 700; color: var(--navy-900); }}
    [data-testid="stMetricLabel"] {{ font-size: .72rem; color: var(--gray-500); text-transform:uppercase; letter-spacing:.05em; }}

    .kpi-card {{ padding: 16px 18px; }}
    .kpi-icon {{
        width: 34px; height: 34px; border-radius: 8px; display:flex; align-items:center; justify-content:center;
        font-size: .95rem; color:#fff; margin-bottom: 10px;
    }}
    .kpi-label {{ font-size: .68rem; color: var(--gray-500); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
    .kpi-value {{ font-size: 1.55rem; font-weight: 700; color: var(--navy-900); margin-top:2px; }}

    .meta-card {{ padding: 16px 20px; margin-bottom: 12px; }}
    .meta-card .meta-title {{ font-size:.68rem; color:var(--gray-500); text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
    .meta-card .meta-val {{ font-size:1.45rem; font-weight:700; color: var(--navy-900); margin-top:3px; }}
    .meta-card .meta-sub {{ font-size:.72rem; color:var(--gray-500); margin-top:3px; }}

    .result-card {{ padding: 14px 18px; margin-bottom: 10px; position:relative; overflow:hidden; }}
    .result-card::before {{ content:''; position:absolute; left:0; top:0; bottom:0; width:4px; border-radius:4px; background: var(--accent-grad); }}
    .result-card .rank {{ font-size:.68rem; color:var(--gray-500); text-transform:uppercase; letter-spacing:.08em; font-weight:600; }}
    .result-card .name {{ font-size:1rem; font-weight:700; color:var(--navy-900); margin:4px 0; }}
    .result-card .dist {{ font-size:.82rem; color: var(--blue-600); font-weight:600; }}
    .result-card .coords {{ font-size:.72rem; color:var(--gray-500); margin-top:2px; }}

    .step-card {{ padding: 16px 18px; }}
    .step-num {{
        width:28px; height:28px; border-radius:10px; background: var(--accent-grad); color:#fff;
        display:flex; align-items:center; justify-content:center; font-size:.78rem; font-weight:700; margin-bottom: 10px;
    }}

    .page-title {{
        display:flex; align-items:center; gap:10px;
        font-size: 1.25rem; font-weight: 700; color: var(--navy-900); margin: 6px 0 16px 0;
    }}
    .page-title-pill {{
        font-size:.62rem; font-weight:700; padding: 3px 10px; border-radius:999px;
        background: rgba(37,99,235,.08); color: var(--blue-700); border:1px solid rgba(37,99,235,.18);
        letter-spacing:.06em; text-transform:uppercase;
    }}
    h2, h3, h4 {{ color: var(--navy-900) !important; }}
    hr {{ border-color: var(--gray-200) !important; }}

    [data-testid="stDataFrame"] {{ border: 1px solid var(--gray-200); border-radius: 14px; overflow:hidden; }}

    [data-testid="stButton"] > button {{
        background: var(--accent-grad);
        color:#fff; border:none; border-radius: 999px; font-weight:600;
        box-shadow: 0 2px 8px rgba(37,99,235,.25);
        transition: box-shadow .2s, transform .12s cubic-bezier(.34,1.56,.64,1);
    }}
    [data-testid="stButton"] > button:hover {{ box-shadow: 0 6px 18px rgba(124,58,237,.35); transform: translateY(-2px) scale(1.02); }}
    [data-testid="stButton"] > button:active {{ transform: translateY(0) scale(.97); }}
    [data-testid="stDownloadButton"] > button {{
        background:#fff; color: var(--navy-900); border:1px solid var(--gray-300);
        border-radius:999px; font-weight:600; transition: all .2s cubic-bezier(.34,1.56,.64,1);
    }}
    [data-testid="stDownloadButton"] > button:hover {{ border-color: var(--blue-500); color: var(--blue-700); transform: translateY(-2px) scale(1.02); }}

    [data-testid="stAlert"] {{ border-radius: 14px; }}

    /* ---------- INPUT WIDGETS (force light, readable surfaces) ---------- */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-baseweb="input"] input,
    [data-baseweb="base-input"] {{
        background-color: #fff !important;
        color: var(--navy-900) !important;
        border: 1px solid var(--gray-300) !important;
        border-radius: 10px !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] * {{ color: var(--navy-900) !important; }}
    [data-testid="stSelectbox"] [data-baseweb="select"] [title] {{ color: var(--navy-900) !important; }}
    [data-testid="stSelectbox"] [data-baseweb="select"] svg {{ fill: var(--gray-600) !important; }}
    [data-testid="stNumberInput"] button {{
        background-color: var(--gray-100) !important; border: 1px solid var(--gray-300) !important;
    }}
    [data-testid="stNumberInput"] button svg {{ fill: var(--navy-900) !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] {{ background-color: #fff !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] li {{ color: var(--navy-900) !important; }}
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{ background-color: var(--gray-100) !important; }}

    [data-testid="stCheckbox"] label p, [data-testid="stCheckbox"] label span {{ color: var(--navy-900) !important; }}
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{ color: var(--gray-600) !important; font-weight: 600; }}

    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--gray-600) !important; }}

    [data-testid="stExpander"] {{
        background: #fff !important; border: 1px solid var(--gray-200) !important;
        border-radius: 14px !important; overflow: hidden;
    }}
    [data-testid="stExpander"] summary {{ color: var(--navy-900) !important; }}
    [data-testid="stExpander"] summary p {{ color: var(--navy-900) !important; font-weight: 600; }}
    [data-testid="stExpander"] summary svg {{ fill: var(--navy-900) !important; }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ background: #fff !important; }}

    /* Generic safety net: any stray widget label/help text left unstyled */
    [data-testid="stMarkdownContainer"] p {{ color: inherit; }}
    section.main label, section.main p, section.main span {{ color: var(--navy-900); }}
    section.main [data-testid="stCaptionContainer"] * {{ color: var(--gray-600) !important; }}

    .app-footer {{
        margin-top: 40px; padding: 18px 4px; border-top: 1px solid var(--gray-200);
        display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
        font-size: .72rem; color: var(--gray-500);
    }}
    .app-footer .fa-solid, .app-footer .fa-brands {{ margin-right:5px; }}

    .icon-badge {{ display:inline-flex; align-items:center; justify-content:center; width:20px; }}
    </style>
    """), unsafe_allow_html=True)


def icon(cls, color=None, size=None):
    style = ""
    if color: style += f"color:{color};"
    if size: style += f"font-size:{size};"
    return f'<i class="{cls}" style="{style}"></i>'


# ─────────────────────────────────────────────────────────────
# WELCOME / LAUNCH SCREEN — clean enterprise splash (subtle fade-in only)
# ─────────────────────────────────────────────────────────────
if "cover_shown" not in st.session_state:
    st.session_state.cover_shown = False

if not st.session_state.cover_shown:
    st.markdown(_dedent(f"""
    <style>
    [data-testid="stAppViewContainer"] {{ background: var(--gray-50, #f8fafc) !important; }}
    [data-testid="stSidebar"] {{ display:none !important; }}
    [data-testid="stHeader"] {{ display:none !important; }}
    footer {{ display:none !important; }}
    </style>
    """), unsafe_allow_html=True)
    inject_global_css()

    st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown(_dedent("""
        <style>
        @keyframes logoPop { 0% { transform: scale(.4) rotate(-15deg); opacity:0; }
            60% { transform: scale(1.12) rotate(4deg); opacity:1; }
            100% { transform: scale(1) rotate(0); opacity:1; } }
        @keyframes ringPulse { 0% { box-shadow: 0 0 0 0 rgba(124,58,237,.35); }
            70% { box-shadow: 0 0 0 16px rgba(124,58,237,0); }
            100% { box-shadow: 0 0 0 0 rgba(124,58,237,0); } }
        @keyframes slideUp { from { opacity:0; transform: translateY(14px); } to { opacity:1; transform: translateY(0); } }
        .launch-logo { animation: logoPop .6s cubic-bezier(.34,1.56,.64,1) both, ringPulse 2.2s ease-out .6s infinite; }
        .launch-text-1 { animation: slideUp .5s ease-out .15s both; }
        .launch-text-2 { animation: slideUp .5s ease-out .28s both; }
        .launch-text-3 { animation: slideUp .5s ease-out .4s both; }
        </style>
        <div style="text-align:center;">
            <div class="launch-logo" style="width:76px;height:76px;border-radius:22px;margin:0 auto 22px auto;
                        background:linear-gradient(135deg,#2563eb 0%,#7c3aed 100%);
                        display:flex;align-items:center;justify-content:center;
                        box-shadow:0 10px 30px rgba(124,58,237,.3);">
                <i class="fa-solid fa-tower-broadcast" style="color:#fff;font-size:2rem;"></i>
            </div>
            <div class="launch-text-1" style="font-size:2.2rem;font-weight:800;color:#0f172a;letter-spacing:-.02em;">ResQ</div>
            <div class="launch-text-2" style="font-size:.95rem;color:#64748b;margin-top:6px;">
                Smart Emergency Services Locator
            </div>
            <div class="launch-text-3" style="font-size:.82rem;color:#94a3b8;margin-top:4px;max-width:420px;margin-left:auto;margin-right:auto;">
                Ball Tree–indexed k-NN spatial search for Hospitals, Police and Fire Stations
            </div>
        </div>
        """), unsafe_allow_html=True)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        for idx, (c, i_cls, lbl) in enumerate([
            (pc1, "fa-solid fa-hospital", "Hospitals"),
            (pc2, "fa-solid fa-shield-halved", "Police"),
            (pc3, "fa-solid fa-fire-flame-curved", "Fire"),
        ]):
            with c:
                delay = 0.5 + idx * 0.1
                st.markdown(_dedent(f"""
                <style>@keyframes popIn{idx} {{ from {{ opacity:0; transform: translateY(10px) scale(.9); }} to {{ opacity:1; transform: translateY(0) scale(1); }} }}
                .pop-{idx} {{ animation: popIn{idx} .4s cubic-bezier(.34,1.56,.64,1) {delay}s both; }}</style>
                <div class="pop-{idx}" style="text-align:center;background:#fff;border:1px solid #e2e8f0;
                            border-radius:16px;padding:12px 8px;box-shadow:0 1px 2px rgba(15,23,42,.05);
                            transition: transform .2s;">
                    <i class="{i_cls}" style="color:#7c3aed;font-size:1.1rem;"></i>
                    <div style="font-size:.72rem;color:#475569;margin-top:6px;font-weight:600;">{lbl}</div>
                </div>
                """), unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        if st.button("Enter Dashboard", use_container_width=True, type="primary", key="enter_btn"):
            st.session_state.cover_shown = True
            st.rerun()

        st.markdown(_dedent("""
        <div style="text-align:center;margin-top:16px;font-size:.68rem;color:#94a3b8;letter-spacing:.05em;text-transform:uppercase;">
            sklearn BallTree &middot; Haversine Metric &middot; Streamlit
        </div>
        """), unsafe_allow_html=True)
    st.stop()

inject_global_css()

# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS FOR NEW FEATURES
# ─────────────────────────────────────────────────────────────
if "search_count" not in st.session_state:
    st.session_state.search_count = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = None

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(_dedent("""
    <div class="brand">
        <div class="brand-mark"><i class="fa-solid fa-tower-broadcast"></i></div>
        <div>
            <div class="brand-name">ResQ</div>
            <div class="brand-sub">Emergency Locator</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["Overview", "Search", "Map"],
        label_visibility="hidden"
    )

    st.markdown('<div class="sidebar-section-label">System Status</div>', unsafe_allow_html=True)
    st.markdown(_dedent(f"""
    <div class="status-panel">
        <div class="status-row"><span class="status-dot ok"></span>System Online</div>
        <div class="status-row"><span class="status-dot info"></span>Ball Tree k-NN Ready</div>
        <div class="status-row"><span class="status-dot warn"></span>Haversine Metric Active</div>
        <div class="status-row"><span class="status-dot info"></span>Searches this session: {st.session_state.search_count}</div>
    </div>
    """), unsafe_allow_html=True)

    if st.session_state.last_query:
        lq = st.session_state.last_query
        st.markdown('<div class="sidebar-section-label">Last Query</div>', unsafe_allow_html=True)
        st.markdown(_dedent(f"""
        <div class="status-panel" style="font-size:.72rem; line-height:1.7;">
            <div>Type: <b style="color:#fff;">{lq['type']}</b></div>
            <div>k: <b style="color:#fff;">{lq['k']}</b></div>
            <div>Coords: <b style="color:#fff;">{lq['lat']:.4f}, {lq['lon']:.4f}</b></div>
            <div>Time: <b style="color:#fff;">{lq['ts']}</b></div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown(_dedent("""
    <div class="sidebar-footer">
        ResQ Analytics Platform &middot; v2.0<br>
        sklearn BallTree &middot; Folium &middot; Streamlit
    </div>
    """), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA — UNCHANGED
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…", ttl=600)
def load_data():
    from sqlalchemy import create_engine
    engine = create_engine(st.secrets["DATABASE_URL"])
    return pd.read_sql("SELECT * FROM services", engine)

data = load_data()

hospital_count = len(data[data["Type"] == "Hospital"])
police_count   = len(data[data["Type"] == "Police"])
fire_count     = len(data[data["Type"] == "Fire"])

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(_dedent(f"""
<div class="app-header fade-in">
    <div class="app-header-left">
        <div class="app-header-icon"><i class="fa-solid fa-tower-broadcast"></i></div>
        <div>
            <h1>ResQ &mdash; Smart Emergency Services Locator</h1>
            <p>Ball Tree spatial search &middot; real-time k-NN &middot; Hospitals, Police &amp; Fire Stations</p>
        </div>
    </div>
    <div class="header-badge"><span class="dot"></span>System Operational</div>
</div>
"""), unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns(4)
with col_a: st.metric("Total Records", f"{len(data):,}")
with col_b: st.metric("Hospitals", f"{hospital_count:,}")
with col_c: st.metric("Police Stations", f"{police_count:,}")
with col_d: st.metric("Fire Stations", f"{fire_count:,}")

st.divider()

# ═══════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f'<div class="page-title">{icon("fa-solid fa-gauge-high", "#1d4ed8")} Overview <span class="page-title-pill">Dashboard</span></div>', unsafe_allow_html=True)
    st.info("**Welcome to ResQ** — a Ball Tree–powered spatial intelligence platform (scikit-learn) for locating the nearest Hospital, Police Station, or Fire Station from any coordinate in milliseconds, with brute-force validation built in. Use the sidebar to navigate.")

    c1, c2, c3 = st.columns(3)
    for c, i_cls, color, tag, title, desc in [
        (c1, "fa-solid fa-hospital", "#dc2626", "Medical", "Hospitals", "Emergency medical care, ICU, trauma response"),
        (c2, "fa-solid fa-shield-halved", "#1d4ed8", "Law Enforcement", "Police Stations", "Crime response and public safety coverage"),
        (c3, "fa-solid fa-fire-flame-curved", "#d97706", "Rescue", "Fire Stations", "Fire response and rescue operations"),
    ]:
        with c:
            st.markdown(_dedent(f"""
            <div class="result-card fade-in">
                <div class="rank">{tag}</div>
                <div class="name">{icon(i_cls, color)} &nbsp;{title}</div>
                <div class="dist" style="color:#475569;font-weight:500;">{desc}</div>
            </div>"""), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### How It Works")
    s1, s2, s3, s4 = st.columns(4)
    for col, num, i_cls, title, desc in [
        (s1, "01", "fa-solid fa-database", "Select Dataset", "Choose real or synthetic data up to 1M records"),
        (s2, "02", "fa-solid fa-location-crosshairs", "Enter Location", "Provide latitude &amp; longitude coordinates"),
        (s3, "03", "fa-solid fa-sitemap", "Ball Tree Query", "k-NN search finds nearest services in microseconds"),
        (s4, "04", "fa-solid fa-map-location-dot", "Explore Results", "View map, metadata, and brute-force validation"),
    ]:
        with col:
            st.markdown(_dedent(f"""<div class="step-card fade-in">
                <div class="step-num">{num}</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f172a;">{icon(i_cls, "#1d4ed8", "0.85rem")} {title}</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:4px;">{desc}</div></div>"""), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Dataset Distribution")
    dist_df = pd.DataFrame({"Count": [hospital_count, police_count, fire_count]},
                           index=["Hospitals", "Police", "Fire"])
    st.bar_chart(dist_df)

# ═══════════════════════════════════════════════════════════
# PAGE: SEARCH
# ═══════════════════════════════════════════════════════════
elif page == "Search":
    st.markdown(f'<div class="page-title">{icon("fa-solid fa-magnifying-glass", "#1d4ed8")} Search <span class="page-title-pill">k-NN Query</span></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        service_type = st.selectbox("Service Type", ["Hospital", "Police", "Fire"])
    with right:
        k = st.number_input("Results (k)", min_value=1, max_value=len(data), value=5, step=1)

    use_gps = st.checkbox("Use Current GPS Location")
    if use_gps:
        loc = streamlit_geolocation()
        if loc and loc.get("latitude") is not None:
            st.session_state["gps_lat"] = loc["latitude"]
            st.session_state["gps_lon"] = loc["longitude"]
        if "gps_lat" in st.session_state:
            user_lat = st.session_state["gps_lat"]
            user_lon = st.session_state["gps_lon"]
            st.success(f"GPS location acquired: **{user_lat:.6f}**, **{user_lon:.6f}**")
            with st.expander("Override coordinates manually"):
                col_lat, col_lon = st.columns(2)
                with col_lat: user_lat = st.number_input("Latitude", value=user_lat, format="%.6f", key="ovr_lat")
                with col_lon: user_lon = st.number_input("Longitude", value=user_lon, format="%.6f", key="ovr_lon")
        else:
            st.info("Click the locator control above and allow location access in your browser.")
            default_lat, default_lon = get_current_location()
            user_lat, user_lon = default_lat, default_lon
    else:
        st.session_state.pop("gps_lat", None)
        st.session_state.pop("gps_lon", None)
        default_lat, default_lon = get_current_location()
        col_lat, col_lon = st.columns(2)
        with col_lat: user_lat = st.number_input("Latitude", value=default_lat, format="%.6f")
        with col_lon: user_lon = st.number_input("Longitude", value=default_lon, format="%.6f")

    st.caption(f"Returning top **{int(k)}** nearest `{service_type}` services for `({user_lat:.4f}, {user_lon:.4f})`")

    if "search_clicked" not in st.session_state:
        st.session_state.search_clicked = False

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        if st.button("Find Nearby Services", use_container_width=True):
            st.session_state.search_clicked = True
    with btn_col2:
        if st.button("Reset", use_container_width=True):
            st.session_state.search_clicked = False
            for key in ["results", "filtered_data", "tree", "coords", "coords_rad",
                        "build_time", "search_time", "selected_type", "k_val",
                        "user_lat", "user_lon", "leaf_size"]:
                st.session_state.pop(key, None)
            st.rerun()

    if st.session_state.search_clicked:
        selected_type = service_type  # "Hospital" | "Police" | "Fire"

        filtered_data = data[data["Type"] == selected_type].copy()
        if len(filtered_data) == 0:
            st.error("No records found for this service type.")
            st.stop()

        if use_gps and "gps_lat" in st.session_state:
            user_lat = st.session_state["gps_lat"]
            user_lon = st.session_state["gps_lon"]

        k_val      = min(int(k), len(filtered_data))
        coords     = filtered_data[["Latitude", "Longitude"]].values
        coords_rad = np.radians(coords)
        LEAF_SIZE  = 40
        EARTH_KM   = 6371.0

        with st.spinner("Building Ball Tree index…"):
            build_start = time.time()
            tree        = BallTree(coords_rad, leaf_size=LEAF_SIZE, metric="haversine")
            build_time  = time.time() - build_start

        with st.spinner("Querying nearest services…"):
            search_start      = time.time()
            user_rad          = np.radians([[user_lat, user_lon]])
            dist_rad, indices = tree.query(user_rad, k=k_val)
            distances_km      = dist_rad[0] * EARTH_KM
            search_time       = time.time() - search_start

        results = filtered_data.iloc[indices[0]].copy()
        results["Distance_km"] = distances_km

        st.session_state.update({
            "results": results, "filtered_data": filtered_data, "tree": tree,
            "coords": coords, "coords_rad": coords_rad, "build_time": build_time,
            "search_time": search_time, "selected_type": selected_type,
            "k_val": k_val, "user_lat": user_lat, "user_lon": user_lon, "leaf_size": LEAF_SIZE,
        })

        st.session_state.search_count += 1
        st.session_state.last_query = {
            "type": selected_type, "k": k_val, "lat": user_lat, "lon": user_lon,
            "ts": datetime.datetime.now().strftime("%H:%M:%S"),
        }

        st.success(f"Found **{len(results)}** nearest `{selected_type}` services  •  Build: `{build_time*1000:.2f} ms`  •  Query: `{search_time*1000:.3f} ms`")
        st.divider()

        top_col, dl_col = st.columns([4, 1])
        with top_col:
            st.markdown("#### Nearest Services")
        with dl_col:
            csv_buf = io.StringIO()
            results[["Name", "Type", "Latitude", "Longitude", "Distance_km"]].to_csv(csv_buf, index=False)
            st.download_button(
                "Export CSV", data=csv_buf.getvalue(),
                file_name=f"resq_{selected_type.lower()}_results.csv",
                mime="text/csv", use_container_width=True,
            )

        for rank, (_, row) in enumerate(results.iterrows(), 1):
            st.markdown(_dedent(f"""<div class="result-card fade-in">
                <div class="rank">#{rank} Nearest {selected_type}</div>
                <div class="name">{row['Name']}</div>
                <div class="dist">{icon('fa-solid fa-ruler', '#2563eb')} &nbsp;Distance: {row['Distance_km']:.3f} km</div>
                <div class="coords">{icon('fa-solid fa-globe', '#94a3b8')} &nbsp;{row['Latitude']:.6f}, {row['Longitude']:.6f}</div>
            </div>"""), unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Full Results Table")
        st.dataframe(results[["Name","Type","Latitude","Longitude","Distance_km"]].reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE: MAP
# ═══════════════════════════════════════════════════════════
elif page == "Map":
    st.markdown(f'<div class="page-title">{icon("fa-solid fa-map-location-dot", "#1d4ed8")} Map View <span class="page-title-pill">Spatial Visualization</span></div>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.info("No results yet. Go to **Search** and run a query first.")
    else:
        results       = st.session_state["results"]
        selected_type = st.session_state["selected_type"]
        user_lat      = st.session_state["user_lat"]
        user_lon      = st.session_state["user_lon"]
        k_val         = st.session_state["k_val"]

        lc1, lc2, lc3 = st.columns(3)
        with lc1: st.info("Your Location (green marker)")
        with lc2: st.info(f"{selected_type} Stations ({k_val})")
        with lc3: st.info("Dashed lines = Ball Tree paths, with distance labels")

        try:
            m = folium.Map(location=[user_lat, user_lon], zoom_start=13, tiles="cartodbpositron")

            user_icon_html = """<div style="width:26px;height:26px;border-radius:50%;
                background:linear-gradient(135deg,#059669,#10b981);
                display:flex;align-items:center;justify-content:center;
                box-shadow:0 0 0 4px rgba(16,185,129,.25),0 2px 6px rgba(0,0,0,.25);">
                <i class="fa-solid fa-house" style="color:#fff;font-size:11px;"></i></div>"""
            folium.Marker(
                [user_lat, user_lon],
                popup=folium.Popup("<b>Your Location</b>", max_width=160),
                tooltip="Your Location",
                icon=folium.DivIcon(html=user_icon_html, icon_size=(26,26), icon_anchor=(13,13))
            ).add_to(m)

            rank_colors = ["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]

            for rank, (_, row) in enumerate(results.iterrows(), 1):
                dist_km    = row['Distance_km']
                dist_label = f"{dist_km:.2f} km" if dist_km >= 1 else f"{dist_km*1000:.0f} m"
                mid_lat    = (user_lat + row["Latitude"]) / 2
                mid_lon    = (user_lon + row["Longitude"]) / 2
                lc         = rank_colors[min(rank-1, len(rank_colors)-1)]

                popup_html = f"""<div style='font-family:Inter,sans-serif;min-width:200px;'>
                  <div style='font-size:.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;'>#{rank} {selected_type}</div>
                  <div style='font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:6px;'>{row['Name']}</div>
                  <div style='font-size:.85rem;color:#2563eb;font-weight:600;'>{dist_label}</div>
                  <div style='font-size:.7rem;color:#9ca3af;margin-top:3px;'>{row['Latitude']:.5f}, {row['Longitude']:.5f}</div>
                </div>"""

                svc_icon = f"""<div style="width:28px;height:28px;border-radius:50%;
                    background:{lc};
                    display:flex;align-items:center;justify-content:center;
                    font-size:12px;font-weight:700;color:white;
                    box-shadow:0 2px 6px rgba(0,0,0,.25),0 0 0 3px {lc}33;
                    font-family:Inter,sans-serif;">{rank}</div>"""

                folium.Marker(
                    [row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(popup_html, max_width=260),
                    tooltip=f"#{rank} {row['Name']} · {dist_label}",
                    icon=folium.DivIcon(html=svc_icon, icon_size=(28,28), icon_anchor=(14,14))
                ).add_to(m)

                folium.PolyLine(
                    [[user_lat, user_lon], [row["Latitude"], row["Longitude"]]],
                    color=lc, weight=2, opacity=0.75, dash_array="8 5"
                ).add_to(m)

                dist_label_html = f"""<div style="background:rgba(255,255,255,.95);
                    border:1px solid {lc}55;border-radius:6px;padding:3px 9px;
                    font-family:Inter,sans-serif;font-size:11px;font-weight:600;
                    color:{lc};white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.12);">
                    #{rank} &middot; {dist_label}</div>"""

                folium.Marker(
                    [mid_lat, mid_lon],
                    icon=folium.DivIcon(html=dist_label_html, icon_size=(110,26), icon_anchor=(55,13))
                ).add_to(m)

            st_folium(m, height=580, use_container_width=True)

        except Exception as e:
            st.error(f"Map Error: {e}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown(_dedent(f"""
<div class="app-footer">
    <div>{icon('fa-solid fa-tower-broadcast', '#1d4ed8')} &nbsp;<b>ResQ</b> Emergency Services Intelligence Platform &middot; v2.0</div>
    <div>Built with scikit-learn BallTree &middot; Folium &middot; Streamlit &nbsp;|&nbsp; Session searches: {st.session_state.search_count}</div>
</div>
"""), unsafe_allow_html=True)
