"""
app.py
------
Streamlit frontend for Music Mood Matcher.
iOS-inspired design with light purple accents.

Run with:
    streamlit run app.py
"""

import os
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "Music Mood Matcher",
    page_icon  = "music_note",
    layout     = "centered",
)

# ─── iOS-inspired styling ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import SF Pro-like font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 680px; }

    /* Purple accent variables */
    :root {
        --purple:       #7B68EE;
        --purple-light: #EDE9FF;
        --purple-mid:   #C4B8FF;
        --gray-1:       #F2F2F7;
        --gray-2:       #E5E5EA;
        --gray-3:       #C7C7CC;
        --gray-text:    #8E8E93;
        --label:        #1C1C1E;
        --radius:       14px;
        --radius-sm:    10px;
    }

    /* Title */
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        color: var(--label) !important;
    }

    /* Subtitle */
    .subtitle {
        font-size: 0.95rem;
        color: var(--gray-text);
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Text area — iOS rounded style */
    .stTextArea textarea {
        border-radius: var(--radius) !important;
        border: 1.5px solid var(--gray-2) !important;
        background: var(--gray-1) !important;
        font-family: inherit !important;
        font-size: 1rem !important;
        padding: 14px 16px !important;
        color: var(--label) !important;
        transition: border-color 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: var(--purple) !important;
        background: white !important;
        box-shadow: 0 0 0 3px rgba(123,104,238,0.12) !important;
    }
    .stTextArea label {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: var(--label) !important;
    }

    /* Primary button — iOS filled style */
    .stButton > button[kind="primary"] {
        background: var(--purple) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-family: inherit !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        letter-spacing: -0.1px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 2px 8px rgba(123,104,238,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #6A56E0 !important;
        box-shadow: 0 4px 14px rgba(123,104,238,0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 4px rgba(123,104,238,0.3) !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] {
        padding-top: 0.5rem !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: var(--purple) !important;
        font-weight: 600 !important;
    }
    .stSlider label {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: var(--label) !important;
    }

    /* Song cards — iOS list cell style */
    .song-card {
        background: white;
        border: 1.5px solid var(--gray-2);
        border-radius: var(--radius);
        padding: 16px 18px;
        margin-bottom: 10px;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .song-card:hover {
        border-color: var(--purple-mid);
        box-shadow: 0 2px 12px rgba(123,104,238,0.1);
    }
    .song-number {
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--purple);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .song-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--label);
        margin-bottom: 2px;
    }
    .song-artist {
        font-size: 0.875rem;
        color: var(--purple);
        font-weight: 500;
        margin-bottom: 8px;
    }
    .song-reason {
        font-size: 0.875rem;
        color: var(--gray-text);
        line-height: 1.5;
    }

    /* Success banner */
    .success-banner {
        background: var(--purple-light);
        border: 1.5px solid var(--purple-mid);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        color: var(--purple);
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Warning banner */
    .warning-banner {
        background: #FFF9E6;
        border: 1.5px solid #FFD60A;
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        color: #7D6200;
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }

    /* Error banner */
    .error-banner {
        background: #FFF0F0;
        border: 1.5px solid #FF3B30;
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        color: #CC0000;
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
    }

    /* Info caption */
    .info-caption {
        font-size: 0.8rem;
        color: var(--gray-text);
        margin-bottom: 0.5rem;
    }

    /* Divider */
    hr {
        border: none !important;
        border-top: 1px solid var(--gray-2) !important;
        margin: 1.5rem 0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #EDE9FF 0%, #F2F2F7 60%) !important;
        border-right: 1px solid #D8D3F8 !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }
    .sidebar-label {
        font-size: 0.8rem;
        color: var(--gray-text);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .sidebar-value {
        font-size: 0.9rem;
        color: var(--label);
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online  { background: #34C759; }
    .status-offline { background: #FF3B30; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("<h1>Music Mood Matcher</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Describe how you feel — we\'ll find the songs that match.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ─── Input ────────────────────────────────────────────────────────────────────

mood = st.text_area(
    label       = "How are you feeling right now?",
    placeholder = "e.g. I feel melancholic and nostalgic, like looking at old photos on a rainy Sunday...",
    height      = 110,
)

col1, col2 = st.columns([3, 1])
with col2:
    k = st.slider("Songs", min_value=1, max_value=10, value=5)
with col1:
    submit = st.button("Find my songs", use_container_width=True, type="primary")

# ─── Results ──────────────────────────────────────────────────────────────────

if submit:
    if not mood.strip():
        st.markdown('<div class="warning-banner">Please describe your mood first.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Finding songs that match your mood..."):
            try:
                resp = requests.post(
                    f"{API_URL}/recommend",
                    json    = {"mood": mood, "k": k},
                    timeout = 60,
                )

                if resp.status_code == 400:
                    st.markdown(
                        f'<div class="error-banner">Input blocked: {resp.json()["detail"]}</div>',
                        unsafe_allow_html=True
                    )

                elif resp.status_code == 503:
                    st.markdown(
                        '<div class="error-banner">LM Studio is not running. Please start it and load a model first.</div>',
                        unsafe_allow_html=True
                    )

                elif resp.status_code == 200:
                    data = resp.json()
                    recs = data["recommendations"]

                    st.markdown(
                        f'<div class="success-banner">Found {len(recs)} song{"s" if len(recs) != 1 else ""} for your mood.</div>',
                        unsafe_allow_html=True
                    )

                    if data.get("fallback_notice"):
                        st.markdown(
                            f'<div class="warning-banner">{data["fallback_notice"]}</div>',
                            unsafe_allow_html=True
                        )

                    if data["removed_count"] > 0:
                        st.markdown(
                            f'<p class="info-caption">{data["removed_count"]} unverified song(s) were filtered out automatically.</p>',
                            unsafe_allow_html=True
                        )

                    st.markdown("<hr>", unsafe_allow_html=True)

                    for i, song in enumerate(recs, 1):
                        st.markdown(f"""
                        <div class="song-card">
                            <div class="song-number">Track {i}</div>
                            <div class="song-title">{song['track_name']}</div>
                            <div class="song-artist">{song['track_artist']}</div>
                            <div class="song-reason">{song['reason']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.markdown(
                        f'<div class="error-banner">Unexpected error: {resp.status_code}</div>',
                        unsafe_allow_html=True
                    )

            except requests.exceptions.ConnectionError:
                st.markdown(
                    '<div class="error-banner">Cannot connect to the backend. Make sure FastAPI is running: uvicorn main:app --reload</div>',
                    unsafe_allow_html=True
                )

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo above About
    try:
        import base64
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
        with open(logo_path, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'gap:10px;padding:1.5rem 0 1rem 0">'
            '<img src="data:image/png;base64,' + _b64 + '" '
            'style="width:96px;height:96px;border-radius:22px;"/>'
            '<span style="font-size:1rem;font-weight:600;color:#3D2FBF;'
            'letter-spacing:-0.2px">Music Mood Matcher</span>'
            '</div>',
            unsafe_allow_html=True
        )
    except Exception:
        pass

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown(
        "Music Mood Matcher uses RAG over song lyrics "
        "combined with a local language model to recommend "
        "songs based on your emotional state."
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Team</p><p class="sidebar-value">The Overfitters</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Model</p><p class="sidebar-value">LLaMA 3.2 3B</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Dataset</p><p class="sidebar-value">Spotify Lyrics + Audio Features</p>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    try:
        h = requests.get(f"{API_URL}/health", timeout=2)
        if h.status_code == 200:
            st.markdown('<p><span class="status-dot status-online"></span><strong>Backend online</strong></p>', unsafe_allow_html=True)
        else:
            st.markdown('<p><span class="status-dot status-offline"></span><strong>Backend error</strong></p>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<p><span class="status-dot status-offline"></span><strong>Backend offline</strong></p>', unsafe_allow_html=True)