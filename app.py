import streamlit as st
import tempfile
import os

from main import run_pipeline

st.set_page_config(
    page_title="DeepScan — Deepfake Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ───────────────────────────── STYLES ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&family=Orbitron:wght@400;700;900&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #050a0f !important;
    color: #c8d8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stToolbar"] { display: none; }
.block-container {
    padding-top: 2rem !important;
    max-width: 780px !important;
}

body::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 170, 0.015) 2px,
        rgba(0, 255, 170, 0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

.hero-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.35em;
    color: #00ffaa;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.75;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    line-height: 1;
    background: linear-gradient(135deg, #e8f4ff 30%, #00ffaa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #4a7a6a;
    margin-top: 1rem;
    letter-spacing: 0.15em;
}
.hero-line {
    width: 120px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ffaa, transparent);
    margin: 1.5rem auto 0;
}

[data-testid="stFileUploader"] {
    background: rgba(0, 255, 170, 0.03) !important;
    border: 1px dashed rgba(0, 255, 170, 0.25) !important;
    border-radius: 4px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 255, 170, 0.55) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #4a8a7a !important;
    letter-spacing: 0.1em;
}
[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid rgba(0, 255, 170, 0.4) !important;
    color: #00ffaa !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em;
    border-radius: 2px !important;
}

[data-testid="stVideo"] {
    border: 1px solid rgba(0, 255, 170, 0.15) !important;
    border-radius: 4px !important;
    overflow: hidden;
}

[data-testid="stButton"] button {
    width: 100%;
    background: transparent !important;
    border: 1px solid #00ffaa !important;
    color: #00ffaa !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase;
    padding: 0.9rem 2rem !important;
    border-radius: 2px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 20px rgba(0, 255, 170, 0.08) !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(0, 255, 170, 0.08) !important;
    box-shadow: 0 0 35px rgba(0, 255, 170, 0.22) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stSpinner"] p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #00ffaa !important;
    letter-spacing: 0.15em;
}

.result-card {
    margin-top: 2rem;
    border: 1px solid rgba(0, 255, 170, 0.18);
    border-radius: 4px;
    background: rgba(0, 255, 170, 0.03);
    padding: 2rem 2.2rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00ffaa, transparent);
}
.result-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.3em;
    color: #4a7a6a;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-verdict {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: 0.08em;
    margin: 0;
}
.verdict-real { color: #00ffaa; text-shadow: 0 0 30px rgba(0,255,170,0.4); }
.verdict-fake { color: #ff4455; text-shadow: 0 0 30px rgba(255,68,85,0.4); }

.result-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(0,255,170,0.25), transparent);
    margin: 1.5rem 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 0.5rem;
}
.metric-cell {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0, 255, 170, 0.1);
    border-radius: 3px;
    padding: 1rem;
    text-align: center;
}
.metric-name {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: #4a7a6a;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #c8f0e0;
}
.metric-bar-bg {
    height: 2px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00ffaa, #00ccff);
    border-radius: 2px;
}
.confidence-big {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #00ffaa;
    letter-spacing: 0.15em;
    margin-top: 0.8rem;
}

.footer {
    text-align: center;
    margin-top: 4rem;
    padding-bottom: 2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #1e3028;
    letter-spacing: 0.2em;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────── HEADER ─────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">[ forensic analysis system v2.4 ]</div>
    <h1 class="hero-title">DEEPSCAN</h1>
    <div class="hero-sub">NEURAL AUTHENTICITY VERIFICATION &nbsp;&middot;&nbsp; rPPG &nbsp;&middot;&nbsp; MICRO-EXPRESSION ANALYSIS</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────── UPLOAD ─────────────────────────────
uploaded_file = st.file_uploader(
    "DROP VIDEO FILE — MP4 · AVI · MOV",
    type=["mp4", "avi", "mov"],
    label_visibility="visible"
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if st.button("▶  INITIATE ANALYSIS"):

        with st.spinner("SCANNING · EXTRACTING NEURAL SIGNATURES · ANALYZING BIOMETRICS ..."):
            result = run_pipeline(
                video_path=video_path,
                use_transformer=False
            )

        if result:
            label = result['final_label'].upper()
            verdict_class = "verdict-real" if label == "REAL" else "verdict-fake"

            confidence_pct = int(result['confidence'] * 100)
            cnn_pct        = int(result['cnn_score'] * 100)
            hr_pct         = int(result['hr_quality_score'] * 100)
            micro_pct      = int(result['micro_expression_score'] * 100)

            st.markdown(f"""
<div class="result-card">
    <div class="result-label">AUTHENTICITY VERDICT</div>
    <div class="result-verdict {verdict_class}">{label}</div>
    <div class="confidence-big">CONFIDENCE &nbsp;&middot;&nbsp; {confidence_pct}%</div>
    <div class="result-divider"></div>
    <div class="result-label" style="margin-bottom:1rem">SIGNAL BREAKDOWN</div>
    <div class="metric-grid">
        <div class="metric-cell">
            <div class="metric-name">CNN SCORE</div>
            <div class="metric-value">{result['cnn_score']:.2f}</div>
            <div class="metric-bar-bg">
                <div class="metric-bar-fill" style="width:{cnn_pct}%"></div>
            </div>
        </div>
        <div class="metric-cell">
            <div class="metric-name">HR QUALITY</div>
            <div class="metric-value">{result['hr_quality_score']:.2f}</div>
            <div class="metric-bar-bg">
                <div class="metric-bar-fill" style="width:{hr_pct}%"></div>
            </div>
        </div>
        <div class="metric-cell">
            <div class="metric-name">MICRO-EXP</div>
            <div class="metric-value">{result['micro_expression_score']:.2f}</div>
            <div class="metric-bar-bg">
                <div class="metric-bar-fill" style="width:{micro_pct}%"></div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        os.remove(video_path)


# ───────────────────────────── FOOTER ─────────────────────────────
st.markdown("""
<div class="footer">
    DEEPSCAN &nbsp;&middot;&nbsp; MULTI-MODAL DEEPFAKE DETECTION &nbsp;&middot;&nbsp; ALL ANALYSIS PERFORMED LOCALLY
</div>
""", unsafe_allow_html=True)