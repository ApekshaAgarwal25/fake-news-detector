import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="",
    layout="centered"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf0 100%);
        min-height: 100vh;
    }

    .main .block-container {
        max-width: 750px;
        padding: 2rem 2.5rem 3rem;
        background: #ffffff;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }

    /* Header section */
    .app-header {
        text-align: center;
        padding: 1rem 0 1.5rem;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 1.8rem;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: #6c757d;
        line-height: 1.6;
        max-width: 520px;
        margin: 0 auto;
    }

    /* Tag badge */
    .model-badge {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 999px;
        margin-top: 0.6rem;
        letter-spacing: 0.03em;
    }

    /* Section label */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    /* Example buttons */
    div[data-testid="column"] .stButton > button {
        width: 100%;
        background: #f8f9ff;
        color: #374151;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        font-size: 0.85rem;
        padding: 0.55rem 1rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #eef2ff;
        border-color: #c7d2fe;
        color: #4338ca;
    }

    /* Text area */
    .stTextArea > div > div > textarea {
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
        color: #1f2937 !important;
        background: #fafafa !important;
        min-height: 140px !important;
        transition: border-color 0.2s;
        resize: vertical;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* Analyze button */
    div[data-testid="stVerticalBlock"] > div:last-child .stButton > button,
    .analyze-btn .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.7rem;
        letter-spacing: 0.02em;
        transition: opacity 0.2s;
        box-shadow: 0 4px 12px rgba(79,70,229,0.3);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.7rem !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        opacity: 0.92 !important;
    }

    /* Result cards */
    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.4rem;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .result-fake {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 1px solid #fecdd3;
    }
    .result-real {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #bbf7d0;
    }
    .result-label {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-label-fake { color: #be123c; }
    .result-label-real { color: #15803d; }
    .result-meta {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }

    /* Score bars */
    .score-row { margin-top: 1.2rem; }
    .score-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 3px;
    }

    /* Stats row */
    .stats-row {
        display: flex;
        gap: 12px;
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #f3f4f6;
    }
    .stat-card {
        flex: 1;
        background: #f9fafb;
        border: 1px solid #f3f4f6;
        border-radius: 10px;
        padding: 0.9rem 0.8rem;
        text-align: center;
    }
    .stat-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 2px;
    }
    .stat-name {
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Warning */
    .low-conf {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.83rem;
        color: #92400e;
        margin-top: 1rem;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.76rem;
        color: #d1d5db;
        text-align: center;
        margin-top: 1.5rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_name = "impactcircle/fakenews-distilbert"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    real_score = probs[0].item()
    fake_score = probs[1].item()
    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(real_score, fake_score) * 100
    return label, confidence, real_score, fake_score


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Fake News Detector</div>
    <div class="app-subtitle">
        Paste any news headline or article to check if it is real or fake.
    </div>
    <span class="model-badge">DistilBERT · Fine-tuned · 20K Articles</span>
</div>
""", unsafe_allow_html=True)

# ── Example Buttons ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("Real news example", use_container_width=True):
        st.session_state.input_text = (
            "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent, "
            "citing elevated inflation and global economic uncertainty as key concerns "
            "for the monetary policy committee."
        )
with col2:
    if st.button("Fake news example", use_container_width=True):
        st.session_state.input_text = (
            "EXPOSED: Government secretly adding mind-control chemicals to tap water — "
            "whistleblower reveals shocking leaked documents!"
        )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── Text Input ────────────────────────────────────────────────────────────────
text_input = st.text_area(
    "Input",
    value=st.session_state.get("input_text", ""),
    height=150,
    placeholder="Paste your news article or headline here...",
    label_visibility="collapsed"
)

# ── Analyze Button ────────────────────────────────────────────────────────────
analyze = st.button("Analyze Article", use_container_width=True, type="primary")

if analyze:
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running model..."):
            try:
                tokenizer, model = load_model()
                label, confidence, real_score, fake_score = predict(
                    text_input, tokenizer, model
                )

                # Result card
                if label == "FAKE":
                    st.markdown(f"""
                    <div class="result-card result-fake">
                        <div class="result-label result-label-fake">Fake News Detected</div>
                        <div class="result-meta">
                            Confidence: <strong>{confidence:.1f}%</strong>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Real score: {real_score*100:.1f}%
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Fake score: {fake_score*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-real">
                        <div class="result-label result-label-real">Likely Real News</div>
                        <div class="result-meta">
                            Confidence: <strong>{confidence:.1f}%</strong>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Real score: {real_score*100:.1f}%
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Fake score: {fake_score*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Score bars
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="score-label">Real score</div>', unsafe_allow_html=True)
                    st.progress(round(real_score, 2))
                with c2:
                    st.markdown('<div class="score-label">Fake score</div>', unsafe_allow_html=True)
                    st.progress(round(fake_score, 2))

                # Low confidence warning
                if confidence < 65:
                    st.markdown(
                        '<div class="low-conf">Low confidence prediction — the model is uncertain. '
                        'Try pasting the full article instead of just the headline for better accuracy.</div>',
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error loading model: {e}")

# ── Model Stats ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card"><div class="stat-val">94%</div><div class="stat-name">Accuracy</div></div>
    <div class="stat-card"><div class="stat-val">0.93</div><div class="stat-name">F1 Score</div></div>
    <div class="stat-card"><div class="stat-val">20K+</div><div class="stat-name">Training Samples</div></div>
    <div class="stat-card"><div class="stat-val">DistilBERT</div><div class="stat-name">Base Model</div></div>
</div>
<div class="disclaimer">
    For educational purposes only. Predictions may vary by region and language.
    Always verify news through trusted sources.
</div>
""", unsafe_allow_html=True)
