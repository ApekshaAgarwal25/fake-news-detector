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
    .stApp { background-color: #f8f9fa; }
    .main .block-container {
        max-width: 720px;
        padding: 2.5rem 2rem;
        background-color: #ffffff;
        border-radius: 12px;
        margin-top: 2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .divider { border: none; border-top: 1px solid #e9ecef; margin: 1.2rem 0; }
    .result-fake {
        background-color: #fff5f5;
        border-left: 4px solid #c0392b;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.2rem;
    }
    .result-real {
        background-color: #f0fff4;
        border-left: 4px solid #1a7a4a;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.2rem;
    }
    .result-label-fake { font-size: 1.1rem; font-weight: 600; color: #c0392b; margin-bottom: 0.3rem; }
    .result-label-real { font-size: 1.1rem; font-weight: 600; color: #1a7a4a; margin-bottom: 0.3rem; }
    .result-confidence { font-size: 0.88rem; color: #6c757d; }
    .score-label {
        font-size: 0.82rem; color: #6c757d; margin-bottom: 2px;
        font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em;
    }
    .stButton > button {
        background-color: #ffffff; color: #1a1a2e;
        border: 1px solid #dee2e6; border-radius: 6px;
        font-size: 0.85rem; padding: 0.4rem 1rem; font-weight: 400;
    }
    .stButton > button:hover { background-color: #f1f3f5; border-color: #adb5bd; }
    .stTextArea textarea {
        font-size: 0.95rem; border-radius: 6px;
        border: 1px solid #dee2e6; background-color: #fdfdfd;
    }
    .metric-row { display: flex; gap: 12px; margin-top: 1rem; }
    .metric-card {
        flex: 1; background: #f8f9fa; border-radius: 8px;
        padding: 0.8rem 1rem; text-align: center;
    }
    .metric-val { font-size: 1.2rem; font-weight: 600; color: #1a1a2e; }
    .metric-name { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
    .disclaimer { font-size: 0.78rem; color: #adb5bd; margin-top: 2rem; line-height: 1.6; text-align: center; }
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    real_score = probs[0].item()
    fake_score = probs[1].item()
    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(real_score, fake_score) * 100
    return label, confidence, real_score, fake_score


# Header
st.markdown('<div class="header-title">Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-sub">Paste any news headline or article below. '
    'The model classifies it as real or fake using a fine-tuned DistilBERT '
    'trained on 20,000+ labeled news articles.</div>',
    unsafe_allow_html=True
)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Example buttons
st.markdown("<small style='color:#6c757d;font-size:0.82rem;'>Try an example</small>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("Real news example"):
        st.session_state.input_text = (
            "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent "
            "on Friday, citing elevated inflation and global economic uncertainty "
            "as key concerns for the monetary policy committee."
        )
with col2:
    if st.button("Fake news example"):
        st.session_state.input_text = (
            "EXPOSED: Government secretly fluoridating water supply to control "
            "citizens minds — whistleblower reveals shocking documents!"
        )

# Text input
text_input = st.text_area(
    "News article or headline",
    value=st.session_state.get("input_text", ""),
    height=160,
    placeholder="Paste your news article or headline here...",
    label_visibility="collapsed"
)

# Analyze button
analyze = st.button("Analyze", use_container_width=True, type="primary")

if analyze:
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                tokenizer, model = load_model()
                label, confidence, real_score, fake_score = predict(text_input, tokenizer, model)

                if label == "FAKE":
                    st.markdown(f"""
                    <div class="result-fake">
                        <div class="result-label-fake">Fake News</div>
                        <div class="result-confidence">
                            Confidence: {confidence:.1f}% &nbsp;|&nbsp;
                            Real: {real_score*100:.1f}% &nbsp;|&nbsp;
                            Fake: {fake_score*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-real">
                        <div class="result-label-real">Real News</div>
                        <div class="result-confidence">
                            Confidence: {confidence:.1f}% &nbsp;|&nbsp;
                            Real: {real_score*100:.1f}% &nbsp;|&nbsp;
                            Fake: {fake_score*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="score-label">Real</div>', unsafe_allow_html=True)
                    st.progress(real_score)
                with c2:
                    st.markdown('<div class="score-label">Fake</div>', unsafe_allow_html=True)
                    st.progress(fake_score)

                if confidence < 65:
                    st.markdown(
                        "<small style='color:#adb5bd;'>Low confidence — the model is uncertain. "
                        "Try adding more context to get a stronger prediction.</small>",
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Model could not be loaded. Error: {e}")

# Footer stats
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="metric-row">
    <div class="metric-card"><div class="metric-val">94%</div><div class="metric-name">Accuracy</div></div>
    <div class="metric-card"><div class="metric-val">0.93</div><div class="metric-name">F1 Score</div></div>
    <div class="metric-card"><div class="metric-val">20K+</div><div class="metric-name">Training Samples</div></div>
    <div class="metric-card"><div class="metric-val">DistilBERT</div><div class="metric-name">Model</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">This tool is intended for educational purposes. '
    'Predictions may vary across news sources, languages, and regions. '
    'Always verify news through trusted sources.</div>',
    unsafe_allow_html=True
)
