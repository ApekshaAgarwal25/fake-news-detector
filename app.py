import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding: 2rem; }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1.5rem;
        font-size: 1.1rem;
    }
    .fake-box {
        background-color: #fff0f0;
        border: 2px solid #ff4b4b;
        color: #cc0000;
    }
    .real-box {
        background-color: #f0fff4;
        border: 2px solid #00cc66;
        color: #006633;
    }
    .confidence-text {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .stTextArea textarea {
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model once and cache it"""
    # If deployed on Streamlit Cloud, load from HuggingFace Hub:
    # model_name = "your-username/fakenews-distilbert"

    # If running locally with saved model:
    model_name = "fakenews-distilbert"  # path to your saved model folder

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ─── Predict Function ────────────────────────────────────────────────────────
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    real_score = probs[0].item()
    fake_score = probs[1].item()
    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(real_score, fake_score) * 100
    return label, confidence, real_score, fake_score


# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article or headline below to check if it's **real or fake** using a fine-tuned DistilBERT model.")
st.markdown("---")

# Example buttons
st.markdown("**Try an example:**")
col1, col2 = st.columns(2)
with col1:
    if st.button("📰 Real news example"):
        st.session_state.input_text = "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, citing continued concerns about inflation and a resilient labor market."
with col2:
    if st.button("🚫 Fake news example"):
        st.session_state.input_text = "BREAKING: Government secretly adding mind-control chemicals to tap water, leaked documents reveal shocking conspiracy!"

# Text input
text_input = st.text_area(
    "Enter news article or headline:",
    value=st.session_state.get("input_text", ""),
    height=180,
    placeholder="Paste your news article or headline here..."
)

# Predict button
if st.button("🔍 Analyze", use_container_width=True, type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                tokenizer, model = load_model()
                label, confidence, real_score, fake_score = predict(text_input, tokenizer, model)

                # Result box
                if label == "FAKE":
                    st.markdown(f"""
                    <div class="result-box fake-box">
                        🚫 <strong>FAKE NEWS</strong><br>
                        <span class="confidence-text">Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box real-box">
                        ✅ <strong>REAL NEWS</strong><br>
                        <span class="confidence-text">Confidence: {confidence:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                # Score bars
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Prediction scores:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Real", f"{real_score*100:.1f}%")
                    st.progress(real_score)
                with col2:
                    st.metric("🚫 Fake", f"{fake_score*100:.1f}%")
                    st.progress(fake_score)

            except Exception as e:
                st.error(f"Error loading model: {e}\n\nMake sure your model folder 'fakenews-distilbert' is in the same directory as app.py")

st.markdown("---")
st.markdown(
    "<small>Built with DistilBERT · Fine-tuned on 20K news articles · "
    "Accuracy ~94% · F1 ~0.93</small>",
    unsafe_allow_html=True
)
