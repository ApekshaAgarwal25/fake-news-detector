import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re

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

    /* Clean white background — no patterns */
    .stApp { background-color: #f4f6f9 !important; }

    .main .block-container {
        max-width: 750px;
        padding: 2rem 2.5rem 3rem;
        background: #ffffff;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 1rem 0 1.5rem;
        border-bottom: 1px solid #f0f0f0;
        margin-bottom: 1.8rem;
    }
    .app-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; letter-spacing: -0.5px; }
    .app-subtitle { font-size: 0.92rem; color: #6c757d; line-height: 1.6; max-width: 520px; margin: 0.4rem auto 0; }
    .model-badge {
        display: inline-block; background: #eef2ff; color: #4338ca;
        font-size: 0.75rem; font-weight: 600; padding: 3px 12px;
        border-radius: 999px; margin-top: 0.6rem; letter-spacing: 0.03em;
    }

    /* Fix radio button text visibility */
    .stRadio > div { gap: 12px !important; }
    .stRadio > div > label {
        background: #f8f9ff !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 0.45rem 1.2rem !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #374151 !important;
        cursor: pointer !important;
    }
    .stRadio > div > label:hover {
        border-color: #c7d2fe !important;
        background: #eef2ff !important;
        color: #4338ca !important;
    }
    .stRadio > div > label[data-checked="true"],
    .stRadio > div > label[aria-checked="true"] {
        background: #eef2ff !important;
        border-color: #6366f1 !important;
        color: #4338ca !important;
    }
    /* Hide the actual radio circle */
    .stRadio > div > label > div:first-child { display: none !important; }

    .section-label {
        font-size: 0.78rem; font-weight: 600; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem;
    }

    /* Example buttons */
    div[data-testid="column"] .stButton > button {
        width: 100%; background: #f8f9ff; color: #374151;
        border: 1px solid #e5e7eb; border-radius: 8px;
        font-size: 0.85rem; padding: 0.55rem 1rem; font-weight: 500;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #eef2ff; border-color: #c7d2fe; color: #4338ca;
    }

    /* Primary analyze button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-size: 1rem !important;
        font-weight: 600 !important; padding: 0.7rem !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.92 !important; }

    /* Text area */
    .stTextArea > div > div > textarea {
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important; font-size: 0.95rem !important;
        padding: 1rem !important; color: #1f2937 !important;
        background: #fafafa !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important; background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* URL input */
    .stTextInput > div > div > input {
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important; font-size: 0.95rem !important;
        padding: 0.7rem 1rem !important; color: #1f2937 !important;
        background: #fafafa !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important; background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* Scraped preview */
    .scraped-label { font-size: 0.75rem; font-weight: 600; color: #6366f1; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .scraped-box {
        background: #f8f9ff; border: 1px solid #e0e7ff; border-radius: 10px;
        padding: 1rem 1.2rem; font-size: 0.87rem; color: #374151;
        line-height: 1.6; margin-top: 0.5rem;
    }

    /* Result cards */
    .result-card { border-radius: 12px; padding: 1.4rem 1.6rem; margin-top: 1.4rem; }
    .result-fake { background: #fff1f2; border: 1px solid #fecdd3; }
    .result-real { background: #f0fdf4; border: 1px solid #bbf7d0; }
    .result-label { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-label-fake { color: #be123c; }
    .result-label-real { color: #15803d; }
    .result-meta { font-size: 0.85rem; color: #6b7280; }
    .score-label {
        font-size: 0.78rem; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px;
    }
    .low-conf {
        background: #fffbeb; border: 1px solid #fde68a;
        border-radius: 8px; padding: 0.7rem 1rem;
        font-size: 0.83rem; color: #92400e; margin-top: 1rem;
    }

    /* Stats */
    .stats-row { display: flex; gap: 12px; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #f3f4f6; }
    .stat-card { flex: 1; background: #f9fafb; border: 1px solid #f3f4f6; border-radius: 10px; padding: 0.9rem 0.8rem; text-align: center; }
    .stat-val { font-size: 1.15rem; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }
    .stat-name { font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; }
    .disclaimer { font-size: 0.76rem; color: #d1d5db; text-align: center; margin-top: 1.5rem; line-height: 1.6; }
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


def scrape_article(url):
    """Scrape article text from a URL with multiple fallback strategies"""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        text = ""

        # Strategy 1: <article> tag
        article_tag = soup.find("article")
        if article_tag:
            text = article_tag.get_text(separator=" ", strip=True)

        # Strategy 2: common article body class names
        if len(text.split()) < 80:
            for cls in ["article-body", "article__body", "story-body",
                        "post-content", "entry-content", "content-body",
                        "article-content", "news-body", "story-content"]:
                div = soup.find(attrs={"class": re.compile(cls, re.I)})
                if div:
                    text = div.get_text(separator=" ", strip=True)
                    if len(text.split()) > 80:
                        break

        # Strategy 3: all <p> tags (most reliable fallback)
        if len(text.split()) < 80:
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < 50:
            return None, (
                "Could not extract enough text from this page. "
                "This may be a paywalled or JavaScript-heavy site. "
                "Try copying the article text and using 'Paste text' mode instead."
            )

        return text[:4000], None

    except requests.exceptions.Timeout:
        return None, "Request timed out. The website took too long to respond."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect. Please check the URL and try again."
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error {e.response.status_code}. This site may block automated access."
    except Exception as e:
        return None, f"Could not fetch article: {str(e)}"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">Fake News Detector</div>
    <div class="app-subtitle">
        Paste a news headline, article text, or a URL to check if it is real or fake.
    </div>
    <span class="model-badge">DistilBERT · Fine-tuned · 20K Articles</span>
</div>
""", unsafe_allow_html=True)

# ── Input mode ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Input method</div>', unsafe_allow_html=True)
mode = st.radio("Input method", ["Paste text", "Paste URL"], horizontal=True, label_visibility="collapsed")

text_to_analyze = ""

# ── TEXT MODE ─────────────────────────────────────────────────────────────────
if mode == "Paste text":
    st.markdown('<div class="section-label" style="margin-top:1rem;">Try an example</div>', unsafe_allow_html=True)
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    text_to_analyze = st.text_area(
        "Article text",
        value=st.session_state.get("input_text", ""),
        height=150,
        placeholder="Paste your news article or headline here...",
        label_visibility="collapsed"
    )

# ── URL MODE ──────────────────────────────────────────────────────────────────
else:
    st.markdown(
        "<small style='color:#9ca3af;font-size:0.82rem;'>"
        "Works with most news sites. Paywalled or JavaScript-heavy sites may not work."
        "</small>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    url_input = st.text_input(
        "Article URL",
        placeholder="https://www.thehindu.com/news/...",
        label_visibility="collapsed"
    )

    if url_input.strip():
        with st.spinner("Fetching article..."):
            scraped_text, error = scrape_article(url_input.strip())

        if error:
            st.error(error)
        else:
            text_to_analyze = scraped_text
            word_count = len(scraped_text.split())
            preview = scraped_text[:350] + "..." if len(scraped_text) > 350 else scraped_text
            st.markdown(
                f'<div class="scraped-label">Article extracted — {word_count} words</div>'
                f'<div class="scraped-box">{preview}</div>',
                unsafe_allow_html=True
            )

# ── Analyze Button ────────────────────────────────────────────────────────────
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
analyze = st.button("Analyze Article", use_container_width=True, type="primary")

if analyze:
    if not text_to_analyze.strip():
        if mode == "Paste URL":
            st.warning("Please enter a valid URL above first.")
        else:
            st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Running model..."):
            try:
                tokenizer, model = load_model()
                label, confidence, real_score, fake_score = predict(text_to_analyze, tokenizer, model)

                if label == "FAKE":
                    st.markdown(f"""
                    <div class="result-card result-fake">
                        <div class="result-label result-label-fake">Fake News Detected</div>
                        <div class="result-meta">
                            Confidence: <strong>{confidence:.1f}%</strong>
                            &nbsp;|&nbsp; Real: {real_score*100:.1f}%
                            &nbsp;|&nbsp; Fake: {fake_score*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-real">
                        <div class="result-label result-label-real">Likely Real News</div>
                        <div class="result-meta">
                            Confidence: <strong>{confidence:.1f}%</strong>
                            &nbsp;|&nbsp; Real: {real_score*100:.1f}%
                            &nbsp;|&nbsp; Fake: {fake_score*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="score-label">Real score</div>', unsafe_allow_html=True)
                    st.progress(round(real_score, 2))
                with c2:
                    st.markdown('<div class="score-label">Fake score</div>', unsafe_allow_html=True)
                    st.progress(round(fake_score, 2))

                if confidence < 65:
                    st.markdown(
                        '<div class="low-conf">Low confidence — the model is uncertain. '
                        'Try pasting the full article text for a stronger prediction.</div>',
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error loading model: {e}")

# ── Stats ─────────────────────────────────────────────────────────────────────
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
