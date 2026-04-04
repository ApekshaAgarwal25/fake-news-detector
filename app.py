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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Force light theme via config
st.markdown("""
<style>
    /* ── Force light theme everywhere ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp {
        background-color: #f4f6f9 !important;
        color: #1f2937 !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer, header { visibility: hidden; }

    .main .block-container {
        max-width: 740px;
        padding: 2rem 2.5rem 3rem;
        background: #ffffff !important;
        border-radius: 16px;
        margin-top: 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        color: #1f2937 !important;
    }

    /* ── All text forced dark ── */
    p, span, div, label, small {
        color: #1f2937;
    }

    /* ── Header ── */
    .app-title {
        font-size: 1.9rem; font-weight: 700;
        color: #1a1a2e !important; text-align: center;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        font-size: 0.92rem; color: #6b7280 !important;
        text-align: center; line-height: 1.6;
        max-width: 500px; margin: 0 auto 0.5rem;
    }
    .badge-row { text-align: center; margin-bottom: 1.5rem; }
    .model-badge {
        display: inline-block; background: #eef2ff !important;
        color: #4338ca !important; font-size: 0.75rem;
        font-weight: 600; padding: 3px 14px;
        border-radius: 999px; letter-spacing: 0.03em;
    }
    .divider { border: none; border-top: 1px solid #e5e7eb; margin: 0 0 1.5rem; }

    /* ── Radio as pill toggle ── */
    div[data-testid="stRadio"] > div {
        display: flex; gap: 10px; flex-direction: row !important;
    }
    div[data-testid="stRadio"] label {
        background: #f3f4f6 !important;
        color: #374151 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 6px 20px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        min-width: 100px;
        text-align: center;
    }
    div[data-testid="stRadio"] label:hover {
        background: #eef2ff !important;
        border-color: #a5b4fc !important;
        color: #4338ca !important;
    }
    /* Hide radio circle dot */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 0.75rem; font-weight: 600;
        color: #9ca3af !important; text-transform: uppercase;
        letter-spacing: 0.08em; margin: 1rem 0 0.4rem;
    }

    /* ── Example buttons ── */
    div[data-testid="column"] button {
        width: 100% !important;
        background: #f8faff !important;
        color: #374151 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        padding: 8px 12px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div[data-testid="column"] button:hover {
        background: #eef2ff !important;
        border-color: #a5b4fc !important;
        color: #4338ca !important;
    }

    /* ── Analyze button ── */
    button[kind="primary"] {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
    }
    button[kind="primary"]:hover { background: #4338ca !important; }

    /* ── Text area ── */
    textarea {
        background: #fafafa !important;
        color: #1f2937 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }
    textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* ── URL input ── */
    input[type="text"] {
        background: #fafafa !important;
        color: #1f2937 !important;
        border: 1.5px solid #e5e7eb !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }
    input[type="text"]:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    /* ── Scraped preview ── */
    .scraped-box {
        background: #f8f9ff !important;
        border: 1px solid #e0e7ff;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.87rem;
        color: #374151 !important;
        line-height: 1.65;
        margin-top: 0.5rem;
    }
    .scraped-label {
        font-size: 0.74rem; font-weight: 600;
        color: #6366f1 !important; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.3rem;
    }

    /* ── Result cards ── */
    .result-card { border-radius: 12px; padding: 1.3rem 1.5rem; margin-top: 1.2rem; }
    .result-fake { background: #fff1f2 !important; border: 1.5px solid #fca5a5; }
    .result-real { background: #f0fdf4 !important; border: 1.5px solid #86efac; }
    .result-title-fake { font-size: 1.2rem; font-weight: 700; color: #b91c1c !important; margin-bottom: 4px; }
    .result-title-real { font-size: 1.2rem; font-weight: 700; color: #15803d !important; margin-bottom: 4px; }
    .result-meta { font-size: 0.84rem; color: #6b7280 !important; }

    /* ── Score bars ── */
    .score-label {
        font-size: 0.76rem; font-weight: 600;
        color: #9ca3af !important; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 3px;
    }

    /* ── Warning box ── */
    .warn-box {
        background: #fffbeb !important; border: 1px solid #fde68a;
        border-radius: 8px; padding: 0.65rem 1rem;
        font-size: 0.83rem; color: #92400e !important; margin-top: 0.8rem;
    }

    /* ── Disclaimer ── */
    .note-box {
        background: #f9fafb !important; border: 1px solid #f3f4f6;
        border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.8rem;
    }
    .note-box p { font-size: 0.82rem; color: #6b7280 !important; margin: 0; line-height: 1.6; }

    /* ── Stats footer ── */
    .stats-row { display: flex; gap: 10px; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #f3f4f6; }
    .stat-card { flex: 1; background: #f9fafb !important; border: 1px solid #f3f4f6; border-radius: 10px; padding: 0.85rem; text-align: center; }
    .stat-val { font-size: 1.1rem; font-weight: 700; color: #1a1a2e !important; }
    .stat-name { font-size: 0.68rem; color: #9ca3af !important; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px; }
    .footer-note { font-size: 0.75rem; color: #d1d5db !important; text-align: center; margin-top: 1.2rem; line-height: 1.6; }
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
        text, return_tensors="pt", truncation=True,
        max_length=256, padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    # 0=FAKE, 1=REAL
    fake_score = probs[0].item()
    real_score = probs[1].item()
    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(real_score, fake_score) * 100
    return label, confidence, real_score, fake_score


def scrape_article(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        resp = requests.get(url, headers=headers, timeout=14)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script","style","nav","footer","header","aside","form","noscript","iframe"]):
            tag.decompose()

        text = ""

        # Try <article> tag first
        article = soup.find("article")
        if article:
            text = article.get_text(" ", strip=True)

        # Try known content div classes
        if len(text.split()) < 80:
            for cls in ["article-body","article__body","story-body","post-content",
                        "entry-content","content-body","article-content","news-body",
                        "story-content","td-post-content","articlebody"]:
                el = soup.find(attrs={"class": re.compile(cls, re.I)})
                if el:
                    text = el.get_text(" ", strip=True)
                    if len(text.split()) > 80:
                        break

        # Fallback: all long paragraphs
        if len(text.split()) < 80:
            paras = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40)

        text = re.sub(r'\s+', ' ', text).strip()

        if len(text.split()) < 50:
            return None, (
                "Could not extract enough text. This site may be paywalled or "
                "require JavaScript. Please copy the article text and use "
                "'Paste text' mode instead."
            )
        return text[:4000], None

    except requests.exceptions.Timeout:
        return None, "Request timed out. Try a different news site or paste the text directly."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 403:
            return None, "Access denied (403). This site blocks automated access. Please copy the article text and paste it manually."
        return None, f"HTTP error {code}. Please try a different URL."
    except requests.exceptions.ConnectionError:
        return None, "Connection failed. Please check the URL and your internet connection."
    except Exception as e:
        return None, f"Could not fetch article: {str(e)}"


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.3rem;">
    <div class="app-title">Fake News Detector</div>
</div>
<div class="app-subtitle">
    Paste a news headline, article text, or URL to check if it is real or fake.
</div>
<div class="badge-row">
    <span class="model-badge">DistilBERT &nbsp;·&nbsp; Fine-tuned &nbsp;·&nbsp; 20K Articles</span>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── Input toggle ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Input method</div>', unsafe_allow_html=True)
mode = st.radio(
    "mode", ["Paste text", "Paste URL"],
    horizontal=True, label_visibility="collapsed"
)

text_to_analyze = ""

# ── TEXT MODE ─────────────────────────────────────────────────────────────────
if mode == "Paste text":
    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Real news example", use_container_width=True):
            st.session_state.demo = (
                "NASA successfully launched the Artemis II mission, "
                "sending four astronauts on a journey around the Moon "
                "in the first crewed lunar flyby since Apollo 17 in 1972."
            )
    with c2:
        if st.button("Fake news example", use_container_width=True):
            st.session_state.demo = (
                "BREAKING: Scientists confirm 5G towers are secretly emitting "
                "radiation that causes memory loss — government whistleblower "
                "leaks classified documents proving mass cover-up!"
            )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    text_to_analyze = st.text_area(
        "text", height=150,
        value=st.session_state.get("demo", ""),
        placeholder="Paste your news article or headline here...",
        label_visibility="collapsed"
    )

# ── URL MODE ──────────────────────────────────────────────────────────────────
else:
    st.markdown(
        "<p style='font-size:0.82rem;color:#9ca3af;margin:4px 0 8px;'>"
        "Works with most public news sites. Paywalled or JavaScript-heavy pages may not work."
        "</p>", unsafe_allow_html=True
    )
    url_input = st.text_input(
        "url", placeholder="https://www.thehindu.com/news/...",
        label_visibility="collapsed"
    )

    if url_input.strip():
        with st.spinner("Fetching article..."):
            scraped, err = scrape_article(url_input.strip())
        if err:
            st.error(err)
        else:
            text_to_analyze = scraped
            wc = len(scraped.split())
            preview = scraped[:300] + "..." if len(scraped) > 300 else scraped
            st.markdown(
                f'<div class="scraped-label">Article extracted &nbsp;·&nbsp; {wc} words</div>'
                f'<div class="scraped-box">{preview}</div>',
                unsafe_allow_html=True
            )

# ── Analyze ───────────────────────────────────────────────────────────────────
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
go = st.button("Analyze Article", use_container_width=True, type="primary")

if go:
    if not text_to_analyze.strip():
        st.warning("Please enter some text or a valid URL above.")
    else:
        with st.spinner("Running model..."):
            try:
                tok, mdl = load_model()
                label, conf, real_s, fake_s = predict(text_to_analyze, tok, mdl)

                if label == "FAKE":
                    st.markdown(f"""
                    <div class="result-card result-fake">
                        <div class="result-title-fake">Fake News Detected</div>
                        <div class="result-meta">
                            Confidence: <strong>{conf:.1f}%</strong>
                            &nbsp;|&nbsp; Real: {real_s*100:.1f}%
                            &nbsp;|&nbsp; Fake: {fake_s*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-real">
                        <div class="result-title-real">Likely Real News</div>
                        <div class="result-meta">
                            Confidence: <strong>{conf:.1f}%</strong>
                            &nbsp;|&nbsp; Real: {real_s*100:.1f}%
                            &nbsp;|&nbsp; Fake: {fake_s*100:.1f}%
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    st.markdown('<div class="score-label">Real score</div>', unsafe_allow_html=True)
                    st.progress(round(real_s, 2))
                with cb:
                    st.markdown('<div class="score-label">Fake score</div>', unsafe_allow_html=True)
                    st.progress(round(fake_s, 2))

                if conf < 65:
                    st.markdown(
                        '<div class="warn-box">Low confidence — the model is uncertain. '
                        'For Indian or regional news, confidence may be lower as the model '
                        'was trained primarily on English-language US news data.</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("""
                <div class="note-box">
                    <p>Note: This model was trained on US English news data.
                    Indian and regional news predictions may be less accurate,
                    especially for political content. A confidence below 65% means
                    the result should be treated as uncertain.</p>
                </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Model error: {e}")

# ── Footer stats ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-val">94%</div>
        <div class="stat-name">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">0.93</div>
        <div class="stat-name">F1 Score</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">20K+</div>
        <div class="stat-name">Training Samples</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">DistilBERT</div>
        <div class="stat-name">Base Model</div>
    </div>
</div>
<div class="footer-note">
    For educational purposes only. Always verify news through trusted sources.
</div>
""", unsafe_allow_html=True)
