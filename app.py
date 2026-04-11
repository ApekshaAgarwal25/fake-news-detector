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

# ── Theme toggle ──────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# Theme colors
if dark:
    bg_page    = "#0f1117"
    bg_card    = "#1a1d27"
    bg_input   = "#22263a"
    bg_stat    = "#1e2130"
    border     = "#2e3350"
    text_main  = "#f1f3f9"
    text_muted = "#8b93b0"
    text_hint  = "#555d7a"
    btn_bg     = "#22263a"
    btn_border = "#2e3350"
    btn_hover  = "#2e3a5c"
    divider    = "#2e3350"
    fake_bg    = "#2d1a1a"
    fake_bdr   = "#7f1d1d"
    fake_txt   = "#fca5a5"
    real_bg    = "#0f2d1a"
    real_bdr   = "#14532d"
    real_txt   = "#86efac"
    warn_bg    = "#2d2200"
    warn_bdr   = "#854d0e"
    warn_txt   = "#fde68a"
    scraped_bg = "#1a1f35"
    scraped_bdr= "#2e3a5c"
    badge_bg   = "#1e2547"
    badge_txt  = "#818cf8"
    toggle_icon= "☀️"
    toggle_lbl = "Light mode"
else:
    bg_page    = "#f4f6f9"
    bg_card    = "#ffffff"
    bg_input   = "#fafafa"
    bg_stat    = "#f9fafb"
    border     = "#e5e7eb"
    text_main  = "#1f2937"
    text_muted = "#6b7280"
    text_hint  = "#9ca3af"
    btn_bg     = "#f8faff"
    btn_border = "#d1d5db"
    btn_hover  = "#eef2ff"
    divider    = "#e5e7eb"
    fake_bg    = "#fff1f2"
    fake_bdr   = "#fca5a5"
    fake_txt   = "#b91c1c"
    real_bg    = "#f0fdf4"
    real_bdr   = "#86efac"
    real_txt   = "#15803d"
    warn_bg    = "#fffbeb"
    warn_bdr   = "#fde68a"
    warn_txt   = "#92400e"
    scraped_bg = "#f8f9ff"
    scraped_bdr= "#e0e7ff"
    badge_bg   = "#eef2ff"
    badge_txt  = "#4338ca"
    toggle_icon= "🌙"
    toggle_lbl = "Dark mode"

st.markdown(f"""
<style>
    #MainMenu, footer, header {{ visibility: hidden; }}

    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {bg_page} !important;
    }}
    .main .block-container {{
        max-width: 740px;
        padding: 2rem 2.5rem 3rem;
        background: {bg_card} !important;
        border-radius: 16px;
        margin-top: 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.10);
        border: 1px solid {border};
    }}
    p, span, div, label, small, h1, h2, h3 {{ color: {text_main} !important; }}

    .app-title {{
        font-size: 1.9rem; font-weight: 700;
        color: {text_main} !important;
        text-align: center; margin-bottom: 0.3rem;
    }}
    .app-subtitle {{
        font-size: 0.92rem; color: {text_muted} !important;
        text-align: center; line-height: 1.6;
        max-width: 500px; margin: 0 auto 0.5rem;
    }}
    .badge-row {{ text-align: center; margin-bottom: 1.2rem; }}
    .model-badge {{
        display: inline-block;
        background: {badge_bg} !important;
        color: {badge_txt} !important;
        font-size: 0.75rem; font-weight: 600;
        padding: 3px 14px; border-radius: 999px;
        letter-spacing: 0.03em; margin: 2px 3px;
    }}
    .divider {{ border: none; border-top: 1px solid {divider}; margin: 0 0 1.5rem; }}

    /* Radio */
    div[data-testid="stRadio"] > div {{
        display: flex !important; gap: 10px !important;
        flex-direction: row !important;
    }}
    div[data-testid="stRadio"] label {{
        background: {btn_bg} !important;
        color: {text_main} !important;
        border: 1.5px solid {btn_border} !important;
        border-radius: 8px !important;
        padding: 6px 20px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        cursor: pointer !important;
    }}
    div[data-testid="stRadio"] label > div:first-child {{ display: none !important; }}

    /* Section labels */
    .section-label {{
        font-size: 0.75rem; font-weight: 600;
        color: {text_hint} !important;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 1rem 0 0.4rem;
    }}

    /* All buttons */
    .stButton > button {{
        background: {btn_bg} !important;
        color: {text_main} !important;
        border: 1.5px solid {btn_border} !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }}

    /* Analyze button */
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        background: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.3) !important;
    }}

    /* Inputs */
    textarea {{
        background: {bg_input} !important;
        color: {text_main} !important;
        border: 1.5px solid {border} !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }}
    input[type="text"] {{
        background: {bg_input} !important;
        color: {text_main} !important;
        border: 1.5px solid {border} !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }}

    /* Scraped preview */
    .scraped-box {{
        background: {scraped_bg} !important;
        border: 1px solid {scraped_bdr};
        border-radius: 10px; padding: 0.9rem 1.1rem;
        font-size: 0.87rem; color: {text_main} !important;
        line-height: 1.65; margin-top: 0.5rem;
    }}
    .scraped-label {{
        font-size: 0.74rem; font-weight: 600;
        color: {badge_txt} !important;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }}

    /* Results */
    .result-card {{ border-radius: 12px; padding: 1.3rem 1.5rem; margin-top: 1.2rem; }}
    .result-fake {{ background: {fake_bg} !important; border: 1.5px solid {fake_bdr}; }}
    .result-real {{ background: {real_bg} !important; border: 1.5px solid {real_bdr}; }}
    .result-title-fake {{ font-size: 1.2rem; font-weight: 700; color: {fake_txt} !important; margin-bottom: 4px; }}
    .result-title-real {{ font-size: 1.2rem; font-weight: 700; color: {real_txt} !important; margin-bottom: 4px; }}
    .result-meta {{ font-size: 0.84rem; color: {text_muted} !important; }}

    .score-label {{
        font-size: 0.76rem; font-weight: 600;
        color: {text_hint} !important;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px;
    }}
    .warn-box {{
        background: {warn_bg} !important; border: 1px solid {warn_bdr};
        border-radius: 8px; padding: 0.65rem 1rem;
        font-size: 0.83rem; color: {warn_txt} !important; margin-top: 0.8rem;
    }}

    /* Coverage cards */
    .coverage-row {{
        display: flex; gap: 10px; margin-top: 1.5rem;
        padding-top: 1.5rem; border-top: 1px solid {divider};
    }}
    .coverage-card {{
        flex: 1; background: {bg_stat} !important;
        border: 1px solid {border}; border-radius: 10px;
        padding: 0.85rem; text-align: center;
    }}
    .coverage-val {{
        font-size: 1.1rem; font-weight: 700;
        color: {text_main} !important; margin-bottom: 2px;
    }}
    .coverage-name {{
        font-size: 0.68rem; color: {text_hint} !important;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 2px;
    }}
    .coverage-sub {{
        font-size: 0.7rem; color: {text_muted} !important;
        margin-top: 3px; line-height: 1.4;
    }}

    /* Training info row */
    .training-row {{
        display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;
    }}
    .training-tag {{
        background: {badge_bg} !important; color: {badge_txt} !important;
        font-size: 0.72rem; font-weight: 600; padding: 3px 10px;
        border-radius: 999px; letter-spacing: 0.03em;
    }}

    .footer-note {{
        font-size: 0.75rem; color: {text_hint} !important;
        text-align: center; margin-top: 1.2rem; line-height: 1.6;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
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
    id2label = model.config.id2label
    score_0 = probs[0].item()
    score_1 = probs[1].item()
    if id2label.get(0) == 'FAKE':
        fake_score, real_score = score_0, score_1
    else:
        real_score, fake_score = score_0, score_1
    label = "FAKE" if fake_score > real_score else "REAL"
    confidence = max(real_score, fake_score) * 100
    return label, confidence, real_score, fake_score


def scrape_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
        session = requests.Session()
        session.headers.update(headers)
        resp = session.get(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside","form","noscript","iframe"]):
            tag.decompose()
        text = ""
        article = soup.find("article")
        if article:
            text = article.get_text(" ", strip=True)
        if len(text.split()) < 80:
            for cls in ["article-body","article__body","story-body","post-content",
                        "entry-content","content-body","article-content","news-body",
                        "story-content","td-post-content","articlebody","article-text"]:
                el = soup.find(attrs={"class": re.compile(cls, re.I)})
                if el:
                    text = el.get_text(" ", strip=True)
                    if len(text.split()) > 80:
                        break
        if len(text.split()) < 80:
            main = soup.find("main")
            if main:
                text = main.get_text(" ", strip=True)
        if len(text.split()) < 80:
            paras = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text.split()) < 50:
            return None, "Could not extract enough text. Please copy the article text and use Paste text mode."
        return text[:4000], None
    except requests.exceptions.Timeout:
        return None, "Request timed out. Try a different news site or paste the text directly."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 403:
            return None, "Access denied (403). Please copy and paste the article text manually."
        return None, f"HTTP error {code}. Please try a different URL."
    except Exception as e:
        return None, f"Could not fetch article: {str(e)}"


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_toggle = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <div style="padding: 0.8rem 0 0.2rem;">
        <div class="app-title">Fake News Detector</div>
    </div>
    <div class="app-subtitle">
        Paste a news headline, article text, or URL to check if it is real or fake.
    </div>
    <div class="badge-row">
        <span class="model-badge">DistilBERT</span>
        <span class="model-badge">99.7% Accuracy</span>
        <span class="model-badge">116K Articles</span>
    </div>
    """, unsafe_allow_html=True)

with col_toggle:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button(toggle_icon, help=toggle_lbl):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f'<hr class="divider">', unsafe_allow_html=True)

# ── Input toggle ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Input method</div>', unsafe_allow_html=True)
mode = st.radio("mode", ["Paste text", "Paste URL"], horizontal=True, label_visibility="collapsed")

text_to_analyze = ""

if mode == "Paste text":
    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Real news example", use_container_width=True):
            st.session_state.demo = (
                "The Reserve Bank of India kept the repo rate unchanged at 6.5 percent, "
                "citing elevated inflation and global economic uncertainty as key concerns."
            )
    with c2:
        if st.button("Fake news example", use_container_width=True):
            st.session_state.demo = (
                "BREAKING: Government secretly adding microchips to COVID vaccines — "
                "whistleblower leaks classified documents proving mass cover-up!"
            )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    text_to_analyze = st.text_area(
        "text", height=150,
        value=st.session_state.get("demo", ""),
        placeholder="Paste your news article or headline here...",
        label_visibility="collapsed"
    )
else:
    st.markdown(
        f"<p style='font-size:0.82rem;color:{text_hint};margin:4px 0 8px;'>"
        "Works with most public news sites. Paywalled pages may not work — use Paste text mode instead.</p>",
        unsafe_allow_html=True
    )
    url_input = st.text_input("url", placeholder="https://www.thehindu.com/news/...", label_visibility="collapsed")
    if url_input.strip():
        with st.spinner("Fetching article..."):
            scraped, err = scrape_article(url_input.strip())
        if err:
            st.error(err)
            st.info("Tip: Copy the article text and switch to Paste text mode.")
        else:
            text_to_analyze = scraped
            wc = len(scraped.split())
            preview = scraped[:350] + "..." if len(scraped) > 350 else scraped
            st.markdown(
                f'<div class="scraped-label">Article extracted · {wc} words</div>'
                f'<div class="scraped-box">{preview}</div>',
                unsafe_allow_html=True
            )

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

                if conf < 70:
                    st.markdown(
                        '<div class="warn-box">Low confidence — try pasting the full article text for a stronger prediction.</div>',
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Model error: {e}")

# ── Coverage + Stats ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="coverage-row">
    <div class="coverage-card">
        <div class="coverage-val">99.7%</div>
        <div class="coverage-name">Accuracy</div>
        <div class="coverage-sub">0.997 F1 Score</div>
    </div>
    <div class="coverage-card">
        <div class="coverage-val">116K+</div>
        <div class="coverage-name">Training Articles</div>
        <div class="coverage-sub">Deduplicated & balanced</div>
    </div>
    <div class="coverage-card">
        <div class="coverage-val">US + India</div>
        <div class="coverage-name">News Coverage</div>
        <div class="coverage-sub">English language news</div>
    </div>
    <div class="coverage-card">
        <div class="coverage-val">DistilBERT</div>
        <div class="coverage-name">Base Model</div>
        <div class="coverage-sub">HuggingFace Transformers</div>
    </div>
</div>
<div style="margin-top:12px;">
    <div class="section-label">Trained on</div>
    <div class="training-row">
        <span class="training-tag">WELFake Dataset</span>
        <span class="training-tag">ISOT Fake News</span>
        <span class="training-tag">Kaggle Fake News</span>
        <span class="training-tag">US Political News</span>
        <span class="training-tag">Indian News</span>
    </div>
</div>
<div class="footer-note">
    For educational purposes only. Best accuracy on English language news.<br>
    Always verify news through trusted sources.
</div>
""", unsafe_allow_html=True)
