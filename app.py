import pickle
import streamlit as st
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

from disease_reference import (
    identify_hen_condition,
    identify_egg_condition,
    DISCLAIMER,
)
from gemini_ai import generate_ai_report, is_gemini_configured


# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Poultry AI — Coop Console",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

IMG_SIZE = 64
DISPLAY_IMG_WIDTH = 420


# ================================================================
# PREMIUM DARK / GOLD UI
# ================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #10110d;
    --panel: #191a14;
    --panel-2: #1d1e16;
    --gold: #f4c400;
    --gold-2: #ffd21c;
    --text: #f5f1e6;
    --muted: #a7a79d;
    --line: rgba(244,196,0,.22);
    --green: #52c477;
    --red: #ef5a35;
    --purple: #8d79e8;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
    background: var(--bg);
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 82% 12%, rgba(244,196,0,.045), transparent 25%),
        linear-gradient(180deg, #10110d 0%, #11120e 100%);
}

/* FIX 1: extra top padding so content clears Streamlit's fixed header
   instead of rendering underneath/behind it */
.block-container {
    max-width: 1180px;
    padding-top: 5rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: rgba(16,17,13,.92);
}

footer { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

h1, h2, h3, h4 {
    font-family: "Space Grotesk", sans-serif;
    color: var(--text);
}

li, label {
    color: #ddd9cd;
}

div[data-testid="stMetric"] {
    background: transparent;
}

div[data-testid="stMetricValue"] {
    color: var(--gold);
    font-family: "Space Grotesk", sans-serif;
}

div[data-testid="stMetricLabel"] {
    color: #aaa99e;
}

/* Top navigation */
.nav-shell {
    border-bottom: 1px solid var(--line);
    padding: .2rem 0 .75rem 0;
    margin-bottom: 2.2rem;
}


.brand {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    line-height: 1.05;
    color: var(--text);
   
}

.brand small {
    display: block;
    color: #8e8d82;
    font-family: "DM Sans", sans-serif;
    font-size: .65rem;
    letter-spacing: .16em;
    margin-top: .22rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    border: 1px solid rgba(82,196,119,.35);
    background: rgba(82,196,119,.08);
    color: #6bdb8e;
    padding: .3rem .55rem;
    border-radius: 999px;
    font-size: .7rem;
    font-weight: 600;
    white-space: nowrap;
    visibility: hidden;
}

.status-dot {
    width: 7px;
    height: 7px;
    background: #52c477;
    border-radius: 50%;
    display: inline-block;
}

/* Hero */
.hero {
    padding: 1.1rem 0 1.5rem 0;
}

.kicker {
    display: inline-block;
    border: 1px solid rgba(244,196,0,.45);
    border-radius: 999px;
    color: var(--gold-2);
    padding: .35rem .75rem;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(3rem, 6vw, 5.2rem);
    line-height: .96;
    letter-spacing: -.055em;
    margin: 1.3rem 0 1.2rem 0;
    max-width: 700px;
}

.hero h1 span {
    color: var(--gold);
}

.hero p {
    max-width: 680px;
    font-size: 1.02rem;
    line-height: 1.75;
    color: #bbb9ae;
}

.gold-button {
    background: var(--gold);
    color: #11120d;
    border-radius: 10px;
    padding: .75rem 1.15rem;
    font-weight: 700;
}

/* Stat strip */
.stat-strip {
    border: 1px solid var(--line);
    background: rgba(29,30,22,.78);
    border-radius: 16px;
    overflow: hidden;
    margin: .5rem 0 2.5rem 0;
}

.stat-box {
    padding: 1.15rem 1.25rem;
    border-right: 1px solid var(--line);
}

.stat-box:last-child {
    border-right: 0;
}

.stat-value {
    color: var(--gold);
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
}

.stat-label {
    color: #aaa99e;
    font-size: .72rem;
    letter-spacing: .03em;
    text-transform: uppercase;
}

/* Sections */
.section-kicker {
    color: #9b9a8e;
    font-size: .72rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .35rem;
}

.section-title {
    font-family: "Space Grotesk", sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: .45rem;
}

.section-copy {
    color: #b5b3a9;
    max-width: 690px;
    line-height: 1.65;
}

/* Cards */
.dark-card {
    background: var(--panel);
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 18px;
    padding: 1.35rem;
}

.gold-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.35rem;
}

.upload-card {
    background: #1b1c15;
    border: 1px solid rgba(244,196,0,.16);
    border-radius: 18px;
    padding: 1.25rem;
}

.result-card {
    background: #1b1c15;
    border: 1px solid rgba(244,196,0,.18);
    border-radius: 18px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
}

.ai-card {
    background: #19171f;
    border: 1px solid rgba(141,121,232,.55);
    border-radius: 18px;
    padding: 1.3rem;
    margin-top: 1rem;
}

.condition-box {
    background: rgba(244,196,0,.10);
    border-left: 4px solid var(--gold);
    border-radius: 10px;
    padding: .9rem 1rem;
    margin-top: .9rem;
}

.care-box {
    background: rgba(239,90,53,.13);
    border-radius: 10px;
    padding: .9rem 1rem;
    margin-top: .8rem;
    color: #e7d8cf;
}

.good-badge, .bad-badge {
    display: inline-block;
    padding: .4rem .8rem;
    border-radius: 999px;
    font-size: .8rem;
    font-weight: 700;
}

.good-badge {
    color: #66d98d;
    border: 1px solid rgba(82,196,119,.4);
    background: rgba(82,196,119,.10);
}

.bad-badge {
    color: #ff8b68;
    border: 1px solid rgba(239,90,53,.5);
    background: rgba(239,90,53,.10);
}

.guide-row {
    border-bottom: 1px solid rgba(255,255,255,.08);
    padding: 1.1rem 0;
}

.guide-number {
    color: var(--gold);
    font-family: "Space Grotesk", sans-serif;
    font-weight: 700;
}

.guide-title {
    color: var(--text);
    font-weight: 700;
    font-size: 1rem;
}

.guide-copy {
    color: #aaa99e;
    margin-top: .25rem;
}

.tip-card {
    background: rgba(244,196,0,.07);
    border: 1px solid rgba(244,196,0,.18);
    border-radius: 14px;
    padding: 1rem;
}

.tip-card strong {
    color: var(--text);
}

.donut {
    width: 190px;
    height: 190px;
    border-radius: 50%;
    margin: 1rem auto;
    display: flex;
    align-items: center;
    justify-content: center;
}

.donut-inner {
    width: 112px;
    height: 112px;
    border-radius: 50%;
    background: #191a14;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.donut-number {
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
}

.donut-label {
    color: #99988e;
    font-size: .68rem;
}

.legend {
    display: flex;
    flex-direction: column;
    gap: .65rem;
    justify-content: center;
    height: 100%;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: .55rem;
    color: #aaa99e;
    font-size: .82rem;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
}

.footer-line {
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(255,255,255,.08);
    color: #8f8e84;
    font-size: .8rem;
}

/* Home / cover page */
.home-wrap {
    padding: .35rem 0 1rem 0;
}

.home-heading {
    text-align: center;
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(1.1rem, 2vw, 1.5rem);
    font-weight: 700;
    letter-spacing: .03em;
    color: var(--gold);
    margin: .5rem 0 2rem 0;
}

.home-name-item {
    display: flex;
    align-items: center;
    gap: .65rem;
    padding: .55rem 0;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    border-bottom: 1px solid rgba(255,255,255,.06);
}

.home-name-item:last-of-type {
    border-bottom: none;
}

.home-name-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--gold);
    flex-shrink: 0;
}

.home-tag-row {
    display: flex;
    gap: .55rem;
    flex-wrap: wrap;
    margin-top: 1.1rem;
}

.home-tag {
    display: inline-block;
    border: 1px solid rgba(244,196,0,.4);
    background: rgba(244,196,0,.08);
    color: var(--gold-2);
    font-size: .74rem;
    font-weight: 700;
    letter-spacing: .03em;
    padding: .4rem .8rem;
    border-radius: 999px;
}

.home-guide {
    margin-top: 1.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid rgba(255,255,255,.08);
}

.home-guide .label {
    color: #8f8e84;
    font-size: .68rem;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.home-guide .name {
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: .25rem;
}

.home-college-banner {
    text-align: center;
    padding: 1.7rem 1.5rem;
}

.home-college-banner .small {
    color: #99988e;
    font-size: .72rem;
    letter-spacing: .15em;
    text-transform: uppercase;
}

.home-college-banner .name {
    font-family: "Space Grotesk", sans-serif;
    font-size: clamp(1.5rem, 3vw, 2.1rem);
    font-weight: 700;
    color: var(--gold);
    margin-top: .4rem;
    letter-spacing: -.01em;
}

@media (max-width: 800px) {
    .home-name-item {
        font-size: 1rem;
    }
}


/* Streamlit controls */
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(244,196,0,.22);
    background: #202117;
    color: #f5f1e6;
    font-weight: 700;
    min-height: 2.7rem;
}

.stButton > button:hover {
    border-color: var(--gold);
    color: var(--gold);
}

.stButton > button[kind="primary"] {
    background: var(--gold);
    color: #11120d;
    border-color: var(--gold);
}

.stButton > button[kind="primary"]:hover {
    background: var(--gold-2);
    color: #11120d;
}

div[data-testid="stFileUploader"] {
    background: transparent;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #202117;
    border: 1px dashed rgba(244,196,0,.35);
    border-radius: 14px;
}

div[data-testid="stFileUploaderDropzone"] button {
    color: #11120d !important;
    background: var(--gold) !important;
    border: none !important;
}

div[data-testid="stImage"] img {
    border-radius: 14px;
}

.stProgress > div > div > div > div {
    background-color: var(--gold);
}

/* Hide radio label and make nav compact */
div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] > div {
    gap: .25rem;
    flex-wrap: nowrap;
}

/* FIX 3: make the nav pills read as clickable buttons — cursor, hover,
   and a visible "active" state — and hide the plain radio dot */
div[data-testid="stRadio"] label {
    border-radius: 8px;
    padding: .45rem .9rem;
    cursor: pointer;
    color: #ddd9cd;
    white-space: nowrap;
    transition: background .15s ease, color .15s ease;
}

div[data-testid="stRadio"] label:hover {
    background: rgba(244,196,0,.10);
    color: var(--gold-2);
}

div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(244,196,0,.16);
    color: var(--gold);
    font-weight: 700;
}

div[data-testid="stRadio"] input {
    display: none;
}

</style>
""",
    unsafe_allow_html=True,
)


# ================================================================
# FEATURE EXTRACTION + MODEL LOADING
# ================================================================
def extract_features(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)

    hist_feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 255))
        hist_feats.extend(hist / max(hist.sum(), 1))

    gray = rgb2gray(arr)
    brightness_mean = gray.mean()
    brightness_std = gray.std()

    gray_u8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_u8,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    features = hist_feats + [
        brightness_mean,
        brightness_std,
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        (gray < (brightness_mean - 0.15)).mean(),
    ]

    return np.array(features).reshape(1, -1)


@st.cache_resource
def load_artifacts():
    with open("hen_health_model.pkl", "rb") as f:
        hen_model = pickle.load(f)
    with open("hen_health_scaler.pkl", "rb") as f:
        hen_scaler = pickle.load(f)
    with open("egg_quality_model.pkl", "rb") as f:
        egg_model = pickle.load(f)
    with open("egg_quality_scaler.pkl", "rb") as f:
        egg_scaler = pickle.load(f)
    return hen_model, hen_scaler, egg_model, egg_scaler


@st.cache_resource
def load_accuracies():
    try:
        with open("model_accuracies.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


hen_model, hen_scaler, egg_model, egg_scaler = load_artifacts()
accuracies = load_accuracies()


# ================================================================
# SESSION STATE
# ================================================================
DEFAULTS = {
    "hen_result": None,
    "hen_confidence": None,
    "hen_condition": None,
    "hen_note": None,
    "hen_ai_report": None,
    "hen_image": None,
    "egg_result": None,
    "egg_confidence": None,
    "egg_condition": None,
    "egg_note": None,
    "egg_ai_report": None,
    "egg_image": None,
    "prediction_history": [],
    "nav": "Home",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ================================================================
# HELPERS
# ================================================================
def accuracy_value(group):
    if not accuracies:
        return "—"
    try:
        return f"{max(accuracies[group].values()):.0f}%"
    except Exception:
        return "—"


def add_history(module, result):
    st.session_state.prediction_history.append(
        {"module": module, "result": result}
    )
    st.session_state.prediction_history = st.session_state.prediction_history[-40:]


def prediction_mix():
    history = st.session_state.prediction_history
    healthy = sum(
        1 for x in history
        if x["result"] in ("Healthy", "Good Quality")
    )
    diseased = len(history) - healthy
    return healthy, diseased, len(history)


def render_footer():
    st.markdown(
        f"""
        <div class="footer-line">
            <b style="color:#e8e3d7;">🐔 Poultry AI — Coop Console</b><br><br>
            {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(
    result,
    confidence,
    condition_name,
    condition_note,
    module,
):
    good = result in ("Healthy", "Good Quality")
    badge_class = "good-badge" if good else "bad-badge"
    icon = "✓" if good else "⚠"
    label = "Likely condition" if module == "hen" else "Likely defect"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    c1, c2 = st.columns([0.25, 0.75], gap="large")

    with c1:
        st.markdown(
            f"""
            <div style="
                width:130px;height:130px;border-radius:50%;
                background:conic-gradient(
                    {'#52c477' if good else '#ef5a35'} {confidence*100:.0f}%,
                    #303129 0
                );
                display:flex;align-items:center;justify-content:center;
                margin:auto;
            ">
                <div style="
                    width:92px;height:92px;border-radius:50%;
                    background:#1b1c15;
                    display:flex;align-items:center;justify-content:center;
                    flex-direction:column;
                ">
                    <div style="font-size:1.45rem;font-weight:700;color:#f5f1e6;">
                        {confidence:.0%}
                    </div>
                    <div style="font-size:.62rem;color:#99988e;">
                        CONFIDENCE
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<span class="{badge_class}">{icon} {result}</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="condition-box">
                <div style="font-weight:700;font-size:1rem;">
                    🔎 {label}: {condition_name}
                </div>
                <div style="margin-top:.35rem;color:#bbb9ae;">
                    {condition_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not good:
            st.markdown(
                '<div class="care-box">🩺 Consider separating the bird '
                'from the flock and consulting a qualified veterinarian.</div>',
                unsafe_allow_html=True,
            )

    st.caption(DISCLAIMER)
    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_report(
    module,
    result,
    confidence,
    condition_name,
    condition_note,
):
    key = "hen_ai_report" if module == "hen" else "egg_ai_report"

    st.markdown('<div class="ai-card">', unsafe_allow_html=True)

    title = (
        "Generate AI Explanation & Care Guidance"
        if module == "hen"
        else "Generate AI Explanation & Egg Guidance"
    )

    st.markdown("### ✨ Generative AI Report")
    st.caption(
        "Plain-language explanation and care guidance, written by Gemini."
    )

    if st.button(
        f"🤖 {title}",
        key=f"generate_{module}_ai",
        use_container_width=True,
    ):
        if not is_gemini_configured():
            st.error(
                "Gemini is not configured. Add GEMINI_API_KEY to your .env file."
            )
        else:
            with st.spinner("✨ Gemini is generating the report..."):
                report = generate_ai_report(
                    subject=module,
                    prediction=result,
                    confidence=confidence,
                    reference_condition=condition_name,
                    reference_note=condition_note,
                )

            if report.get("ok"):
                st.session_state[key] = report["text"]
            else:
                st.session_state[key] = None
                st.error(report.get("error", "Unable to generate the report."))

    if st.session_state[key]:
        st.markdown(
            '<div style="border-top:1px solid rgba(255,255,255,.10);'
            'margin-top:1rem;padding-top:1rem;">',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state[key])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def predict_hen(file):
    img = Image.open(file).convert("RGB")
    feats = extract_features(img)
    feats_scaled = hen_scaler.transform(feats)

    pred = hen_model.predict(feats_scaled)[0]
    proba = hen_model.predict_proba(feats_scaled)[0]

    if pred == 1:
        confidence = float(proba[1])
        result = "Healthy"
        condition_name = "No specific condition detected by the classifier"
        condition_note = (
            "Continue normal observation, hygiene, nutrition, "
            "and access to clean water."
        )
    else:
        confidence = float(proba[0])
        result = "Diseased"
        condition = identify_hen_condition(img)
        condition_name = condition["name"]
        condition_note = condition["note"]

    st.session_state.hen_result = result
    st.session_state.hen_confidence = confidence
    st.session_state.hen_condition = condition_name
    st.session_state.hen_note = condition_note
    st.session_state.hen_ai_report = None
    st.session_state.hen_image = img

    add_history("Hen Health", result)


def predict_egg(file):
    img = Image.open(file).convert("RGB")
    feats = extract_features(img)
    feats_scaled = egg_scaler.transform(feats)

    pred = egg_model.predict(feats_scaled)[0]
    proba = egg_model.predict_proba(feats_scaled)[0]

    if pred == 1:
        confidence = float(proba[1])
        result = "Good Quality"
        condition_name = "No specific defect detected by the classifier"
        condition_note = (
            "Keep eggs clean, handle them gently, and follow "
            "appropriate food-safety storage practices."
        )
    else:
        confidence = float(proba[0])
        result = "Poor Quality"
        condition = identify_egg_condition(img)
        condition_name = condition["name"]
        condition_note = condition["note"]

    st.session_state.egg_result = result
    st.session_state.egg_confidence = confidence
    st.session_state.egg_condition = condition_name
    st.session_state.egg_note = condition_note
    st.session_state.egg_ai_report = None
    st.session_state.egg_image = img

    add_history("Egg Quality", result)


# ================================================================
# TOP NAVIGATION
# ================================================================
if "language" not in st.session_state:
    st.session_state.language = "English"

# FIX 2: give the nav radio group more room (was [2.6, 4.9, 1.7, 1.1])
# so all five pills render on one line instead of wrapping/colliding
nav_left, nav_mid, nav_right, nav_lang = st.columns(
    [2.2, 5.6, 1.3, 1.4], vertical_alignment="center"
)

with nav_left:
    st.markdown(
        """
        <div class="brand">
            🐔 AI Hens Health
            <small>COOP CONSOLE</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_mid:
    st.session_state.nav = st.radio(
        "Navigation",
        ["Home", "Overview", "Hen Health", "Egg Quality", "Model"],
        index=["Home", "Overview", "Hen Health", "Egg Quality", "Model"].index(
            st.session_state.nav
        ),
        horizontal=True,
        key="navigation_radio",
        label_visibility="collapsed",
    )

with nav_right:
    status_text = "Model online" if is_gemini_configured() else "AI key needed"
    status_color = "#52c477" if is_gemini_configured() else "#f4c400"
    st.markdown(
        f"""
        <div style="text-align:right;">
            <span class="status-pill"
                  style="color:{status_color};border-color:{status_color}55;">
                <span class="status-dot"
                      style="background:{status_color};"></span>
                {status_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_lang:
    st.session_state.language = st.selectbox(
        "🌐 Language",
        ["English", "తెలుగు"],
        index=["English", "తెలుగు"].index(st.session_state.language),
        key="language_select",
        label_visibility="collapsed",
    )

st.markdown('<div class="nav-shell"></div>', unsafe_allow_html=True)

# ================================================================
# HOME PAGE TRANSLATIONS
# Only the Home page's text is translated — every other page/feature
# is unchanged and stays in English regardless of this selector.
# ================================================================
HOME_TRANSLATIONS = {
    "English": {
        "kicker": "🐔 AICW · PROJECT",
        "hero_title": "ARTIFICIAL INTELLIGENCE CAREER FOR WOMEN (AICW)",
        "hero_sub": (
            "An Artificial Intelligence project developed under AICW "
            "(Artificial Intelligence Career for Women), combining machine "
            "learning and generative AI for poultry health and egg quality assessment."
        ),
        "team_kicker": "STUDENT TEAM",
        "team_lines": ["Kaki Vasavi", "Kuripudi Hema Maha Lakshmi", "Medisetti Hema Sri"],
        "guide_label": "Under Guide",
        "guide_name": "MR.Abdul Aziz Md",
        "project_kicker": "PROJECT",
        "project_title": "Hen Health &amp; Egg Quality Prediction",
        "project_sub": "Hen Health and Egg Quality Prediction* is an AI-based project designed "
        "to monitor the health condition of hens and predict the quality of eggs using image "
        "processing and machine learning techniques. The system analyzes hen and egg images to"
        " identify whether a hen is healthy or diseased and whether an egg is healthy or"
        " damaged. By using AI and deep learning, the project helps farmers detect health "
        "problems at an early stage and assess egg quality efficiently. This can reduce manual"
        " inspection, improve poultry farm management, and support better decision-making."
        " Overall, the system provides a smart and automated solution for poultry health monitoring and egg quality analysis.",
        "tag_1": "Machine Learning",
        "tag_2": "Generative AI",
        "college_small": "PRESENTED AT",
        "college_name": "VSM Engineering College",
    },
    "తెలుగు": {
        "kicker": "🐔 AICW · ఫైనల్ ఇయర్ ప్రాజెక్ట్",
        "hero_title": "కృత్రిమ మేధస్సు వృత్తి కోసం మహిళలు (AICW)",
        "hero_sub": (
            "AICW (మహిళల కోసం కృత్రిమ మేధస్సు వృత్తి) కింద అభివృద్ధి చేయబడిన ఒక "
            "కృత్రిమ మేధస్సు ప్రాజెక్ట్, ఇది కోడి ఆరోగ్యం మరియు గుడ్డు నాణ్యత అంచనా కోసం "
            "మెషిన్ లెర్నింగ్ మరియు జనరేటివ్ AI లను మిళితం చేస్తుంది."
        ),
        "team_kicker": "విద్యార్థి బృందం",
        "team_lines": ["కాకి వాసవి", "కురిపూడి హెమ మహా లక్ష్మీ", "మెడిశెట్టి హెమ శ్రీ"],
        "guide_label": "మార్గదర్శకుడు",
        "guide_name": "శ్రీ అబ్దుల్ అజీజ్ మహమ్మద్",
        "project_kicker": "ప్రాజెక్ట్",
        "project_title": "కోడి ఆరోగ్యం &amp; గుడ్డు నాణ్యత అంచనా",
        "project_sub": "కోడి ఆరోగ్యం మరియు గుడ్డు నాణ్యత అంచనా* అనేది కోడిపిల్లల ఆరోగ్య పరిస్థితిని పర్యవేక్షించడానికి మరియు చిత్రాన్ని ఉపయోగించి గుడ్ల నాణ్యతను అంచనా వేయడానికి రూపొందించిన AI ఆధారిత ప్రాజెక్ట్. ప్రాసెసింగ్ మరియు మెషిన్ లెర్నింగ్ సాంకేతికతలు. ఈ వ్యవస్థ కోడి మరియు గుడ్ల చిత్రాలను విశ్లేషిస్తుంది, కోడి ఆరోగ్యంగా ఉందా లేదా వ్యాధిగ్రస్తమైందో, గుడ్డు ఆరోగ్యంగా ఉందా లేదా దెబ్బతిన్నదో గుర్తిస్తుంది. AI మరియు డీప్ లెర్నింగ్‌ను ఉపయోగించడం ద్వారా, ఈ ప్రాజెక్ట్ రైతులు ఆరోగ్య సమస్యలను ప్రారంభ దశలో గుర్తించడంలో మరియు గుడ్ల నాణ్యతను సమర్థవంతంగా అంచనా వేయడంలో సహాయపడుతుంది. ఇది మాన్యువల్ తనిఖీని తగ్గించవచ్చు, పౌల్ట్రీ ఫారం నిర్వహణను మెరుగుపరచవచ్చు మరియు మెరుగైన నిర్ణయాలను తీసుకోవడానికి మద్దతు ఇవ్వవచ్చు. మొత్తం మీద, ఈ వ్యవస్థ పౌల్ట్రీ ఆరోగ్య పర్యవేక్షణ మరియు గుడ్ల నాణ్యత విశ్లేషణ కోసం ఒక స్మార్ట్ మరియు ఆటోమేటెడ్ పరిష్కారాన్ని అందిస్తుంది.",
        "tag_1": "మెషిన్ లెర్నింగ్",
        "tag_2": "జనరేటివ్ AI",
        "college_small": "సమర్పించిన సంస్థ",
        "college_name": "VSM ఇంజనీరింగ్ కళాశాల",
    },
}


def home_text(key):
    return HOME_TRANSLATIONS[st.session_state.language][key]


# ================================================================
# HOME — SIMPLE PROJECT COVER PAGE
# ================================================================
if st.session_state.nav == "Home":

    st.markdown('<div class="home-wrap">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="hero" style="padding-top:.3rem;">
            <span class="kicker">{home_text('kicker')}</span>
            <h1 style="font-size:clamp(2.1rem, 4.2vw, 3.4rem); max-width:900px;">
                {home_text('hero_title')}
            </h1>
            <p>{home_text('hero_sub')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        team_items = "".join(
            f'<div class="home-name-item"><span class="home-name-dot"></span>{name}</div>'
            for name in home_text("team_lines")
        )
        st.markdown(
            f"""
            <div class="gold-card">
                <div class="section-kicker">{home_text('team_kicker')}</div>
                {team_items}
                <div class="home-guide">
                    <div class="label">{home_text('guide_label')}</div>
                    <div class="name">{home_text('guide_name')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="gold-card">
                <div class="section-kicker">{home_text('project_kicker')}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.35rem;
                            font-weight:700; color:var(--text); line-height:1.4;">
                    {home_text('project_title')}
                </div>
                <div style="color:#aaa99e; margin-top:.6rem; line-height:1.6; font-size:.9rem;">
                    {home_text('project_sub')}
                </div>
                <div class="home-tag-row">
                    <span class="home-tag">{home_text('tag_1')}</span>
                    <span class="home-tag">{home_text('tag_2')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="stat-strip" style="margin:0;">
            <div class="home-college-banner">
                <div class="small">{home_text('college_small')}</div>
                <div class="name">{home_text('college_name')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()



# ================================================================
# OVERVIEW
# ================================================================
elif st.session_state.nav == "Overview":

    st.markdown(
        """
        <div class="hero">
            <span class="kicker">🐔 VISION MODEL · V2</span>
            <h1>Read the flock <span>before</span><br>it tells you.</h1>
            <p>
                Upload a photo of a hen or an egg and get an instant ML read
                on health and quality — then a plain-language report with care
                guidance, generated in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.2, 1.2])
    with c1:
        if st.button("Upload a photo", type="primary", key="overview_upload"):
            st.session_state.nav = "Hen Health"
            st.rerun()
    with c2:
        if st.button("See how scoring works ↓", key="overview_guide"):
            st.session_state.nav = "Model"
            st.rerun()

    hen_acc = accuracy_value("hen")
    egg_acc = accuracy_value("egg")

    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (hen_acc, "HEN MODEL ACCURACY"),
        (egg_acc, "EGG MODEL ACCURACY"),
        ("2", "CLASSIFIERS RUNNING"),
        ("<2s", "AVG. PREDICTION TIME"),
    ]

    for col, (value, label) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-strip" style="margin:0;">
                    <div class="stat-box" style="border-right:0;">
                        <div class="stat-value">{value}</div>
                        <div class="stat-label">{label}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">MODULE 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🐔 Hen Health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Feed in a clear, well-lit photo of a hen. '
        'The classifier scores visual features and flags a likely condition '
        'if something looks off.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Fill the frame</strong><br>
                <span style="color:#aaa99e;">Keep the hen centered and large in the shot.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Even light</strong><br>
                <span style="color:#aaa99e;">Avoid harsh shadows or backlighting.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Plain background</strong><br>
                <span style="color:#aaa99e;">Cuts down on visual noise the model has to filter out.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Stay in focus</strong><br>
                <span style="color:#aaa99e;">Blur is a major cause of low-confidence reads.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">PREDICTION MIX — LAST 40 UPLOADS</div>', unsafe_allow_html=True)

    healthy, diseased, total = prediction_mix()
    if total == 0:
        healthy_pct = 67
        diseased_pct = 33
        center_text = "0"
        center_label = "UPLOADS"
    else:
        healthy_pct = round((healthy / total) * 100)
        diseased_pct = 100 - healthy_pct
        center_text = str(total)
        center_label = "SCANS"

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            f"""
            <div class="gold-card">
                <div class="donut"
                     style="background:conic-gradient(#52c477 0 {healthy_pct}%,
                                                     #ef5a35 {healthy_pct}% 100%);">
                    <div class="donut-inner">
                        <div class="donut-number">{center_text}</div>
                        <div class="donut-label">{center_label}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="gold-card" style="height:100%;">
                <div class="legend">
                    <div class="legend-item">
                        <span class="legend-dot" style="background:#52c477;"></span>
                        Healthy / Good — {healthy_pct}%
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot" style="background:#ef5a35;"></span>
                        Diseased / Poor — {diseased_pct}%
                    </div>
                    <div style="color:#77766d;font-size:.75rem;margin-top:1rem;">
                        Live session history. Uploads are not stored permanently by this UI.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">GUIDE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How a scan works</div>', unsafe_allow_html=True)

    guides = [
        ("01", "Pick a module", "Choose Hen Health or Egg Quality — each runs its own trained classifier."),
        ("02", "Upload a photo", "Sharp, well-lit, plain-background shots score most reliably."),
        ("03", "Read the result", "Get a status badge, confidence gauge, and the likely condition or defect."),
        ("04", "Generate the AI report", "Turn the raw prediction into a plain-language explanation with next steps."),
    ]

    for number, title, copy in guides:
        st.markdown(
            f"""
            <div class="guide-row">
                <div class="guide-number">{number}</div>
                <div class="guide-title">{title}</div>
                <div class="guide-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()


# ================================================================
# HEN HEALTH
# ================================================================
elif st.session_state.nav == "Hen Health":

    st.markdown('<div class="section-kicker">MODULE 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🐔 Hen Health</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload a clear hen image, run the trained classifier, '
        'then optionally generate a Gemini explanation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([0.95, 1.35], gap="large")

    with left:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Upload a photo")
        hen_file = st.file_uploader(
            "Choose a hen image",
            type=["png", "jpg", "jpeg"],
            key="hen_upload_v2",
        )

        if hen_file:
            img = Image.open(hen_file).convert("RGB")
            st.image(img, caption="Uploaded hen image", use_container_width=True)

            if st.button(
                "🔎 Predict Hen Health",
                type="primary",
                key="predict_hen_v2",
                use_container_width=True,
            ):
                with st.spinner("Analyzing hen image..."):
                    predict_hen(hen_file)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if st.session_state.hen_result:
            render_result_card(
                st.session_state.hen_result,
                st.session_state.hen_confidence,
                st.session_state.hen_condition,
                st.session_state.hen_note,
                "hen",
            )

            render_ai_report(
                "hen",
                st.session_state.hen_result,
                st.session_state.hen_confidence,
                st.session_state.hen_condition,
                st.session_state.hen_note,
            )
        else:
            st.markdown(
                """
                <div class="dark-card" style="min-height:390px;
                    display:flex;align-items:center;justify-content:center;
                    text-align:center;color:#77766d;">
                    <div>
                        <div style="font-size:2rem;">🐔</div>
                        <h3 style="color:#8f8e84;">Ready for a scan</h3>
                        <p>Upload a photo and click Predict.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


# ================================================================
# EGG QUALITY
# ================================================================
elif st.session_state.nav == "Egg Quality":

    st.markdown('<div class="section-kicker">MODULE 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🥚 Egg Quality</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload an egg image. The classifier evaluates '
        'visual features and flags likely quality defects before handling or grading.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([0.95, 1.35], gap="large")

    with left:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Upload a photo")
        egg_file = st.file_uploader(
            "Choose an egg image",
            type=["png", "jpg", "jpeg"],
            key="egg_upload_v2",
        )

        if egg_file:
            img = Image.open(egg_file).convert("RGB")
            st.image(img, caption="Uploaded egg image", use_container_width=True)

            if st.button(
                "🔎 Predict Egg Quality",
                type="primary",
                key="predict_egg_v2",
                use_container_width=True,
            ):
                with st.spinner("Analyzing egg image..."):
                    predict_egg(egg_file)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if st.session_state.egg_result:
            render_result_card(
                st.session_state.egg_result,
                st.session_state.egg_confidence,
                st.session_state.egg_condition,
                st.session_state.egg_note,
                "egg",
            )

            render_ai_report(
                "egg",
                st.session_state.egg_result,
                st.session_state.egg_confidence,
                st.session_state.egg_condition,
                st.session_state.egg_note,
            )
        else:
            st.markdown(
                """
                <div class="dark-card" style="min-height:390px;
                    display:flex;align-items:center;justify-content:center;
                    text-align:center;color:#77766d;">
                    <div>
                        <div style="font-size:2rem;">🥚</div>
                        <h3 style="color:#8f8e84;">Ready for a scan</h3>
                        <p>Upload an egg photo and click Predict.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


# ================================================================
# MODEL PAGE
# ================================================================
else:

    st.markdown('<div class="section-kicker">MODEL</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Scoring & model status</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The application keeps the existing trained '
        'classifiers and feature-extraction pipeline. This page only changes how '
        'their status and guidance are presented.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="gold-card">
                <div class="section-kicker">HEN MODEL</div>
                <div style="font-size:2.4rem;color:#f4c400;
                            font-family:'Space Grotesk',sans-serif;font-weight:700;">
                    {accuracy_value("hen")}
                </div>
                <div style="color:#99988e;">Cross-validated accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="gold-card">
                <div class="section-kicker">EGG MODEL</div>
                <div style="font-size:2.4rem;color:#f4c400;
                            font-family:'Space Grotesk',sans-serif;font-weight:700;">
                    {accuracy_value("egg")}
                </div>
                <div style="color:#99988e;">Cross-validated accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        ai_state = "Connected" if is_gemini_configured() else "Not configured"
        ai_color = "#52c477" if is_gemini_configured() else "#f4c400"
        st.markdown(
            f"""
            <div class="gold-card">
                <div class="section-kicker">GENERATIVE AI</div>
                <div style="font-size:1.65rem;color:{ai_color};
                            font-family:'Space Grotesk',sans-serif;font-weight:700;">
                    {ai_state}
                </div>
                <div style="color:#99988e;">Gemini report generation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">GUIDE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How scoring works</div>', unsafe_allow_html=True)

    guides = [
        ("01", "Feature extraction",
         "Images are resized and converted into visual features including color histograms, brightness statistics and texture measurements."),
        ("02", "Classifier",
         "The saved Hen Health or Egg Quality model receives the scaled feature vector and returns the predicted class."),
        ("03", "Confidence",
         "The displayed percentage comes from the classifier probability for the predicted class."),
        ("04", "Reference layer",
         "For a negative prediction, the project’s existing reference module supplies a likely condition/defect and care note."),
        ("05", "Generative report",
         "Gemini turns the structured prediction into a plain-language educational explanation and guidance."),
    ]

    for number, title, copy in guides:
        st.markdown(
            f"""
            <div class="guide-row">
                <div class="guide-number">{number}</div>
                <div class="guide-title">{title}</div>
                <div class="guide-copy">{copy}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Fill the frame</strong><br>
                <span style="color:#aaa99e;">Keep the subject large and centered.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Even light</strong><br>
                <span style="color:#aaa99e;">Avoid harsh shadows, glare and backlighting.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    t3, t4 = st.columns(2)
    with t3:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Plain background</strong><br>
                <span style="color:#aaa99e;">Reduce unrelated visual information.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            """
            <div class="tip-card">
                <strong>Stay in focus</strong><br>
                <span style="color:#aaa99e;">Blur can reduce prediction reliability.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()