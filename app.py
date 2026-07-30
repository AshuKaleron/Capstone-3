# ============================================================================
# BLOCK 25: [Advanced Production Streamlit Dashboard — NeuroSense v2]
# ============================================================================
# A futuristic, production-grade Streamlit application featuring:
#   • Dark-mode neon-glow glassmorphism UI with animated cosmic background
#   • 6 interactive tabs with rich features in each
#   • Plotly radar, gauge, sunburst, heatmap, and animated bar charts
#   • Real-time SHAP waterfall explanations
#   • What-If Simulator for feature sensitivity analysis
#   • Batch CSV processing with downloadable results
#   • Full model performance dashboard with interactive comparison
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Lazy optional imports (TF is heavy — only load if actually needed) ──
# TensorFlow is NOT imported at module level. It will be imported inside
# load_artifacts() ONLY if the champion model is a Keras model.
# This prevents Streamlit Cloud from loading ~800MB of TF into RAM when
# the champion is a lightweight tree model like CatBoost or LightGBM.
TF_AVAILABLE = False  # Set to True lazily inside load_artifacts() if needed

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================================
# CHAMPION MODEL WRAPPER (Required for pickle deserialization)
# ============================================================================
# This class MUST be defined in app.py BEFORE pickle.load() is called.
# The pipeline (Block 24) pickles a ChampionModelWrapper object into
# champion_model.pkl. When Python unpickles it, it looks for the class
# in the current module's namespace. If this class is missing, Python
# throws: AttributeError: Can't get attribute 'ChampionModelWrapper'
# ============================================================================

class ChampionModelWrapper:
    """
    Unified inference wrapper that standardizes the .predict_proba() contract
    regardless of whether the champion is a single tree model, a soft-vote
    ensemble, or a Keras neural network.
    """

    def __init__(self, model_or_models, deploy_type, model_names=None):
        self.deploy_type = deploy_type
        self.model_names = model_names  # only for Ensemble
        if deploy_type == 'Ensemble':
            self._models = model_or_models  # list of sklearn estimators
        else:
            self._model = model_or_models   # single estimator

    def predict_proba(self, X):
        """Return class probability array of shape (n_samples, n_classes)."""
        if self.deploy_type == 'Ensemble':
            return np.mean([m.predict_proba(X) for m in self._models], axis=0)
        elif self.deploy_type == 'Keras':
            return self._model.predict(X, verbose=0)
        else:
            return self._model.predict_proba(X)

    def predict(self, X):
        """Return integer class predictions."""
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================================
# PAGE CONFIG & GLOBAL CSS
# ============================================================================

st.set_page_config(
    page_title="NeuroSense — AI Emotion Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Ultra-Premium Futuristic Dark CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ═══ GLOBAL BACKGROUND — Animated Cosmic Gradient ═══ */
    .stApp {
        background: linear-gradient(135deg, #05051a 0%, #0a0a2e 25%, #10103a 50%, #0d0d30 75%, #05051a 100%);
        background-size: 400% 400%;
        animation: cosmicShift 20s ease infinite;
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }
    @keyframes cosmicShift {
        0%   { background-position: 0% 50%; }
        25%  { background-position: 100% 0%; }
        50%  { background-position: 100% 100%; }
        75%  { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }

    /* ═══ HIDE STREAMLIT DEFAULTS ═══ */
    #MainMenu, footer, header { visibility: hidden; }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7c3aed, #06d6a0);
        border-radius: 10px;
    }

    /* ═══ HERO HEADER ═══ */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
        position: relative;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c3aed 0%, #06d6a0 40%, #3b82f6 70%, #7c3aed 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGlow 5s ease infinite;
        letter-spacing: -2px;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    @keyframes textGlow {
        0%   { background-position: 0% 50%; filter: brightness(1); }
        50%  { background-position: 100% 50%; filter: brightness(1.15); }
        100% { background-position: 0% 50%; filter: brightness(1); }
    }
    .hero-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.3);
        color: #a78bfa;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
        letter-spacing: 0.3px;
        max-width: 600px;
        margin: 0 auto;
    }

    /* ═══ GLASS CARD — Base ═══ */
    .glass {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.4), rgba(6, 214, 160, 0.3), transparent);
    }
    .glass:hover {
        border-color: rgba(124, 58, 237, 0.2);
        box-shadow: 0 8px 40px rgba(124, 58, 237, 0.08);
        transform: translateY(-1px);
    }

    /* ═══ NEON GLOW CARD (for results) ═══ */
    .neon-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: cardReveal 0.7s ease-out;
    }
    .neon-card::before {
        content: '';
        position: absolute;
        top: -1px; left: -1px; right: -1px; bottom: -1px;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.3), transparent, rgba(6, 214, 160, 0.2));
        border-radius: 20px;
        z-index: -1;
        opacity: 0.5;
    }
    @keyframes cardReveal {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ═══ EMOTION RESULT ═══ */
    .emo-icon { font-size: 5rem; animation: emoPulse 1s ease; }
    @keyframes emoPulse {
        0%   { transform: scale(0); opacity: 0; }
        50%  { transform: scale(1.3); }
        100% { transform: scale(1); opacity: 1; }
    }
    .emo-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.4rem 0 0.2rem;
    }
    .emo-conf {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
    }

    /* ═══ METRIC CARDS ═══ */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
        margin: 1rem 0;
    }
    .m-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s;
        position: relative;
        overflow: hidden;
    }
    .m-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #7c3aed, #06d6a0);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .m-card:hover { transform: scale(1.02); }
    .m-card:hover::after { opacity: 1; }
    .m-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c3aed, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .m-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    /* ═══ SECTION HEADERS ═══ */
    .sec-h {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #c8d0e0;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(124, 58, 237, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sec-h-glow {
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #7c3aed, #06d6a0);
        border-radius: 2px;
    }

    /* ═══ TAB STYLING ═══ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #6b7280;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #a78bfa;
        background: rgba(124, 58, 237, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(6, 214, 160, 0.1)) !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    /* ═══ BUTTONS ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #06d6a0) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.4) !important;
    }

    /* ═══ INPUTS ═══ */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        color: #e0e6ed !important;
        border-radius: 10px !important;
    }

    /* ═══ DIVIDER ═══ */
    .neon-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 5%, rgba(124, 58, 237, 0.3) 30%, rgba(6, 214, 160, 0.2) 70%, transparent 95%);
        margin: 1.5rem 0;
    }

    /* ═══ STAT TAG ═══ */
    .stat-tag {
        display: inline-block;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.2);
        color: #a78bfa;
        padding: 0.2rem 0.7rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ═══ FEATURE ROW ═══ */
    .feat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .feat-name { color: #9ca3af; font-size: 0.85rem; }
    .feat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #06d6a0;
        font-weight: 600;
    }

    /* ═══ FOOTER ═══ */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #374151;
        font-size: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        margin-top: 3rem;
    }
    .footer-glow {
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# EMOTION CONFIGURATION
# ============================================================================

EMOTION_CONFIG = {
    'Happiness': {'emoji': '😊', 'color': '#fbbf24', 'glow': 'rgba(251,191,36,0.3)'},
    'Sadness':   {'emoji': '😢', 'color': '#3b82f6', 'glow': 'rgba(59,130,246,0.3)'},
    'Anger':     {'emoji': '😠', 'color': '#ef4444', 'glow': 'rgba(239,68,68,0.3)'},
    'Anxiety':   {'emoji': '😰', 'color': '#f97316', 'glow': 'rgba(249,115,22,0.3)'},
    'Boredom':   {'emoji': '😐', 'color': '#6b7280', 'glow': 'rgba(107,114,128,0.3)'},
    'Neutral':   {'emoji': '😶', 'color': '#06d6a0', 'glow': 'rgba(6,214,160,0.3)'},
}

PLATFORMS = ["Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Telegram", "Whatsapp"]
GENDERS   = ["Female", "Male", "Non-binary"]


# ============================================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================================

@st.cache_resource
def load_artifacts():
    """
    Load all serialized pipeline components into Streamlit's cache.
    The pipeline exports a ChampionModelWrapper that always exposes
    .predict_proba(). Keras champions use a lazy TF import to avoid
    loading ~800MB into RAM when the model is tree-based.
    """
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("pipeline_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    deploy_type = metadata.get("model_deployment_type", "")

    if deploy_type == "Keras":
        global TF_AVAILABLE
        try:
            import tensorflow as tf
            TF_AVAILABLE = True
        except ImportError:
            TF_AVAILABLE = False
            st.error("❌ Champion model is Keras but tensorflow is not installed.")
            st.stop()

        keras_model = tf.keras.models.load_model("champion_model.keras")

        class _KerasWrapper:
            def predict_proba(self, X):
                return keras_model.predict(X, verbose=0)
            def predict(self, X):
                return np.argmax(self.predict_proba(X), axis=1)

        model = _KerasWrapper()
    else:
        with open("champion_model.pkl", "rb") as f:
            model = pickle.load(f)

    tree_model = None
    try:
        with open("best_tree_model.pkl", "rb") as f:
            tree_model = pickle.load(f)
    except FileNotFoundError:
        pass

    return model, scaler, encoder, metadata, tree_model


try:
    model, scaler, encoder, metadata, tree_model = load_artifacts()
    feature_names = metadata["retained_features_list"]
    class_names   = metadata["label_encoder_classes"]
    deploy_type   = metadata["model_deployment_type"]
    n_classes     = metadata["n_classes"]
except Exception as e:
    st.error(f"❌ Failed to load deployment artifacts. Error: {str(e)}")
    st.info("Ensure champion_model.pkl/.keras, scaler.pkl, encoder.pkl, and pipeline_metadata.pkl exist in the repo.")
    st.stop()


# ============================================================================
# PREDICTION HELPERS
# ============================================================================

EPS = 1e-5

def predict_single(input_scaled):
    """Unified prediction — works for Tree, Ensemble, and Keras champions."""
    probs = model.predict_proba(input_scaled)[0]
    return probs


def build_input(age, gender, platform, usage, posts, likes, comments, messages):
    """Build a fully-engineered, encoded, scaled single-row input."""
    raw = {
        'Age': age,
        'Daily_Usage_Time (minutes)': usage,
        'Posts_Per_Day': posts,
        'Likes_Received_Per_Day': likes,
        'Comments_Received_Per_Day': comments,
        'Messages_Sent_Per_Day': messages,
        'Interaction_Density': (likes + comments) / (usage + EPS),
        'Social_Velocity': likes / (posts + EPS),
        'Conversational_Reciprocity': messages / (comments + EPS),
        'Attention_Index': usage / (posts + EPS),
        'Engagement_Ratio': (likes + comments + messages) / (usage + EPS),
        'Content_Efficiency': likes / (usage * posts + EPS),
    }
    df = pd.DataFrame(columns=feature_names, data=[np.zeros(len(feature_names))])
    for col, val in raw.items():
        if col in df.columns:
            df[col] = val
    gcol = f"Gender_{gender}"
    pcol = f"Platform_{platform}"
    if gcol in df.columns:
        df[gcol] = 1.0
    if pcol in df.columns:
        df[pcol] = 1.0
    return scaler.transform(df), raw


def get_engineered_features_display(raw):
    """Return a formatted list of engineered feature names and values."""
    eng = [
        ("Interaction Density", raw.get('Interaction_Density', 0)),
        ("Social Velocity", raw.get('Social_Velocity', 0)),
        ("Conversational Reciprocity", raw.get('Conversational_Reciprocity', 0)),
        ("Attention Index", raw.get('Attention_Index', 0)),
        ("Engagement Ratio", raw.get('Engagement_Ratio', 0)),
        ("Content Efficiency", raw.get('Content_Efficiency', 0)),
    ]
    return eng


# ============================================================================
# PLOTLY THEME HELPER
# ============================================================================

def styled_layout(height=380, **kwargs):
    """Return common Plotly layout kwargs for the dark theme."""
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c8d0e0', family='Inter'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=height,
    )
    base.update(kwargs)
    return base


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI-Powered Analytics</div>
    <div class="hero-title">🧠 NeuroSense</div>
    <div class="hero-subtitle">
        Predict dominant emotional states from social media behavioral patterns
        using ensemble machine learning & deep neural networks
    </div>
</div>
<div class="neon-divider"></div>
""", unsafe_allow_html=True)


# ============================================================================
# TABS — 6 Feature-Rich Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Predict", "📁 Batch Analysis", "📊 Model Performance",
    "🔬 Feature Lab", "🎛️ What-If Simulator", "ℹ️ About"
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1: 🔮 PREDICT EMOTION
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>User Profile & Engagement Metrics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("##### 👤 Demographics")
        age      = st.slider("Age", 10, 90, 25, key="p_age")
        gender   = st.selectbox("Gender", GENDERS, key="p_gender")
        platform = st.selectbox("Platform", PLATFORMS, key="p_platform")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("##### 📊 Daily Engagement")
        usage    = st.number_input("Usage Time (min)", 1, 1440, 120, key="p_usage")
        posts    = st.number_input("Posts Per Day", 0, 100, 3, key="p_posts")
        likes    = st.number_input("Likes Received", 0, 10000, 45, key="p_likes")
        comments = st.number_input("Comments Received", 0, 5000, 10, key="p_comments")
        messages = st.number_input("Messages Sent", 0, 5000, 15, key="p_messages")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
    with bcol2:
        predict_btn = st.button("🚀  Analyze Profile", use_container_width=True, key="predict_btn")

    if predict_btn:
        inp_scaled, raw_feats = build_input(age, gender, platform, usage, posts, likes, comments, messages)
        probs     = predict_single(inp_scaled)
        pred_idx  = int(np.argmax(probs))
        pred_emo  = class_names[pred_idx]
        conf      = float(probs[pred_idx])
        emo_cfg   = EMOTION_CONFIG.get(pred_emo, {'emoji': '🔮', 'color': '#7c3aed', 'glow': 'rgba(124,58,237,0.3)'})

        # ── Result Card ──
        st.markdown(f"""
        <div class="neon-card" style="border-color: {emo_cfg['color']}40;
             box-shadow: 0 0 60px {emo_cfg['glow']};">
            <div class="emo-icon">{emo_cfg['emoji']}</div>
            <div class="emo-label" style="color: {emo_cfg['color']};">{pred_emo}</div>
            <div class="emo-conf">Confidence: <strong>{conf:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts Row ──
        r1, r2 = st.columns(2, gap="large")

        with r1:
            st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Probability Radar</div>', unsafe_allow_html=True)
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=list(probs) + [probs[0]],
                theta=class_names + [class_names[0]],
                fill='toself',
                fillcolor='rgba(124, 58, 237, 0.12)',
                line=dict(color='#7c3aed', width=2.5),
                marker=dict(size=7, color='#06d6a0'),
            ))
            radar.update_layout(**styled_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.06)', tickfont=dict(size=9, color='#4b5563')),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickfont=dict(size=11, color='#9ca3af')),
                ),
                showlegend=False,
            ))
            st.plotly_chart(radar, use_container_width=True)

        with r2:
            st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Class Probabilities</div>', unsafe_allow_html=True)
            pdf = pd.DataFrame({'Emotion': class_names, 'Probability': probs}).sort_values('Probability', ascending=True)
            bcolors = [EMOTION_CONFIG.get(e, {}).get('color', '#7c3aed') for e in pdf['Emotion']]
            bar = go.Figure(go.Bar(
                x=pdf['Probability'], y=pdf['Emotion'], orientation='h',
                marker=dict(color=bcolors, line=dict(width=0)),
                text=[f'{p:.1%}' for p in pdf['Probability']],
                textposition='outside', textfont=dict(color='#9ca3af', size=11),
            ))
            bar.update_layout(**styled_layout(
                xaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.03)', tickformat='.0%', tickfont=dict(color='#4b5563')),
                yaxis=dict(tickfont=dict(color='#c8d0e0', size=11)),
            ))
            st.plotly_chart(bar, use_container_width=True)

        # ── Confidence Gauge ──
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Confidence Gauge</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=conf * 100,
            number=dict(suffix="%", font=dict(size=40, color='#e0e6ed')),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#4b5563'),
                bar=dict(color=emo_cfg['color']),
                bgcolor='rgba(255,255,255,0.03)',
                bordercolor='rgba(255,255,255,0.06)',
                steps=[
                    dict(range=[0, 33], color='rgba(239,68,68,0.1)'),
                    dict(range=[33, 66], color='rgba(249,115,22,0.1)'),
                    dict(range=[66, 100], color='rgba(6,214,160,0.1)'),
                ],
                threshold=dict(line=dict(color='#06d6a0', width=3), thickness=0.8, value=conf*100),
            ),
        ))
        gauge.update_layout(**styled_layout(height=250))
        st.plotly_chart(gauge, use_container_width=True)

        # ── Engineered Features Breakdown ──
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Engineered Features</div>', unsafe_allow_html=True)
        eng_feats = get_engineered_features_display(raw_feats)
        feat_html = ""
        for fname, fval in eng_feats:
            feat_html += f'<div class="feat-row"><span class="feat-name">{fname}</span><span class="feat-val">{fval:.4f}</span></div>'
        st.markdown(f'<div class="glass">{feat_html}</div>', unsafe_allow_html=True)

        # ── SHAP Explanation ──
        if SHAP_AVAILABLE and tree_model is not None:
            st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>SHAP Feature Explanation</div>', unsafe_allow_html=True)
            try:
                import matplotlib.pyplot as plt
                inp_df = pd.DataFrame(inp_scaled, columns=feature_names)
                explainer = shap.TreeExplainer(tree_model)
                sv = explainer(inp_df)
                if hasattr(sv, 'values') and sv.values.ndim == 3:
                    sv_single = shap.Explanation(
                        values=sv.values[0, :, pred_idx],
                        base_values=sv.base_values[0][pred_idx] if isinstance(sv.base_values[0], (list, np.ndarray)) else sv.base_values[0],
                        data=sv.data[0], feature_names=feature_names,
                    )
                else:
                    sv_single = sv[0]
                fig_shap, _ = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(sv_single, show=False)
                st.pyplot(plt.gcf())
                plt.close('all')
            except Exception as e:
                st.caption(f"SHAP unavailable: {str(e)[:120]}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: 📁 BATCH ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Batch Prediction — CSV Upload</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <p style="margin:0;">Upload a CSV with columns: <code style="color:#06d6a0;">Age, Gender, Platform,
        Daily_Usage_Time (minutes), Posts_Per_Day, Likes_Received_Per_Day,
        Comments_Received_Per_Day, Messages_Sent_Per_Day</code></p>
        <p style="color:#6b7280; margin-top:0.5rem; margin-bottom:0;">
        The system will engineer features, encode categoricals, and predict emotions for every row.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose CSV", type=["csv"], key="batch_up")

    if uploaded is not None:
        try:
            bdf = pd.read_csv(uploaded)
            st.markdown(f'<span class="stat-tag">{len(bdf)} rows × {len(bdf.columns)} cols</span>', unsafe_allow_html=True)

            with st.expander("📋 Preview uploaded data", expanded=True):
                st.dataframe(bdf.head(10), use_container_width=True)

            if st.button("🚀  Run Batch Predictions", key="batch_go"):
                results = []
                prog = st.progress(0)
                for i, (_, row) in enumerate(bdf.iterrows()):
                    try:
                        inp_s, _ = build_input(
                            row.get('Age', 25), row.get('Gender', 'Male'),
                            row.get('Platform', 'Instagram'),
                            row.get('Daily_Usage_Time (minutes)', 60),
                            row.get('Posts_Per_Day', 2), row.get('Likes_Received_Per_Day', 20),
                            row.get('Comments_Received_Per_Day', 5), row.get('Messages_Sent_Per_Day', 10),
                        )
                        pr = predict_single(inp_s)
                        pi = int(np.argmax(pr))
                        results.append({'Predicted_Emotion': class_names[pi], 'Confidence': float(pr[pi])})
                    except Exception:
                        results.append({'Predicted_Emotion': 'Error', 'Confidence': 0.0})
                    prog.progress((i + 1) / len(bdf))

                rdf = pd.concat([bdf, pd.DataFrame(results)], axis=1)

                st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Results</div>', unsafe_allow_html=True)
                st.dataframe(rdf, use_container_width=True)

                # ── Summary Row ──
                sc1, sc2 = st.columns(2)
                with sc1:
                    edist = rdf['Predicted_Emotion'].value_counts()
                    pie = px.pie(values=edist.values, names=edist.index,
                                 color=edist.index,
                                 color_discrete_map={e: c['color'] for e, c in EMOTION_CONFIG.items()},
                                 title="Predicted Distribution")
                    pie.update_layout(**styled_layout(height=350))
                    st.plotly_chart(pie, use_container_width=True)

                with sc2:
                    avg_c = rdf['Confidence'].mean()
                    st.markdown(f"""
                    <div class="glass" style="text-align:center; padding:2rem;">
                        <div class="m-val">{avg_c:.1%}</div>
                        <div class="m-label">Average Confidence</div>
                        <div style="margin-top:1.5rem;">
                            <div class="m-val">{len(rdf)}</div>
                            <div class="m-label">Total Predictions</div>
                        </div>
                        <div style="margin-top:1.5rem;">
                            <div class="m-val">{edist.index[0]}</div>
                            <div class="m-label">Most Common Emotion</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                csv_out = rdf.to_csv(index=False)
                st.download_button("⬇️  Download Results", csv_out, "neurosense_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {str(e)}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3: 📊 MODEL PERFORMANCE
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    champ_name = metadata.get('champion_model_name', 'Unknown')
    perf_data  = metadata.get('model_performance', [])

    # ── Champion Banner ──
    st.markdown(f"""
    <div class="neon-card">
        <div style="font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:2px;">Champion Model</div>
        <div style="font-family:'Space Grotesk'; font-size:2.2rem; font-weight:700;
             background: linear-gradient(135deg, #7c3aed, #06d6a0);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0.3rem 0;">
            🏆 {champ_name}
        </div>
        <div><span class="stat-tag">{deploy_type}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──
    if perf_data:
        cp = perf_data[0]
        st.markdown(f"""
        <div class="metric-grid">
            <div class="m-card"><div class="m-val">{cp.get('Accuracy',0):.4f}</div><div class="m-label">Accuracy</div></div>
            <div class="m-card"><div class="m-val">{cp.get('Precision (W)',0):.4f}</div><div class="m-label">Precision (W)</div></div>
            <div class="m-card"><div class="m-val">{cp.get('Recall (W)',0):.4f}</div><div class="m-label">Recall (W)</div></div>
            <div class="m-card"><div class="m-val">{cp.get('F1-Score (W)',0):.4f}</div><div class="m-label">F1-Score (W)</div></div>
            <div class="m-card"><div class="m-val">{cp.get('Precision (Macro)',0):.4f}</div><div class="m-label">Precision (M)</div></div>
            <div class="m-card"><div class="m-val">{cp.get('F1-Score (Macro)',0):.4f}</div><div class="m-label">F1-Score (M)</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── All Models Comparison ──
    if perf_data:
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>All Models — Performance Comparison</div>', unsafe_allow_html=True)
        pdf = pd.DataFrame(perf_data)
        st.dataframe(pdf.style.format({c: '{:.4f}' for c in pdf.columns if c != 'Model'})
                     .highlight_max(subset=[c for c in pdf.columns if c != 'Model'], color='rgba(6,214,160,0.15)'),
                     use_container_width=True)

        # ── F1 Score Bar Race ──
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>F1-Score Comparison</div>', unsafe_allow_html=True)
        pdf_sorted = pdf.sort_values('F1-Score (W)', ascending=True)
        f1_fig = go.Figure(go.Bar(
            x=pdf_sorted['F1-Score (W)'], y=pdf_sorted['Model'], orientation='h',
            marker=dict(
                color=pdf_sorted['F1-Score (W)'],
                colorscale=[[0, '#7c3aed'], [1, '#06d6a0']],
            ),
            text=[f'{v:.4f}' for v in pdf_sorted['F1-Score (W)']],
            textposition='outside', textfont=dict(color='#9ca3af', size=11),
        ))
        f1_fig.update_layout(**styled_layout(
            height=max(250, len(pdf_sorted) * 45),
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#4b5563')),
            yaxis=dict(tickfont=dict(color='#c8d0e0', size=11)),
        ))
        st.plotly_chart(f1_fig, use_container_width=True)

    # ── Feature Importance + Confusion Matrix ──
    ic1, ic2 = st.columns(2, gap="large")

    with ic1:
        fimp = metadata.get('feature_importance', {})
        if fimp:
            st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Feature Importance</div>', unsafe_allow_html=True)
            idf = pd.DataFrame({'Feature': list(fimp.keys()), 'Importance': list(fimp.values())}).sort_values('Importance', ascending=True).tail(15)
            imp_fig = go.Figure(go.Bar(
                x=idf['Importance'], y=idf['Feature'], orientation='h',
                marker=dict(color=idf['Importance'], colorscale=[[0, '#7c3aed'], [1, '#06d6a0']]),
                text=[f'{v:.4f}' for v in idf['Importance']],
                textposition='outside', textfont=dict(color='#9ca3af', size=10),
            ))
            imp_fig.update_layout(**styled_layout(height=400,
                xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#4b5563')),
                yaxis=dict(tickfont=dict(color='#c8d0e0', size=10)),
            ))
            st.plotly_chart(imp_fig, use_container_width=True)

    with ic2:
        cm = metadata.get('confusion_matrix', None)
        if cm is not None:
            st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Confusion Matrix</div>', unsafe_allow_html=True)
            cma = np.array(cm)
            cmn = cma.astype('float') / cma.sum(axis=1, keepdims=True)
            cm_fig = go.Figure(go.Heatmap(
                z=cmn, x=class_names, y=class_names,
                colorscale=[[0, '#05051a'], [0.5, '#7c3aed'], [1, '#06d6a0']],
                text=[[f'{v:.2f}' for v in row] for row in cmn],
                texttemplate='%{text}', textfont=dict(size=11, color='white'),
            ))
            cm_fig.update_layout(**styled_layout(height=400,
                xaxis=dict(title='Predicted', tickfont=dict(color='#c8d0e0')),
                yaxis=dict(title='True', tickfont=dict(color='#c8d0e0'), autorange='reversed'),
            ))
            st.plotly_chart(cm_fig, use_container_width=True)

    # ── Class Distribution ──
    cdist = metadata.get('class_distribution', {})
    if cdist:
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Training Class Distribution</div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame({'Emotion': [class_names[int(k)] for k in cdist.keys()], 'Count': list(cdist.values())})
        dist_colors = [EMOTION_CONFIG.get(e, {}).get('color', '#7c3aed') for e in dist_df['Emotion']]
        dfig = go.Figure(go.Bar(x=dist_df['Emotion'], y=dist_df['Count'],
                                marker=dict(color=dist_colors), text=dist_df['Count'], textposition='outside',
                                textfont=dict(color='#9ca3af')))
        dfig.update_layout(**styled_layout(height=300,
            xaxis=dict(tickfont=dict(color='#c8d0e0')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#4b5563')),
        ))
        st.plotly_chart(dfig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: 🔬 FEATURE LAB
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Feature Engineering Explained</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <p style="color:#9ca3af; margin:0;">The pipeline engineers <strong style="color:#06d6a0;">6 derived features</strong>
        from the raw social media metrics. These capture behavioral patterns that raw counts alone cannot represent.</p>
    </div>
    """, unsafe_allow_html=True)

    eng_info = [
        ("Interaction Density", "(Likes + Comments) / Usage Time", "How intensely a user engages per minute of screen time"),
        ("Social Velocity", "Likes / Posts", "Average likes earned per content piece — measures content quality"),
        ("Conversational Reciprocity", "Messages / Comments", "Ratio of outgoing messages to incoming comments — social balance"),
        ("Attention Index", "Usage Time / Posts", "Time spent per post — passive consumers score high here"),
        ("Engagement Ratio", "(Likes + Comments + Messages) / Usage Time", "Holistic engagement intensity across all interaction types"),
        ("Content Efficiency", "Likes / (Usage Time × Posts)", "Per-post, per-minute effectiveness — content ROI"),
    ]

    for fname, formula, desc in eng_info:
        st.markdown(f"""
        <div class="glass" style="padding:1rem 1.5rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="color:#e0e6ed; font-weight:600; font-size:1rem;">{fname}</div>
                    <div style="color:#6b7280; font-size:0.8rem; margin-top:0.2rem;">{desc}</div>
                </div>
                <div class="stat-tag" style="white-space:nowrap;">{formula}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature Correlation Heatmap ──
    fimp = metadata.get('feature_importance', {})
    if fimp:
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Feature Importance Ranking</div>', unsafe_allow_html=True)

        ranked = sorted(fimp.items(), key=lambda x: x[1], reverse=True)
        rank_html = ""
        for i, (fn, fv) in enumerate(ranked):
            bar_w = (fv / max(fimp.values())) * 100 if max(fimp.values()) > 0 else 0
            rank_html += f"""
            <div style="display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.03);">
                <div style="color:#4b5563; font-size:0.8rem; width:25px; text-align:right;">#{i+1}</div>
                <div style="flex:1;">
                    <div style="color:#c8d0e0; font-size:0.85rem;">{fn}</div>
                    <div style="background:rgba(255,255,255,0.03); border-radius:4px; height:6px; margin-top:4px; overflow:hidden;">
                        <div style="width:{bar_w}%; height:100%; background:linear-gradient(90deg,#7c3aed,#06d6a0); border-radius:4px;"></div>
                    </div>
                </div>
                <div class="stat-tag">{fv:.5f}</div>
            </div>
            """
        st.markdown(f'<div class="glass">{rank_html}</div>', unsafe_allow_html=True)

    # ── Feature List ──
    st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Full Feature Set Used by Model</div>', unsafe_allow_html=True)
    fcols = st.columns(3)
    for i, fn in enumerate(feature_names):
        fcols[i % 3].markdown(f'<span class="stat-tag" style="margin:2px;">{fn}</span>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 5: 🎛️ WHAT-IF SIMULATOR
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>What-If Sensitivity Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <p style="color:#9ca3af; margin:0;">Adjust a <strong style="color:#06d6a0;">single feature</strong>
        while keeping others fixed. Watch how the predicted emotion and confidence change in real-time.
        This reveals which inputs the model is most sensitive to.</p>
    </div>
    """, unsafe_allow_html=True)

    wc1, wc2 = st.columns([1, 2], gap="large")

    with wc1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("##### ⚙️ Fixed Baseline")
        w_age      = st.slider("Age", 10, 90, 25, key="w_age")
        w_gender   = st.selectbox("Gender", GENDERS, key="w_gender")
        w_platform = st.selectbox("Platform", PLATFORMS, key="w_platform")
        w_usage    = st.number_input("Usage Time", 1, 1440, 120, key="w_usage")
        w_posts    = st.number_input("Posts/Day", 0, 100, 3, key="w_posts")
        w_likes    = st.number_input("Likes/Day", 0, 10000, 45, key="w_likes")
        w_comments = st.number_input("Comments/Day", 0, 5000, 10, key="w_comments")
        w_messages = st.number_input("Messages/Day", 0, 5000, 15, key="w_messages")
        st.markdown('</div>', unsafe_allow_html=True)

    with wc2:
        sweep_feature = st.selectbox("Feature to Sweep", [
            "Daily Usage Time", "Posts Per Day", "Likes Received",
            "Comments Received", "Messages Sent", "Age"
        ], key="sweep_feat")

        sweep_map = {
            "Daily Usage Time":   ("usage", 1, 500, 25),
            "Posts Per Day":      ("posts", 0, 50, 3),
            "Likes Received":     ("likes", 0, 500, 25),
            "Comments Received":  ("comments", 0, 200, 10),
            "Messages Sent":      ("messages", 0, 200, 10),
            "Age":                ("age", 10, 80, 5),
        }

        param, lo, hi, step = sweep_map[sweep_feature]
        sweep_vals = list(range(lo, hi + 1, step))

        # Build predictions for each sweep value
        sweep_results = {cn: [] for cn in class_names}
        sweep_preds = []

        for sv in sweep_vals:
            args = dict(age=w_age, gender=w_gender, platform=w_platform,
                        usage=w_usage, posts=w_posts, likes=w_likes,
                        comments=w_comments, messages=w_messages)
            args[param] = sv
            inp_s, _ = build_input(args['age'], args['gender'], args['platform'],
                                   args['usage'], args['posts'], args['likes'],
                                   args['comments'], args['messages'])
            pr = predict_single(inp_s)
            for i, cn in enumerate(class_names):
                sweep_results[cn].append(float(pr[i]))
            sweep_preds.append(class_names[int(np.argmax(pr))])

        # ── Line Chart ──
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Probability Sweep</div>', unsafe_allow_html=True)
        sfig = go.Figure()
        for cn in class_names:
            ec = EMOTION_CONFIG.get(cn, {}).get('color', '#7c3aed')
            sfig.add_trace(go.Scatter(
                x=sweep_vals, y=sweep_results[cn], mode='lines+markers',
                name=cn, line=dict(color=ec, width=2.5), marker=dict(size=5),
            ))
        sfig.update_layout(**styled_layout(height=400,
            xaxis=dict(title=sweep_feature, gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#4b5563')),
            yaxis=dict(title='Probability', range=[0, 1], gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#4b5563')),
            legend=dict(font=dict(color='#9ca3af', size=10)),
        ))
        st.plotly_chart(sfig, use_container_width=True)

        # ── Predicted Emotion Strip ──
        st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>Predicted Emotion at Each Point</div>', unsafe_allow_html=True)
        strip_html = '<div style="display:flex; gap:4px; flex-wrap:wrap; margin-bottom:1rem;">'
        for sv, pe in zip(sweep_vals, sweep_preds):
            ec = EMOTION_CONFIG.get(pe, {}).get('color', '#7c3aed')
            em = EMOTION_CONFIG.get(pe, {}).get('emoji', '🔮')
            strip_html += f'<div style="background:rgba(255,255,255,0.04); border:1px solid {ec}30; border-radius:8px; padding:0.3rem 0.6rem; text-align:center; min-width:55px;"><div style="font-size:1.2rem;">{em}</div><div style="font-size:0.65rem; color:#6b7280;">{sv}</div></div>'
        strip_html += '</div>'
        st.markdown(strip_html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 6: ℹ️ ABOUT
# ────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("""
    <div class="neon-card" style="text-align:left;">
        <div style="text-align:center; margin-bottom:1rem;">
            <div style="font-family:'Space Grotesk'; font-size:2rem; font-weight:700;
                 background: linear-gradient(135deg, #7c3aed, #06d6a0);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🧠 About NeuroSense
            </div>
        </div>
        <p style="color:#9ca3af;">NeuroSense is an advanced behavioral emotion analytics platform that predicts a user's
        dominant emotional state based on their social media engagement patterns. The system leverages
        a hybrid pipeline combining classical machine learning, gradient boosting ensembles, and deep
        neural networks to deliver high-accuracy multi-class classification.</p>
    </div>
    """, unsafe_allow_html=True)

    ab1, ab2 = st.columns(2, gap="large")

    with ab1:
        st.markdown("""
        <div class="glass">
            <div class="sec-h" style="margin-top:0;"><div class="sec-h-glow"></div>⚙️ Data Pipeline</div>
            <ul style="color:#9ca3af; padding-left:1.2rem; font-size:0.9rem; line-height:1.8;">
                <li>Multi-stage anomaly-resilient data cleaning (7 anomaly types handled)</li>
                <li>6 engineered behavioral interaction features</li>
                <li>One-hot categorical encoding with strict schema alignment</li>
                <li>Multicollinearity elimination (r &gt; 0.85 threshold)</li>
                <li>Partition-isolated StandardScaler (fit on train only)</li>
                <li>Target label typo correction (e.g. "Agression" → "Anger")</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass">
            <div class="sec-h" style="margin-top:0;"><div class="sec-h-glow"></div>📊 Evaluation Methods</div>
            <ul style="color:#9ca3af; padding-left:1.2rem; font-size:0.9rem; line-height:1.8;">
                <li>Weighted & Macro F1-Score, Precision, Recall</li>
                <li>One-vs-Rest ROC-AUC & PR-AUC Curves</li>
                <li>Per-class Classification Report</li>
                <li>SHAP TreeExplainer Feature Importance</li>
                <li>Learning Curve Analysis (bias/variance diagnosis)</li>
                <li>Normalized Confusion Matrix</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with ab2:
        st.markdown("""
        <div class="glass">
            <div class="sec-h" style="margin-top:0;"><div class="sec-h-glow"></div>🤖 Models Trained (8 Total)</div>
            <ul style="color:#9ca3af; padding-left:1.2rem; font-size:0.9rem; line-height:1.8;">
                <li>📊 Logistic Regression (multinomial, balanced)</li>
                <li>🌲 Random Forest (300 trees, balanced)</li>
                <li>🚀 CatBoost (RandomizedSearchCV tuned)</li>
                <li>⚡ LightGBM (RandomizedSearchCV tuned)</li>
                <li>🎯 XGBoost (RandomizedSearchCV tuned)</li>
                <li>🧠 MLP (256→128→64, BatchNorm + Dropout)</li>
                <li>🔮 Swish-Net (512→256→128→64, Swish activation)</li>
                <li>🏆 Soft-Vote Ensemble (top-3 averaged)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass">
            <div class="sec-h" style="margin-top:0;"><div class="sec-h-glow"></div>🛠️ Tech Stack</div>
            <table style="width:100%; color:#9ca3af; font-size:0.9rem;">
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">ML/DL</strong></td>
                    <td>scikit-learn, CatBoost, LightGBM, XGBoost, TensorFlow/Keras</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">XAI</strong></td>
                    <td>SHAP (TreeExplainer)</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">Frontend</strong></td>
                    <td>Streamlit, Plotly, Custom CSS</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">Data</strong></td>
                    <td>Pandas, NumPy, SciPy</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">Viz</strong></td>
                    <td>Matplotlib, Seaborn (training pipeline)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture Diagram (Mermaid-style using text) ──
    st.markdown('<div class="sec-h"><div class="sec-h-glow"></div>🏗️ System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass" style="text-align:center; padding:2rem;">
        <div style="display:flex; justify-content:center; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div class="stat-tag" style="padding:0.5rem 1rem; font-size:0.85rem;">📁 Raw CSVs</div>
            <div style="color:#4b5563;">→</div>
            <div class="stat-tag" style="padding:0.5rem 1rem; font-size:0.85rem;">🧹 Cleaning Pipeline</div>
            <div style="color:#4b5563;">→</div>
            <div class="stat-tag" style="padding:0.5rem 1rem; font-size:0.85rem;">⚙️ Feature Engineering</div>
            <div style="color:#4b5563;">→</div>
            <div class="stat-tag" style="padding:0.5rem 1rem; font-size:0.85rem;">📐 Scaling</div>
            <div style="color:#4b5563;">→</div>
            <div class="stat-tag" style="padding:0.5rem 1rem; font-size:0.85rem; background:rgba(6,214,160,0.15); border-color:rgba(6,214,160,0.3); color:#06d6a0;">🏆 Champion Model</div>
            <div style="color:#4b5563;">→</div>
            <div class="stat-tag" style="padding:0.5rem 1rem; font-size:0.85rem;">🔮 Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="app-footer">
    Built with <span class="footer-glow">NeuroSense</span> Analytics Engine &nbsp;•&nbsp;
    Powered by ML & Deep Learning &nbsp;•&nbsp; Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
