# ============================================================================
# BLOCK 25: [Advanced Production Streamlit Dashboard]
# ============================================================================
# A production-grade Streamlit application with dark-mode glassmorphism UI,
# animated gradients, multiple tabs (Predict / Batch / Insights / About),
# Plotly radar charts, SHAP waterfall explanations, and batch CSV processing.
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

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
    page_title="NeuroSense — Behavioral Emotion Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium Dark-Mode Glassmorphism CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0d0d2b 70%, #0a0a1a 100%);
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #6c63ff; border-radius: 10px; }

    /* ── Animated Gradient Header ── */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c63ff, #00d4aa, #6c63ff);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #8892a4;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    @keyframes gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }

    /* ── Emotion Result Card ── */
    .emotion-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        animation: card-appear 0.6s ease-out;
    }
    @keyframes card-appear {
        from { opacity: 0; transform: translateY(20px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    .emotion-emoji {
        font-size: 4.5rem;
        margin-bottom: 0.5rem;
        animation: bounce-in 0.8s ease;
    }
    @keyframes bounce-in {
        0%   { transform: scale(0); }
        60%  { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    .emotion-label {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        color: #8892a4;
        font-weight: 400;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: scale(1.03); }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c63ff, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #c8d0e0;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(108, 99, 255, 0.3);
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #8892a4;
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108, 99, 255, 0.2) !important;
        color: #ffffff !important;
    }

    /* ── Button Styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #00d4aa) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(108, 99, 255, 0.5) !important;
    }

    /* ── Input Styling ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #e0e6ed !important;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.4), transparent);
        margin: 1.5rem 0;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        color: #5a6270;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EMOTION MAPPING
# ============================================================================

EMOTION_CONFIG = {
    'Happiness': {'emoji': '😊', 'color': '#FFD700', 'gradient': 'linear-gradient(135deg, #FFD700, #FFA500)'},
    'Sadness':   {'emoji': '😢', 'color': '#4169E1', 'gradient': 'linear-gradient(135deg, #4169E1, #6495ED)'},
    'Anger':     {'emoji': '😠', 'color': '#FF4444', 'gradient': 'linear-gradient(135deg, #FF4444, #FF6B6B)'},
    'Anxiety':   {'emoji': '😰', 'color': '#FF8C00', 'gradient': 'linear-gradient(135deg, #FF8C00, #FFB347)'},
    'Boredom':   {'emoji': '😐', 'color': '#9E9E9E', 'gradient': 'linear-gradient(135deg, #9E9E9E, #BDBDBD)'},
    'Neutral':   {'emoji': '😶', 'color': '#00CED1', 'gradient': 'linear-gradient(135deg, #00CED1, #48D1CC)'},
}

# ============================================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================================

@st.cache_resource
def load_artifacts():
    """
    Load all serialized pipeline components into Streamlit's cache.

    The pipeline exports a ChampionModelWrapper that always exposes
    .predict_proba() regardless of whether the champion is a single tree
    model, a soft-vote ensemble, or a Keras neural network.
    For Keras champions, the model weights are saved separately in .keras
    format; for everything else, the wrapper itself is the pickle.
    """
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("pipeline_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    deploy_type = metadata.get("model_deployment_type", "")

    if deploy_type == "Keras":
        # ── Lazy TF import: only load TensorFlow when actually needed ──
        # This saves ~800MB of RAM on Streamlit Cloud for tree-based champions.
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
            """Lightweight wrapper to match the ChampionModelWrapper API."""
            def predict_proba(self, X):
                return keras_model.predict(X, verbose=0)
            def predict(self, X):
                return np.argmax(self.predict_proba(X), axis=1)

        model = _KerasWrapper()
    else:
        # Tree or Ensemble — the pickle IS the ChampionModelWrapper
        with open("champion_model.pkl", "rb") as f:
            model = pickle.load(f)

    # Load best tree model for SHAP (if available)
    tree_model = None
    try:
        with open("best_tree_model.pkl", "rb") as f:
            tree_model = pickle.load(f)
    except FileNotFoundError:
        pass

    return model, scaler, encoder, metadata, tree_model


try:
    model, scaler, encoder, metadata, tree_model = load_artifacts()
    feature_names  = metadata["retained_features_list"]
    class_names    = metadata["label_encoder_classes"]
    deploy_type    = metadata["model_deployment_type"]
    n_classes      = metadata["n_classes"]
except Exception as e:
    st.error(f"❌ Failed to load deployment artifacts. Error: {str(e)}")
    st.info("Ensure champion_model.pkl/.keras, scaler.pkl, encoder.pkl, and pipeline_metadata.pkl exist.")
    st.stop()


# ============================================================================
# PREDICTION HELPER
# ============================================================================

def predict_single(input_df_scaled):
    """
    Generate prediction from the loaded model.

    Thanks to the ChampionModelWrapper (or the Keras wrapper), every model
    type exposes the same .predict_proba() interface — no type-sniffing needed.
    """
    probs = model.predict_proba(input_df_scaled)[0]
    return probs


def build_input_dataframe(age, gender, platform, usage_time, posts_per_day,
                          likes_received, comments_received, messages_sent):
    """Construct a fully-engineered, encoded, and aligned input DataFrame."""
    eps = 1e-5

    raw_data = {
        'Age': age,
        'Daily_Usage_Time (minutes)': usage_time,
        'Posts_Per_Day': posts_per_day,
        'Likes_Received_Per_Day': likes_received,
        'Comments_Received_Per_Day': comments_received,
        'Messages_Sent_Per_Day': messages_sent,
        'Interaction_Density': (likes_received + comments_received) / (usage_time + eps),
        'Social_Velocity': likes_received / (posts_per_day + eps),
        'Conversational_Reciprocity': messages_sent / (comments_received + eps),
        'Attention_Index': usage_time / (posts_per_day + eps),
        'Engagement_Ratio': (likes_received + comments_received + messages_sent) / (usage_time + eps),
        'Content_Efficiency': likes_received / (usage_time * posts_per_day + eps),
    }

    input_df = pd.DataFrame(columns=feature_names, data=[np.zeros(len(feature_names))])

    for col, val in raw_data.items():
        if col in input_df.columns:
            input_df[col] = val

    gender_col = f"Gender_{gender}"
    platform_col = f"Platform_{platform}"
    if gender_col in input_df.columns:
        input_df[gender_col] = 1.0
    if platform_col in input_df.columns:
        input_df[platform_col] = 1.0

    return scaler.transform(input_df)


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="hero-header">
    <div class="hero-title">🧠 NeuroSense</div>
    <div class="hero-subtitle">
        Behavioral Emotion Analytics Platform — Powered by Ensemble ML & Deep Learning
    </div>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)


# ============================================================================
# TABS
# ============================================================================

tab_predict, tab_batch, tab_insights, tab_about = st.tabs([
    "🔮 Predict Emotion", "📁 Batch Analysis", "📊 Model Insights", "ℹ️ About"
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1: PREDICT EMOTION
# ────────────────────────────────────────────────────────────────────────────
with tab_predict:

    st.markdown('<div class="section-header">👤 User Profile & Social Metrics</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 👤 Demographics")
        age = st.slider("Age", min_value=10, max_value=90, value=25, step=1,
                         key="pred_age")
        gender = st.selectbox("Gender", options=["Female", "Male", "Non-binary"],
                               key="pred_gender")
        platform = st.selectbox("Primary Platform",
                                 options=["Instagram", "Twitter", "Facebook", "LinkedIn",
                                          "Snapchat", "Telegram", "Whatsapp"],
                                 key="pred_platform")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 📊 Engagement Metrics")
        usage_time        = st.number_input("Daily Usage Time (min)", 1, 1440, 120, key="pred_usage")
        posts_per_day     = st.number_input("Posts Per Day", 0, 100, 3, key="pred_posts")
        likes_received    = st.number_input("Likes Received Per Day", 0, 10000, 45, key="pred_likes")
        comments_received = st.number_input("Comments Received Per Day", 0, 5000, 10, key="pred_comments")
        messages_sent     = st.number_input("Messages Sent Per Day", 0, 5000, 15, key="pred_messages")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    predict_col1, predict_col2, predict_col3 = st.columns([1, 1, 1])
    with predict_col2:
        predict_btn = st.button("🚀  Analyze Behavioral Profile", use_container_width=True,
                                 key="predict_button")

    if predict_btn:
        input_scaled = build_input_dataframe(
            age, gender, platform, usage_time, posts_per_day,
            likes_received, comments_received, messages_sent
        )
        probs = predict_single(input_scaled)
        pred_idx = int(np.argmax(probs))
        pred_emotion = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        emo_cfg = EMOTION_CONFIG.get(pred_emotion, {'emoji': '🔮', 'color': '#6c63ff',
                                                     'gradient': 'linear-gradient(135deg, #6c63ff, #00d4aa)'})

        # ── Result Card ──
        st.markdown(f"""
        <div class="emotion-card" style="border: 2px solid {emo_cfg['color']}40;">
            <div class="emotion-emoji">{emo_cfg['emoji']}</div>
            <div class="emotion-label" style="color: {emo_cfg['color']};">{pred_emotion}</div>
            <div class="confidence-text">Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts: Radar + Bar side by side ──
        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            st.markdown('<div class="section-header">🎯 Probability Radar</div>',
                        unsafe_allow_html=True)
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=list(probs) + [probs[0]],
                theta=class_names + [class_names[0]],
                fill='toself',
                fillcolor='rgba(108, 99, 255, 0.15)',
                line=dict(color='#6c63ff', width=2.5),
                marker=dict(size=6, color='#00d4aa'),
                name='Probability'
            ))
            radar_fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)',
                                    tickfont=dict(color='#8892a4', size=10)),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                                     tickfont=dict(color='#c8d0e0', size=11)),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                margin=dict(l=60, r=60, t=30, b=30),
                height=380,
                showlegend=False,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

        with chart_col2:
            st.markdown('<div class="section-header">📊 Class Probabilities</div>',
                        unsafe_allow_html=True)
            prob_df = pd.DataFrame({
                'Emotion': class_names,
                'Probability': probs
            }).sort_values('Probability', ascending=True)

            bar_colors = [EMOTION_CONFIG.get(e, {}).get('color', '#6c63ff') for e in prob_df['Emotion']]
            bar_fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Emotion'],
                orientation='h',
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f'{p:.1%}' for p in prob_df['Probability']],
                textposition='outside',
                textfont=dict(color='#c8d0e0', size=12),
            ))
            bar_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                xaxis=dict(range=[0, 1], gridcolor='rgba(255,255,255,0.05)',
                           tickformat='.0%', tickfont=dict(color='#8892a4')),
                yaxis=dict(tickfont=dict(color='#c8d0e0', size=12)),
                margin=dict(l=10, r=80, t=30, b=30),
                height=380,
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # ── SHAP Explanation (if available) ──
        if SHAP_AVAILABLE and tree_model is not None:
            st.markdown('<div class="section-header">🔍 SHAP Feature Explanation</div>',
                        unsafe_allow_html=True)
            try:
                input_df_shap = pd.DataFrame(input_scaled, columns=feature_names)
                explainer = shap.TreeExplainer(tree_model)
                shap_vals = explainer(input_df_shap)

                import matplotlib.pyplot as plt

                # Get SHAP values for the predicted class
                if hasattr(shap_vals, 'values') and len(shap_vals.values.shape) == 3:
                    sv = shap.Explanation(
                        values=shap_vals.values[0, :, pred_idx],
                        base_values=shap_vals.base_values[0, pred_idx] if hasattr(shap_vals.base_values, '__len__') and len(np.array(shap_vals.base_values).shape) > 1 else shap_vals.base_values[0],
                        data=shap_vals.data[0],
                        feature_names=feature_names,
                    )
                else:
                    sv = shap_vals[0]

                fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(sv, show=False)
                st.pyplot(plt.gcf())
                plt.close('all')
            except Exception as e:
                st.caption(f"SHAP explanation unavailable: {str(e)[:100]}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 2: BATCH ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="section-header">📁 Batch Prediction — CSV Upload</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <p>Upload a CSV file with the following columns:</p>
        <code>Age, Gender, Platform, Daily_Usage_Time (minutes), Posts_Per_Day,
        Likes_Received_Per_Day, Comments_Received_Per_Day, Messages_Sent_Per_Day</code>
        <p style="color:#8892a4; margin-top:0.5rem;">The system will automatically engineer features,
        encode categoricals, and generate emotion predictions for every row.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Uploaded:** {len(batch_df)} rows × {len(batch_df.columns)} columns")

            with st.expander("📋 Preview uploaded data", expanded=True):
                st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("🚀  Run Batch Predictions", key="batch_predict"):
                results_list = []

                for _, row in batch_df.iterrows():
                    try:
                        input_scaled = build_input_dataframe(
                            age=row.get('Age', 25),
                            gender=row.get('Gender', 'Male'),
                            platform=row.get('Platform', 'Instagram'),
                            usage_time=row.get('Daily_Usage_Time (minutes)', 60),
                            posts_per_day=row.get('Posts_Per_Day', 2),
                            likes_received=row.get('Likes_Received_Per_Day', 20),
                            comments_received=row.get('Comments_Received_Per_Day', 5),
                            messages_sent=row.get('Messages_Sent_Per_Day', 10),
                        )
                        probs = predict_single(input_scaled)
                        pred_idx = int(np.argmax(probs))
                        results_list.append({
                            'Predicted_Emotion': class_names[pred_idx],
                            'Confidence': float(probs[pred_idx]),
                        })
                    except Exception:
                        results_list.append({
                            'Predicted_Emotion': 'Error',
                            'Confidence': 0.0,
                        })

                results_df = pd.concat([batch_df, pd.DataFrame(results_list)], axis=1)

                st.markdown('<div class="section-header">📊 Prediction Results</div>',
                            unsafe_allow_html=True)
                st.dataframe(results_df, use_container_width=True)

                # ── Summary ──
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    emotion_dist = results_df['Predicted_Emotion'].value_counts()
                    dist_fig = px.pie(
                        values=emotion_dist.values,
                        names=emotion_dist.index,
                        color=emotion_dist.index,
                        color_discrete_map={e: cfg['color'] for e, cfg in EMOTION_CONFIG.items()},
                        title="Predicted Emotion Distribution",
                    )
                    dist_fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e6ed'),
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)

                with summary_col2:
                    avg_conf = results_df['Confidence'].mean()
                    st.markdown(f"""
                    <div class="glass-card" style="margin-top: 1rem;">
                        <div class="metric-value">{avg_conf:.1%}</div>
                        <div class="metric-label">Average Confidence</div>
                        <br/>
                        <div class="metric-value">{len(results_df)}</div>
                        <div class="metric-label">Total Predictions</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Download ──
                csv_output = results_df.to_csv(index=False)
                st.download_button(
                    label="⬇️  Download Results CSV",
                    data=csv_output,
                    file_name="emotion_predictions.csv",
                    mime="text/csv",
                    key="download_results"
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL INSIGHTS
# ────────────────────────────────────────────────────────────────────────────
with tab_insights:
    st.markdown('<div class="section-header">📊 Model Performance & Insights</div>',
                unsafe_allow_html=True)

    # ── Champion Card ──
    champion_name = metadata.get('champion_model_name', 'Unknown')
    perf_data = metadata.get('model_performance', [])

    if perf_data:
        champion_perf = perf_data[0]  # sorted by F1
        m_cols = st.columns(4)
        metrics_display = [
            ('Accuracy', champion_perf.get('Accuracy', 0)),
            ('Precision', champion_perf.get('Precision (W)', 0)),
            ('Recall', champion_perf.get('Recall (W)', 0)),
            ('F1-Score', champion_perf.get('F1-Score (W)', 0)),
        ]
        for col, (label, value) in zip(m_cols, metrics_display):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value:.4f}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; margin: 1rem 0; color: #8892a4;">
            🏆 Champion: <strong style="color:#00d4aa;">{champion_name}</strong> |
            Deployment Type: <strong style="color:#6c63ff;">{deploy_type}</strong>
        </div>
        """, unsafe_allow_html=True)

    # ── Model Comparison Table ──
    if perf_data:
        st.markdown('<div class="section-header">📈 All Models — Performance Comparison</div>',
                    unsafe_allow_html=True)
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df.style.format({
            col: '{:.4f}' for col in perf_df.columns if col != 'Model'
        }).highlight_max(subset=[c for c in perf_df.columns if c != 'Model'],
                         color='rgba(0, 212, 170, 0.2)'),
                     use_container_width=True)

    insight_col1, insight_col2 = st.columns(2, gap="large")

    # ── Feature Importance ──
    with insight_col1:
        feat_imp = metadata.get('feature_importance', {})
        if feat_imp:
            st.markdown('<div class="section-header">🎯 Feature Importance</div>',
                        unsafe_allow_html=True)
            imp_df = pd.DataFrame({
                'Feature': list(feat_imp.keys()),
                'Importance': list(feat_imp.values())
            }).sort_values('Importance', ascending=True).tail(15)

            imp_fig = go.Figure(go.Bar(
                x=imp_df['Importance'],
                y=imp_df['Feature'],
                orientation='h',
                marker=dict(
                    color=imp_df['Importance'],
                    colorscale=[[0, '#6c63ff'], [1, '#00d4aa']],
                    line=dict(width=0),
                ),
                text=[f'{v:.4f}' for v in imp_df['Importance']],
                textposition='outside',
                textfont=dict(color='#c8d0e0', size=10),
            ))
            imp_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)',
                           tickfont=dict(color='#8892a4')),
                yaxis=dict(tickfont=dict(color='#c8d0e0', size=10)),
                margin=dict(l=10, r=80, t=10, b=10),
                height=400,
            )
            st.plotly_chart(imp_fig, use_container_width=True)

    # ── Confusion Matrix ──
    with insight_col2:
        cm_data = metadata.get('confusion_matrix', None)
        if cm_data is not None:
            st.markdown('<div class="section-header">🗺️ Confusion Matrix</div>',
                        unsafe_allow_html=True)
            cm_array = np.array(cm_data)
            cm_norm = cm_array.astype('float') / cm_array.sum(axis=1, keepdims=True)

            cm_fig = go.Figure(data=go.Heatmap(
                z=cm_norm,
                x=class_names,
                y=class_names,
                colorscale=[[0, '#0a0a1a'], [0.5, '#6c63ff'], [1, '#00d4aa']],
                text=[[f'{v:.2f}' for v in row] for row in cm_norm],
                texttemplate='%{text}',
                textfont=dict(size=11, color='white'),
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>Value: %{z:.3f}<extra></extra>',
            ))
            cm_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                xaxis=dict(title='Predicted', tickfont=dict(color='#c8d0e0')),
                yaxis=dict(title='True', tickfont=dict(color='#c8d0e0'), autorange='reversed'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=400,
            )
            st.plotly_chart(cm_fig, use_container_width=True)

    # ── Training Class Distribution ──
    class_dist = metadata.get('class_distribution', {})
    if class_dist:
        st.markdown('<div class="section-header">📦 Training Class Distribution</div>',
                    unsafe_allow_html=True)
        dist_df = pd.DataFrame({
            'Emotion': [class_names[int(k)] for k in class_dist.keys()],
            'Count': list(class_dist.values())
        })
        dist_colors = [EMOTION_CONFIG.get(e, {}).get('color', '#6c63ff') for e in dist_df['Emotion']]
        dist_fig = go.Figure(go.Bar(
            x=dist_df['Emotion'], y=dist_df['Count'],
            marker=dict(color=dist_colors, line=dict(width=0)),
            text=dist_df['Count'], textposition='outside',
            textfont=dict(color='#c8d0e0'),
        ))
        dist_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e6ed'),
            xaxis=dict(tickfont=dict(color='#c8d0e0')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#8892a4')),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        st.plotly_chart(dist_fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4: ABOUT
# ────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="border: none; margin-top: 0;">🧠 About NeuroSense</div>
        <p>NeuroSense is an advanced behavioral emotion analytics platform that predicts a user's
        dominant emotional state based on their social media engagement patterns. The system leverages
        a hybrid pipeline combining classical machine learning, gradient boosting ensembles, and deep
        neural networks.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="border: none; margin-top: 0;">⚙️ Methodology</div>
        <p><strong>Data Pipeline:</strong></p>
        <ul>
            <li>Multi-stage anomaly-resilient data cleaning (handles 6 types of CSV artifacts)</li>
            <li>6 engineered behavioral interaction features (Interaction Density, Social Velocity,
                Conversational Reciprocity, Attention Index, Engagement Ratio, Content Efficiency)</li>
            <li>One-hot categorical encoding with strict schema alignment</li>
            <li>Multicollinearity elimination (r > 0.85 threshold)</li>
            <li>Partition-isolated StandardScaler (fit on train only)</li>
        </ul>
        <p><strong>Models Trained:</strong></p>
        <ul>
            <li>📊 Logistic Regression (multinomial, balanced)</li>
            <li>🌲 Random Forest (300 trees, balanced)</li>
            <li>🚀 CatBoost (RandomizedSearchCV tuned)</li>
            <li>⚡ LightGBM (RandomizedSearchCV tuned)</li>
            <li>🎯 XGBoost (RandomizedSearchCV tuned)</li>
            <li>🧠 MLP (256→128→64, BatchNorm + Dropout)</li>
            <li>🔮 Swish-Net (512→256→128→64, Swish activation)</li>
            <li>🏆 Soft-Vote Ensemble (top-3 sklearn models)</li>
        </ul>
        <p><strong>Evaluation:</strong> Weighted & Macro F1-Score, ROC-AUC (OvR), PR-AUC, SHAP explainability</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="section-header" style="border: none; margin-top: 0;">🛠️ Tech Stack</div>
        <table style="width:100%; color:#c8d0e0;">
            <tr><td><strong>ML/DL:</strong></td><td>scikit-learn, CatBoost, LightGBM, XGBoost, TensorFlow/Keras</td></tr>
            <tr><td><strong>Explainability:</strong></td><td>SHAP (TreeExplainer)</td></tr>
            <tr><td><strong>Frontend:</strong></td><td>Streamlit, Plotly</td></tr>
            <tr><td><strong>Data:</strong></td><td>Pandas, NumPy</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="app-footer">
    Built with 🧠 NeuroSense Analytics Engine &nbsp;|&nbsp; Powered by ML & DL &nbsp;|&nbsp;
    Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
