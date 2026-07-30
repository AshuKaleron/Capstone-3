# ============================================================================
# BLOCK 25 — NeuroSense v3: Production Streamlit Dashboard
# ============================================================================
# Futuristic dark-mode analytics platform with 6 tabs.
# All critical deployment precautions are preserved:
#   1. ChampionModelWrapper defined BEFORE pickle.load()
#   2. TensorFlow imported lazily (only if champion is Keras)
#   3. Unified .predict_proba() contract — no type-sniffing
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ── Lazy TF — NOT imported at module level ──
TF_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================================
# CHAMPION MODEL WRAPPER  (pickle needs this class in the module namespace)
# ============================================================================

class ChampionModelWrapper:
    """Unified wrapper: always exposes .predict_proba() and .predict()."""

    def __init__(self, model_or_models, deploy_type, model_names=None):
        self.deploy_type = deploy_type
        self.model_names = model_names
        if deploy_type == "Ensemble":
            self._models = model_or_models
        else:
            self._model = model_or_models

    def predict_proba(self, X):
        if self.deploy_type == "Ensemble":
            return np.mean([m.predict_proba(X) for m in self._models], axis=0)
        elif self.deploy_type == "Keras":
            return self._model.predict(X, verbose=0)
        else:
            return self._model.predict_proba(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="NeuroSense — AI Emotion Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CSS — Futuristic Dark Neon Theme
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --accent-1: #7c3aed;
    --accent-2: #06d6a0;
    --accent-3: #3b82f6;
    --bg-dark: #05051a;
    --bg-card: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.06);
    --text-primary: #e0e6ed;
    --text-secondary: #6b7280;
    --text-muted: #4b5563;
}

/* ── Animated Background ── */
.stApp {
    background: linear-gradient(135deg, var(--bg-dark) 0%, #0a0a2e 30%, #10103a 60%, #0d0d30 80%, var(--bg-dark) 100%);
    background-size: 400% 400%;
    animation: drift 25s ease infinite;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}
@keyframes drift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--accent-1), var(--accent-2)); border-radius: 10px; }

/* ── Hero ── */
.hero { text-align: center; padding: 2rem 1rem 0.5rem; }
.hero-badge {
    display: inline-block;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.25);
    color: #a78bfa;
    padding: 0.25rem 1rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2), var(--accent-3), var(--accent-1));
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 6s ease infinite;
    letter-spacing: -2px;
    line-height: 1.1;
}
@keyframes shimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 580px;
    margin: 0.4rem auto 0;
}

/* ── Divider ── */
.ndiv {
    height: 1px;
    background: linear-gradient(90deg, transparent 5%, rgba(124,58,237,0.3) 35%, rgba(6,214,160,0.2) 65%, transparent 95%);
    margin: 1.2rem 0;
}

/* ── Glass Card ── */
.gl {
    background: var(--bg-card);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
}
.gl::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.35), rgba(6,214,160,0.25), transparent);
}
.gl:hover {
    border-color: rgba(124,58,237,0.18);
    box-shadow: 0 8px 35px rgba(124,58,237,0.07);
    transform: translateY(-1px);
}

/* ── Neon Result Card ── */
.neon {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    overflow: hidden;
    animation: neonIn 0.6s ease-out;
}
@keyframes neonIn {
    from { opacity: 0; transform: translateY(25px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.emo-icon { font-size: 5rem; animation: pop 0.8s ease; }
@keyframes pop {
    0%   { transform: scale(0); }
    60%  { transform: scale(1.25); }
    100% { transform: scale(1); }
}
.emo-lbl {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.3rem 0 0.15rem;
}
.emo-conf { font-size: 1rem; color: var(--text-secondary); }

/* ── Metric Cards ── */
.mg { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.7rem; margin: 0.8rem 0; }
.mc {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    transition: all 0.25s;
    position: relative;
    overflow: hidden;
}
.mc::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    opacity: 0;
    transition: opacity 0.3s;
}
.mc:hover { transform: scale(1.02); }
.mc:hover::after { opacity: 1; }
.mv {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ml {
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.25rem;
}

/* ── Section Header ── */
.sh {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #c8d0e0;
    margin: 1.3rem 0 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(124,58,237,0.18);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sh-bar {
    width: 3px;
    height: 18px;
    background: linear-gradient(180deg, var(--accent-1), var(--accent-2));
    border-radius: 2px;
    flex-shrink: 0;
}

/* ── Tag ── */
.tag {
    display: inline-block;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.2);
    color: #a78bfa;
    padding: 0.2rem 0.65rem;
    border-radius: 8px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Feature Row ── */
.fr {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.035);
}
.fn { color: #9ca3af; font-size: 0.82rem; }
.fv {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent-2);
    font-weight: 600;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.85rem;
    padding: 8px 14px;
    transition: all 0.3s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #a78bfa;
    background: rgba(124,58,237,0.06);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(6,214,160,0.08)) !important;
    color: #fff !important;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.2rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 18px rgba(124,58,237,0.22) !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.35) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.07) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}

/* ── Footer ── */
.foot {
    text-align: center;
    padding: 1.8rem 0 0.8rem;
    color: #374151;
    font-size: 0.72rem;
    border-top: 1px solid rgba(255,255,255,0.03);
    margin-top: 2.5rem;
}
.foot b {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# EMOTION CONFIG
# ============================================================================

EMO = {
    "Happiness": {"emoji": "😊", "c": "#fbbf24", "g": "rgba(251,191,36,0.25)"},
    "Sadness":   {"emoji": "😢", "c": "#3b82f6", "g": "rgba(59,130,246,0.25)"},
    "Anger":     {"emoji": "😠", "c": "#ef4444", "g": "rgba(239,68,68,0.25)"},
    "Anxiety":   {"emoji": "😰", "c": "#f97316", "g": "rgba(249,115,22,0.25)"},
    "Boredom":   {"emoji": "😐", "c": "#6b7280", "g": "rgba(107,114,128,0.25)"},
    "Neutral":   {"emoji": "😶", "c": "#06d6a0", "g": "rgba(6,214,160,0.25)"},
}
PLATFORMS = ["Instagram", "Twitter", "Facebook", "LinkedIn", "Snapchat", "Telegram", "Whatsapp"]
GENDERS = ["Female", "Male", "Non-binary"]


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

@st.cache_resource
def load_artifacts():
    """Load pipeline artifacts.  Keras uses lazy TF import."""
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("pipeline_metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    dt = meta.get("model_deployment_type", "")

    if dt == "Keras":
        global TF_AVAILABLE
        try:
            import tensorflow as tf
            TF_AVAILABLE = True
        except ImportError:
            st.error("❌ Champion is Keras but tensorflow is not installed.")
            st.stop()
        km = tf.keras.models.load_model("champion_model.keras")

        class _KW:
            def predict_proba(self, X):
                return km.predict(X, verbose=0)
            def predict(self, X):
                return np.argmax(self.predict_proba(X), axis=1)

        mdl = _KW()
    else:
        with open("champion_model.pkl", "rb") as f:
            mdl = pickle.load(f)

    tm = None
    try:
        with open("best_tree_model.pkl", "rb") as f:
            tm = pickle.load(f)
    except FileNotFoundError:
        pass

    return mdl, scaler, encoder, meta, tm


try:
    model, scaler, encoder, META, tree_model = load_artifacts()
    FEATURES    = list(META.get("retained_features_list", []))
    CLASSES     = list(META.get("label_encoder_classes", []))
    DEPLOY_TYPE = META.get("model_deployment_type", "Unknown")
    N_CLASSES   = META.get("n_classes", len(CLASSES))
except Exception as e:
    st.error(f"❌ Artifact loading failed: {e}")
    st.info("Upload champion_model.pkl/.keras, scaler.pkl, encoder.pkl, pipeline_metadata.pkl.")
    st.stop()


# ============================================================================
# HELPERS
# ============================================================================

EPS = 1e-5


def _predict(X_scaled):
    """Return probability vector (1-D)."""
    return model.predict_proba(X_scaled)[0]


def _build(age, gender, platform, usage, posts, likes, comments, messages):
    """Engineer features → one-hot → scale.  Returns (scaled_array, raw_dict)."""
    raw = {
        "Age": age,
        "Daily_Usage_Time (minutes)": usage,
        "Posts_Per_Day": posts,
        "Likes_Received_Per_Day": likes,
        "Comments_Received_Per_Day": comments,
        "Messages_Sent_Per_Day": messages,
        "Interaction_Density": (likes + comments) / (usage + EPS),
        "Social_Velocity": likes / (posts + EPS),
        "Conversational_Reciprocity": messages / (comments + EPS),
        "Attention_Index": usage / (posts + EPS),
        "Engagement_Ratio": (likes + comments + messages) / (usage + EPS),
        "Content_Efficiency": likes / (usage * posts + EPS),
    }
    row = pd.DataFrame(columns=FEATURES, data=[np.zeros(len(FEATURES))])
    for c, v in raw.items():
        if c in row.columns:
            row[c] = v
    gc = f"Gender_{gender}"
    pc = f"Platform_{platform}"
    if gc in row.columns:
        row[gc] = 1.0
    if pc in row.columns:
        row[pc] = 1.0
    return scaler.transform(row), raw


def _layout(h=380, **kw):
    """Dark Plotly layout defaults."""
    d = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d0e0", family="Inter"),
        margin=dict(l=20, r=20, t=25, b=20),
        height=h,
    )
    d.update(kw)
    return d


def _emo(name):
    """Safe emotion config lookup."""
    return EMO.get(name, {"emoji": "🔮", "c": "#7c3aed", "g": "rgba(124,58,237,0.25)"})


def _sec(title):
    """Render a section header."""
    st.markdown(f'<div class="sh"><div class="sh-bar"></div>{title}</div>', unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI · ML · Deep Learning</div>
    <div class="hero-title">🧠 NeuroSense</div>
    <div class="hero-sub">Predict dominant emotional states from social media behaviour using ensemble ML &amp; neural networks</div>
</div>
<div class="ndiv"></div>
""", unsafe_allow_html=True)


# ============================================================================
# 6  TABS
# ============================================================================

t1, t2, t3, t4, t5, t6 = st.tabs([
    "🔮 Predict", "📁 Batch", "📊 Performance",
    "🔬 Feature Lab", "🎛️ What-If", "ℹ️ About",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════
with t1:
    _sec("👤  User Profile & Engagement Metrics")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="gl">', unsafe_allow_html=True)
        st.markdown("##### Demographics")
        p_age  = st.slider("Age", 10, 90, 25, key="p_age")
        p_gen  = st.selectbox("Gender", GENDERS, key="p_gen")
        p_plat = st.selectbox("Platform", PLATFORMS, key="p_plat")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="gl">', unsafe_allow_html=True)
        st.markdown("##### Daily Engagement")
        p_use = st.number_input("Usage Time (min)", 1, 1440, 120, key="p_use")
        p_pos = st.number_input("Posts", 0, 100, 3, key="p_pos")
        p_lik = st.number_input("Likes Received", 0, 10000, 45, key="p_lik")
        p_com = st.number_input("Comments Received", 0, 5000, 10, key="p_com")
        p_msg = st.number_input("Messages Sent", 0, 5000, 15, key="p_msg")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ndiv"></div>', unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        go_pred = st.button("🚀  Analyze Profile", use_container_width=True, key="go_pred")

    if go_pred:
        xs, raw = _build(p_age, p_gen, p_plat, p_use, p_pos, p_lik, p_com, p_msg)
        probs   = _predict(xs)
        idx     = int(np.argmax(probs))
        emo_n   = CLASSES[idx]
        conf    = float(probs[idx])
        ec      = _emo(emo_n)

        # ── result card ──
        st.markdown(f"""
        <div class="neon" style="border-color:{ec['c']}40; box-shadow:0 0 50px {ec['g']};">
            <div class="emo-icon">{ec['emoji']}</div>
            <div class="emo-lbl" style="color:{ec['c']};">{emo_n}</div>
            <div class="emo-conf">Confidence: <strong>{conf:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        # ── radar + bar ──
        r1, r2 = st.columns(2, gap="large")
        with r1:
            _sec("🎯 Probability Radar")
            rfig = go.Figure(go.Scatterpolar(
                r=list(probs) + [probs[0]],
                theta=CLASSES + [CLASSES[0]],
                fill="toself",
                fillcolor="rgba(124,58,237,0.1)",
                line=dict(color="#7c3aed", width=2.5),
                marker=dict(size=6, color="#06d6a0"),
            ))
            rfig.update_layout(**_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(size=9, color="#4b5563")),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                                     tickfont=dict(size=10, color="#9ca3af")),
                ), showlegend=False,
            ))
            st.plotly_chart(rfig, use_container_width=True)

        with r2:
            _sec("📊 Class Probabilities")
            bdf = pd.DataFrame({"Emotion": CLASSES, "P": probs}).sort_values("P")
            bcolors = [_emo(e)["c"] for e in bdf["Emotion"]]
            bfig = go.Figure(go.Bar(
                x=bdf["P"], y=bdf["Emotion"], orientation="h",
                marker=dict(color=bcolors),
                text=[f"{p:.1%}" for p in bdf["P"]],
                textposition="outside",
                textfont=dict(color="#9ca3af", size=11),
            ))
            bfig.update_layout(**_layout(
                xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.03)",
                           tickformat=".0%", tickfont=dict(color="#4b5563")),
                yaxis=dict(tickfont=dict(color="#c8d0e0", size=11)),
            ))
            st.plotly_chart(bfig, use_container_width=True)

        # ── gauge ──
        _sec("⚡ Confidence Gauge")
        gfig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf * 100,
            number=dict(suffix="%", font=dict(size=38, color="#e0e6ed")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#4b5563"),
                bar=dict(color=ec["c"]),
                bgcolor="rgba(255,255,255,0.02)",
                bordercolor="rgba(255,255,255,0.05)",
                steps=[
                    dict(range=[0, 33],  color="rgba(239,68,68,0.08)"),
                    dict(range=[33, 66], color="rgba(249,115,22,0.08)"),
                    dict(range=[66, 100], color="rgba(6,214,160,0.08)"),
                ],
                threshold=dict(line=dict(color="#06d6a0", width=3),
                               thickness=0.8, value=conf * 100),
            ),
        ))
        gfig.update_layout(**_layout(h=240))
        st.plotly_chart(gfig, use_container_width=True)

        # ── engineered features ──
        _sec("🧬 Engineered Features")
        eng_names = [
            ("Interaction Density", "Interaction_Density"),
            ("Social Velocity", "Social_Velocity"),
            ("Conversational Reciprocity", "Conversational_Reciprocity"),
            ("Attention Index", "Attention_Index"),
            ("Engagement Ratio", "Engagement_Ratio"),
            ("Content Efficiency", "Content_Efficiency"),
        ]
        fhtml = ""
        for label, key in eng_names:
            fhtml += f'<div class="fr"><span class="fn">{label}</span><span class="fv">{raw.get(key, 0):.4f}</span></div>'
        st.markdown(f'<div class="gl">{fhtml}</div>', unsafe_allow_html=True)

        # ── SHAP ──
        if SHAP_AVAILABLE and tree_model is not None:
            _sec("🔍 SHAP Feature Explanation")
            try:
                import matplotlib.pyplot as plt
                idf = pd.DataFrame(xs, columns=FEATURES)
                expl = shap.TreeExplainer(tree_model)
                sv   = expl(idf)
                if hasattr(sv, "values") and sv.values.ndim == 3:
                    bv = sv.base_values[0]
                    if isinstance(bv, (list, np.ndarray)):
                        bv = bv[idx]
                    sv0 = shap.Explanation(
                        values=sv.values[0, :, idx],
                        base_values=bv,
                        data=sv.data[0],
                        feature_names=FEATURES,
                    )
                else:
                    sv0 = sv[0]
                shap.plots.waterfall(sv0, show=False)
                st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as ex:
                st.caption(f"SHAP unavailable: {str(ex)[:120]}")

        # ── all-class sunburst ──
        _sec("🌐 Probability Sunburst")
        sb_df = pd.DataFrame({"Emotion": CLASSES, "Probability": probs})
        sb_fig = px.sunburst(
            sb_df, path=["Emotion"], values="Probability",
            color="Probability",
            color_continuous_scale=[[0, "#7c3aed"], [1, "#06d6a0"]],
        )
        sb_fig.update_layout(**_layout(h=350))
        sb_fig.update_traces(textfont=dict(color="white"))
        st.plotly_chart(sb_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with t2:
    _sec("📁 Batch Prediction — CSV Upload")

    st.markdown("""
    <div class="gl">
        <p style="margin:0; color:#9ca3af;">Upload a CSV with columns:
        <code style="color:#06d6a0;">Age, Gender, Platform, Daily_Usage_Time (minutes),
        Posts_Per_Day, Likes_Received_Per_Day, Comments_Received_Per_Day,
        Messages_Sent_Per_Day</code></p>
    </div>
    """, unsafe_allow_html=True)

    up = st.file_uploader("Choose CSV", type=["csv"], key="batch_up")

    if up is not None:
        try:
            bdf = pd.read_csv(up)
            st.markdown(
                f'<span class="tag">{len(bdf)} rows × {len(bdf.columns)} cols</span>',
                unsafe_allow_html=True,
            )
            with st.expander("📋 Preview", expanded=True):
                st.dataframe(bdf.head(10), use_container_width=True)

            if st.button("🚀  Run Batch", key="batch_go"):
                res = []
                bar = st.progress(0, text="Predicting…")
                for i, (_, row) in enumerate(bdf.iterrows()):
                    try:
                        xs, _ = _build(
                            row.get("Age", 25),
                            row.get("Gender", "Male"),
                            row.get("Platform", "Instagram"),
                            row.get("Daily_Usage_Time (minutes)", 60),
                            row.get("Posts_Per_Day", 2),
                            row.get("Likes_Received_Per_Day", 20),
                            row.get("Comments_Received_Per_Day", 5),
                            row.get("Messages_Sent_Per_Day", 10),
                        )
                        pr = _predict(xs)
                        pi = int(np.argmax(pr))
                        res.append({"Predicted_Emotion": CLASSES[pi],
                                    "Confidence": float(pr[pi])})
                    except Exception:
                        res.append({"Predicted_Emotion": "Error", "Confidence": 0.0})
                    bar.progress((i + 1) / len(bdf))
                bar.empty()

                rdf = pd.concat([bdf, pd.DataFrame(res)], axis=1)
                _sec("📊 Results")
                st.dataframe(rdf, use_container_width=True)

                sc1, sc2 = st.columns(2)
                with sc1:
                    ed = rdf["Predicted_Emotion"].value_counts()
                    pfig = px.pie(
                        values=ed.values, names=ed.index, color=ed.index,
                        color_discrete_map={e: v["c"] for e, v in EMO.items()},
                        title="Distribution",
                    )
                    pfig.update_layout(**_layout(h=340))
                    st.plotly_chart(pfig, use_container_width=True)
                with sc2:
                    ac = rdf["Confidence"].mean()
                    top_emo = ed.index[0] if len(ed) > 0 else "N/A"
                    st.markdown(f"""
                    <div class="gl" style="text-align:center; padding:2rem;">
                        <div class="mv">{ac:.1%}</div><div class="ml">Avg Confidence</div>
                        <div style="margin-top:1.2rem;"><div class="mv">{len(rdf)}</div><div class="ml">Total Rows</div></div>
                        <div style="margin-top:1.2rem;"><div class="mv">{top_emo}</div><div class="ml">Most Frequent</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                csv_out = rdf.to_csv(index=False)
                st.download_button("⬇️  Download CSV", csv_out,
                                   "neurosense_predictions.csv", "text/csv")

        except Exception as ex:
            st.error(f"Error: {ex}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with t3:
    champ = META.get("champion_model_name", "Unknown")
    perf  = META.get("model_performance", [])

    # champion banner
    st.markdown(f"""
    <div class="neon">
        <div style="font-size:0.65rem; color:#6b7280; text-transform:uppercase; letter-spacing:2.5px;">Champion</div>
        <div style="font-family:'Space Grotesk'; font-size:2rem; font-weight:700;
             background:linear-gradient(135deg,#7c3aed,#06d6a0);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0.3rem 0;">
            🏆 {champ}
        </div>
        <span class="tag">{DEPLOY_TYPE}</span>
    </div>
    """, unsafe_allow_html=True)

    # metric cards
    if perf:
        cp = perf[0]
        keys = [
            ("Accuracy", "Accuracy"), ("Precision (W)", "Precision W"),
            ("Recall (W)", "Recall W"), ("F1-Score (W)", "F1 Weighted"),
            ("Precision (Macro)", "Precision M"), ("F1-Score (Macro)", "F1 Macro"),
        ]
        cards = ""
        for k, label in keys:
            v = cp.get(k, 0)
            cards += f'<div class="mc"><div class="mv">{v:.4f}</div><div class="ml">{label}</div></div>'
        st.markdown(f'<div class="mg">{cards}</div>', unsafe_allow_html=True)

    # all models table
    if perf:
        _sec("📈 All Models Comparison")
        pdf = pd.DataFrame(perf)
        num_cols = [c for c in pdf.columns if c != "Model"]
        st.dataframe(
            pdf.style
            .format({c: "{:.4f}" for c in num_cols})
            .highlight_max(subset=num_cols, color="rgba(6,214,160,0.12)"),
            use_container_width=True,
        )

        # F1 bar
        _sec("📊 F1-Score (Weighted) Ranking")
        f1col = "F1-Score (W)"
        if f1col in pdf.columns:
            ps = pdf.sort_values(f1col, ascending=True)
            ff = go.Figure(go.Bar(
                x=ps[f1col], y=ps["Model"], orientation="h",
                marker=dict(color=ps[f1col],
                            colorscale=[[0, "#7c3aed"], [1, "#06d6a0"]]),
                text=[f"{v:.4f}" for v in ps[f1col]],
                textposition="outside",
                textfont=dict(color="#9ca3af", size=11),
            ))
            ff.update_layout(**_layout(
                h=max(250, len(ps) * 42),
                xaxis=dict(gridcolor="rgba(255,255,255,0.03)",
                           tickfont=dict(color="#4b5563")),
                yaxis=dict(tickfont=dict(color="#c8d0e0", size=11)),
            ))
            st.plotly_chart(ff, use_container_width=True)

    # feature importance + confusion matrix
    ic1, ic2 = st.columns(2, gap="large")

    fimp = META.get("feature_importance", {})
    with ic1:
        if fimp:
            _sec("🎯 Feature Importance")
            idf = (pd.DataFrame({"F": list(fimp.keys()), "I": list(fimp.values())})
                     .sort_values("I", ascending=True).tail(15))
            ifig = go.Figure(go.Bar(
                x=idf["I"], y=idf["F"], orientation="h",
                marker=dict(color=idf["I"],
                            colorscale=[[0, "#7c3aed"], [1, "#06d6a0"]]),
                text=[f"{v:.4f}" for v in idf["I"]],
                textposition="outside",
                textfont=dict(color="#9ca3af", size=10),
            ))
            ifig.update_layout(**_layout(h=400,
                xaxis=dict(gridcolor="rgba(255,255,255,0.03)",
                           tickfont=dict(color="#4b5563")),
                yaxis=dict(tickfont=dict(color="#c8d0e0", size=10)),
            ))
            st.plotly_chart(ifig, use_container_width=True)

    cm_raw = META.get("confusion_matrix")
    with ic2:
        if cm_raw is not None:
            _sec("🗺️ Confusion Matrix")
            cma = np.array(cm_raw, dtype=float)
            row_sums = cma.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1          # prevent /0
            cmn = cma / row_sums
            cfig = go.Figure(go.Heatmap(
                z=cmn, x=CLASSES, y=CLASSES,
                colorscale=[[0, "#05051a"], [0.5, "#7c3aed"], [1, "#06d6a0"]],
                text=[[f"{v:.2f}" for v in r] for r in cmn],
                texttemplate="%{text}",
                textfont=dict(size=11, color="white"),
            ))
            cfig.update_layout(**_layout(h=400,
                xaxis=dict(title="Predicted", tickfont=dict(color="#c8d0e0")),
                yaxis=dict(title="True", tickfont=dict(color="#c8d0e0"),
                           autorange="reversed"),
            ))
            st.plotly_chart(cfig, use_container_width=True)

    # class distribution
    cdist = META.get("class_distribution", {})
    if cdist:
        _sec("📦 Training Class Distribution")
        try:
            dist_labels = [CLASSES[int(k)] if int(k) < len(CLASSES) else str(k)
                           for k in cdist.keys()]
        except (ValueError, IndexError):
            dist_labels = list(cdist.keys())
        dist_vals = list(cdist.values())
        dc = [_emo(l)["c"] for l in dist_labels]
        dfig = go.Figure(go.Bar(
            x=dist_labels, y=dist_vals,
            marker=dict(color=dc),
            text=dist_vals, textposition="outside",
            textfont=dict(color="#9ca3af"),
        ))
        dfig.update_layout(**_layout(h=280,
            xaxis=dict(tickfont=dict(color="#c8d0e0")),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)",
                       tickfont=dict(color="#4b5563")),
        ))
        st.plotly_chart(dfig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE LAB
# ═══════════════════════════════════════════════════════════════════════════
with t4:
    _sec("🧬 Engineered Features — Explained")
    st.markdown("""
    <div class="gl"><p style="color:#9ca3af; margin:0;">
    The pipeline derives <strong style="color:#06d6a0;">6 behavioural interaction features</strong>
    from raw social media metrics.  These capture engagement patterns that raw counts alone cannot express.
    </p></div>
    """, unsafe_allow_html=True)

    eng_cards = [
        ("Interaction Density", "(Likes + Comments) / Usage", "Engagement intensity per minute of screen time"),
        ("Social Velocity", "Likes / Posts", "Average likes per content piece — content quality signal"),
        ("Conversational Reciprocity", "Messages / Comments", "Outgoing-to-incoming ratio — social balance"),
        ("Attention Index", "Usage / Posts", "Time per post — passive consumers score high"),
        ("Engagement Ratio", "(L + C + M) / Usage", "Holistic engagement across all interaction types"),
        ("Content Efficiency", "Likes / (Usage × Posts)", "Per-post, per-minute effectiveness — content ROI"),
    ]
    for name, formula, desc in eng_cards:
        st.markdown(f"""
        <div class="gl" style="padding:1rem 1.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div style="color:#e0e6ed; font-weight:600;">{name}</div>
                    <div style="color:#6b7280; font-size:0.78rem; margin-top:0.15rem;">{desc}</div>
                </div>
                <span class="tag" style="white-space:nowrap;">{formula}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # importance ranking
    if fimp:
        _sec("🏅 Feature Importance Ranking")
        ranked = sorted(fimp.items(), key=lambda x: x[1], reverse=True)
        max_val = ranked[0][1] if ranked else 1
        rhtml = ""
        for i, (fn, fv) in enumerate(ranked):
            pct = (fv / max_val * 100) if max_val > 0 else 0
            rhtml += f"""
            <div style="display:flex; align-items:center; gap:0.7rem; padding:0.45rem 0; border-bottom:1px solid rgba(255,255,255,0.03);">
                <div style="color:#4b5563; font-size:0.75rem; width:22px; text-align:right;">#{i+1}</div>
                <div style="flex:1;">
                    <div style="color:#c8d0e0; font-size:0.82rem;">{fn}</div>
                    <div style="background:rgba(255,255,255,0.03); border-radius:3px; height:5px; margin-top:3px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%; background:linear-gradient(90deg,#7c3aed,#06d6a0); border-radius:3px;"></div>
                    </div>
                </div>
                <span class="tag">{fv:.5f}</span>
            </div>"""
        st.markdown(f'<div class="gl">{rhtml}</div>', unsafe_allow_html=True)

    # feature set tags
    _sec("📋 Full Feature Set")
    cols = st.columns(3)
    for i, fn in enumerate(FEATURES):
        cols[i % 3].markdown(f'<span class="tag" style="margin:2px;">{fn}</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════
with t5:
    _sec("🎛️ What-If Sensitivity Simulator")
    st.markdown("""
    <div class="gl"><p style="color:#9ca3af; margin:0;">
    Adjust a <strong style="color:#06d6a0;">single feature</strong> while keeping others fixed.
    Watch how predicted probabilities shift — revealing which inputs the model is most sensitive to.
    </p></div>
    """, unsafe_allow_html=True)

    wl, wr = st.columns([1, 2], gap="large")
    with wl:
        st.markdown('<div class="gl">', unsafe_allow_html=True)
        st.markdown("##### ⚙️ Baseline")
        w_age  = st.slider("Age", 10, 90, 25, key="w_age")
        w_gen  = st.selectbox("Gender", GENDERS, key="w_gen")
        w_plat = st.selectbox("Platform", PLATFORMS, key="w_plat")
        w_use  = st.number_input("Usage (min)", 1, 1440, 120, key="w_use")
        w_pos  = st.number_input("Posts", 0, 100, 3, key="w_pos")
        w_lik  = st.number_input("Likes", 0, 10000, 45, key="w_lik")
        w_com  = st.number_input("Comments", 0, 5000, 10, key="w_com")
        w_msg  = st.number_input("Messages", 0, 5000, 15, key="w_msg")
        st.markdown("</div>", unsafe_allow_html=True)

    with wr:
        sweep_feat = st.selectbox("Feature to Sweep", [
            "Daily Usage Time", "Posts Per Day", "Likes Received",
            "Comments Received", "Messages Sent", "Age",
        ], key="sw_feat")

        sweep_cfg = {
            "Daily Usage Time":  ("usage", 1,  500, 25),
            "Posts Per Day":     ("posts", 0,   50, 3),
            "Likes Received":    ("likes", 0,  500, 25),
            "Comments Received": ("comments", 0, 200, 10),
            "Messages Sent":     ("messages", 0, 200, 10),
            "Age":               ("age",  10,  80, 5),
        }
        param, lo, hi, step = sweep_cfg[sweep_feat]
        vals = list(range(lo, hi + 1, step))

        traces = {cn: [] for cn in CLASSES}
        preds  = []

        base = dict(age=w_age, gender=w_gen, platform=w_plat,
                    usage=w_use, posts=w_pos, likes=w_lik,
                    comments=w_com, messages=w_msg)

        for v in vals:
            a = base.copy()
            a[param] = v
            xs, _ = _build(a["age"], a["gender"], a["platform"],
                           a["usage"], a["posts"], a["likes"],
                           a["comments"], a["messages"])
            pr = _predict(xs)
            for ci, cn in enumerate(CLASSES):
                traces[cn].append(float(pr[ci]))
            preds.append(CLASSES[int(np.argmax(pr))])

        # line chart
        _sec("📈 Probability Sweep")
        lfig = go.Figure()
        for cn in CLASSES:
            lfig.add_trace(go.Scatter(
                x=vals, y=traces[cn], mode="lines+markers",
                name=cn, line=dict(color=_emo(cn)["c"], width=2.5),
                marker=dict(size=5),
            ))
        lfig.update_layout(**_layout(h=400,
            xaxis=dict(title=sweep_feat,
                       gridcolor="rgba(255,255,255,0.03)",
                       tickfont=dict(color="#4b5563")),
            yaxis=dict(title="Probability", range=[0, 1],
                       gridcolor="rgba(255,255,255,0.03)",
                       tickfont=dict(color="#4b5563")),
            legend=dict(font=dict(color="#9ca3af", size=10)),
        ))
        st.plotly_chart(lfig, use_container_width=True)

        # emoji strip
        _sec("🔮 Predicted Emotion at Each Point")
        strip = '<div style="display:flex; gap:3px; flex-wrap:wrap;">'
        for sv, pe in zip(vals, preds):
            ec = _emo(pe)
            strip += (
                f'<div style="background:rgba(255,255,255,0.03); border:1px solid {ec["c"]}25;'
                f'border-radius:8px; padding:0.25rem 0.5rem; text-align:center; min-width:50px;">'
                f'<div style="font-size:1.1rem;">{ec["emoji"]}</div>'
                f'<div style="font-size:0.6rem; color:#6b7280;">{sv}</div></div>'
            )
        strip += "</div>"
        st.markdown(strip, unsafe_allow_html=True)

        # sensitivity summary
        _sec("📐 Sensitivity Summary")
        changes = {}
        for cn in CLASSES:
            arr = traces[cn]
            changes[cn] = max(arr) - min(arr) if arr else 0
        sens_sorted = sorted(changes.items(), key=lambda x: x[1], reverse=True)
        shtml = ""
        for cn, delta in sens_sorted:
            ec = _emo(cn)
            pct = delta * 100
            shtml += f"""
            <div class="fr">
                <span class="fn">{ec['emoji']} {cn}</span>
                <span class="fv" style="color:{ec['c']};">Δ {pct:.1f}%</span>
            </div>"""
        st.markdown(f'<div class="gl">{shtml}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
with t6:
    st.markdown(f"""
    <div class="neon" style="text-align:left;">
        <div style="text-align:center; margin-bottom:1rem;">
            <div style="font-family:'Space Grotesk'; font-size:2rem; font-weight:700;
                 background:linear-gradient(135deg,#7c3aed,#06d6a0);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🧠 About NeuroSense
            </div>
        </div>
        <p style="color:#9ca3af;">NeuroSense predicts a user's dominant emotional state from social
        media engagement patterns.  It combines classical ML, gradient boosting ensembles, and deep
        neural networks in a fully automated training pipeline with production-grade deployment.</p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("""
        <div class="gl">
            <div class="sh" style="margin-top:0;"><div class="sh-bar"></div>⚙️ Data Pipeline</div>
            <ul style="color:#9ca3af; padding-left:1.1rem; font-size:0.85rem; line-height:1.9;">
                <li>Multi-stage anomaly-resilient cleaning (7 anomaly types)</li>
                <li>6 engineered behavioural features</li>
                <li>One-hot encoding + strict schema alignment</li>
                <li>Multicollinearity filter (r &gt; 0.85)</li>
                <li>Train-only StandardScaler (leak-proof)</li>
                <li>Label typo correction ("Agression" → "Anger")</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="gl">
            <div class="sh" style="margin-top:0;"><div class="sh-bar"></div>📊 Evaluation</div>
            <ul style="color:#9ca3af; padding-left:1.1rem; font-size:0.85rem; line-height:1.9;">
                <li>Weighted &amp; Macro F1, Precision, Recall</li>
                <li>OvR ROC-AUC &amp; PR-AUC Curves</li>
                <li>Per-class Classification Report</li>
                <li>SHAP TreeExplainer Importance</li>
                <li>Learning Curve Diagnostics</li>
                <li>Normalized Confusion Matrix</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="gl">
            <div class="sh" style="margin-top:0;"><div class="sh-bar"></div>🤖 Models (8 Total)</div>
            <ul style="color:#9ca3af; padding-left:1.1rem; font-size:0.85rem; line-height:1.9;">
                <li>📊 Logistic Regression (multinomial, balanced)</li>
                <li>🌲 Random Forest (300 trees, balanced)</li>
                <li>🚀 CatBoost (RandomizedSearchCV, 20 iter)</li>
                <li>⚡ LightGBM (RandomizedSearchCV, 20 iter)</li>
                <li>🎯 XGBoost (RandomizedSearchCV, 20 iter)</li>
                <li>🧠 MLP (256→128→64, BN + Dropout)</li>
                <li>🔮 Swish-Net (512→256→128→64, Swish)</li>
                <li>🏆 Soft-Vote Ensemble (top-3 avg)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="gl">
            <div class="sh" style="margin-top:0;"><div class="sh-bar"></div>🛠️ Tech Stack</div>
            <table style="width:100%; color:#9ca3af; font-size:0.85rem;">
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">ML / DL</strong></td>
                    <td>scikit-learn · CatBoost · LightGBM · XGBoost · TF/Keras</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">XAI</strong></td>
                    <td>SHAP (TreeExplainer)</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">Frontend</strong></td>
                    <td>Streamlit · Plotly · Custom CSS</td></tr>
                <tr><td style="padding:0.3rem 0;"><strong style="color:#c8d0e0;">Data</strong></td>
                    <td>Pandas · NumPy · SciPy</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # architecture flow
    _sec("🏗️ System Architecture")
    st.markdown("""
    <div class="gl" style="text-align:center; padding:1.8rem;">
        <div style="display:flex; justify-content:center; align-items:center; gap:0.8rem; flex-wrap:wrap;">
            <span class="tag" style="padding:0.4rem 0.9rem; font-size:0.8rem;">📁 Raw CSVs</span>
            <span style="color:#4b5563;">→</span>
            <span class="tag" style="padding:0.4rem 0.9rem; font-size:0.8rem;">🧹 Cleaning</span>
            <span style="color:#4b5563;">→</span>
            <span class="tag" style="padding:0.4rem 0.9rem; font-size:0.8rem;">⚙️ Feature Eng</span>
            <span style="color:#4b5563;">→</span>
            <span class="tag" style="padding:0.4rem 0.9rem; font-size:0.8rem;">📐 Scaling</span>
            <span style="color:#4b5563;">→</span>
            <span class="tag" style="padding:0.4rem 0.9rem; font-size:0.8rem; background:rgba(6,214,160,0.12); border-color:rgba(6,214,160,0.25); color:#06d6a0;">🏆 Champion</span>
            <span style="color:#4b5563;">→</span>
            <span class="tag" style="padding:0.4rem 0.9rem; font-size:0.8rem;">🔮 Predict</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # deployment info
    _sec("🚀 Deployment Precautions")
    st.markdown(f"""
    <div class="gl">
        <div class="fr"><span class="fn">Champion Model</span><span class="fv">{champ}</span></div>
        <div class="fr"><span class="fn">Deployment Type</span><span class="fv">{DEPLOY_TYPE}</span></div>
        <div class="fr"><span class="fn">Pickle Contract</span><span class="fv">ChampionModelWrapper</span></div>
        <div class="fr"><span class="fn">TF Strategy</span><span class="fv">Lazy (only if Keras wins)</span></div>
        <div class="fr"><span class="fn">Python Version</span><span class="fv">3.10 (.python-version)</span></div>
        <div class="fr"><span class="fn">TF Package</span><span class="fv">tensorflow-cpu (memory-safe)</span></div>
        <div class="fr" style="border:none;"><span class="fn">Features</span><span class="fv">{len(FEATURES)} retained</span></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="foot">
    Built with <b>NeuroSense</b> Analytics Engine &nbsp;•&nbsp;
    ML &amp; Deep Learning &nbsp;•&nbsp; Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
