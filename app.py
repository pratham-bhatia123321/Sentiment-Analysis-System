"""
Streamlit Application for Advanced Sentiment Analysis System.

A modern, interactive web interface for real-time sentiment prediction
using 5 trained ML/DL models.

Usage:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODELS_DIR, OUTPUTS_DIR, MAX_SEQUENCE_LENGTH, W2V_VECTOR_SIZE,
    LR_MODEL_PATH, XGB_MODEL_PATH, RF_MODEL_PATH,
    LSTM_MODEL_PATH, BILSTM_MODEL_PATH,
    TFIDF_PATH, TOKENIZER_PATH, W2V_MODEL_PATH,
)
from src.preprocessing import preprocess_text

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Result cards */
    .result-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .result-confidence {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }

    /* Model card */
    .model-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Metrics */
    .metric-box {
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
        background: #f1f3f5;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #495057;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #868e96;
        text-transform: uppercase;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 3rem;
        color: #868e96;
        font-size: 0.85rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML/DL models...")
def load_all_models():
    """Load all trained models and vectorizers. Cached across sessions."""
    models = {}
    try:
        if os.path.exists(LR_MODEL_PATH):
            models['Logistic Regression'] = joblib.load(LR_MODEL_PATH)
        if os.path.exists(XGB_MODEL_PATH):
            models['XGBoost'] = joblib.load(XGB_MODEL_PATH)
        if os.path.exists(RF_MODEL_PATH):
            models['Random Forest'] = joblib.load(RF_MODEL_PATH)

        from tensorflow.keras.models import load_model
        if os.path.exists(LSTM_MODEL_PATH):
            models['LSTM'] = load_model(LSTM_MODEL_PATH)
        if os.path.exists(BILSTM_MODEL_PATH):
            models['BiLSTM'] = load_model(BILSTM_MODEL_PATH)

        vectorizer = joblib.load(TFIDF_PATH) if os.path.exists(TFIDF_PATH) else None
        dl_tokenizer = joblib.load(TOKENIZER_PATH) if os.path.exists(TOKENIZER_PATH) else None

        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load(W2V_MODEL_PATH) if os.path.exists(W2V_MODEL_PATH) else None

        return models, vectorizer, dl_tokenizer, w2v_model
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure models exist in the `models/` directory.")
        st.stop()


def predict_with_model(model, name, cleaned_text, vectorizer,
                       dl_tokenizer, w2v_model):
    """Run prediction through a single model."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if name in ('Logistic Regression', 'XGBoost'):
        features = vectorizer.transform([cleaned_text])
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] >= 0.5)
        confidence = max(proba)

    elif name == 'Random Forest':
        tokens = cleaned_text.split()
        vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
        doc_vec = np.mean(vecs, axis=0) if vecs else np.zeros(W2V_VECTOR_SIZE)
        features = doc_vec.reshape(1, -1)
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] >= 0.5)
        confidence = max(proba)

    elif name in ('LSTM', 'BiLSTM'):
        seq = dl_tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH,
                               padding='post', truncating='post')
        proba_val = model.predict(padded, verbose=0)[0][0]
        prediction = int(proba_val >= 0.5)
        confidence = float(proba_val) if prediction == 1 else float(1 - proba_val)

    else:
        return 0, 0.5

    return prediction, float(confidence)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_options = [
        'All Models',
        'Logistic Regression',
        'XGBoost',
        'Random Forest',
        'LSTM',
        'BiLSTM',
    ]
    selected_model = st.selectbox("Select Model", model_options, index=0)

    st.markdown("---")
    st.markdown("### 📊 Quick Examples")

    example_texts = {
        "Positive Review": "This movie was absolutely fantastic! The acting, "
                           "the plot, everything was perfect. A true masterpiece!",
        "Negative Review": "Terrible movie. Waste of time. The acting was wooden "
                           "and the plot was predictable and boring.",
        "Mixed Review": "The visuals were stunning but the story was weak. "
                        "Some good performances but overall disappointing.",
    }

    for label, text in example_texts.items():
        if st.button(f"Try: {label}", use_container_width=True):
            st.session_state['input_text'] = text

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This system uses **5 trained models** to analyze "
        "sentiment in movie reviews. Built with TF-IDF, Word2Vec, "
        "LSTM, and BiLSTM architectures."
    )


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Sentiment Analysis System</h1>
    <p>Advanced ML & Deep Learning Pipeline for Movie Review Analysis</p>
</div>
""", unsafe_allow_html=True)

# Load models
models, vectorizer, dl_tokenizer, w2v_model = load_all_models()

# Input area
col_input, col_spacer, col_info = st.columns([3, 0.2, 1.5])

with col_input:
    st.markdown("#### ✏️ Enter Your Review")
    default_text = st.session_state.get('input_text', '')
    user_input = st.text_area(
        "Type or paste a movie review below:",
        value=default_text,
        height=150,
        placeholder="e.g., 'This movie was absolutely incredible! The performances were outstanding...'",
        label_visibility="collapsed",
    )

    predict_btn = st.button("🔍 Analyze Sentiment", type="primary",
                            use_container_width=True)

with col_info:
    st.markdown("#### 📈 Model Info")
    st.metric("Models Available", len(models))
    st.metric("Selected", selected_model)
    if user_input:
        st.metric("Input Length", f"{len(user_input)} chars")

# ─── Prediction ──────────────────────────────────────────
if predict_btn and user_input.strip():
    cleaned = preprocess_text(user_input)

    if not cleaned.strip():
        st.warning("After preprocessing, the text is empty. Please provide a longer review.")
    else:
        with st.spinner("Analyzing sentiment..."):
            # Determine which models to use
            if selected_model == 'All Models':
                models_to_run = models
            else:
                models_to_run = {selected_model: models[selected_model]}

            results = {}
            for name, model in models_to_run.items():
                pred, conf = predict_with_model(
                    model, name, cleaned, vectorizer, dl_tokenizer, w2v_model,
                )
                results[name] = {
                    'sentiment': 'Positive' if pred == 1 else 'Negative',
                    'confidence': conf,
                }

        st.markdown("---")

        # ─── Display Primary Result ─────────────────────
        # Use majority vote for "All Models"
        sentiments = [r['sentiment'] for r in results.values()]
        majority = max(set(sentiments), key=sentiments.count)
        avg_confidence = np.mean([r['confidence'] for r in results.values()])

        if majority == 'Positive':
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-label">👍 POSITIVE</div>
                <div class="result-confidence">
                    Average Confidence: {avg_confidence:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-label">👎 NEGATIVE</div>
                <div class="result-confidence">
                    Average Confidence: {avg_confidence:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ─── Per-Model Results ───────────────────────────
        st.markdown("#### 🔬 Per-Model Results")

        cols = st.columns(min(len(results), 3))
        for idx, (name, result) in enumerate(results.items()):
            with cols[idx % len(cols)]:
                sentiment_emoji = "✅" if result['sentiment'] == 'Positive' else "❌"
                st.markdown(f"""
                <div class="model-card">
                    <strong>{name}</strong><br>
                    {sentiment_emoji} {result['sentiment']}<br>
                    Confidence: <strong>{result['confidence']:.1%}</strong>
                </div>
                """, unsafe_allow_html=True)

        # ─── Confidence Bar Chart ────────────────────────
        if len(results) > 1:
            st.markdown("#### 📊 Confidence Comparison")

            chart_data = pd.DataFrame([
                {'Model': name, 'Confidence': r['confidence'],
                 'Sentiment': r['sentiment']}
                for name, r in results.items()
            ])

            colors = ['#2ECC71' if s == 'Positive' else '#E74C3C'
                      for s in chart_data['Sentiment']]

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(chart_data['Model'], chart_data['Confidence'],
                           color=colors, edgecolor='white', height=0.5)

            for bar, conf in zip(bars, chart_data['Confidence']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2.,
                        f'{conf:.1%}', va='center', fontweight='bold', fontsize=11)

            ax.set_xlim(0, 1.15)
            ax.set_xlabel('Confidence', fontsize=12)
            ax.set_title('Model Confidence Scores', fontsize=14, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ─── Preprocessing Info ──────────────────────────
        with st.expander("🔧 Preprocessing Details"):
            st.markdown(f"**Original Text:** {user_input[:200]}...")
            st.markdown(f"**Cleaned Text:** {cleaned[:200]}...")
            st.markdown(f"**Original Length:** {len(user_input)} characters")
            st.markdown(f"**Cleaned Length:** {len(cleaned)} characters")

elif predict_btn and not user_input.strip():
    st.warning("Please enter a movie review to analyze.")


# ═══════════════════════════════════════════════════════════════
# MODEL COMPARISON SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Model Performance Comparison")

results_csv = os.path.join(OUTPUTS_DIR, 'model_results.csv')
if os.path.exists(results_csv):
    perf_df = pd.read_csv(results_csv)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Show comparison chart if available
    comparison_img = os.path.join(OUTPUTS_DIR, 'plots', 'model_comparison.png')
    if os.path.exists(comparison_img):
        st.image(comparison_img, caption="Model Performance Comparison",
                 use_container_width=True)
else:
    st.info("Run `python main.py` first to generate model performance data.")


# ─── Footer ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>Advanced Sentiment Analysis System | Built with Streamlit, TensorFlow, scikit-learn</p>
</div>
""", unsafe_allow_html=True)
