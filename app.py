import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_input, Feature_50
from utils.plots import plot_training_history, plot_confusion
from utils.model import show_comparison_table

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("model/cnn_blstm_model_v1.h5")

model = load_trained_model()

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
    }
    .centered-input > div {
        display: flex;
        justify-content: center;
    }
    .highlight-box {
        background-color: #f4f9f9;
        border-left: 5px solid #1abc9c;
        padding: 15px;
        margin: 20px 0;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header & Description ---
st.markdown("<div class='big-font'>🧬 Peptide Classification with CNN-BLSTM</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>This app predicts whether a peptide is <b>Hormonal</b> or <b>Non-Hormonal</b> using a CNN-BLSTM deep learning model.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Main Prediction UI ---
st.markdown("### 🔮 Predict Your Peptide")
st.markdown("<p><b>Enter a peptide sequence of amino acids (e.g., TDIELEIYGMEGIPEK):</b></p>", unsafe_allow_html=True)

with st.container():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        user_seq = st.text_input("🧬 Peptide sequence (AAs only)", placeholder="e.g. TDIELEIYGMEGIPEK", max_chars=50, label_visibility="collapsed")

if user_seq:
    input_vector = preprocess_input(user_seq)
    prediction = model.predict(input_vector)[0][0]
    label = "🟢 Hormonal (Positive)" if prediction > 0.5 else "🔴 Non-Hormonal (Negative)"
    
    st.markdown(f"""
    <div class="highlight-box">
        <b>🔍 Prediction Result:</b> <br>
        <b>{label}</b><br>
        <i>Model confidence:</i> <code>{prediction:.2f}</code>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- Additional Tabs ---
with st.expander("📈 View Model Performance"):
    plot_training_history("model/training_history.json")

with st.expander("📊 Compare with Classical Models"):
    show_comparison_table("model/classical_results.csv")

# with st.expander("🧪 View Confusion Matrix"):
#     plot_confusion()

# --- Footer ---
st.markdown("---")
st.markdown("""
📘 <b>About This Project</b>: This classification system uses a CNN-BLSTM model trained on peptide sequence embeddings to predict hormone classification. Evaluation metrics include <b>AUC</b>, <b>MCC</b>, <b>Accuracy</b>, and <b>F1-score</b>.
""", unsafe_allow_html=True)
