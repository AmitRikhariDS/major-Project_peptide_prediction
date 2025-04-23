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

# UI
st.title("🧬 Peptide Classification with CNN-BLSTM")
st.markdown("This app predicts whether a peptide is **hormonal** or **non-hormonal** using a deep learning model trained on peptide sequences.")



# Visualizations
with st.expander("📈 View Model Performance"):
    plot_training_history("model/training_history.json")

with st.expander("📊 Compare with Classical Models"):
    show_comparison_table("model/classical_results.csv")

with st.expander("🔬 Confusion Matrix"):
    plot_confusion()

# User input
user_seq = st.text_input("Enter peptide sequence (AAs only):", max_chars=50)

if user_seq:
    # Preprocess and predict
    input_vector =preprocess_input(user_seq)
    prediction = model.predict(input_vector)[0][0]
    label = "Hormonal (Positive)" if prediction > 0.5 else "Non-Hormonal (Negative)"
    
    st.success(f"Prediction: **{label}** ({prediction:.2f})")

st.markdown("---")
st.markdown("📘 **About This Project**: This classification system uses a CNN-BLSTM model trained on peptide sequence embeddings to predict hormone classification. Evaluation metrics include AUC, MCC, Accuracy, and F1.")
