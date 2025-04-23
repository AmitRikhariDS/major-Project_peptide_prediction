# utils/plots.py
import matplotlib.pyplot as plt
import json
import seaborn as sns
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix

# Plot training/validation accuracy and loss
def plot_training_history(history_path):
    with open(history_path) as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()

    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Loss')
    ax2.legend()

    st.pyplot(fig)

# Dummy Confusion Matrix Plot
# Replace with real predictions when model is evaluated

def plot_confusion():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]  # Example ground truth
    y_pred = [0, 1, 0, 1, 0, 0, 1, 1]  # Example predictions

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    st.pyplot()

