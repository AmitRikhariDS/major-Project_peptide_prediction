# 🧬 Peptide Classification with CNN-BLSTM

This project is a deep learning-based web application that classifies peptides as **hormonal** or **non-hormonal** using a hybrid **CNN-BLSTM** model. Built with **TensorFlow/Keras** and deployed using **Streamlit**, this app allows real-time predictions from user-input peptide sequences.
**Webapp link**: https://predictpeptide.streamlit.app/

---

## 🚀 Features

- ✅ Predict peptide classification from user input
- ✅ Load and use trained CNN-BLSTM model
- ✅ View training & validation performance curves
- ✅ Visualize confusion matrix
- ✅ Compare deep learning performance with classical models
- ✅ Lightweight, interactive web UI powered by Streamlit

---

## 🧠 Model Architecture

The model architecture includes:
- Embedding layer for sequence learning
- CNN layers to capture local patterns
- Bidirectional LSTMs for sequential context
- Dense layers with dropout for classification

---

## 📊 Classical Model Comparison

We evaluated and benchmarked against:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- K-Nearest Neighbors
- XGBoost

> 📈 Best Accuracy: **0.86** using hybrid model CNN-BLSTM
---

## 📁 File Structure

```
├── app.py                    # Streamlit app
├── model/
│   ├── cnn_blstm_model.h5    # Trained model
│   ├── training_history.json # Metrics history
│   └── classical_results.csv # Performance table
├── data/
│   └── Feature_50.pkl        # Amino acid embedding dict
├── utils/
│   ├── preprocess.py         # Preprocess function
│   ├── plots.py              # Graph & matrix plotting
│   └── compare_models.py     # Table viewer
└── requirements.txt
```

---

## ▶️ How to Run

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

```

---

## 🧬 Sample Input

```
Peptide Sequence: TDIELEIYGMEGIPEK
```

> 🟢 Output: **Hormonal (0.84 confidence)**

---

## 📌 About

This project demonstrates the power of deep learning in bioinformatics for classification tasks using sequence data. CNN captures local motifs while BLSTM encodes order and context.

---

## 📷 Screenshots

<img width="592" alt="image" src="https://github.com/user-attachments/assets/9d291592-4a62-47ab-871c-27ee1536321b" />

---

## ✨ Acknowledgements

- TensorFlow/Keras
- Streamlit
- Scikit-learn
