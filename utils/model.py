# utils/compare_models.py
import pandas as pd
import streamlit as st

def show_comparison_table(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df)