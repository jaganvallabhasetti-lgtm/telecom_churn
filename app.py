import streamlit as st
import pandas as pd

st.title("Telecom Customer Churn Prediction")

st.write("ML Project Deployment Successful 🚀")

uploaded_file = st.file_uploader("Upload CSV File")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())