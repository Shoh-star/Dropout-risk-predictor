import streamlit as st
import pandas as pd
import joblib
import random

# Load your trained model and feature list
model = joblib.load(r"C:\Users\PRO-TEC\IdeaProjects\dropout_model.pkl")
feature_columns = joblib.load(
    r"C:\Users\PRO-TEC\IdeaProjects\model_features.pkl")

st.set_page_config(page_title="Dropout Risk Predictor", layout="centered")
st.title("🎓 Student Dropout Risk Predictor")

st.markdown("""
Upload a CSV file with student data to predict dropout risk.  
Make sure the file has the same columns used during model training.
""")

uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file, delimiter=';')
    # Drop target column if present
    if 'Target' in data.columns:
        data = data.drop('Target', axis=1)
    elif 'Dropout' in data.columns:
        data = data.drop('Dropout', axis=1)

    # Separate identifiers and features
    identifiers = data[['Student_ID', 'Full_Name']]
    features = data.drop(['Student_ID', 'Full_Name'], axis=1)

    # Filter to expected model features
    features = features[feature_columns]

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    try:
        predictions = model.predict(features)
        results = identifiers.copy()
        results['Dropout Risk'] = predictions

        st.subheader("✅ Prediction Results")
        st.dataframe(results)
        high_risk = results[results['Dropout Risk'] == 1]
        st.subheader("⚠️ High-Risk Students")
        st.dataframe(high_risk)
        # Download button
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='dropout_predictions.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
