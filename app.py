import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("student_performance_model.pkl")

# Judul aplikasi
st.title("Student Performance Prediction")
st.markdown("Masukkan informasi berikut untuk memprediksi kelulusan (Pass/Fail):")

# Input kolom
student_id = st.text_input("Student ID")
gender = st.selectbox("Gender", ["Male", "Female"])
weekly_study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100)
attendance = st.number_input("Attendance Rate (0-100)", min_value=0, max_value=100)
past_exam_scores = st.number_input("Past Exam Scores (0-100)", min_value=0, max_value=100)
final_exam_score = st.number_input("Final Exam Score (0-100)", min_value=0, max_value=100)
parental_education_level = st.selectbox("Parental Education Level", ["High School", "Bachelors", "Masters", "PhD"])
internet_access_at_home = st.selectbox("Internet Access at Home", ["Yes", "No"])
extracurricular_activities = st.selectbox("Join Extracurricular Activities?", ["Yes", "No"])

# Tombol prediksi
if st.button("Predict"):

    # Buat DataFrame dari input pengguna
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Study_Hours_per_Week": weekly_study_hours,
        "Attendance_Rate": attendance,
        "Past_Exam_Scores": past_exam_scores,
        "Parental_Education_Level": parental_education_level,
        "Internet_Access_at_Home": internet_access_at_home,
        "Extracurricular_Activities": extracurricular_activities,
        "Final_Exam_Score": final_exam_score
    }])

    # One-hot encoding kolom kategorikal
    categorical_cols = ['Gender', 'Parental_Education_Level', 'Internet_Access_at_Home', 'Extracurricular_Activities']
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols)

    # Pastikan kolom input sama dengan yang digunakan saat pelatihan model
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0  # Tambah kolom yang hilang

    input_encoded = input_encoded[expected_columns]  # urutkan sesuai model

    # Lakukan prediksi
    prediction = model.predict(input_encoded)[0]
    result = "Pass" if prediction == 1 else "Fail"

    # Tampilkan hasil
    st.success(f"Prediction: {result}")
