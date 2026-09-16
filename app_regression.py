import streamlit as st
import pandas as pd
import requests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="Salary Predictor", layout="wide")

st.title("💰 Student Salary Prediction")
gender = st.selectbox("Select your gender", ["🧑‍🎓 Male", "👩‍🎓 Female"])

# ================= LAYOUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Academic")

    branch = st.selectbox("Branch", ["CE", "CSE", "IT", "ECE"])
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
    twelfth = st.slider("12th Percentage", 0.0, 100.0, 60.0)
    backlogs = st.number_input("Backlogs", 0, 10, 0)

    st.subheader("💻 Skills")

    coding = st.slider("Coding Skill", 0, 10, 5)
    aptitude = st.slider("Aptitude Skill", 0, 10, 5)

with col2:
    st.subheader("📁 Experience")

    projects = st.number_input("Projects", 0, 10, 1)
    internships = st.number_input("Internships", 0, 10, 0)
    hackathons = st.number_input("Hackathons", 0, 10, 0)
    certs = st.number_input("Certifications", 0, 10, 0)

    st.subheader("📈 Study & Activity")

    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0)
    internet = st.selectbox("Internet Access", ["Yes", "No"])

data = {
    "branch": branch,
    "cgpa": cgpa,
    "twelfth_percentage": twelfth,
    "backlogs": backlogs,
    "projects_completed": projects,
    "internships_completed": internships,
    "coding_skill_rating": coding,
    "aptitude_skill_rating": aptitude,
    "hackathons_participated": hackathons,
    "certifications_count": certs,
    "internet_access": internet,
    "study_hours_per_day": study_hours,
}

# ================= PREDICT =================
st.divider()
st.subheader("📊 Prediction Result")

if st.button("💰 Predict Salary", use_container_width=True):
    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=data)
        result = res.json()

        pred = result["salary_lpa"]
        pred = max(0, pred)

        if pred < 2:
            st.warning("Salary is very low")
        else:
            st.success(f"💸 Estimated Salary: {pred:.2f} LPA")

    except Exception as e:
        st.error(f"Error: {e}")