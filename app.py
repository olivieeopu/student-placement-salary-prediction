import streamlit as st
import pandas as pd
import joblib
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# load model
model_clf = joblib.load("best_model_classification.pkl")
model_reg = joblib.load("best_model_regression.pkl")

st.set_page_config(page_title="Student Prediction", layout="wide")

st.title("🎓 Student Placement & Salary Prediction")
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

# ================= JSON =================
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
    "study_hours_per_day": study_hours
}

json_data = json.dumps(data)

# ================= PREDICT =================
st.divider()

if st.button("🚀 Predict", use_container_width=True):
    try:
        input_dict = json.loads(json_data)
        input_df = pd.DataFrame([input_dict])

        pred_class = model_clf.predict(input_df)[0]
        pred_salary = model_reg.predict(input_df)[0]
        pred_salary = max(0, pred_salary)

        placement = "Placed" if pred_class == 1 else "Not Placed"

        st.subheader("📊 Prediction Result")

        if placement == "Placed":
            st.success(f"🎯 Placement Status: {placement}")
        else:
            st.error(f"❌ Placement Status: {placement}")

        st.info(f"💰 Estimated Salary: {pred_salary:.2f} LPA")

    except Exception as e:
        st.error(f"Error: {e}")