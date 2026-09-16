from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.preprocessing import FeatureEngineer, FeatureDropper, create_preprocessor

app = FastAPI()

model = joblib.load("best_model_regression.pkl")


class StudentInput(BaseModel):
    branch: str
    cgpa: float
    twelfth_percentage: float
    backlogs: int
    projects_completed: int
    internships_completed: int
    coding_skill_rating: int
    aptitude_skill_rating: int
    hackathons_participated: int
    certifications_count: int
    internet_access: str
    study_hours_per_day: float


@app.get("/")
def home():
    return {"message": "Student Salary API is running"}


@app.post("/predict")
def predict(data: StudentInput):
    input_df = pd.DataFrame([data.dict()])

    pred = model.predict(input_df)[0]
    pred = max(0, pred)

    return {
        "salary_lpa": float(pred)
    }