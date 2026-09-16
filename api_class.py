from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.preprocessing import FeatureEngineer, FeatureDropper, create_preprocessor

app = FastAPI()

model = joblib.load("best_model_classification.pkl")


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


@app.get("/")
def home():
    return {"message": "Student Placement API is running"}


@app.post("/predict")
def predict(data: StudentInput):
    input_df = pd.DataFrame([data.dict()])

    pred = model.predict(input_df)[0]

    return {
        "placement": int(pred),
        "result": "Placed" if pred == 1 else "Not Placed"
    }