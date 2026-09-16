from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

app = FastAPI()

model_clf = joblib.load("best_model_classification.pkl")
model_reg = joblib.load("best_model_regression.pkl")

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

@app.post("/predict")
def predict(data: StudentInput):
    try:
        # JSON → DataFrame
        input_df = pd.DataFrame([data.dict()])

        # prediction
        pred_class = model_clf.predict(input_df)[0]
        pred_salary = model_reg.predict(input_df)[0]

        pred_salary = max(0, pred_salary)

        return {
            "placement": int(pred_class),
            "placement_label": "Placed" if pred_class == 1 else "Not Placed",
            "salary_lpa": float(pred_salary)
        }

    except Exception as e:
        return {"error": str(e)}