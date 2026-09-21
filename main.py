from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ml_model = {} # {"model": "credit_risk_model.pkl"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['model'] = joblib.load(BASE_DIR / 'credit_risk_model.pkl')
    ml_model['threshold'] = joblib.load(BASE_DIR / 'best_threshold.pkl')

    yield

    ml_model.clear()

app = FastAPI(lifespan = lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The only columns that user will see and provide inputs.
class LoanApplication(BaseModel): # Pydantic Model(Validation)
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.post("/predict")
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.dict()])

    prediction_proba = ml_model['model'].predict_proba(input_df)[:,1][0]

    prediction = int(prediction_proba >= ml_model["threshold"])

    return {
        "default_probability": prediction_proba,
        "default_prediction": prediction,
        "threshold": ml_model["threshold"],
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }

app.mount(
    "/",
    StaticFiles(directory=BASE_DIR / "static", html=True),
    name="static"
)