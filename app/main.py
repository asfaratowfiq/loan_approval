from __future__ import annotations

import os

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional dependency guard
    FastAPI = None

import requests
from pydantic import BaseModel

from pipeline import run_training_pipeline

ML_ENDPOINT_URL = os.environ.get("ML_ENDPOINT_URL")
ML_ENDPOINT_KEY = os.environ.get("ML_ENDPOINT_KEY")


class LoanApplication(BaseModel):
    income: float
    loan_amount: float
    property_value: float
    debt_to_income_ratio: float
    combined_loan_to_value_ratio: float
    interest_rate: float
    loan_term: int
    loan_type: str
    occupancy_type: str
    state_code: str
    applicant_age: str
    loan_to_income_ratio: float
    loan_to_property_value_ratio: float
    high_dti_flag: int
    high_cltv_flag: int


def create_app() -> "FastAPI":
    """Create the FastAPI application instance."""
    if FastAPI is None:
        raise ImportError("FastAPI is required to run the API. Install the project dependencies first.")

    app = FastAPI(title="Loan Approval Underwriting Assistant")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/pipeline")
    def pipeline_status() -> dict:
        return run_training_pipeline()

    @app.post("/predict")
    def predict(application: LoanApplication) -> dict:
        if not ML_ENDPOINT_URL or not ML_ENDPOINT_KEY:
            return {"error": "ML_ENDPOINT_URL and ML_ENDPOINT_KEY environment variables must be set."}

        data_dict = application.model_dump()
        payload = {
            "input_data": {
                "columns": list(data_dict.keys()),
                "data": [list(data_dict.values())],
            }
        }
        headers = {
            "Authorization": f"Bearer {ML_ENDPOINT_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(ML_ENDPOINT_URL, json=payload, headers=headers, timeout=30)

        if response.status_code != 200:
            return {"error": response.text, "status_code": response.status_code}

        result = response.json()
        prediction = result[0] if isinstance(result, list) else result
        decision = "approved" if int(prediction) == 1 else "denied"

        return {"decision": decision}

    return app


app = create_app() if FastAPI is not None else None
