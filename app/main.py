from __future__ import annotations

import pickle
from pathlib import Path

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional dependency guard
    FastAPI = None

import pandas as pd
from pydantic import BaseModel

from pipeline import run_training_pipeline
from src.utils.config_loader import load_config
from src.utils.paths import find_project_root


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


def _latest_model_path(project_root: Path, models_dir: str) -> Path:
    """Find the most recently saved model artifact."""
    model_dir = project_root / models_dir
    candidates = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No model artifact found in {model_dir}. Run /pipeline first to train and save a model."
        )
    return candidates[0]


def _load_fair_model():
    project_root = find_project_root()
    config = load_config()
    model_path = _latest_model_path(project_root, config["paths"]["models"])
    with model_path.open("rb") as handle:
        return pickle.load(handle)


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
        fair_model = _load_fair_model()

        row = pd.DataFrame([application.model_dump()])
        sensitive_features = row["applicant_age"].astype(str)

        prediction = fair_model.predict(row, sensitive_features=sensitive_features, random_state=42)
        probability = fair_model.predict_proba(row)[:, 1]

        decision = "approved" if int(prediction[0]) == 1 else "denied"

        return {
            "decision": decision,
            "approval_probability": float(probability[0]),
            "fairness_strategy": fair_model.fairness_strategy,
        }

    return app
app = create_app() if FastAPI is not None else None
