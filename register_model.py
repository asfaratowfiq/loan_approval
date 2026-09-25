import mlflow
import mlflow.sklearn
import pickle

with open("models/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

mlflow.set_tracking_uri(
    "azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/c3d7d280-955f-4da3-ad72-3ea1608f4fea/"
    "resourceGroups/Mortgage_application/providers/Microsoft.MachineLearningServices/workspaces/mortgageML"
)

metrics = {
    "accuracy": 0.9960232486999082,
    "precision": 1.0,
    "recall": 0.9951182876455126,
    "f1": 0.997553171466215,
    "roc_auc": 0.9999312173049825,
    "expected_value": 2795.0,
    "approval_rate": 0.8106454573263995,
    "false_approval_rate": 0.0,
    "false_rejection_rate": 0.00488171235448742,
}

conda_env = {
    "channels": ["conda-forge", "defaults"],
    "dependencies": [
        "python=3.10",
        "pip",
        {
            "pip": [
                "setuptools<81",
                "mlflow==2.15.1",
                "mlflow-skinny==2.15.1",
                "cloudpickle==3.1.2",
                "scikit-learn==1.3.2",
                "xgboost==2.1.4",
                "fairlearn==0.13.0",
                "numpy==1.26.4",
                "pandas==2.3.3",
                "scipy",
                "inference-schema[numpy-support]==1.5.0",
                "azureml-defaults",
                "azureml-inference-server-http==0.8.4",
                "azureml-ai-monitoring",
                "azureml-contrib-services",
                "azureml-monitoring",
                "applicationinsights",
            ]
        },
    ],
    "name": "loan-model-env",
}

with mlflow.start_run(run_name="loan_model_fairlearn_equalized_odds") as run:
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    mlflow.set_tag("fairness_strategy", "fairlearn_equalized_odds")
    mlflow.set_tag("fairness_attribute", "applicant_age")
    mlflow.set_tag(
        "mlflow.note.content",
        "Metrics sourced from reports/pipeline_report.json, matching models/loan_model.pkl."
    )
    mlflow.log_artifact("reports/pipeline_report.json")

    mlflow.sklearn.log_model(model, artifact_path="model", conda_env=conda_env)
    run_id = run.info.run_id

result = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="loan-model")
print(f"Registered: {result.name} v{result.version}, run_id={run_id}")
