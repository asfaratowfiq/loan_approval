FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir fastapi streamlit uvicorn pandas scikit-learn xgboost pyyaml fairlearn imbalanced-learn

EXPOSE 8000
EXPOSE 8501

CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=8501 & \
    wait
