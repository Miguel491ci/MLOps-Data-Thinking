import time
from datetime import datetime

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    MONITORING_DIR,
)
from src.features import FEATURE_COLUMNS


app = FastAPI(title="Mall Customers Classification API")


# =========================
# MÉTRICAS PROMETHEUS
# =========================

PREDICTION_COUNT = Counter(
    "mall_customer_prediction_count",
    "Número total de predicciones de segmentos de clientes"
)

PREDICTION_LATENCY = Histogram(
    "mall_customer_prediction_latency_seconds",
    "Latencia de predicción del modelo en segundos"
)

SEGMENT_COUNTER = Counter(
    "mall_customer_segment_predictions_total",
    "Predicciones por segmento",
    ["segment"]
)


# =========================
# MODELO GLOBAL
# =========================

model = None


# =========================
# REQUEST MODEL
# =========================

class PredictionRequest(BaseModel):
    gender: str
    age: int
    annual_income_k: int
    spending_score: int


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def load_champion_model():
    global model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    model = mlflow.pyfunc.load_model(model_uri)

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    print("Modelo cargado correctamente")


# =========================
# ROOT
# =========================

@app.get("/")
def root():
    return {"message": "Mall Customers API activa"}


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": REGISTERED_MODEL_NAME,
        "alias": "champion",
    }


# =========================
# PREDICCIÓN
# =========================

@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.time()

    if model is None:
        return {"error": "El modelo todavía no ha sido cargado"}

    # -------------------------
    # Validar gender
    # -------------------------

    gender_value = request.gender.strip().lower()

    if gender_value not in ["male", "female"]:
        return {
            "error": "gender debe ser Male o Female"
        }

    gender_encoded = 1 if gender_value == "male" else 0

    # -------------------------
    # Crear dataframe
    # -------------------------

    input_df = pd.DataFrame([{
        "gender": int(gender_encoded),
        "age": int(request.age),
        "annual_income_k$": int(request.annual_income_k),
        "spending_score_1_100": int(request.spending_score),
    }])

    # Asegurar tipos exactos para MLflow
    input_df = input_df.astype({
        "gender": "int64",
        "age": "int64",
        "annual_income_k$": "int64",
        "spending_score_1_100": "int64",
    })

    # Orden correcto de columnas
    input_df = input_df[FEATURE_COLUMNS]

    # -------------------------
    # Predicción
    # -------------------------

    prediction = int(model.predict(input_df)[0])

    latency = time.time() - start_time

    # -------------------------
    # Métricas
    # -------------------------

    PREDICTION_COUNT.inc()

    PREDICTION_LATENCY.observe(latency)

    SEGMENT_COUNTER.labels(
        segment=str(prediction)
    ).inc()

    # -------------------------
    # Logging
    # -------------------------

    log_prediction(
        input_df=input_df,
        prediction=prediction,
        latency=latency,
    )

    # -------------------------
    # Response
    # -------------------------

    return {
        "customer_segment": prediction,
        "message": f"El cliente pertenece al segmento {prediction}",
        "latency_seconds": latency,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =========================
# MÉTRICAS PROMETHEUS
# =========================

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# =========================
# LOG PREDICCIONES
# =========================

def log_prediction(
    input_df: pd.DataFrame,
    prediction: int,
    latency: float,
):

    log_file = MONITORING_DIR / "api_predictions_log.csv"

    row = input_df.copy()

    row["customer_segment"] = prediction
    row["latency_seconds"] = latency
    row["timestamp"] = datetime.utcnow().isoformat()

    if log_file.exists():

        row.to_csv(
            log_file,
            mode="a",
            header=False,
            index=False,
        )

    else:

        row.to_csv(
            log_file,
            index=False,
        )