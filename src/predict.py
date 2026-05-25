import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    return mlflow.pyfunc.load_model(model_uri)


def predict_customer():
    model = load_model()

    input_df = pd.DataFrame([{
        "gender": 1,
        "age": 35,
        "annual_income_k$": 120,
        "spending_score_1_100": 50,
    }])

    prediction = model.predict(input_df)

    print("Segmento predicho:", int(prediction[0]))


if __name__ == "__main__":
    predict_customer()