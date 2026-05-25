import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError("No existe el experimento")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No hay runs disponibles")

    best_run = runs[0]
    run_id = best_run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=result.version
    )

    print("Modelo registrado correctamente")
    print("Run ID:", run_id)
    print("Modelo:", REGISTERED_MODEL_NAME)
    print("Versión:", result.version)
    print("Alias: champion")


if __name__ == "__main__":
    register_best_model()