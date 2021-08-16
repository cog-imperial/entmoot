import os

from celery import Celery

from api import core
from api.schema import DataFrame

CELERY_BACKEND_URL = os.environ.get("CELERY_BACKEND_URL", "redis://127.0.0.1")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "amqp://127.0.0.1")

app = Celery("tasks", backend=CELERY_BACKEND_URL, broker=CELERY_BROKER_URL)

AsyncResult = app.AsyncResult


@app.task()
def ask_for_proposals(session_id: str, n_proposals: int) -> DataFrame:
    return core.ask_for_proposals(session_id, n_proposals)


@app.task()
def predict(session_id: str, data: dict) -> dict:
    return core.predict(session_id, data)


@app.task()
def predict_pareto_front(session_id: str, n_levels: int) -> dict:
    return core.predict_pareto_front(session_id, n_levels)


@app.task()
def cross_validate_endpoint(session_id: str, n_splits: int) -> dict:
    return core.cross_validate_endpoint(session_id, n_splits)


@app.task()
def plot_data(session_id: str, color_by: str = None) -> str:
    return core.plot_data(session_id, color_by)


@app.task()
def plot_model_predictions(
    session_id: str, n_samples: int, color_by: str = None
) -> str:
    return core.plot_model_predictions(session_id, n_samples, color_by)


@app.task()
def plot_model_parameters(
    session_id: str,
    n_splits: int,
    run_cross_validation: bool = False,
) -> str:
    return core.plot_model_parameters(session_id, n_splits, run_cross_validation)


@app.task()
def plot_model_prediction_residuals(
    session_id: str,
    n_splits: int,
    run_cross_validation: bool = False,
):
    return core.plot_model_prediction_residuals(
        session_id, n_splits, run_cross_validation
    )


@app.task()
def plot_all(
    session_id: str,
    n_splits: int,
    n_samples: int,
):
    return core.plot_all(session_id, n_splits, n_samples)
