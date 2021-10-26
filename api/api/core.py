import io

import pandas as pd

import bo
from api.db import get_session_optimizer
from api.schema import DataFrame
from bo.metric import cross_validate


def ask_for_proposals(session_id: str, n_proposals: int) -> DataFrame:
    optimizer = get_session_optimizer(session_id)
    proposal = optimizer.propose(n_proposals).to_dict(orient="split")
    return proposal


def predict(session_id: str, data: dict) -> dict:
    """Predict the posterior mean and std at given inputs."""
    optimizer = get_session_optimizer(session_id)
    data = pd.DataFrame(**data)
    return optimizer.predict(data).to_dict(orient="split")


def predict_pareto_front(session_id: str, n_levels: int) -> dict:
    """Sample the Pareto front using the model posterior mean."""
    optimizer = get_session_optimizer(session_id)
    return optimizer.predict_pareto_front(n_levels).to_dict(orient="split")


def cross_validate_endpoint(session_id: str, n_splits: int) -> dict:
    """Perform k-fold shuffled cross-validation."""
    optimizer = get_session_optimizer(session_id)
    results = cross_validate(optimizer, n_splits)
    return {k: v.to_dict(orient="split") for k, v in results.items()}


def plot_data(session_id: str, color_by: str = None) -> str:
    """Parallel coordinates plot of the data."""
    optimizer = get_session_optimizer(session_id)
    fig = bo.plot.parallel_data(optimizer._problem, color_by=color_by)
    return fig.to_html(include_plotlyjs="cdn")


def plot_model_predictions(
    session_id: str, n_samples: int, color_by: str = None
) -> str:
    """Parallel coordinates plot of the model predictions."""
    optimizer = get_session_optimizer(session_id)
    fig = bo.plot.parallel_model(optimizer, color_by=color_by, n_samples=n_samples)
    return fig.to_html(include_plotlyjs="cdn")


def plot_model_parameters(
    session_id: str,
    n_splits: int,
    run_cross_validation: bool = False,
) -> str:
    """Parallel coordinates plot of the model parameters."""
    optimizer = get_session_optimizer(session_id)
    cv_results = cross_validate(optimizer, n_splits) if run_cross_validation else None
    fig = bo.plot.parallel_parameters(optimizer, cv_results)
    return fig.to_html(include_plotlyjs="cdn")


def plot_model_prediction_residuals(
    session_id: str,
    n_splits: int,
    run_cross_validation: bool = False,
):
    """Plot of the model prediction residuals."""
    optimizer = get_session_optimizer(session_id)
    cv_results = cross_validate(optimizer, n_splits) if run_cross_validation else None
    figs = bo.plot.residuals(optimizer, cv_results)
    # combine to a single html output
    s = io.StringIO()
    for fig in figs:
        s.write(fig.to_html(include_plotlyjs="cdn"))
    return s.getvalue()


def plot_all(
    session_id: str,
    n_splits: int,
    n_samples: int,
):
    """Generate a graphical report of the data and model."""
    optimizer = get_session_optimizer(session_id)

    s = io.StringIO()
    s.write('<html>\n<head><meta charset="utf-8" /></head>\n<body>\n')

    s.write("<h2>Data</h2>\n")
    fig = bo.plot.parallel_data(optimizer._problem)
    s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    fig = bo.plot.scatter_data(optimizer._problem)
    s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    fig = bo.plot.correlation_data(optimizer._problem)
    s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    s.write("<h2>Model</h2>\n")
    fig = bo.plot.parallel_model(optimizer, n_samples=n_samples)
    s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    try:
        cv_results = cross_validate(optimizer, n_splits)
    except NotImplementedError:
        cv_results = None

    if cv_results:
        fig = bo.plot.parallel_parameters(optimizer, cv_results)
        s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    figs = bo.plot.residuals(optimizer)
    for fig in figs:
        s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    if cv_results:
        figs = bo.plot.residuals(optimizer, cv_results)
        for fig in figs:
            s.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    s.write("</body>\n</html>")
    return s.getvalue()
