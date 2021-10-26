from typing import List

import opti
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bo.algorithm import Algorithm
from bo.metric import goodness_of_fit


def _parallel(data: pd.DataFrame, inputs: opti.Parameters, color_by: str) -> go.Figure:
    """Parallel plot with handling of categorical variables.
    Waiting for https://github.com/plotly/plotly.py/issues/2143
    """
    data = data.copy().reset_index(drop=True)

    cats = {}
    for p in inputs:
        if isinstance(p, opti.Categorical):
            c = pd.Categorical(data[p.name])
            cats[p.name] = c
            data[p.name] = c.codes

    fig = px.parallel_coordinates(data, color=color_by)

    for dim in fig.data[0]["dimensions"]:
        name = dim["label"]
        if name in cats:
            dim["tickvals"] = cats[name].unique().codes
            dim["ticktext"] = cats[name].categories
    fig.update_yaxes()
    fig.update_layout(coloraxis_showscale=False)
    return fig


def parallel_data(problem: opti.Problem, color_by: str = None) -> go.Figure:
    """Parallel plot of the data."""
    data = problem.data[problem.inputs.names + problem.outputs.names]
    if color_by is None:
        color_by = problem.outputs.names[0]
    return _parallel(data, problem.inputs, color_by)


def scatter_data(problem: opti.Problem) -> go.Figure:
    """Scatter plots of the data for each combination of parameters."""
    fig = px.scatter_matrix(
        problem.data,
        dimensions=problem.inputs.names + problem.outputs.names,
    )
    fig.update_traces(diagonal_visible=False)
    return fig


def correlation_data(problem: opti.Problem) -> go.Figure:
    """Heatmap plot of the Pearson and Spearman correlation for the data."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Linear correlation (Pearson)",
            "Monotonous correlation (Spearman)",
        ),
    )
    df = problem.data[problem.inputs.names + problem.outputs.names]
    fig.add_trace(px.imshow(df.corr(method="pearson")).data[0], row=1, col=1)
    fig.add_trace(px.imshow(df.corr(method="spearman")).data[0], row=1, col=2)
    fig.update_traces(
        {"hovertemplate": "%{x}<br>%{y}<br>correlation: %{z:.3f}<extra></extra>"}
    )
    fig.update_layout(
        coloraxis_showscale=False,
        coloraxis_colorscale="RdBu_r",
        coloraxis_cmid=0,
        yaxis_autorange="reversed",
        yaxis2_autorange="reversed",
    )
    return fig


def parallel_model(
    optimizer: Algorithm, color_by: str = None, n_samples: int = 10000
) -> go.Figure:
    """Parallel plot of model predictions over entire design space."""
    problem = optimizer._problem
    X = opti.sampling.constrained_sampling(
        n_samples=n_samples, parameters=problem.inputs, constraints=problem.constraints
    )
    Y = optimizer.predict(X)
    data = pd.concat([X, Y], axis=1)
    if color_by is None:
        color_by = Y.columns[0]
    fig = _parallel(data, problem.inputs, color_by)
    fig.update_layout(title=dict(text="Model predictions", y=0.99, yanchor="top"))
    return fig


def parallel_parameters(
    optimizer: Algorithm = None, cv_results: dict = None
) -> go.Figure():
    """Parallel plot of all model parameters.

    If cv_results is given the "parameters" dataframe from it will be used.
    Otherwise optimizer.get_model_parameters() will be used.
    """
    if cv_results is not None:
        parameters = cv_results["parameters"].copy()
        n_splits = parameters["cv_split"].max() + 1
        title = f"Model parameters ({n_splits}-fold CV)"
        parameters.drop(columns=["cv_split"], inplace=True)
    else:
        parameters = optimizer.get_model_parameters()
        title = "Model parameters"

    # set index ("output") as first column
    parameters.reset_index(inplace=True)
    c = pd.Categorical(parameters["output"])
    parameters["output"] = c.codes

    fig = px.parallel_coordinates(parameters.reset_index(), color="output")

    # style specific axes
    dims = fig.data[0]["dimensions"]
    for d in dims:
        if d["label"] in ("noise", "scale"):
            d["range"] = [0, 2]
        if d["label"].startswith("ls_"):
            d["range"] = [0, 10]
        if d["label"] == "output":
            d["tickvals"] = c.unique().codes
            d["ticktext"] = c.categories
    fig.update_yaxes()

    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(title=dict(text=title, y=0.99, yanchor="top"))
    return fig


def residuals(optimizer: Algorithm, cv_results: dict = None) -> List[go.Figure]:
    """Plot the residuals of the model predictions against the data.

    If cv_results is given, the "predictions" and "metrics" from it will be used.
    Otherwise the model predictions will be compared against training data.
    """
    problem = optimizer._problem

    if cv_results is not None:
        predictions = cv_results["predictions"]
        metrics = cv_results["metrics"]
    else:
        Y = optimizer.data[problem.outputs.names]
        Yp = optimizer.predict(optimizer.data)
        predictions = pd.concat([Y, Yp], axis=1)
        metrics = goodness_of_fit(predictions)

    figs = []
    for name in problem.outputs.names:
        error_y = predictions[f"std_{name}"] if f"std_{name}" in predictions else None
        fig = px.scatter(
            x=predictions[name],
            y=predictions[f"mean_{name}"] - predictions[name],
            text=predictions.index,
            error_y=error_y,
            template="simple_white",
            labels={"x": name, "y": "residuals", "text": "id"},
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=0,
            xref="paper",
            line=dict(
                color="Red",
                width=3,
                dash="dot",
            ),
        )
        title = f"{name}: R² = {metrics.loc[name, 'R2']:.3f}"
        if cv_results is not None:
            n = predictions["cv_split"].max() + 1
            title += f" ({n}-fold CV)"
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "y": 0.9,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        figs.append(fig)
    return figs
