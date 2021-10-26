import inspect
import os
from typing import Any, Dict, List, Optional

import opti
import orjson
import pandas as pd
from basf_auth.authorization.accessit_api import AccessITAuthorization
from basf_auth.framework.fastapi_auth import BASFAuth, make_session_params
from basf_auth.user import UserInfo
from celery.states import FAILURE, PENDING, SUCCESS
from fastapi import Depends, FastAPI, Query
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

import api.celery_tasks as tasks
from api import core
from api.celery_tasks import AsyncResult
from api.db import (
    add_session,
    delete_session,
    get_session_config,
    get_session_problem,
    list_sessions,
    update_session_data,
)
from api.schema import DataFrame, MethodName, Problem, SessionInfo, methods


class ORJSONResponse(JSONResponse):
    """Custom JSON encoder to handle nan and inf,
    see https://github.com/tiangolo/fastapi/issues/459
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)


app = FastAPI(
    root_path="/brain" if os.getenv("APPSTORE_ENV") is not None else "/",
    title="Brain",
    description="API for Bayesian optimization & adaptive experimentation",
    version="0.3.1",
    docs_url="/",
    default_response_class=ORJSONResponse,
)

# Authentication
app.add_middleware(SessionMiddleware, **make_session_params())
auth = BASFAuth()
auth_optional = BASFAuth(error_mode="none")
auth_admin = BASFAuth(
    authorization_scheme=AccessITAuthorization(
        application_id=3860,
        authorized_roles_id=[11006385],
    ),
    error_mode="json",
)


@app.exception_handler(Exception)
async def exception_handler(request, err):
    """Handler for all uncaught exceptions."""
    return JSONResponse(str(err), status_code=400)


@app.get("/ping")
def ping_pong():
    return "pong"


@app.get("/method")
def list_all_methods():
    """List all optimization methods."""
    return list(methods.keys())


@app.post("/method")
def list_suitable_methods(problem: Problem = None):
    """List suitable optimization methods for a given problem."""
    if problem is None:
        return list(methods.keys())

    problem = opti.Problem.from_config(problem.dict())

    suitable_methods = []
    for name, method in methods.items():
        try:
            method(problem)
            suitable_methods.append(name)
        except BaseException:
            pass
    return suitable_methods


@app.get("/method/{method}")
def list_method_parameters(method: MethodName):
    """List all parameters of a given optimization method."""
    if method not in methods:
        raise HTTPException(404, "Method not found")
    arguments_string = str(inspect.signature(methods[method].__init__))
    arguments_list = [s.strip() for s in arguments_string[1:-1].split(",")]
    return arguments_list[1:]  # omit "self"


@app.post("/check_problem")
def check_problem(problem: Problem):
    """Check if the problem definition is valid."""
    opti.Problem.from_config(problem.dict())
    return "all good"


@app.post("/session/create")
def create_session(
    method: MethodName,
    problem: Problem,
    parameters: Optional[Dict[str, Any]] = None,
    user: UserInfo = Depends(auth_optional.dependency),
):
    """Create a new optimization session for a given problem."""
    username = user.get_username() if user else None
    try:
        problem = opti.Problem.from_config(problem.dict())
        if parameters is None:
            parameters = {}
        optimizer = methods[method](problem, **parameters)
    except Exception as err:
        raise HTTPException(400, str(err))
    session_id = add_session(optimizer.to_config(), username)
    return {"session_id": session_id}


@app.get("/session/list", response_model=List[SessionInfo])
def list_user_sessions(user: UserInfo = Depends(auth.dependency)):
    """List all your sessions."""
    return list_sessions(user.get_username())


@app.get("/session/list_all", response_model=List[SessionInfo])
def list_all_sessions(user: UserInfo = Depends(auth_admin.dependency)):
    """List all sessions (admin only)."""
    return list_sessions()


@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """Get all information on a given session."""
    return get_session_config(session_id)


@app.delete("/session/{session_id}")
def destroy_session(session_id: str):
    """Remove an existing session."""
    delete_session(session_id)
    return "deleted"


@app.get("/session/{session_id}/data", response_model=DataFrame)
def get_all_data_points(session_id: str):
    """Get all experimental data points."""
    data = get_session_config(session_id)["problem"].get("data", None)
    if data:
        return data
    else:
        return JSONResponse(status_code=404, content={"message": "No data in session"})


@app.get("/session/{session_id}/ask_async")
def ask_for_proposals_async(session_id: str, n_proposals: int = Query(1, ge=1, le=50)):
    """Get proposals for the next experiments."""
    run = tasks.ask_for_proposals.apply_async([session_id, n_proposals])
    return str(run)


@app.get("/session/{session_id}/ask", response_model=DataFrame)
def ask_for_proposals(session_id: str, n_proposals: int = Query(1, ge=1, le=50)):
    """Get proposals for the next experiments."""
    return core.ask_for_proposals(session_id, n_proposals)


@app.get("/session/result/{run_id}")
def result(run_id: str):
    """Get the result of an asynchronous endpoint call."""
    if AsyncResult(run_id).state == SUCCESS:
        return AsyncResult(run_id).get()
    elif AsyncResult(run_id).state == PENDING:
        return PENDING
    elif AsyncResult(run_id).state == FAILURE:
        return FAILURE
    else:
        raise ValueError(f"Unknown celery state {AsyncResult(run_id).state}.")


@app.post("/session/{session_id}/tell")
def tell_experimental_results(session_id: str, data: DataFrame, replace: bool = False):
    """Add/replace experimental results, the surrogate model will be refit after
    loading."""
    data = pd.DataFrame(**data.dict())
    problem = get_session_problem(session_id)
    if replace:
        problem.set_data(data)
    else:
        problem.add_data(data)
    update_session_data(session_id, problem.data.to_dict(orient="split"))


@app.post("/session/{session_id}/predict_async", response_model=str)
def predict_async(session_id: str, data: DataFrame):
    """Predict the posterior mean and std at given inputs."""
    run = tasks.predict.apply_async([session_id, data.dict()])
    return str(run)


@app.post("/session/{session_id}/predict", response_model=DataFrame)
def predict(session_id: str, data: DataFrame):
    """Predict the posterior mean and std at given inputs."""
    return core.predict(session_id, data.dict())


@app.get("/session/{session_id}/predict_front_async", response_model=str)
def predict_pareto_front_async(session_id: str, n_levels: int = Query(5, ge=2, le=20)):
    """Sample the Pareto front using the model posterior mean."""
    run = tasks.predict_pareto_front.apply_async([session_id, n_levels])
    return str(run)


@app.get("/session/{session_id}/predict_front", response_model=DataFrame)
def predict_pareto_front(session_id: str, n_levels: int = Query(5, ge=2, le=20)):
    """Sample the Pareto front using the model posterior mean."""
    return core.predict_pareto_front(session_id, n_levels)


@app.get("/session/{session_id}/cross_validate_async", response_model=str)
def cross_validate_endpoint_async(
    session_id: str, n_splits: int = Query(5, ge=1, le=100)
):
    """Perform k-fold shuffled cross-validation."""
    run = tasks.cross_validate_endpoint.apply_async([session_id, n_splits])
    return str(run)


@app.get("/session/{session_id}/cross_validate", response_model=dict)
def cross_validate_endpoint(session_id: str, n_splits: int = Query(5, ge=1, le=100)):
    """Perform k-fold shuffled cross-validation."""
    return core.cross_validate_endpoint(session_id, n_splits)


@app.get("/session/{session_id}/plot/data_async", response_model=str)
def plot_data_async(session_id: str, color_by: str = None):
    """Parallel coordinates plot of the data."""
    run = tasks.plot_data.apply_async([session_id, color_by])
    return str(run)


@app.get("/session/{session_id}/plot/data", response_class=HTMLResponse)
def plot_data(session_id: str, color_by: str = None):
    """Parallel coordinates plot of the data."""
    return core.plot_data(session_id, color_by)


@app.get("/session/{session_id}/plot/model_async", response_model=str)
def plot_model_predictions_async(
    session_id: str,
    color_by: str = None,
    n_samples: int = Query(10000, ge=1, le=100000),
):
    """Parallel coordinates plot of the model predictions."""
    run = tasks.plot_model_predictions.apply_async([session_id, n_samples, color_by])
    return str(run)


@app.get("/session/{session_id}/plot/model", response_class=HTMLResponse)
def plot_model_predictions(
    session_id: str,
    color_by: str = None,
    n_samples: int = Query(10000, ge=1, le=100000),
):
    """Parallel coordinates plot of the model predictions."""
    return core.plot_model_predictions(session_id, n_samples, color_by)


@app.get("/session/{session_id}/plot/parameters_async", response_model=str)
def plot_model_parameters_async(
    session_id: str,
    run_cross_validation: bool = False,
    n_splits: int = Query(5, ge=1, le=100),
):
    """Parallel coordinates plot of the model parameters."""
    run = tasks.plot_model_predictions.apply_async(
        [session_id, n_splits, run_cross_validation]
    )
    return str(run)


@app.get("/session/{session_id}/plot/parameters", response_class=HTMLResponse)
def plot_model_parameters(
    session_id: str,
    run_cross_validation: bool = False,
    n_splits: int = Query(5, ge=1, le=100),
):
    """Parallel coordinates plot of the model parameters."""
    return core.plot_model_parameters(session_id, n_splits, run_cross_validation)


@app.get("/session/{session_id}/plot/residuals_async", response_model=str)
def plot_model_prediction_residuals_async(
    session_id: str,
    run_cross_validation: bool = False,
    n_splits: int = Query(5, ge=1, le=100),
):
    """Plot of the model prediction residuals."""
    run = tasks.plot_model_prediction_residuals.apply_async(
        [session_id, n_splits, run_cross_validation]
    )
    return str(run)


@app.get("/session/{session_id}/plot/residuals", response_class=HTMLResponse)
def plot_model_prediction_residuals(
    session_id: str,
    run_cross_validation: bool = False,
    n_splits: int = Query(5, ge=1, le=100),
):
    """Plot of the model prediction residuals."""
    return core.plot_model_prediction_residuals(
        session_id, n_splits, run_cross_validation
    )


@app.get("/session/{session_id}/plot/all_async", response_model=str)
def plot_all_async(
    session_id: str,
    n_splits: int = Query(5, ge=1, le=100),
    n_samples: int = Query(10000, ge=1, le=100000),
):
    """Generate a graphical report of the data and surrogate model."""

    run = tasks.plot_all.apply_async([session_id, n_splits, n_samples])
    return str(run)


@app.get("/session/{session_id}/plot/all", response_class=HTMLResponse)
def plot_all(
    session_id: str,
    n_splits: int = Query(5, ge=1, le=100),
    n_samples: int = Query(10000, ge=1, le=100000),
):
    """Generate a graphical report of the data and surrogate model."""
    return core.plot_all(session_id, n_splits, n_samples)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
