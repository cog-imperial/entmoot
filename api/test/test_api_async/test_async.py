import os
from time import sleep, time

import opti
import pandas as pd
import requests
from celery.states import FAILURE, PENDING

BRAINHOST = os.getenv("BRAINHOST", "http://host.docker.internal:5000")
HEADERS = {"X-Gitlab-Token": os.getenv("GITLAB_TOKEN")}


def _create(method, problem):
    response = requests.post(
        f"{BRAINHOST}/session/create?method={method}",
        json={"problem": problem.to_config()},
        headers=HEADERS,
    )
    return response.json()["session_id"]


def _poll(endpoint, pause=1, timeout=20, json=None):
    if json is None:
        response = requests.get(f"{BRAINHOST}/session/{endpoint}")
    else:
        response = requests.post(f"{BRAINHOST}/session/{endpoint}", json=json)
    assert response.ok
    run_id = response.json()
    print(f"run {run_id} triggered")
    result = PENDING
    start = time()
    elapsed = 0
    while result == PENDING and elapsed < timeout:
        response = requests.get(f"{BRAINHOST}/session/result/{run_id}")
        assert response.ok
        result = response.json()
        sleep(pause)
        elapsed = time() - start
        print(f"{elapsed} elapsed, pending...")
    assert result != PENDING and result != FAILURE
    print("Result")
    print(result)
    print("---")
    return result


def test_ask():
    problem = opti.problems.ZDT1(n_inputs=10)
    session_id = _create("RandomSearch", problem)

    proposal = _poll(f"{session_id}/ask_async?n_proposals=4")
    df = pd.DataFrame(**proposal)
    assert len(df) == 4
    assert problem.inputs.contains(df).all()


def test_predict():
    problem = opti.problems.ZDT1(3)
    problem.create_initial_data(5)
    session_id = _create("ParEGO", problem)

    X = problem.inputs.sample(10)
    prediction = _poll(f"{session_id}/predict_async", json=X.to_dict("split"))
    Y = pd.DataFrame(**prediction)
    assert len(X) == len(Y)
    assert (X.index == Y.index).all()


def test_cross_validate():
    # ParEGO
    problem = opti.problems.ZDT1(3)
    problem.create_initial_data(5)
    session_id = _create("ParEGO", problem)

    cv_res = _poll(f"{session_id}/cross_validate_async")
    predictions = pd.DataFrame(**cv_res["predictions"])
    assert len(predictions) == 5

    # SOBO
    problem = opti.problems.Ackley(3)
    problem.create_initial_data(5)
    session_id = _create("SOBO", problem)

    cv_res = _poll(f"{session_id}/cross_validate_async")
    predictions = pd.DataFrame(**cv_res["predictions"])
    assert len(predictions) == 5


def test_plot():
    problem = opti.problems.Ackley(3)
    problem.create_initial_data(5)
    session_id = _create("SOBO", problem)

    html = _poll(f"{session_id}/plot/data_async")
    assert html.startswith("<html>")

    html = _poll(f"{session_id}/plot/model_async")
    assert html.startswith("<html>")

    html = _poll(f"{session_id}/plot/residuals_async")
    assert html.startswith("<html>")

    html = _poll(f"{session_id}/plot/residuals_async?cross_validate=true")
    assert html.startswith("<html>")
