from unittest.mock import patch

import opti
import pandas as pd
from fastapi.testclient import TestClient
from mongomock import MongoClient

import api.main as main
from api.main import app

mock_sessions = MongoClient("mongodb://localhost:27017/").admin["sessions"]

client = TestClient(app)


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"


def test_list_methods():
    # list all available methods
    methods = client.get("/method").json()
    assert set(main.methods.keys()) == set(methods)

    # list method arguments for each method
    for m in methods:
        response = client.get(f"/method/{m}")
        assert response.ok


def test_list_suitable_methods():
    # problem without data
    problem = opti.problems.ZDT1(n_inputs=2)
    methods = client.post("/method", json=problem.to_config()).json()
    assert methods[0] == "RandomSearch"

    # unconstrained optimization problem with data
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(5)
    methods = client.post("/method", json=problem.to_config()).json()
    assert methods[0] == "ParEGO"

    # single-objective problem
    problem = opti.problems.Ackley()
    problem.create_initial_data(5)
    methods = client.post("/method", json=problem.to_config()).json()
    assert methods[0] == "SOBO"


def test_check_problem():
    # problem with all continuous variables
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(5)
    response = client.post("/check_problem", json=problem.to_config())
    assert response.ok

    # problem with categorical variables
    problem = opti.problems.Zakharov_categorical(n_inputs=3)
    problem.create_initial_data(5)
    response = client.post("/check_problem", json=problem.to_config())
    assert response.ok

    # problem with close-to-target objective
    problem = opti.problems.Cake()
    response = client.post("/check_problem", json=problem.to_config())
    assert response.ok


@patch("api.db.sessions", new=mock_sessions)
def test_session_handling():
    # create
    response = client.post(
        "/session/create?method=RandomSearch",
        json={"problem": opti.problems.Detergent().to_config()},
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    # retrieve
    assert client.get(f"/session/{session_id}").status_code == 200

    # delete
    assert client.delete(f"/session/{session_id}").status_code == 200

    # make sure it's gone
    assert client.get(f"/session/{session_id}").status_code == 404


@patch("api.db.sessions", new=mock_sessions)
def test_inf():
    problem = opti.Problem(
        inputs=[opti.Continuous("x")], outputs=[opti.Continuous("y")]
    )
    response = client.post(
        "/session/create?method=RandomSearch",
        json={"problem": problem.to_config()},
    )
    session_id = response.json()["session_id"]

    response = client.get(f"/session/{session_id}")
    assert response.status_code == 200


@patch("api.db.sessions", new=mock_sessions)
def test_closetotarget():
    problem = opti.Problem(
        inputs=[opti.Continuous("x", [0, 1])],
        outputs=[opti.Continuous("y")],
        objectives=[opti.objective.CloseToTarget("y", target=0.5)],
    )
    response = client.post(
        "/session/create?method=RandomSearch",
        json={"problem": problem.to_config()},
    )
    session_id = response.json()["session_id"]

    response = client.get(f"/session/{session_id}")
    problem = opti.Problem.from_config(response.json()["problem"])
    assert problem.objectives[0].target == 0.5


@patch("api.db.sessions", new=mock_sessions)
def test_tell():
    problem = opti.problems.ZDT1(3)
    session_id = client.post(
        "/session/create?method=RandomSearch", json={"problem": problem.to_config()}
    ).json()["session_id"]

    # add data
    problem.create_initial_data(5)
    response = client.post(
        f"/session/{session_id}/tell", json=problem.data.to_dict("split")
    )
    assert response.status_code == 200
    response = client.get(f"/session/{session_id}/data")
    assert response.status_code == 200
    df = pd.DataFrame(**response.json())
    assert len(df) == 5

    # replace data
    problem.create_initial_data(3)
    response = client.post(
        f"/session/{session_id}/tell?replace=true", json=problem.data.to_dict("split")
    )
    assert response.status_code == 200
    response = client.get(f"/session/{session_id}/data")
    assert response.status_code == 200
    df = pd.DataFrame(**response.json())
    assert len(df) == 3


@patch("api.db.sessions", new=mock_sessions)
def test_predict():
    problem = opti.problems.ZDT1(3)
    problem.create_initial_data(5)

    session_id = client.post(
        "/session/create?method=ParEGO", json={"problem": problem.to_config()}
    ).json()["session_id"]

    # create some inputs and retrieve predictions
    X = problem.inputs.sample(10)
    response = client.post(f"/session/{session_id}/predict", json=X.to_dict("split"))
    Y = pd.DataFrame(**response.json())
    assert len(X) == len(Y)
    assert (X.index == Y.index).all()


@patch("api.db.sessions", new=mock_sessions)
def test_cross_validate():
    # ParEGO
    problem = opti.problems.ZDT1(3)
    problem.create_initial_data(5)

    session_id = client.post(
        "/session/create?method=ParEGO",
        json={"problem": problem.to_config()},
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/cross_validate").json()
    predictions = pd.DataFrame(**response["predictions"])
    assert len(predictions) == 5

    # SOBO
    problem = opti.problems.Ackley(3)
    problem.create_initial_data(5)

    session_id = client.post(
        "/session/create?method=SOBO",
        json={"problem": problem.to_config()},
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/cross_validate").json()
    predictions = pd.DataFrame(**response["predictions"])
    assert len(predictions) == 5


@patch("api.db.sessions", new=mock_sessions)
def test_plot():
    problem = opti.problems.Ackley(3)
    problem.create_initial_data(5)

    session_id = client.post(
        "/session/create?method=SOBO",
        json={"problem": problem.to_config()},
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/plot/data")
    assert response.ok

    response = client.get(f"/session/{session_id}/plot/model")
    assert response.ok

    response = client.get(f"/session/{session_id}/plot/model")
    assert response.ok

    response = client.get(f"/session/{session_id}/plot/residuals")
    assert response.ok

    response = client.get(f"/session/{session_id}/plot/residuals?cross_validate=true")
    assert response.ok


@patch("api.db.sessions", new=mock_sessions)
def test_method_parameters():
    problem = opti.problems.ZDT1(3)
    problem.create_initial_data(5)

    # create session with method arguments
    session_id = client.post(
        "/session/create?method=ParEGO",
        json={
            "problem": problem.to_config(),
            "parameters": {"mc_samples": 42, "restarts": 5},
        },
    ).json()["session_id"]

    # query session info
    info = client.get(f"/session/{session_id}").json()
    assert "problem" in info
    assert info["method"] == "ParEGO"
    assert info["parameters"]["mc_samples"] == 42
    assert info["parameters"]["restarts"] == 5


if __name__ == "__main__":
    test_session_handling()
