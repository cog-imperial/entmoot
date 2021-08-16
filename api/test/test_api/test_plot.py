import opti
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def problem():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(10)
    return problem


def test_plot_data(problem):
    r = client.post("/session/create?method=SOBO", json={"problem": problem.to_config()})
    session_id = r.json()["session_id"]
    response = client.get(f"/session/{session_id}/plot/data")
    assert response.ok
    assert response.headers["content-type"].startswith("text/html")


def test_plot_model(problem):
    session_id = client.post(
        "/session/create?method=SOBO",
        json={"problem": problem.to_config()},
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/plot/model")
    assert response.ok
    assert response.headers["content-type"].startswith("text/html")


def test_plot_parameters(problem):
    session_id = client.post(
        "/session/create?method=SOBO",
        json={"problem": problem.to_config()},
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/plot/model")
    assert response.ok
    assert response.headers["content-type"].startswith("text/html")


def test_plot_parameters(problem):
    session_id = client.post(
        "/session/create?method=SOBO",
        json={"problem": problem.to_config()},
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/plot/residuals")
    assert response.ok
    assert response.headers["content-type"].startswith("text/html")
 
    response = client.get(f"/session/{session_id}/plot/residuals?cross_validate=true")
    assert response.ok
    assert response.headers["content-type"].startswith("text/html")
