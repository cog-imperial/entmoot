from unittest.mock import patch

import opti
import pandas as pd
from fastapi.testclient import TestClient
from mongomock.mongo_client import MongoClient

from api.main import app

mock_sessions = MongoClient("mongodb://localhost:27017/").admin["sessions"]

client = TestClient(app)


@patch("api.db.sessions", new=mock_sessions)
def test_sphere():
    problem = opti.problems.Sphere(3)
    problem.create_initial_data(5)

    session_id = client.post(
        "/session/create?method=SOBO", json={"problem": problem.to_config()}
    ).json()["session_id"]

    # ask
    response = client.get(f"/session/{session_id}/ask?n_proposals=2")
    assert response.ok

    # tell
    X = pd.DataFrame(**response.json())
    Y = problem.eval(X)
    data = pd.concat([X, Y], axis=1)
    response = client.post(f"/session/{session_id}/tell", json=data.to_dict("split"))
    assert response.ok

    # predict
    X = problem.inputs.sample(20)
    response = client.post(f"/session/{session_id}/predict", json=X.to_dict("split"))
    assert response.ok

    # get all data
    response = client.get(f"/session/{session_id}/data")
    data = pd.DataFrame(**response.json())

    # cross-validate
    response = client.get(f"/session/{session_id}/cross_validate?n_splits=3")
    predictions = pd.DataFrame(**response.json()["predictions"])
    parameters = pd.DataFrame(**response.json()["parameters"])
    assert len(predictions) == len(data)
    assert len(parameters) == 3
