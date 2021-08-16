from unittest.mock import patch

import opti
import pandas as pd
from fastapi.testclient import TestClient
from mongomock.mongo_client import MongoClient

from api.main import app

client = TestClient(app)

mock_sessions = MongoClient("mongodb://localhost:27017/").admin["sessions"]


@patch("api.db.sessions", new=mock_sessions)
def test_zhakarov():
    # Zakharov problem with 1 categorical variable
    problem = opti.problems.Zakharov_categorical(n_inputs=3)
    problem.create_initial_data(5)

    response = client.post(
        "/session/create?method=ENTMOOT", json={"problem": problem.to_config()}
    )
    print(response.json())
    session_id = response.json()["session_id"]

    # ask
    response = client.get(f"/session/{session_id}/ask")
    df = pd.DataFrame(**response.json())
    assert len(df) == 1
    assert problem.inputs.contains(df).all()

    # plot
    response = client.get(f"/session/{session_id}/plot/all")
    assert response.headers["content-type"].startswith("text/html")


if __name__ == "__main__":
    test_zhakarov()
