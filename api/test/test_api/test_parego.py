from unittest.mock import patch

import numpy as np
import opti
import pandas as pd
from fastapi.testclient import TestClient
from mongomock.mongo_client import MongoClient

from api.main import app

mock_sessions = MongoClient("mongodb://localhost:27017/").admin["sessions"]
client = TestClient(app)


@patch("api.db.sessions", new=mock_sessions)
def test_zdt1():
    # ZDT1 problem with 10 initial points and some missing data
    problem = opti.problems.ZDT1(3)
    problem.create_initial_data(10)
    problem.data.loc[0, "y0"] = np.nan
    problem.data.loc[0, "y1"] = np.nan
    problem.data.loc[1, "y1"] = np.nan

    session_id = client.post(
        "/session/create?method=ParEGO", json={"problem": problem.to_config()}
    ).json()["session_id"]

    response = client.get(f"/session/{session_id}/ask")
    df = pd.DataFrame(**response.json())
    assert len(df) == 1
    assert problem.inputs.contains(df).all()

    # create some inputs and retrieve predictions
    response = client.get(f"/session/{session_id}/predict_front")
    front = pd.DataFrame(**response.json())
    assert len(front) == 5
