from unittest.mock import patch

import pytest
from mongomock import MongoClient

from api.db import (
    add_session,
    delete_session,
    get_session_config,
    list_sessions,
    update_session_data,
)

mock_sessions = MongoClient("mongodb://localhost:27017/").admin["sessions"]


@patch("api.db.sessions", new=mock_sessions)
def test_add_get_delete():
    # without username
    session_id = add_session({"problem": {"name": "foo"}})
    config = get_session_config(session_id)
    assert config["problem"] == {"name": "foo"}
    assert "created" in config

    # with username
    session_id = add_session({"problem": {"name": "foo"}}, username="RaynorJ")
    config = get_session_config(session_id)
    assert config["problem"] == {"name": "foo"}
    assert config["user"] == "RaynorJ"
    assert "created" in config

    delete_session(session_id)
    with pytest.raises(Exception):
        get_session_config(session_id)


@patch("api.db.sessions", new=mock_sessions)
def test_update():
    config = {
        "parameters": {"mc_samples": 1},
        "problem": {"name": "foo", "data": [1, 2]},
    }
    session_id = add_session(config)
    update_session_data(session_id, [1, 2, 3])
    config2 = get_session_config(session_id)
    assert config2["parameters"] == config["parameters"]
    assert config2["problem"]["data"] == [1, 2, 3]


@patch("api.db.sessions", new=mock_sessions)
def test_list():
    add_session({"problem": {"name": "foo"}}, username="BayesT")
    add_session({"problem": {"name": "bar"}}, username="BayesT")

    # list all
    assert len(list_sessions()) >= 2

    # list for given user
    assert len(list_sessions("BayesT")) == 3  # including the demo session


@patch("api.db.sessions", new=mock_sessions)
def test_demo():
    # users should always see the demo project
    assert len(list_sessions("I'm new here")) == 1

    config = get_session_config("demo")
    assert "problem" in config
    assert config["method"] == "ParEGO"
