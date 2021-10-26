import os
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import opti
from fastapi.exceptions import HTTPException
from pymongo import MongoClient

import bo.algorithm
from api.schema import methods

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")

db = MongoClient(MONGODB_URL).admin
sessions = db["sessions"]


def get_demo_session(with_problem=False):
    now = f"{datetime.utcnow()}"
    session = {
        "_id": "demo",
        "name": "Demo Project",
        "method": "ParEGO",
        "created": now,
        "last_modified": now,
    }
    if with_problem:
        session["problem"] = opti.problems.Coating().to_config()
    return session


def add_session(config: dict, username: Optional[str] = None) -> str:
    session_id = uuid4().hex
    config["_id"] = session_id
    if username:
        config["user"] = username
    timestamp = f"{datetime.utcnow()}"
    config["name"] = config["problem"]["name"]
    config["created"] = timestamp
    config["last_changed"] = timestamp
    sessions.insert_one(config)
    return session_id


def check_session_exists(session_id):
    if not sessions.find_one({"_id": session_id}):
        raise HTTPException(status_code=404, detail="Session not found")


def list_sessions(username: Optional[str] = None) -> List[str]:
    projection = {"user", "name", "method", "created", "last_changed"}
    if username:
        ss = list(sessions.find(filter={"user": username}, projection=projection))
        return [get_demo_session()] + ss
    return list(sessions.find(projection=projection))


def delete_session(session_id: str):
    sessions.find_one_and_delete({"_id": session_id})


def get_session_config(session_id) -> dict:
    if session_id == "demo":
        return get_demo_session(with_problem=True)
    check_session_exists(session_id)
    return sessions.find_one({"_id": session_id})


def get_session_optimizer(session_id: str) -> bo.algorithm.Algorithm:
    config = get_session_config(session_id)
    return methods[config["method"]].from_config(config)


def get_session_problem(session_id: str) -> opti.Problem:
    config = get_session_config(session_id)
    return opti.Problem.from_config(config["problem"])


def update_session_data(session_id: str, data: dict):
    check_session_exists(session_id)
    sessions.update_one(
        {"_id": session_id},
        {"$set": {"problem.data": data, "last_changed": f"{datetime.utcnow()}"}},
    )
