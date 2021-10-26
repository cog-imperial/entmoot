from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Extra, Field

import bo


class Parameter(BaseModel):
    name: str
    type: str
    domain: List

    class Config:
        extra = Extra.allow
        schema_extra = {
            "example": {
                "name": "x1",
                "type": "continuous",
                "domain": [0, 1],
            }
        }


class Objective(BaseModel):
    name: str
    type: str

    class Config:
        extra = Extra.allow
        schema_extra = {
            "example": {
                "name": "x1",
                "type": "minimize",
            }
        }


class Constraint(BaseModel):
    type: str
    names: List[str]

    class Config:
        extra = Extra.allow
        schema_extra = {
            "example": {
                "type": "linear-inequality",
                "names": ["x1", "x2"],
                "lhs": [1, 1, 1],
                "rhs": 1,
            }
        }


class DataFrame(BaseModel):
    index: List[Union[int, str]]
    columns: List[str]
    data: List[List]

    class Config:
        schema_extra = {
            "index": [0, 1, 2],
            "columns": ["x1", "x2", "y1"],
            "data": [[0, 1, 2], [0.5, 0.5, 3], [1, 0, 4]],
        }


class Problem(BaseModel):
    inputs: List[Parameter]
    outputs: List[Parameter]
    objectives: List[Objective] = None
    constraints: List[Constraint] = None
    data: DataFrame = None

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "name": "x1",
                        "type": "continuous",
                        "domain": [0, 1],
                    },
                    {
                        "name": "x2",
                        "type": "continuous",
                        "domain": [0, 1],
                    },
                ],
                "outputs": [
                    {
                        "name": "y1",
                        "type": "continuous",
                        "domain": [0, 1],
                    }
                ],
            }
        }


class SessionInfo(BaseModel):
    # https://github.com/samuelcolvin/pydantic/issues/288
    id: str = Field(..., alias="_id")
    user: Optional[str] = None
    name: Optional[str] = None
    method: str
    created: datetime
    last_changed: datetime


class MethodName(str, Enum):
    parego = "ParEGO"
    tsemo = "TSEMO"
    sobo = "SOBO"
    entmoot = "ENTMOOT"
    randomsearch = "RandomSearch"
    pls = "PLS"


methods = {
    "ParEGO": bo.algorithm.ParEGO,
    "TSEMO": bo.algorithm.TSEMO,
    "SOBO": bo.algorithm.SOBO,
    "ENTMOOT": bo.algorithm.Entmoot,
    "RandomSearch": bo.algorithm.RandomSearch,
    "PLS": bo.algorithm.PLS,
}
