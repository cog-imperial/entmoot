# flake8: noqa: F401
from bo.algorithm.algorithm import Algorithm
from bo.algorithm.pls import PLS
from bo.algorithm.random import RandomSearch

try:
    from bo.algorithm.parego import ParEGO
    from bo.algorithm.sobo import SOBO
    from bo.algorithm.tsemo import TSEMO
except ImportError:
    ParEGO = None
    SOBO = None
    TSEMO = None
try:
    from bo.algorithm.entmoot import Entmoot
except ModuleNotFoundError:
    Entmoot = None

all = ["Algorithm", "RandomSearch", "PLS"]
dep_entmoot = ["Entmoot"] if Entmoot is not None else []
dep_botorch = ["TSEMO", "ParEGO", "SOBO"] if ParEGO is not None else []

__all__ = all + dep_entmoot + dep_botorch
