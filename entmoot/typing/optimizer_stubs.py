"""Stubs for Pyomo and Gurobi models"""

import gurobipy
import pyomo.environ as pyo
from pyomo.core.base.var import IndexedVar, _GeneralVarData


class PyomoModelT(pyo.ConcreteModel):
    _all_feat: list[_GeneralVarData | dict[int, _GeneralVarData]]
    indices_features: pyo.Set
    x: IndexedVar


class GurobiModelT(gurobipy.Model):
    _all_feat: list[gurobipy.Var | dict[int, gurobipy.Var]]
