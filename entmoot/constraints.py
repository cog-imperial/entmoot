from typing import TYPE_CHECKING, Callable
from abc import ABC, abstractmethod

import pyomo.environ as pyo

if TYPE_CHECKING:
    from problem_config import FeatureType

ConstraintFunctionType = Callable[[pyo.ConcreteModel, int], pyo.Expression]

class Constraint(ABC):
    def __init__(self, features_keys: list[str]):
        self.feature_keys = features_keys

    def _get_feature_vars(
        self, model: pyo.ConcreteModel, feat_list: list["FeatureType"]
    ) -> list[pyo.Var]:
        """Return a list of all the pyo.Vars, in the order of the constraint definition"""
        all_keys = [feat.name for feat in feat_list]
        feat_idxs = [all_keys.index(key) for key in self.feature_keys]
        features = [model._all_feat[i] for i in feat_idxs]
        return features


class ExpressionConstraint(Constraint):
    """Constraints defined by pyomo.Expressions.

    For constraints that can be simply defined by an expression of variables.    
    """
    def as_pyomo_constraint(self, model: pyo.ConcreteModel, feat_list: list["FeatureType"]) -> pyo.Constraint:
        features = self._get_feature_vars(model, feat_list)
        return pyo.Constraint(expr=self._get_expr(features))

    @abstractmethod
    def _get_expr(self, features) -> pyo.Expression:
        pass


class FunctionalConstraint(Constraint):
    """A constraint that uses a functional approach.
    
    For constraints that require creating intermediate variables and access to the model."""
    def as_pyomo_constraint(self, model: pyo.ConcreteModel, feat_list: list["FeatureType"]) -> pyo.Constraint:
        features = self._get_feature_vars(model, feat_list)
        return pyo.Constraint(rule=self._get_function(model, features))

    @abstractmethod
    def _get_function(self, features) -> ConstraintFunctionType:
        pass


class LinearConstraint(ExpressionConstraint):
    """Constraint that is a function of X @ C, where X is the feature list, and C
    is the list of coefficients."""

    def __init__(self, feature_keys: list[str], coefficients: list[float], rhs: float):
        self.coefficients = coefficients
        self.rhs = rhs
        super().__init__(feature_keys)

    def _get_lhs(self, features: pyo.ConcreteModel) -> pyo.Expression:
        """Get the left-hand side of the linear constraint"""
        return sum(f * c for f, c in zip(features, self.coefficients))


class LinearEqualityConstraint(LinearConstraint):
    def _get_expr(self, features):
        return self._get_lhs(features) == self.rhs


class LinearInequalityConstraint(LinearConstraint):
    def _get_expr(self, features):
        return self._get_lhs(features) <= self.rhs


class NChooseKConstraint(FunctionalConstraint):
    tol: float = 1e-6
    M: float = 1e6
    def __init__(self, feature_keys: list[str], min_count: int, max_count: int, none_also_valid: bool = False):
        self.min_count = min_count
        self.max_count = max_count
        self.none_also_valid = none_also_valid
        super().__init__(feature_keys)

    def _get_function(self, model, features):
        model.feat_selected = pyo.Var(range(len(features)), domain=pyo.Binary, initialize=0)
        model.ub_selected = pyo.ConstraintList()
        model.lb_selected = pyo.ConstraintList()
        for i in range(len(features)):
            model.ub_selected.add(expr=model.feat_selected[i]*self.M >= features[i])
            model.lb_selected.add(expr=model.feat_selected[i]*self.tol <= features[i])
        
        def inner(model, i):
            return sum(model.feat_selected.values()) <= self.max_count

        return inner