from typing import TYPE_CHECKING, Callable
from abc import ABC, abstractmethod

import pyomo.environ as pyo

if TYPE_CHECKING:
    from problem_config import FeatureType


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

    def as_pyomo_constraint(
        self, model: pyo.ConcreteModel, feat_list: list["FeatureType"]
    ):
        features = self._get_feature_vars(model, feat_list)
        return self._as_pyomo_constraint(features)

    @abstractmethod
    def _as_pyomo_constraint(self, features: list[pyo.Var]) -> pyo.Constraint:
        pass


class ExpressionConstraint(Constraint):
    def _as_pyomo_constraint(self, features: list[pyo.Var]) -> pyo.Constraint:
        return pyo.Constraint(expr=self._get_expr(features))

    @abstractmethod
    def _get_expr(self, features) -> pyo.Expression:
        pass


class FunctionalConstraint(Constraint):
    def _as_pyomo_constraint(self, features: list[pyo.Var]) -> pyo.Constraint:
        return pyo.Constraint(rule=self._get_function(features))

    @abstractmethod
    def _get_function(self, features) -> Callable[..., pyo.Expression]:
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
