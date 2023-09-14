from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import pyomo.environ as pyo

if TYPE_CHECKING:
    from problem_config import FeatureType

class Constraint(ABC):
    def __init__(self, features: list[str]):
        self.features = features

    def _get_feature_idxs(self, feat_list: list["FeatureType"]):
        """Get the index of each of the features in the constraint expression"""
        all_keys = [feat.name for feat in feat_list]
        feat_idxs = [all_keys.index(key) for key in self.features]
        return feat_idxs
    
    def as_pyomo_constraint(self):
        return pyo.Constraint()

class ConstraintType(ABC):
    """Contains the type of constraint - whether it is an expression, or a function"""
    @abstractmethod
    def as_pyomo_constraint(self):
        pass


class ExpressionConstraint:
    def as_pyomo_constraint(self):
        return pyo.Constraint(rule=self._get_expr())
    
    @abstractmethod
    def _get_expr(self):
        pass
    
class FunctionalConstraint:
    def as_pyomo_constraint(self):
        return pyo.Constraint(rule=self._get_function())
    
    @abstractmethod
    def _get_expr(self):
        pass


