"""Dataclasses containing the parameters for Enting models"""

from typing import Literal
from dataclasses import dataclass, field

class ParamValidationError(ValueError):
    """A model parameter takes an invalid value."""
    pass

@dataclass
class UncParams:
    """
    This dataclass contains all uncertainty parameters.
    """
    #: weight for penalty/exploration part in objective function
    beta: float = 1.96
    #: the predictions of the GBT model are cut off, if their absolute value exceeds
    #: bound_coeff * variance of the y-values.
    bound_coeff: float = 0.5
    #: "exploration": try to find good points far away from known training data,
    #: "penalty": stay close to explored areas and try to find even better points there.
    acq_sense: Literal["exploration", "penalty"] = "exploration"
    #: controls two different types of transformations by scaling/shifting.
    #: "normal": shift by lower bound, scale by difference of smalles and largest value
    #: "standard": shift by mean, scale by standard deviation
    dist_trafo: Literal["normal", "standard"] = "normal"
    #: compute distance measure using the l_1, the l_2 or the squared l_2 norm.
    dist_metric: Literal["euclidean_squared", "l1", "l2"] = "euclidean_squared"
    #: different ways to compute the distance of categorical features
    cat_metric: Literal["overlap", "of", "goodall4"] = "overlap"

    def __post_init__(self):
        if self.beta < 0.0:
            raise ParamValidationError(
                f"Value for 'beta' is {self.beta}; must be positive."
            )
        
        if self.acq_sense not in ("exploration", "penalty"):
            raise ParamValidationError(
                f"Value for 'acq_sense' is '{self.acq_sense}'; must be in ('exploration', 'penalty')."
            )
        

@dataclass
class TrainParams:
    """
    This dataclass contains all hyperparameters that are used by lightbm during training and
    documented here https://lightgbm.readthedocs.io/en/latest/Parameters.html
    """
    # lightgbm training hyperparameters
    objective: str = "regression"
    metric: str = "rmse"
    boosting: str = "gbdt"
    num_boost_round: int = 100
    max_depth: int = 3
    min_data_in_leaf: int = 1
    min_data_per_group: int = 1
    verbose: int = -1


@dataclass
class TreeTrainParams:
    """
    This dataclass contains all parameters needed for the tree training.
    """
    train_params: "TrainParams" = field(default_factory=dict) # type: ignore
    train_lib: Literal["lgbm"] = "lgbm"

    def __post_init__(self):
        if isinstance(self.train_params, dict):
            self.train_params = TrainParams(**self.train_params)
        
        if self.train_lib not in ("lgbm",):
            raise ParamValidationError(
                f"Value for 'train_lib' is {self.train_lib}; must be in ('lgbm',)"
            )


@dataclass
class EntingParams:
    """Contains parameters for a mean and uncertainty model.
    
    Provides a structured dataclass for the parameters of an Enting model, 
    alongside default values and some light data validation."""
    unc_params: "UncParams" = field(default_factory=dict) # type: ignore
    tree_train_params: "TreeTrainParams" = field(default_factory=dict) # type: ignore
    
    def __post_init__(self):
        if isinstance(self.unc_params, dict):
            self.unc_params = UncParams(**self.unc_params)

        if isinstance(self.tree_train_params, dict):
            self.tree_train_params = TreeTrainParams(**self.tree_train_params)