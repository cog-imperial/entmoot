"""Dataclasses containing the parameters for Enting models"""

from typing import Literal
from dataclasses import dataclass, field

@dataclass
class UncParams:
    beta: float = 1.96
    bound_coeff: float = 0.5
    acq_sense: Literal["exploration", "penalty"] = "exploration"
    dist_trafo: Literal["normal", "standard"] = "normal"
    dist_metric: Literal["euclidean_squared", "l1", "l2"] = "euclidean_squared"
    cat_metric: Literal["overlap", "of", "goodall4"] = "overlap"

    def __post_init__(self):
        if self.beta < 0.0:
            raise ValueError(f"Value for 'beta' is {self.beta} but must be '>= 0.0'.")

@dataclass
class TrainParams:
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
    train_params: "TrainParams" = field(default_factory=TrainParams)
    train_lib: Literal["lgbm"] = "lgbm"

    @staticmethod
    def fromdict(d: dict):
        d_train_params = d.get("train_params", {})
        d_tree_train_params = {k: v for k, v in d.items() if k!="train_params"}
        return TreeTrainParams(
            train_params=TrainParams(**d_train_params),
            **d_tree_train_params
        )


@dataclass
class EntingParams:
    unc_params: "UncParams" = field(default_factory=UncParams)
    tree_train_params: "TreeTrainParams" = field(default_factory=TreeTrainParams)

    @staticmethod
    def fromdict(d: dict):
        d_unc_params = d.get("unc_params", {})
        d_tree_train_params = d.get("tree_train_params", {})

        return EntingParams(
            unc_params=UncParams(**d_unc_params),
            tree_train_params=TreeTrainParams.fromdict(d_tree_train_params)
        )