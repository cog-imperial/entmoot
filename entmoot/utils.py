from collections import namedtuple
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy.special import comb


OptResult = namedtuple(
    "OptResult",
    ["opt_point", "opt_val", "mu_unscaled", "unc_unscaled", "active_leaf_enc"],
)



@dataclass
class UncParams:
    beta: float = 1.96 # >0
    acq_sense: Literal["exploration", "penalty"] = "exploration"
    dist_trafo: Literal["normal", "standard"] = "normal"
    dist_metric: Literal["euclidean_squared", "l1", "l2"] = "euclidean_squared"
    cat_metric: Literal["overlap", "of", "goodall4"] = "overlap"

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


@dataclass
class EntingParams:
    unc_params: "UncParams" = field(default_factory=UncParams)
    tree_train_params: "TreeTrainParams" = field(default_factory=TreeTrainParams)

    @staticmethod
    def from_dict(d: dict):
        d_unc_params = d.get("unc_params", {})
        d_tree_train_params = d.get("tree_train_params", {})
        d_train_params = d_tree_train_params.get("train_params", {})
        d_tree_train_params = {k: v for k, v in d_tree_train_params.items() if k!="train_params"}

        return EntingParams(
            unc_params=UncParams(**d_unc_params),
            tree_train_params=TreeTrainParams(
                train_params=TrainParams(**d_train_params),
                **d_tree_train_params
            )
        )



def grid(dimension: int, levels: int) -> np.ndarray:
    """Construct a regular grid on the unit simplex.

    The number of grid points is L = (n+m-1)!/(n!*(m-1)!)
    where m is the dimension and n+1 is the number of levels.

    Args:
        dimension (int): Number of variables.
        levels (int): Number of levels for each variable.

    Returns:
       array: Regularily spaced points on the unit simplex.

    Examples:
        --> simplex_grid(3, 3)
        array([
            [0. , 0. , 1. ],
            [0. , 0.5, 0.5],
            [0. , 1. , 0. ],
            [0.5, 0. , 0.5],
            [0.5, 0.5, 0. ],
            [1. , 0. , 0. ]
        ])

    References:
        Nijenhuis and Wilf, Combinatorial Algorithms, Chapter 5, Academic Press, 1978.
    """
    m = dimension
    n = levels - 1
    L = int(comb(dimension - 1 + levels - 1, dimension - 1, exact=True))

    x = np.zeros(m, dtype=int)
    x[-1] = n

    out = np.empty((L, m), dtype=int)
    out[0] = x

    h = m
    for i in range(1, L):
        h -= 1

        val = x[h]
        x[h] = 0
        x[h - 1] += 1
        x[-1] = val - 1

        if val != 1:
            h = m

        out[i] = x

    return out / n


def sample(dimension: int, n_samples: int = 1) -> np.ndarray:
    """Sample uniformly from the unit simplex.

    Args:
        dimension (int): Number of dimensions.
        n_samples (int): Number of samples to draw.

    Returns:
        array, shape=(n_samples, dimesnion): Random samples from the unit simplex.
    """
    s = np.random.standard_exponential((n_samples, dimension))
    return (s.T / s.sum(axis=1)).T
