from collections import namedtuple
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from scipy.special import comb


OptResult = namedtuple(
    "OptResult",
    ["opt_point", "opt_val", "mu_unscaled", "unc_unscaled", "active_leaf_enc"],
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
