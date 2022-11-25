import numpy as np
from numpy.typing import ArrayLike
from typing import Iterable


def eval_multi_obj_cat_testfunc(X: ArrayLike, n_obj: int = 2) -> np.ndarray:
    """
    Benchmark function with at least four input variables and one or two outputs
    :param X: Usually a numpy array or a list of tuples. Each row (or tuple) consists of at least four entries with the
    following properties:
    - The first one is a categorical variable with the possible values "blue", "orange" and "gray".
    - The second one is an integer variable.
    - The third one is a binary variable.
    - The fourth one and all additional variables are real numbers.
    :param n_obj: number of objectives (one or two)
    :return: objective value(s) for each data point
    """

    # without the dtype=object paramer, each entry of X is converted into a string
    X = np.array(X, dtype=object)

    def compute_objectives(xi: Iterable):
        if xi[0] == "blue":
            return(
                xi[1]*xi[2]*np.sin(sum(xi[3:])) + xi[1]*(1-xi[2])*np.cos(sum(xi[3:])),
                xi[1]*xi[2]*sum(xi[3:]) - xi[1]*(1-xi[2])*sum(xi[3:])
            )
        elif xi[0] == "orange":
            return(
                xi[1]*xi[2]*(sum(xi[3:])/len(xi[3:]))**2 + xi[1]*(1-xi[2])*(sum(xi[3:])/len(xi[3:]))**3,
                np.sqrt(abs(xi[1]*xi[2]*sum(xi[3:]) + xi[1]*(1-xi[2])*sum(xi[3:])))
            )
        elif xi[0] == "gray":
            return(
                xi[2]*xi[3]**xi[1],
                -(1-xi[2]) * xi[3] ** xi[1]
            )
        else:
            raise IOError(f"You provided the illegal value {xi[0]} for the categorical variable. Allowed values are "
                          f"'blue', 'orange' and 'gray'")
    if n_obj == 2:
        return np.array([compute_objectives(xi) for xi in X])
    elif n_obj == 1:
        return np.array([[sum(compute_objectives(xi))/2] for xi in X])
    else:
        raise IOError(f"You provided the illegal value {n_obj} for the number of objectives. "
                      f"Allowed values are 1 and 2")
