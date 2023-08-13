import numpy as np
from numpy.typing import ArrayLike
from typing import Iterable
from entmoot import ProblemConfig


def build_small_single_obj_categorical_problem(
    problem_config: ProblemConfig, no_cat=False
):
    """
    Builds a small test example which is frequently used by the tests.
    :param problem_config: ProblemConfig object where features and objectives are added
    :param n_obj: Number of objectives
    :return: None, the problem definition happens "inplace"
    """
    if not no_cat:
        problem_config.add_feature("categorical", ("blue", "orange", "gray"))
    problem_config.add_feature("integer", (5, 6))
    problem_config.add_feature("binary")
    problem_config.add_feature("real", (5.0, 6.0))

    problem_config.add_min_objective()


def eval_small_single_obj_cat_testfunc(X: ArrayLike, no_cat=False) -> np.ndarray:
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

    def compute_objectives(xi: Iterable, no_cat=False):
        if no_cat:
            return (
                xi[1] * xi[2] * np.sin(sum(xi[3:]))
                + xi[1] * (1 - xi[2]) * np.cos(sum(xi[3:])),
                xi[1] * xi[2] * sum(xi[3:]) - xi[1] * (1 - xi[2]) * sum(xi[3:]),
            )

        if xi[0] == "blue":
            return (
                xi[1] * xi[2] * np.sin(sum(xi[3:]))
                + xi[1] * (1 - xi[2]) * np.cos(sum(xi[3:])),
                xi[1] * xi[2] * sum(xi[3:]) - xi[1] * (1 - xi[2]) * sum(xi[3:]),
            )
        elif xi[0] == "orange":
            return (
                xi[1] * xi[2] * (sum(xi[3:]) / len(xi[3:])) ** 2
                + xi[1] * (1 - xi[2]) * (sum(xi[3:]) / len(xi[3:])) ** 3,
                np.sqrt(
                    abs(xi[1] * xi[2] * sum(xi[3:]) + xi[1] * (1 - xi[2]) * sum(xi[3:]))
                ),
            )
        elif xi[0] == "gray":
            return (xi[2] * xi[3] ** xi[1], -(1 - xi[2]) * xi[3] ** xi[1])
        else:
            raise IOError(
                f"You provided the illegal value {xi[0]} for the categorical variable. Allowed values are "
                f"'blue', 'orange' and 'gray'"
            )

    return np.array([[sum(compute_objectives(xi, no_cat=no_cat)) / 2] for xi in X])


def build_multi_obj_categorical_problem(
    problem_config: ProblemConfig, n_obj: int = 2, no_cat=False
):
    """
    Builds a small test example which is frequently used by the tests.
    :param problem_config: ProblemConfig object where features and objectives are added
    :param n_obj: Number of objectives
    :return: None, the problem definition happens "inplace"
    """
    if not no_cat:
        problem_config.add_feature("categorical", ("blue", "orange", "gray"))
    problem_config.add_feature("integer", (5, 6))
    problem_config.add_feature("binary")
    problem_config.add_feature("real", (5.0, 6.0))
    problem_config.add_feature("real", (4.6, 6.0))
    problem_config.add_feature("real", (5.0, 6.0))

    for _ in range(n_obj):
        problem_config.add_min_objective()


def eval_multi_obj_cat_testfunc(
    X: ArrayLike, n_obj: int = 2, no_cat=False
) -> np.ndarray:
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

    def compute_objectives(xi: Iterable, no_cat=False):
        if no_cat:
            return (
                xi[1] * xi[2] * np.sin(sum(xi[3:]))
                + xi[1] * (1 - xi[2]) * np.cos(sum(xi[3:])),
                xi[1] * xi[2] * sum(xi[3:]) - xi[1] * (1 - xi[2]) * sum(xi[3:]),
            )

        if xi[0] == "blue":
            return (
                xi[1] * xi[2] * np.sin(sum(xi[3:]))
                + xi[1] * (1 - xi[2]) * np.cos(sum(xi[3:])),
                xi[1] * xi[2] * sum(xi[3:]) - xi[1] * (1 - xi[2]) * sum(xi[3:]),
            )
        elif xi[0] == "orange":
            return (
                xi[1] * xi[2] * (sum(xi[3:]) / len(xi[3:])) ** 2
                + xi[1] * (1 - xi[2]) * (sum(xi[3:]) / len(xi[3:])) ** 3,
                np.sqrt(
                    abs(xi[1] * xi[2] * sum(xi[3:]) + xi[1] * (1 - xi[2]) * sum(xi[3:]))
                ),
            )
        elif xi[0] == "gray":
            return (xi[2] * xi[3] ** xi[1], -(1 - xi[2]) * xi[3] ** xi[1])
        else:
            raise IOError(
                f"You provided the illegal value {xi[0]} for the categorical variable. Allowed values are "
                f"'blue', 'orange' and 'gray'"
            )

    if n_obj == 2:
        return np.array([compute_objectives(xi, no_cat=no_cat) for xi in X])
    elif n_obj == 1:
        return np.array([[sum(compute_objectives(xi, no_cat=no_cat)) / 2] for xi in X])
    else:
        raise IOError(
            f"You provided the illegal value {n_obj} for the number of objectives. "
            f"Allowed values are 1 and 2"
        )
