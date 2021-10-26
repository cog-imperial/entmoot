import numpy as np
import opti
import pandas as pd
import pytest

import bo
from bo.metric import cross_validate


def test_cv_sobo():
    # test cross-validation on a single objective GP
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1])],
        outputs=[opti.Continuous("y1", [0, 1])],
        data=pd.DataFrame(
            {
                "x1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
                "y1": [-0.0694, 0.215, 0.4062, 0.505, 0.8006, 0.9487, 0.9531],
            }
        ),
    )
    optimizer = bo.algorithm.SOBO(problem)
    n = len(problem.data)

    # 5-fold cross-validation
    results = cross_validate(optimizer, n_splits=5)
    assert len(results["predictions"]) == n
    assert len(results["parameters"]) == 5

    # leave-one-out cross-validation
    results = cross_validate(optimizer, n_splits=7)
    assert len(results["predictions"]) == n
    assert len(results["parameters"]) == 7

    # still leave-one-out cross-validation
    results = cross_validate(optimizer, n_splits=10)
    assert len(results["predictions"]) == n
    assert len(results["parameters"]) == 7

    # ensure that cross validation works on data with duplicate row-index
    optimizer._problem.data.index = [0, 0, 0, 0, 0, 0, 0]
    cross_validate(optimizer, n_splits=3)
    assert len(results["predictions"]) == n

    # error: not enough data
    optimizer._problem.data = optimizer.data.iloc[0:2]
    with pytest.raises(ValueError):
        cross_validate(
            optimizer,
        )


def test_cv_parego():
    # test leave-one-out cross validation on a multi-objective GP
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1])],
        outputs=[opti.Continuous("y1", [0, 1]), opti.Continuous("y2", [0, 1])],
    )
    problem.data = pd.DataFrame(
        {
            "x1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0],
            "y1": [-0.0694, 0.215, 0.4062, 0.505, 0.8006, 0.9487, 0.9531],
            "y2": [1.003, 0.9761, 0.9407, 0.7561, 0.5717, 0.3662, 0.3612],
        }
    )
    optimizer = bo.algorithm.ParEGO(problem)
    results = cross_validate(optimizer, n_splits=5)
    assert len(results["predictions"]) == len(problem.data)
    assert len(results["parameters"]) == 2 * 5

    # test cross-validation on a ModelListGP
    problem.data = pd.DataFrame(
        {
            "x1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "y1": [np.nan, 0.215, 0.4062, 0.505, 0.8006, 0.9487],
            "y2": [1.003, np.nan, np.nan, 0.7561, 0.5717, 0.3662],
        }
    )
    optimizer = bo.algorithm.ParEGO(problem)
    results = cross_validate(
        optimizer,
    )
    assert len(results["predictions"]) == len(problem.data)
    assert len(results["parameters"]) == 2 * 5


def test_cv_randomsearch():
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1])],
        outputs=[opti.Continuous("y1", [0, 1]), opti.Continuous("y2", [0, 1])],
    )
    optimizer = bo.algorithm.RandomSearch(problem)
    with pytest.raises(ValueError):
        cross_validate(
            optimizer,
        )
