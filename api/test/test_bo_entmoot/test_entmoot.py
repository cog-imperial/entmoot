import pytest

try:
    # running these tests requires entmoot, gurobipy and access to the Gurobi cloud
    import entmoot  # noqa: F401
    import gurobipy  # noqa: F401
except ImportError:
    pytest.skip("Gurobi not available", allow_module_level=True)

import numpy as np
import opti
import pandas as pd

from bo.algorithm import Entmoot


def test_entmoot_singleobj_categorical():
    problem = opti.problems.Zakharov_categorical(n_inputs=3)
    n_inital_points = 5
    problem.create_initial_data(n_inital_points)
    optimizer = Entmoot(problem)

    # predict
    x = pd.DataFrame(
        [[3.5, 0.21, "one"], [3, -1.1115, "two"]], columns=problem.inputs.names
    )
    y = optimizer.predict(x)
    assert len(y) == 2

    # propose
    optimizer.propose(n_proposals=3)

    # run with several proposals per iteration
    n_steps = 3
    n_proposals = 2
    optimizer.run(n_proposals=n_proposals, n_steps=n_steps)
    assert len(optimizer.data) == n_inital_points + n_steps * n_proposals


def test_entmoot_singleobj_nchoosek():
    problem = opti.problems.Zakharov_NChooseKConstraint(n_inputs=5, n_max_active=3)

    data = pd.DataFrame(
        [
            [2, 0, -2, 0, 0],
            [0, 3, -2, 0, 0],
            [0, 0, -2, 0, 0],
            [2, 0, -2, 6, 0],
            [0, -3, 2, 0, -4],
        ],
        columns=problem.inputs.names,
    )
    data["y"] = problem.eval(data)
    problem.data = data

    optimizer = Entmoot(problem)

    # propose
    X_next = optimizer.propose(n_proposals=3)
    assert len(X_next) == 3

    # run with one proposal per iteration
    n_steps = 3
    optimizer.run(n_proposals=1, n_steps=n_steps)
    assert len(optimizer.data) == len(data) + n_steps

    # run with several proposals per iteration
    n_proposals = 2
    optimizer.run(n_proposals=n_proposals, n_steps=n_steps)
    assert len(optimizer.data) == len(data) + n_steps + n_steps * n_proposals

    # predict
    X_pred = pd.DataFrame(
        [[0, 0.2, 0.2, 0.2, 0], [0.1, 0.1, 0.1, 0, 0]], columns=problem.inputs.names
    )
    Y_pred = optimizer.predict(X_pred)
    assert len(Y_pred) == 2
    assert "mean_y" in Y_pred.columns
    assert "std_y" in Y_pred.columns
    assert np.all(X_pred.index == Y_pred.index)
