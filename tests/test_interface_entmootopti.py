import opti
import pandas as pd

from entmoot.optimizer import EntmootOpti
from entmoot.utils import get_gurobi_env


def test_api():
    """
    Single objective test problem with two continuous input variables and one categorical input variable
    """
    # Definition of test problem
    test_problem = opti.problems.Zakharov_Categorical(n_inputs=3)
    n_init = 15
    test_problem.create_initial_data(n_init)

    base_estimator_params = {
        "lgbm_params": {"min_child_samples": 2},
        "unc_metric": 'exploration',
        "unc_scaling": "standard",
        "dist_metric": "manhattan"
    }

    entmoot = EntmootOpti(
        problem=test_problem,
        base_est_params=base_estimator_params,
        gurobi_env=get_gurobi_env,
    )

    X_pred = pd.DataFrame(
        [
            {"x0": 5, "x1": 5, "expon_switch": "one"},
            {"x0": 0, "x1": 0, "expon_switch": "two"},
        ]
    )

    # Prediction based on surrogate model
    y_pred = entmoot.predict(X_pred)
    assert len(y_pred) == 2

    # Optimize acquisition function
    X_next: pd.DataFrame = entmoot.propose(n_proposals=10)
    assert len(X_next) == 10

    # Run Bayesian Optimization loop
    n_steps = 3
    n_proposals = 7
    entmoot.run(n_steps=3, n_proposals=7)
    assert len(entmoot.problem.data) == n_init + n_steps * n_proposals


def test_mixed_constraints():
    # single objective, linear-equality + n-choose-k constraints
    problem = opti.problems.Photodegradation()
    entmoot = EntmootOpti(problem=problem, gurobi_env=get_gurobi_env)

    X_pred = problem.data[problem.inputs.names]
    y_mean, y_std = entmoot.predict(X_pred)
    assert len(y_std) == len(X_pred)
    assert len(y_mean) == len(X_pred)

    X_next = entmoot.propose(n_proposals=2)
    assert len(X_next) == 2


def test_no_initial_data():
    # Using Entmoot on a problem without data should raise an informative error.
    problem = opti.problems.Zakharov_Categorical(n_inputs=3)

    try:
        EntmootOpti(problem)
    except ValueError:
        assert True
    else:
        assert False


def test_biobjective():
    # opti.problems.ReizmanSuzuki -> bi-objective, cat + cont variables
    problem = opti.problems.ReizmanSuzuki()

    entmoot = EntmootOpti(problem=problem, gurobi_env=get_gurobi_env)

    X_pred = problem.data[problem.inputs.names]
    y_mean, y_std = entmoot.predict(X_pred)
    assert len(y_std) == len(X_pred)
    for y_mean_obj in y_mean:
        assert len(y_mean_obj) == len(X_pred)

    X_next = entmoot.propose(n_proposals=2)
    assert len(X_next) == 2


def test_with_missing_data():
    # In the multi-objective case we the model should handle missing data, i.e. missing entries in the output columns
    pass
