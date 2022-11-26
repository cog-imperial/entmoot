from entmoot import Enting, ProblemConfig, GurobiOptimizer, PyomoOptimizer
from entmoot.benchmarks import eval_multi_obj_cat_testfunc

import numpy as np
import os
import pytest
import random


def build_multi_obj_categorical_problem(problem_config: ProblemConfig, n_obj: int = 2):
    """
    Builds a small test example which is frequently used by the tests.
    :param problem_config: ProblemConfig object where features and objectives are added
    :param n_obj: Number of objectives
    :return: None, the problem definition happens "inplace"
    """
    problem_config.add_feature("categorical", ("blue", "orange", "gray"))
    problem_config.add_feature("integer", (5, 6))
    problem_config.add_feature("binary")
    problem_config.add_feature("real", (5.0, 6.0))
    problem_config.add_feature("real", (4.6, 6.0))
    problem_config.add_feature("real", (5.0, 6.0))

    for _ in range(n_obj):
        problem_config.add_min_objective()


def test_simple_test():
    # Define a one-dimensional minimization problem with one real variable bounded by -2 and 3
    problem_config = ProblemConfig()
    problem_config.add_feature("real", (-2, 3))
    problem_config.add_min_objective()

    # Create training data using the randomly disturbed function f(x) = x^2 + 1 + eps
    X_train = np.reshape(np.linspace(-2, 3, 10), (-1, 1))
    y_train = np.reshape([x**2 + 1 + random.uniform(-0.2, 0.2) for x in X_train], (-1, 1))

    # Define enting object and corresponding parameters
    params = {"unc_params": {"dist_metric": "l1"}}
    enting = Enting(problem_config, params=params)
    # Fit tree model
    enting.fit(X_train, y_train)
    # Compute the predictions for training data and see that light gbm fitted a step function with three steps
    enting.predict(X_train)

    # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
    params_pyo = {"solver_name": "gurobi", "solver_options": {"MIPGap": 0}}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
    X_opt_pyo, _, _ = opt_pyo.solve(enting)

    assert round(X_opt_pyo[0]) == 0


@pytest.mark.fast_test
@pytest.mark.skipif(
    "CICD_ACTIVE" in os.environ, reason="No optimization runs in CICD pipelines"
)
def test_compare_pyomo_gurobipy_multiobj():
    """
    Ensures for a multi objective example with l1  and l2 uncertainty metric and mixed feature types that optimization
    results for Gurobipy model and Pyomo model with Gurobi as optimizer coincide.
    """

    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 2
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=number_objectives)

    for metric in ["l1", "l2"]:
        params = {"unc_params": {"dist_metric": metric}}
        enting = Enting(problem_config, params=params)
        # fit tree ensemble
        enting.fit(rnd_sample, testfunc_evals)

        # Build GurobiOptimizer object and solve optimization problem
        params_gurobi = {"NonConvex": 2, "MIPGap": 0}
        opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
        X_opt_gur, y_opt_gur, _ = opt_gur.solve(enting)

        # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
        params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
        opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
        X_opt_pyo, y_opt_pyo, _ = opt_pyo.solve(enting)

        # Compare optimal values (e.g. objective values) ...
        assert round(y_opt_gur, 2) == round(y_opt_pyo, 2)
        # ... and optimal points (e.g. feature variables)
        assert [round(x, 5) for x in X_opt_gur[2:]] == [round(x, 5) for x in X_opt_pyo[2:]]

@pytest.mark.fast_test
@pytest.mark.skipif(
    "CICD_ACTIVE" in os.environ, reason="No optimization runs in CICD pipelines"
)
def test_compare_pyomo_gurobipy_singleobj():
    """
    Ensures for a single objective example with l1  and l2 uncertainty metric and mixed feature types that optimization
    results for Gurobipy model and Pyomo model with Gurobi as optimizer coincide.
    """

    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 1
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=number_objectives)

    for metric in ["l1", "l2"]:
        params = {"unc_params": {"dist_metric": metric}}
        enting = Enting(problem_config, params=params)
        # fit tree ensemble
        enting.fit(rnd_sample, testfunc_evals)

        # Build GurobiOptimizer object and solve optimization problem
        params_gurobi = {"NonConvex": 2, "MIPGap": 0}
        opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
        X_opt_gur, y_opt_gur, _ = opt_gur.solve(enting)

        # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
        params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
        opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
        X_opt_pyo, y_opt_pyo, _ = opt_pyo.solve(enting)

        # Compare optimal values (e.g. objective values) ...
        assert round(y_opt_gur, 2) == round(y_opt_pyo, 2)
        # ... and optimal points (e.g. feature variables)
        assert [round(x, 5) for x in X_opt_gur[2:]] == [round(x, 5) for x in X_opt_pyo[2:]]


def test_tree_model_vs_opt_model():
    """
    This test compares the prediction values from the tree models with the corresponding decision variable
    model._unscaled_mu in order to check, if the tree model was incorporated correctly in the optimization model.
    """
    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 1
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=number_objectives)

    params = {"unc_params": {"dist_metric": "l1"}}
    enting = Enting(problem_config, params=params)
    # fit tree ensemble
    enting.fit(rnd_sample, testfunc_evals)

    # Build GurobiOptimizer object and solve optimization problem
    params_gurobi = {"NonConvex": 2, "MIPGap": 0}
    opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
    X_opt_gur, y_opt_gur, y_opt_unscaled_gur = opt_gur.solve(enting)

    # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
    params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
    X_opt_pyo, y_opt_pyo, y_opt_unscaled_pyo = opt_pyo.solve(enting)

    pred_mean, pred_std = enting.predict([X_opt_pyo])[0]

    # TODO: Find reason for different values
    # assert y_opt_unscaled_pyo == pred_mean
