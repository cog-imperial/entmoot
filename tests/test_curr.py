import math

from entmoot import Enting, ProblemConfig, GurobiOptimizer, PyomoOptimizer
from entmoot.models.model_params import EntingParams, UncParams
from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
    eval_multi_obj_cat_testfunc,
)

import numpy as np
import pytest
import random
import pyomo.environ # noqa: F401

@pytest.mark.pipeline_test
def test_core_model_copy():
    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 2
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    core_model_gurobi = problem_config.get_gurobi_model_core()
    core_model_gurobi_copy = problem_config.copy_gurobi_model_core(core_model_gurobi)

    assert len(core_model_gurobi.getVars()) == len(core_model_gurobi_copy.getVars())
    assert len(core_model_gurobi._all_feat) == len(core_model_gurobi_copy._all_feat)

    core_model_pyomo = problem_config.get_pyomo_model_core()
    core_model_pyomo_copy = problem_config.copy_pyomo_model_core(core_model_pyomo)

    assert len(core_model_pyomo.x) == len(core_model_pyomo_copy.x)
    assert len(core_model_pyomo._all_feat) == len(core_model_pyomo_copy._all_feat)


@pytest.mark.pipeline_test
def test_multiobj_constraints():
    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 2
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=number_objectives)

    params = EntingParams(unc_params=UncParams(
        dist_metric="l1",
        acq_sense="exploration"
    ))
    enting = Enting(problem_config, params=params)
    # fit tree ensemble
    enting.fit(rnd_sample, testfunc_evals)

    # Add constraints

    # Gurobi version
    # get optimization model
    model_gur = problem_config.get_gurobi_model_core()
    # extract decision variables
    x = model_gur._all_feat[3]
    y = model_gur._all_feat[4]
    z = model_gur._all_feat[5]
    # add constraint that all variables should coincide
    model_gur.addConstr(x == y)
    model_gur.addConstr(y == z)
    # Update model
    model_gur.update()

    # Build GurobiOptimizer object and solve optimization problem
    params_gurobi = {"MIPGap": 1e-3}
    opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)

    res_gur = opt_gur.solve(enting, model_core=model_gur)
    x_opt, y_opt, z_opt = res_gur.opt_point[3:]

    assert round(x_opt, 5) == round(y_opt, 5) and round(y_opt, 5) == round(z_opt, 5)

    # Pyomo version
    import pyomo.environ as pyo

    model_pyo = problem_config.get_pyomo_model_core()
    # extract decision variables
    x = model_pyo._all_feat[3]
    y = model_pyo._all_feat[4]
    z = model_pyo._all_feat[5]
    # add constraint that all variables should coincide
    model_pyo.xy_equal_constr = pyo.Constraint(expr=x == y)
    model_pyo.yz_equal_constr = pyo.Constraint(expr=y == z)

    # Build GurobiOptimizer object and solve optimization problem
    params_pyomo = {"solver_name": "gurobi"}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyomo)

    res_pyo = opt_pyo.solve(enting, model_core=model_pyo)
    x_opt, y_opt, z_opt = res_pyo.opt_point[3:]

    assert round(x_opt, 5) == round(y_opt, 5) and round(y_opt, 5) == round(z_opt, 5)


@pytest.mark.pipeline_test
def test_simple_test():
    def my_func(x: float) -> float:
        return x**2 + 1 + random.uniform(-0.2, 0.2)

    # Define a one-dimensional minimization problem with one real variable bounded by -2 and 3
    problem_config = ProblemConfig()
    problem_config.add_feature("real", (-2, 3))
    problem_config.add_min_objective()

    # Create training data using the randomly disturbed function f(x) = x^2 + 1 + eps
    X_train = np.reshape(np.linspace(-2, 3, 10), (-1, 1))
    y_train = np.reshape([my_func(x) for x in X_train], (-1, 1))

    # Define enting object and corresponding parameters
    params = EntingParams(unc_params=UncParams(
        dist_metric="l1",
        acq_sense="exploration"
    ))
    enting = Enting(problem_config, params=params)
    # Fit tree model
    enting.fit(X_train, y_train)

    # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
    params_pyo = {"solver_name": "gurobi"}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
    res = opt_pyo.solve(enting)

    assert round(res.opt_point[0]) == 0


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

    for metric in ["l1", "l2", "euclidean_squared"]:
        for acq_sense in ["exploration", "penalty"]:
            params = EntingParams(unc_params=UncParams(
                dist_metric=metric,
                acq_sense=acq_sense
            ))
            enting = Enting(problem_config, params=params)
            # fit tree ensemble
            enting.fit(rnd_sample, testfunc_evals)

            # Build GurobiOptimizer object and solve optimization problem
            params_gurobi = {"NonConvex": 2, "MIPGap": 0}
            opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
            res_gur = opt_gur.solve(enting, weights=(0.4, 0.6))

            # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
            params_pyo = {
                "solver_name": "gurobi",
                "solver_options": {"NonConvex": 2, "MIPGap": 0},
            }
            opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
            res_pyo = opt_pyo.solve(enting, weights=(0.4, 0.6))

            # Assert that active leaves coincide for both models
            assert math.isclose(res_gur.opt_val, res_pyo.opt_val, abs_tol=0.01)


@pytest.mark.pipeline_test
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

    for metric in ["l1", "l2", "euclidean_squared"]:
        for acq_sense in ["exploration", "penalty"]:
            params = EntingParams(unc_params=UncParams(
                dist_metric=metric,
                acq_sense=acq_sense
            ))            
            enting = Enting(problem_config, params=params)
            # fit tree ensemble
            enting.fit(rnd_sample, testfunc_evals)

            # Build GurobiOptimizer object and solve optimization problem
            params_gurobi = {"NonConvex": 2, "MIPGap": 1e-3}
            opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
            res_gur = opt_gur.solve(enting)

            # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
            params_pyo = {
                "solver_name": "gurobi",
                "solver_options": {"NonConvex": 2, "MIPGap": 1e-3},
            }
            opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
            res_pyo = opt_pyo.solve(enting)

            # Compare optimal values (e.g. objective values) ...
            assert round(res_gur.opt_val / res_pyo.opt_val, 5) <= 1.001
            # ... and optimal points (e.g. feature variables)
            assert [round(x) for x in res_gur.opt_point[2:]] == [
                round(x) for x in res_pyo.opt_point[2:]
            ]
