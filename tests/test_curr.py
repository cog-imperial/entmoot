from entmoot.problem_config import ProblemConfig
from entmoot.models.enting import Enting
from entmoot.optimizers.gurobi_opt import GurobiOptimizer
from entmoot.optimizers.pyomo_opt import PyomoOptimizer
import pytest
import os
from entmoot.benchmarks import eval_multi_obj_cat_testfunc


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
        optval_gur = opt_gur.solve(enting, weights=(0.5, 0.5))

        # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
        params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
        pyo_opt = PyomoOptimizer(problem_config, params=params_pyo)
        optval_pyo = pyo_opt.solve(enting, weights=(0.5, 0.5))

        assert round(optval_gur, 2) == round(optval_pyo, 2)


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
        optval_gur = opt_gur.solve(enting)

        # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
        params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
        pyo_opt = PyomoOptimizer(problem_config, params=params_pyo)
        optval_pyo = pyo_opt.solve(enting)

        assert round(optval_gur, 2) == round(optval_pyo, 2)


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
    optval_gur = opt_gur.solve(enting)

    # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
    params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
    pyo_opt = PyomoOptimizer(problem_config, params=params_pyo)
    optval_pyo = pyo_opt.solve(enting)

    # TODO: This line of code is not working
    # print(enting.predict(opt_gur.get_curr_sol))
