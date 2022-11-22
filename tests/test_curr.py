from entmoot.problem_config import ProblemConfig
import numpy as np
import pytest
import pyomo.environ as pyo
import os
from entmoot.benchmarks import multi_obj_categorical


@pytest.mark.fast_test
def test_tree_model_definition_multiobj_l2():

    # define problem
    problem_config = ProblemConfig(rnd_seed=73)

    problem_config.add_feature("categorical", ("blue", "orange", "gray"))
    problem_config.add_feature("integer", (5, 6))
    problem_config.add_feature("binary")
    problem_config.add_feature("real", (5.0, 6.0))
    problem_config.add_feature("real", (4.6, 6.0))
    problem_config.add_feature("real", (5.0, 6.0))

    problem_config.add_min_objective()
    problem_config.add_min_objective()

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    pred = multi_obj_categorical(rnd_sample)

    # fit tree ensemble
    from entmoot.models.enting import Enting

    params = {"unc_params": {"dist_metric": "l2"}}
    enting = Enting(problem_config, params=params)
    enting.fit(rnd_sample, pred)

    # Build GurobiOptimizer object and solve optimization problem
    from entmoot.optimizers.gurobi_opt import GurobiOptimizer
    params_gurobi = {"NonConvex": 2, "MIPGap": 0}
    opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
    optval_gur = opt_gur.solve(enting, weights=(0.5, 0.5))

    # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
    from entmoot.optimizers.pyomo_opt import PyomoOptimizer
    params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
    pyo_opt = PyomoOptimizer(problem_config, params=params_pyo)
    optval_pyo = pyo_opt.solve(enting, weights=(0.5, 0.5))

    assert round(optval_gur, 3) == round(optval_pyo, 3)


@pytest.mark.fast_test
def test_tree_model_definition_singleobj_l2():
    pass


@pytest.mark.fast_test
def test_tree_model_definition_multiobj_l1():
    def test_func_multi_obj(X):
        y0 = np.sin([xi[0] for xi in X])
        y1 = np.cos([xi[0] for xi in X])
        return np.squeeze(np.column_stack([y0, y1]))

        # define problem

    problem_config = ProblemConfig(rnd_seed=73)

    problem_config.add_feature("real", (5.0, 6.0))
    problem_config.add_feature("real", (4.6, 6.0))
    problem_config.add_feature("real", (5.0, 6.0))
    problem_config.add_feature("categorical", ("blue", "orange", "gray"))
    problem_config.add_feature("integer", (5, 6))
    problem_config.add_feature("binary")

    problem_config.add_min_objective()
    problem_config.add_min_objective()

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    pred = test_func_multi_obj(rnd_sample)

    # fit tree ensemble
    from entmoot.models.enting import Enting

    params = {"unc_params": {"dist_metric": "l1"}}
    enting = Enting(problem_config, params=params)
    enting.fit(rnd_sample, pred)

    # test prediction
    test_points = problem_config.get_rnd_sample_list(num_samples=10)
    enting.predict_acq(test_points)

    # build Gurobi core model
    model_core_gurobi = problem_config.get_gurobi_model_core()
    # build Pyomo core model
    model_core_pyomo = problem_config.get_pyomo_model_core()

    # Assert that both models contain the same number of variables:
    assert len(model_core_gurobi.getVars()) == len(model_core_pyomo.x)

    # Enrich Gurobi and Pyomo models by constraints from tree model and objective function. The objective function
    # contains an acquisition and an uncertainty part.
    enting.add_to_gurobipy_model(model_core_gurobi)
    enting.add_to_pyomo_model(model_core_pyomo)

    # TODO: Implement Big M Method for gurobi and compare again
    # Assert that both models contain the same number of variables:
    #assert len(model_core_gurobi.getVars()) == sum(
    #    len(x) for x in model_core_pyomo.component_objects(pyo.Var)
    #)
    # Assert that both models contain the same number of constraints:
    #assert len(model_core_gurobi.getConstrs()) + len(
    #    model_core_gurobi.getQConstrs()
    #) == sum(len(x) for x in model_core_pyomo.component_objects(pyo.Constraint))

    return model_core_gurobi, model_core_pyomo

@pytest.mark.skipif(
    "CICD_ACTIVE" in os.environ, reason="No optimization runs in CICD pipelines"
)
def test_compare_pyomogurobi_gurobipy_optimization_results_multi_obj_l1():
    import logging
    logger = logging.getLogger("infeas_pyomo")

    model_gurobipy, model_pyomo = test_tree_model_definition_multiobj_l1()

    gurobi_solver = pyo.SolverFactory("gurobi")
    gurobi_solver.options["NonCOnvex"] = 2
    gurobi_solver.options["MIPGap"] = 0.0

    results = gurobi_solver.solve(model_pyomo, tee=True)

    model_gurobipy.params.NonConvex = 2
    model_gurobipy.params.MIPGap = 0.0
    model_gurobipy.optimize()

    # Activate this line code as soon as it is possible to fix the moo_weights
    # assert round(pyo.value(model_pyomo.obj), 5) == round(model_gurobipy.ObjVal, 5)


@pytest.mark.fast_test
def test_tree_model_definition_singleobj_l1():
    pass
