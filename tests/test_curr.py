from entmoot.problem_config import ProblemConfig
import numpy as np
import pytest
import pyomo.environ as pyo


@pytest.mark.fast_test
def test_tree_model_definition_multiobj_l2():
    def test_func(X):
        X = np.atleast_2d(X)
        return np.sin(X[:, 0])

    def test_func_multi_obj(X):
        y0 = np.sin([xi[0] for xi in X])
        y1 = np.cos([xi[0] for xi in X])
        return np.squeeze(np.column_stack([y0,y1]))

    # define problem
    problem_config = ProblemConfig(rnd_seed=73)

    problem_config.add_feature('real', (5.0, 6.0))
    problem_config.add_feature('real', (4.6, 6.0))
    problem_config.add_feature('real', (5.0, 6.0))
    problem_config.add_feature('categorical', ("blue", "orange", "gray"))
    problem_config.add_feature('integer', (5, 6))
    problem_config.add_feature('binary')

    problem_config.add_min_objective()
    problem_config.add_min_objective()

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    pred = test_func_multi_obj(rnd_sample)

    # fit tree ensemble
    from entmoot.models.enting import Enting
    params = {"unc_params": {"dist_metric": "l2"}}
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

    # Assert that both models contain the same number of variables:
    assert len(model_core_gurobi.getVars()) == sum(len(x) for x in model_core_pyomo.component_objects(pyo.Var))
    # Assert that both models contain the same number of constraints:
    assert len(model_core_gurobi.getConstrs()) + len(model_core_gurobi.getQConstrs()) == \
           sum(len(x) for x in model_core_pyomo.component_objects(pyo.Constraint))

    model_core_gurobi.params.NonConvex = 2
    model_core_gurobi.params.MIPGap = 0.0
    model_core_gurobi.optimize()

    gurobi_solver = pyo.SolverFactory("gurobi")
    gurobi_solver.options["NonCOnvex"] = 2
    gurobi_solver.options["MIPGap"] = 0.0
    gurobi_solver.solve(model_core_pyomo)

    assert round(pyo.value(model_core_pyomo.obj), 1) == round(model_core_gurobi.ObjVal, 1)


@pytest.mark.fast_test
def test_tree_model_definition_singleobj_l2():
    pass


@pytest.mark.fast_test
def test_tree_model_definition_multiobj_l1():
    pass

@pytest.mark.fast_test
def test_tree_model_definition_singleobj_l1():
    pass
