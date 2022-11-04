from entmoot.problem_config import ProblemConfig
import numpy as np
import pytest

@pytest.mark.fast_test
def test_tree_model_definition():
    def test_func(X):
        X = np.atleast_2d(X)
        return np.sin(X[:, 0])

    def test_func_multi_obj(X):
        X = np.atleast_2d(X)
        y0 = np.sin(X[:, 0])
        y1 = np.cos(X[:, 0])
        return np.squeeze(np.column_stack([y0, y1]))

    # define problem
    problem_config = ProblemConfig()

    problem_config.add_feature('real', (5.0, 6.0))
    problem_config.add_feature('real', (4.6, 6.0))
    problem_config.add_feature('real', (5.0, 6.0))
    problem_config.add_feature('categorical', ("blue", "orange", "gray"))
    problem_config.add_feature('integer', (5, 6))
    problem_config.add_feature('binary')

    problem_config.add_min_objective()
    problem_config.add_min_objective()

    # sample data
    rnd_sample = problem_config.get_rnd_sample_numpy(num_samples=20)
    pred = test_func_multi_obj(rnd_sample)

    # fit tree ensemble
    from entmoot.models.mean_models.tree_ensemble import TreeEnsemble
    tree = TreeEnsemble(problem_config)
    tree.fit(rnd_sample, pred)

    # build Gurobi core model
    model_core_gurobi = problem_config.get_gurobi_model_core()
    # build Pyomo core model
    model_core_pyomo = problem_config.get_pyomo_model_core()

    # Assert that both models contain the same number of variables:
    assert len(model_core_gurobi.getVars()) == len(model_core_pyomo.x)

    # Enrich Gurobi model by constraints from tree model
    tree._add_to_gurobipy_model(model_core_gurobi)
    # Enrich Pyomo model by constraints from tree model
    tree._add_to_pyomo_model(model_core_pyomo)


    # Test 1: pyomo with gurobi yields same results as gurobipy
    # Test 2: pyomo with arbitrary solver should find good points on benchmark problems
    # Test 3: Compare BO-Loops