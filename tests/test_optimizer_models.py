import pytest

from entmoot import ProblemConfig
from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
)


@pytest.mark.pipeline_test
def test_core_model_copy():
    def features_equal(model, model_copy, predicate):
        for f, f_copy in zip(model._all_feat, model_copy._all_feat):
            if type(f) is not type(f_copy):
                return False
            elif isinstance(f, dict):
                for v, v_copy in zip(f.values(), f_copy.values()):
                    if not predicate(v, v_copy):
                        return False
            else:
                if not predicate(f, f_copy):
                    return False
        return True

    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    build_multi_obj_categorical_problem(problem_config, n_obj=2)

    core_model_gurobi = problem_config.get_gurobi_model_core()
    core_model_gurobi_copy = problem_config.copy_gurobi_model_core(core_model_gurobi)

    assert len(core_model_gurobi.getVars()) == len(core_model_gurobi_copy.getVars())
    assert len(core_model_gurobi._all_feat) == len(core_model_gurobi_copy._all_feat)
    predicate = lambda u, v: all((u.lb == v.lb, u.ub == v.ub, u.v_type == v.v_type))
    assert features_equal(core_model_gurobi, core_model_gurobi_copy, predicate)

    core_model_pyomo = problem_config.get_pyomo_model_core()
    core_model_pyomo_copy = problem_config.copy_pyomo_model_core(core_model_pyomo)

    assert len(core_model_pyomo.x) == len(core_model_pyomo_copy.x)
    assert len(core_model_pyomo._all_feat) == len(core_model_pyomo_copy._all_feat)
    predicate = lambda u, v: all((u.lb == v.lb, u.ub == v.ub, u.domain == v.domain))
    assert features_equal(core_model_pyomo, core_model_pyomo_copy, predicate)
