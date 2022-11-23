from entmoot.problem_config import ProblemConfig
import pytest
import os
from entmoot.benchmarks import multi_obj_categorical


@pytest.mark.fast_test
@pytest.mark.skipif(
    "CICD_ACTIVE" in os.environ, reason="No optimization runs in CICD pipelines"
)
def test_compare_pyomo_gurobipy_multiobj():

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
    for metric in ["l1", "l2"]:
        params = {"unc_params": {"dist_metric": metric}}
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

        assert round(optval_gur, 2) == round(optval_pyo, 2)


@pytest.mark.fast_test
@pytest.mark.skipif(
    "CICD_ACTIVE" in os.environ, reason="No optimization runs in CICD pipelines"
)
def test_compare_pyomo_gurobipy_singleobj():

    # define problem
    problem_config = ProblemConfig(rnd_seed=73)

    problem_config.add_feature("categorical", ("blue", "orange", "gray"))
    problem_config.add_feature("integer", (5, 6))
    problem_config.add_feature("binary")
    problem_config.add_feature("real", (5.0, 6.0))
    problem_config.add_feature("real", (4.6, 6.0))
    problem_config.add_feature("real", (5.0, 6.0))

    problem_config.add_min_objective()

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    pred = multi_obj_categorical(rnd_sample, n_obj=1)

    # fit tree ensemble
    from entmoot.models.enting import Enting
    for metric in ["l1", "l2"]:
        params = {"unc_params": {"dist_metric": metric}}
        enting = Enting(problem_config, params=params)
        enting.fit(rnd_sample, pred)

        # Build GurobiOptimizer object and solve optimization problem
        from entmoot.optimizers.gurobi_opt import GurobiOptimizer
        params_gurobi = {"NonConvex": 2, "MIPGap": 0}
        opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
        optval_gur = opt_gur.solve(enting)

        # Build PyomoOptimizer object with Gurobi as solver and solve optimization problem
        from entmoot.optimizers.pyomo_opt import PyomoOptimizer
        params_pyo = {"solver_name": "gurobi", "solver_options": {"NonConvex": 2, "MIPGap": 0}}
        pyo_opt = PyomoOptimizer(problem_config, params=params_pyo)
        optval_pyo = pyo_opt.solve(enting)

        assert round(optval_gur, 2) == round(optval_pyo, 2)
