from pytest import approx

from entmoot import Enting, ProblemConfig, PyomoOptimizer
from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
    eval_multi_obj_cat_testfunc,
)


def test_max_predictions_equal_min_predictions():
    """The sign of the predicted objective is independent of max/min."""
    problem_config = ProblemConfig(rnd_seed=73)
    build_multi_obj_categorical_problem(problem_config, n_obj=1)
    problem_config.add_min_objective()

    problem_config_max = ProblemConfig(rnd_seed=73)
    build_multi_obj_categorical_problem(problem_config_max, n_obj=1)
    problem_config_max.add_max_objective()

    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=2)

    params = {"unc_params": {"dist_metric": "l1", "acq_sense": "exploration"}}
    enting = Enting(problem_config, params=params)
    enting.fit(rnd_sample, testfunc_evals)

    enting_max = Enting(problem_config_max, params=params)
    enting_max.fit(rnd_sample, testfunc_evals)

    sample = problem_config.get_rnd_sample_list(num_samples=3)
    pred = enting.predict(sample)
    pred_max = enting_max.predict(sample)

    for ((m1, u1), (m2, u2)) in zip(pred, pred_max):
        print(">", m1, m2)
        assert m1 == approx(m2, rel=1e-5)
        assert u1 == approx(u2, rel=1e-5)

def test_max_objective_equals_minus_min_objective():
    """Assert that the solution found by the minimiser is the same as that of the maximiser for the negative objective function"""
    problem_config = ProblemConfig(rnd_seed=73)
    build_multi_obj_categorical_problem(problem_config, n_obj=1)
    problem_config.add_min_objective()

    problem_config_max = ProblemConfig(rnd_seed=73)
    build_multi_obj_categorical_problem(problem_config_max, n_obj=0)
    problem_config_max.add_max_objective()
    problem_config_max.add_max_objective()

    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=2)

    params = {"unc_params": {"dist_metric": "l1", "acq_sense": "penalty"}}
    enting = Enting(problem_config, params=params)
    enting.fit(rnd_sample, testfunc_evals)
    # pass negative test evaluations to the maximiser
    enting_max = Enting(problem_config, params=params)
    enting_max.fit(rnd_sample, -testfunc_evals)

    params_pyomo = {"solver_name": "gurobi"}
    res = PyomoOptimizer(problem_config, params=params_pyomo).solve(enting)
    res_max = PyomoOptimizer(problem_config_max, params=params_pyomo).solve(enting)

    assert res.opt_point == approx(res_max.opt_point, rel=1e-5)



