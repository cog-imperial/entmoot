import pyomo.environ as pyo
import pytest

from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
    build_reals_only_problem,
    eval_multi_obj_cat_testfunc,
    eval_reals_only_testfunc,
)
from entmoot.constraints import (
    ConstraintList,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from entmoot.models.enting import Enting
from entmoot.models.model_params import EntingParams, UncParams
from entmoot.optimizers.pyomo_opt import PyomoOptimizer
from entmoot.problem_config import ProblemConfig

PARAMS = EntingParams(
    unc_params=UncParams(dist_metric="l1", acq_sense="exploration")
)

def test_linear_equality_constraint():
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 2
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=number_objectives)

    enting = Enting(problem_config, params=PARAMS)
    # fit tree ensemble
    enting.fit(rnd_sample, testfunc_evals)

    model_pyo = problem_config.get_pyomo_model_core()
    # define the constraint
    # then immediately apply it to the model
    model_pyo.xy_equal = LinearEqualityConstraint(
        feature_keys=["feat_3", "feat_4"], coefficients=[1, -1], rhs=0
    ).as_pyomo_constraint(model_pyo, problem_config.feat_list)

    model_pyo.yz_equal = LinearEqualityConstraint(
        feature_keys=["feat_4", "feat_5"], coefficients=[1, -1], rhs=0
    ).as_pyomo_constraint(model_pyo, problem_config.feat_list)

    # optimise the model
    params_pyomo = {"solver_name": "gurobi"}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyomo)
    res_pyo = opt_pyo.solve(enting, model_core=model_pyo)
    x_opt, y_opt, z_opt = res_pyo.opt_point[3:]

    assert round(x_opt, 5) == round(y_opt, 5) and round(y_opt, 5) == round(z_opt, 5)


@pytest.mark.parametrize(
    "min_count,max_count",
    [
        (1, 3),
        (0, 5),
        (1, 1),
        (5, 5),
    ],
)
def test_nchoosek_constraint(min_count, max_count):
    # standard setting up of problem
    problem_config = ProblemConfig(rnd_seed=73)
    build_reals_only_problem(problem_config)
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=50)
    testfunc_evals = eval_reals_only_testfunc(rnd_sample)

    enting = Enting(problem_config, params=PARAMS)
    # fit tree ensemble
    enting.fit(rnd_sample, testfunc_evals)

    model_pyo = problem_config.get_pyomo_model_core()

    # define the constraint
    # then immediately apply it to the model
    model_pyo.nchoosek = NChooseKConstraint(
        feature_keys=["x1", "x2", "x3", "x4", "x5"],
        min_count=min_count,
        max_count=max_count,
        none_also_valid=False,
    ).as_pyomo_constraint(model_pyo, problem_config.feat_list)

    # optimise the model
    params_pyomo = {"solver_name": "gurobi"}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyomo)
    res_pyo = opt_pyo.solve(enting, model_core=model_pyo)

    assert min_count <= sum(x > 1e-6 for x in res_pyo.opt_point) <= max_count


def test_constraint_list():
    problem_config = ProblemConfig(rnd_seed=73)
    build_reals_only_problem(problem_config)
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=50)
    testfunc_evals = eval_reals_only_testfunc(rnd_sample)

    enting = Enting(problem_config, params=PARAMS)
    # fit tree ensemble
    enting.fit(rnd_sample, testfunc_evals)

    model_pyo = problem_config.get_pyomo_model_core()

    # define the constraints
    constraints = [
        NChooseKConstraint(
            feature_keys=["x1", "x2", "x3", "x4", "x5"], 
            min_count=1,
            max_count=3,
            none_also_valid=True
        ),
        LinearInequalityConstraint(
            feature_keys=["x3", "x4", "x5"],
            coefficients=[1, 1, 1],
            rhs=10.0
        )
    ]

    # apply constraints to the model
    model_pyo.problem_constraints = pyo.ConstraintList()
    ConstraintList(constraints).apply_pyomo_constraints(
        model_pyo, problem_config.feat_list, model_pyo.problem_constraints
    )

    # optimise the model
    params_pyomo = {"solver_name": "gurobi"}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyomo)
    res_pyo = opt_pyo.solve(enting, model_core=model_pyo)

    print(res_pyo.opt_point)
    assert 1 <= sum(x > 1e-6 for x in res_pyo.opt_point) <= 3
    assert sum(res_pyo.opt_point[2:]) < 10.0