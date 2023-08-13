import pytest

import random
import math

from entmoot import Enting, ProblemConfig, PyomoOptimizer
from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
    eval_multi_obj_cat_testfunc,
)


def get_leaf_center(bnds):
    return [
        (b[0] + b[1]) / 2 if not isinstance(b, set) else random.choice(list(b))
        for b in bnds
    ]


def run_pyomo(rnd_seed, n_obj, params, params_opt, num_samples=20, no_cat=False):
    # define benchmark problem
    problem_config = ProblemConfig(rnd_seed=rnd_seed)
    build_multi_obj_categorical_problem(problem_config, n_obj=n_obj, no_cat=no_cat)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=num_samples)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=n_obj, no_cat=no_cat)

    # build model
    enting = Enting(problem_config, params=params)
    enting.fit(rnd_sample, testfunc_evals)

    # solve gurobi
    opt = PyomoOptimizer(problem_config, params=params_opt)

    params_opt["solver_options"]["Timelimit"] = 120

    x_sol, obj, mu, unc, leafs = opt.solve(enting)

    # predict enting model with solution
    x_sol_enc = problem_config.encode([x_sol])

    leaf_bnd = [
        enting.leaf_bnd_predict(f"obj_{obj}", opt.get_active_leaf_sol()[obj])
        for obj in range(n_obj)
    ]

    leaf_mid = [get_leaf_center(b) for b in leaf_bnd]

    mu_pred = [enting.mean_model.predict([m])[0][i] for i, m in enumerate(leaf_mid)]
    unc_pred = enting.unc_model.predict([x_sol_enc])[0]

    # compare model mean and uncertainty to prediction
    for m_opt, m_pred in zip(mu, mu_pred):
        assert math.isclose(
            m_opt, m_pred, rel_tol=0.0001
        ), f"`{m_opt}` and `{m_pred}` prediction values are too different from each other"

    assert (
        unc > 0.001 and unc_pred > 0.001
    ), f"`{unc}` and `{unc_pred}` are too small to test"
    assert math.isclose(
        unc, unc_pred, rel_tol=0.0001
    ), f"`{unc}` and `{unc_pred}` unc values are not the same"


@pytest.mark.parametrize("dist_metric", ["l1", "l2", "euclidean_squared"])
@pytest.mark.parametrize("cat_metric", ["overlap", "of", "goodall4"])
@pytest.mark.parametrize("acq_sense", ["exploration"])
@pytest.mark.parametrize("rnd_seed", [100, 101, 102])
@pytest.mark.parametrize("n_obj", [1, 2])
def test_pyomo_consistency1(rnd_seed, n_obj, acq_sense, dist_metric, cat_metric):
    # define model params
    params = {
        "unc_params": {
            "dist_metric": dist_metric,
            "acq_sense": acq_sense,
            "dist_trafo": "normal",
            "cat_metric": cat_metric,
        }
    }
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"NonConvex": 2, "MIPGap": 0},
    }
    run_pyomo(rnd_seed, n_obj, params, params_opt, num_samples=200)


@pytest.mark.parametrize("dist_metric", ["l1", "euclidean_squared"])
@pytest.mark.parametrize("cat_metric", ["overlap", "of", "goodall4"])
@pytest.mark.parametrize("acq_sense", ["penalty"])
@pytest.mark.parametrize("rnd_seed", [100, 101, 102])
@pytest.mark.parametrize("n_obj", [1, 2])
def test_gurobi_consistency2(rnd_seed, n_obj, acq_sense, dist_metric, cat_metric):
    # define model params
    params = {
        "unc_params": {
            "dist_metric": dist_metric,
            "acq_sense": acq_sense,
            "dist_trafo": "normal",
            "cat_metric": cat_metric,
        },
    }
    params["unc_params"]["beta"] = 0.05
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"NonConvex": 2, "MIPGap": 0},
    }

    run_pyomo(rnd_seed, n_obj, params, params_opt, num_samples=300)


@pytest.mark.parametrize("dist_metric", ["l2"])
@pytest.mark.parametrize("cat_metric", ["overlap", "of", "goodall4"])
@pytest.mark.parametrize("acq_sense", ["penalty"])
@pytest.mark.parametrize("rnd_seed", [100, 101, 102])
@pytest.mark.parametrize("n_obj", [1, 2])
def test_pyomo_consistency3(rnd_seed, n_obj, acq_sense, dist_metric, cat_metric):
    # define model params
    params = {}
    params["unc_params"] = {
        "dist_metric": dist_metric,
        "acq_sense": acq_sense,
        "dist_trafo": "normal",
        "cat_metric": cat_metric,
    }

    # make tree model smaller to reduce testing time
    params["tree_train_params"] = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "num_boost_round": 2,
        "max_depth": 2,
        "min_data_in_leaf": 1,
        "min_data_per_group": 1,
        "verbose": -1,
    }
    params["unc_params"]["beta"] = 0.05
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"MIPGap": 0, "LogToConsole": 1, "NonConvex": 2},
    }

    if n_obj == 1:
        params_opt["solver_options"]["MIPGap"] = 0
    else:
        # gurobi takes a long time to fully prove optimality here
        params_opt["solver_options"]["MIPGapAbs"] = 0.001

    run_pyomo(rnd_seed, n_obj, params, params_opt, num_samples=300)


@pytest.mark.parametrize("dist_metric", ["l1", "l2", "euclidean_squared"])
@pytest.mark.parametrize("acq_sense", ["exploration"])
@pytest.mark.parametrize("rnd_seed", [100, 101, 102])
def test_gurobi_consistency4(rnd_seed, acq_sense, dist_metric):
    params = {}
    params["unc_params"] = {
        "dist_metric": dist_metric,
        "acq_sense": acq_sense,
        "dist_trafo": "standard",
    }
    params["unc_params"]["beta"] = 0.1
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"NonConvex": 2, "MIPGap": 1e-5},
    }

    run_pyomo(rnd_seed, 1, params, params_opt, num_samples=20, no_cat=True)


@pytest.mark.parametrize("dist_metric", ["l1", "euclidean_squared"])
@pytest.mark.parametrize("acq_sense", ["penalty"])
@pytest.mark.parametrize("rnd_seed", [100, 101, 102])
def test_gurobi_consistency5(rnd_seed, acq_sense, dist_metric):
    params = {}
    params["unc_params"] = {
        "dist_metric": dist_metric,
        "acq_sense": acq_sense,
        "dist_trafo": "standard",
    }
    params["unc_params"]["beta"] = 0.1
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"NonConvex": 2, "MIPGap": 1e-5},
    }

    run_pyomo(rnd_seed, 1, params, params_opt, num_samples=200, no_cat=True)


@pytest.mark.parametrize("dist_metric", ["l2"])
@pytest.mark.parametrize("acq_sense", ["penalty"])
@pytest.mark.parametrize("rnd_seed", [100, 101, 102])
def test_gurobi_consistency6(rnd_seed, acq_sense, dist_metric):
    params = {}
    params["unc_params"] = {
        "dist_metric": dist_metric,
        "acq_sense": acq_sense,
        "dist_trafo": "standard",
    }
    params["unc_params"]["beta"] = 0.05
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"NonConvex": 2, "MIPGap": 1e-5},
    }

    run_pyomo(rnd_seed, 1, params, params_opt, num_samples=200, no_cat=True)
