import math

import numpy as np
import pytest

from entmoot import Enting, ProblemConfig, PyomoOptimizer
from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
    build_small_single_obj_categorical_problem,
    eval_multi_obj_cat_testfunc,
    eval_small_single_obj_cat_testfunc,
)
from entmoot.models.model_params import EntingParams, UncParams


def run_pyomo(
    rnd_seed,
    params,
    params_opt,
    num_samples,
    no_cat=False,
    rtol=1e-2,
    num_opt_samples=250000,
    smaller_problem=False,
):
    # define benchmark problem
    problem_config = ProblemConfig(rnd_seed=rnd_seed)
    if smaller_problem:
        build_small_single_obj_categorical_problem(problem_config, no_cat=no_cat)
    else:
        build_multi_obj_categorical_problem(problem_config, n_obj=1, no_cat=no_cat)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=num_samples)
    if smaller_problem:
        testfunc_evals = eval_small_single_obj_cat_testfunc(rnd_sample, no_cat=no_cat)
    else:
        testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=1, no_cat=no_cat)

    # build model
    enting = Enting(problem_config, params=params)
    enting.fit(rnd_sample, testfunc_evals)

    # manual optimization estimate via sampling
    samples = problem_config.get_rnd_sample_numpy(num_samples=num_opt_samples)
    preds = enting.predict_acq(samples, is_enc=True)
    min_idx = np.argmin(preds)

    est_pred_sol = preds[min_idx][0]
    est_x_sol = samples[min_idx]

    # solve pyomo model
    opt = PyomoOptimizer(problem_config, params=params_opt)

    x_sol, obj, mu, unc, leafs = opt.solve(enting)
    x_sol_enc = problem_config.encode([x_sol])

    # check that solutions are sufficiently close
    assert np.allclose(x_sol_enc, est_x_sol, rtol=rtol), (
        f"solutions deviate by more than {rtol} relative tolerance, "
        f"`{x_sol_enc}` vs. `{est_x_sol}`"
    )

    # check that optimal values are sufficiently close
    assert math.isclose(
        est_pred_sol, obj, abs_tol=1e-1
    ), f"`{est_pred_sol}` and `{obj}` optimal values are not the same"

    assert est_pred_sol >= obj, "estimated value is smaller than solver solution"

    # check that mu values are sufficiently close
    est_mu = enting.mean_model.predict([samples[min_idx]])[0]
    assert math.isclose(
        est_mu[0], mu[0], abs_tol=1e-1
    ), f"`{est_mu}` and `{mu}` mean values are not the same"

    # check that unc values are sufficiently large
    est_unc = enting.unc_model.predict([samples[min_idx]])[0]
    assert (
        unc > 0.001 and est_unc > 0.001
    ), f"`{unc}` and `{est_unc}` are too small to test"
    # check that mu values are sufficiently close
    est_unc = enting.unc_model.predict([samples[min_idx]])[0]
    assert math.isclose(
        est_unc, unc, abs_tol=1e-1
    ), f"`{est_unc}` and `{unc}` uncertainty values are not the same"


@pytest.mark.parametrize("dist_metric", ["l1", "l2", "euclidean_squared"])
@pytest.mark.parametrize("cat_metric", ["overlap", "of", "goodall4"])
@pytest.mark.parametrize("acq_sense", ["exploration"])
@pytest.mark.parametrize("rnd_seed", [100, 101])
def test_pyomo_optimality(rnd_seed, acq_sense, dist_metric, cat_metric):
    # define model params
    params = EntingParams(
        unc_params=UncParams(
            dist_metric=dist_metric,
            acq_sense=acq_sense,
            dist_trafo="normal",
            cat_metric=cat_metric,
        )
    )
    params_opt = {
        "solver_name": "gurobi",
        "solver_options": {"MIPGap": 0, "LogToConsole": 1, "NonConvex": 2},
    }
    run_pyomo(rnd_seed, params, params_opt, num_samples=20)
