from entmoot import Enting, ProblemConfig, GurobiOptimizer, PyomoOptimizer
from entmoot.benchmarks import (
    build_multi_obj_categorical_problem,
    eval_multi_obj_cat_testfunc,
)


def test_bring_your_own_constraints():
    # define problem
    problem_config = ProblemConfig(rnd_seed=73)
    # number of objectives
    number_objectives = 2
    build_multi_obj_categorical_problem(problem_config, n_obj=number_objectives)

    # sample data
    rnd_sample = problem_config.get_rnd_sample_list(num_samples=20)
    testfunc_evals = eval_multi_obj_cat_testfunc(rnd_sample, n_obj=number_objectives)

    params = {"unc_params": {"dist_metric": "l1", "acq_sense": "exploration"}}
    enting = Enting(problem_config, params=params)
    # fit tree ensemble
    enting.fit(rnd_sample, testfunc_evals)

    # Gurobi Version

    # get optimization model
    model_gur = problem_config.get_gurobi_model_core()
    # extract decision variables
    x = model_gur._all_feat[3]
    y = model_gur._all_feat[4]
    z = model_gur._all_feat[5]
    # add constraint that all variables should coincide
    model_gur.addConstr(x == y)
    model_gur.addConstr(y == z)

    model_gur.update()

    # Build GurobiOptimizer object and solve optimization problem
    params_gurobi = {"MIPGap": 0}
    opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)

    res_gur = opt_gur.solve(enting, model_core=model_gur)
    x_opt, y_opt, z_opt = res_gur.opt_point[3:]

    assert round(x_opt, 5) == round(y_opt, 5) and round(y_opt, 5) == round(z_opt, 5)