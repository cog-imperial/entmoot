import numpy as np
import opti
import pytest

from bo.algorithm import ParEGO


def test_config():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(6)
    optimizer = ParEGO(problem, restarts=1)
    config = optimizer.to_config()
    assert config["method"] == "ParEGO"


def test_unconstrained_problem():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(6)

    optimizer = ParEGO(problem, restarts=1)

    # predict
    X = problem.inputs.sample(10)
    Ypred = optimizer.predict(X)
    assert len(Ypred) == 10

    # propose
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()

    # run
    optimizer.run(n_proposals=1, n_steps=3)
    assert len(optimizer.data) == 6 + 3

    # check convergence to true pareto front
    Y = optimizer.data[problem.outputs.names].values
    exp_front = opti.metric.pareto_front(Y)
    true_front = problem.get_optima().values
    assert opti.metric.generational_distance(exp_front, true_front) < 0.5


def test_problem_with_output_constraint():
    problem = opti.problems.Detergent_OutputConstraint()
    problem.create_initial_data(6)

    optimizer = ParEGO(problem, restarts=1)

    # predict
    X = problem.inputs.sample(10)
    Ypred = optimizer.predict(X)
    assert len(Ypred) == 10

    # propose
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()


def test_missing_output():
    problem = opti.problems.ZDT1(n_inputs=3)
    problem.create_initial_data(10)
    problem.data.loc[0, "y0"] = np.nan
    problem.data.loc[9, "y1"] = np.nan
    optimizer = ParEGO(problem)

    # experiements with missing observations are not pruned
    assert len(optimizer.data) == 10

    # proposals work
    X = optimizer.propose(n_proposals=2)
    assert len(X) == 2
    assert problem.inputs.contains(X).all()


def test_unsuitable_problem():
    # expect error for problem with no initial data
    problem = opti.problems.Detergent()
    with pytest.raises(ValueError):
        ParEGO(problem)

    # expect error for problem with non-continuous inputs
    problem = opti.problems.Bread()
    with pytest.raises(ValueError):
        ParEGO(problem)

    # expect error for problem with non-linear constraints
    problem = opti.problems.Qapi1()
    problem.create_initial_data(10)
    with pytest.raises(ValueError):
        ParEGO(problem)

    # expect error for single-objective problem
    problem = opti.problems.Sphere(n_inputs=3)
    problem.create_initial_data(5)
    with pytest.raises(ValueError):
        ParEGO(problem)


def test_scaling():
    # things should not go wrong when the domain is not [0, 1]
    problem = opti.Problem(
        inputs=[
            opti.Continuous("x0", domain=[1, 5]),
            opti.Continuous("x1", domain=[5, 15]),
        ],
        outputs=[
            opti.Continuous("y0", [0, np.inf]),
            opti.Continuous("y1", [0, np.inf]),
        ],
        f=lambda x: np.stack([x[:, 0] ** 2, x[:, 1] ** 2], axis=1),
    )
    problem.create_initial_data(10)
    optimizer = ParEGO(problem)
    X = optimizer.propose(4)
    assert problem.inputs.contains(X).all()


def test_calc_pareto_front():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(20)
    optimizer = ParEGO(problem, restarts=1)
    front = optimizer.predict_pareto_front(n_levels=4)
    assert len(front) == 4
