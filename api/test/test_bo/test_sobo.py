import opti
import pandas as pd
import pytest

from bo.algorithm import SOBO


def test_init():
    # expect error for multi-objective problem
    problem = opti.problems.ZDT1(n_inputs=3)
    problem.create_initial_data(5)
    with pytest.raises(ValueError):
        SOBO(problem)

    # expect error for problem with non-continuous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1]), opti.Discrete("x2", [0, 1])],
        outputs=[opti.Continuous("y1", [0, 1])],
    )
    with pytest.raises(ValueError):
        SOBO(problem)

    # expect error for problem with non-linear constraints
    problem = opti.Problem(
        inputs=[opti.Continuous("x", [0, 1])],
        outputs=[opti.Continuous("y", [0, 1])],
        constraints=[opti.constraint.NonlinearInequality(lambda x: x < 0.5)],
        data=pd.DataFrame({"x": [0.2, 0.4], "y": [0.9, 0.3]}),
    )
    with pytest.raises(ValueError):
        SOBO(problem)

    # expect error for problem with no initial data
    problem = opti.problems.Ackley()
    with pytest.raises(ValueError):
        SOBO(problem)


def test_sphere():
    problem = opti.problems.Sphere(n_inputs=3)
    problem.create_initial_data(n_samples=5)

    optimizer = SOBO(problem)

    # fit & get parameters
    optimizer._fit_model()
    params = optimizer.get_model_parameters()
    assert len(params) == 1

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
    assert len(optimizer.data) == 5 + 3

    # to_config
    config = optimizer.to_config()
    assert config["method"] == "SOBO"
