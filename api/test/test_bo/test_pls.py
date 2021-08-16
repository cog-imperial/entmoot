import opti
import pytest

from bo.algorithm import PLS


def test_init():
    # expect error for problem with non-continuous inputs
    problem = opti.Problem(
        inputs=[opti.Continuous("x1", [0, 1]), opti.Discrete("x2", [0, 1])],
        outputs=[opti.Continuous("y1", [0, 1])],
    )
    with pytest.raises(ValueError):
        PLS(problem)

    # single-objective problems without data should work
    PLS(opti.problems.Ackley())

    # multi-objective problems without data should work
    PLS(opti.problems.ZDT1())

    # init with model_params should work
    PLS(
        opti.problems.ZDT1(),
        model_params={
            "poly__degree": 1,
            "poly__interaction_only": False,
            "pls__n_components": 4,
        },
    )


def test_detergent():
    problem = opti.problems.Detergent()
    problem.create_initial_data(n_samples=10)

    optimizer = PLS(problem)

    # predict
    X = problem.inputs.sample(10)
    Ypred = optimizer.predict(X)
    assert len(Ypred) == 10

    # predict pareto front
    front = optimizer.predict_pareto_front()
    assert len(front) > 0

    # get parameters
    params = optimizer.get_model_parameters()
    assert len(params) == len(problem.outputs)

    # to_config
    config = optimizer.to_config()
    assert config["method"] == "PLS"
