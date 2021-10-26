import opti

from bo.algorithm import RandomSearch


def test_unconstrained_problem():
    problem = opti.problems.ZDT1(n_inputs=5)
    optimizer = RandomSearch(problem)

    X = optimizer.propose(n_proposals=10)
    assert len(X) == 10
    assert problem.inputs.contains(X).all()

    optimizer.run(n_proposals=3, n_steps=10)
    assert len(optimizer.data) == 30

    config = optimizer.to_config()
    assert config["method"] == "RandomSearch"


def test_constrained_problem():
    problem = opti.problems.Hyperellipsoid()
    optimizer = RandomSearch(problem)

    X = optimizer.propose(n_proposals=10)
    assert len(X) == 10
    assert problem.inputs.contains(X).all()


def test_single_objective_problem():
    problem = opti.problems.Sphere(n_inputs=3)
    optimizer = RandomSearch(problem)

    X = optimizer.propose(n_proposals=10)
    assert len(X) == 10
    assert problem.inputs.contains(X).all()
