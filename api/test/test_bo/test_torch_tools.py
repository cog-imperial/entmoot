import numpy as np
import opti
import torch

from bo.torch_tools import (
    fit_gp,
    fit_gp_list,
    make_constraints,
    make_objective,
    predict,
    torch_kwargs,
)


def test_fit_gp():
    X = np.random.rand(10, 3)
    Y = np.random.randn(10, 2)
    fit_gp(X, Y)


def test_fit_gp_list():
    X = np.random.rand(10, 3)
    Y = np.random.randn(10, 2)
    Y[0, 0] = np.nan
    fit_gp_list(X, Y)


def test_predict():
    X = np.random.rand(10, 3)
    Y = np.random.randn(10, 2)
    model = fit_gp(X, Y)

    # non-batched prediction
    n = 999
    X = np.random.rand(n, 3)
    Ymean, Ystd = predict(model, X)
    assert Ymean.shape == (n, 2)
    assert Ystd.shape == (n, 2)

    # batched prediction
    n = 4001
    X = np.random.rand(n, 3)
    Ymean, Ystd = predict(model, X)
    assert Ymean.shape == (n, 2)
    assert Ystd.shape == (n, 2)


def test_make_constraints():
    problem = opti.Problem(
        inputs=[opti.Continuous(f"x{i}", [10, 100]) for i in range(4)],
        outputs=[opti.Continuous(f"y{i}", [0, 1]) for i in range(2)],
        constraints=[
            opti.constraint.LinearEquality(names=["x0"], lhs=[1], rhs=10),
            opti.constraint.LinearInequality(names=["x1", "x2"], lhs=[1, 2], rhs=100),
        ],
    )

    constraints = make_constraints(problem, normalize=False)
    assert np.allclose(constraints["bounds"].numpy(), problem.inputs.bounds)
    x1 = torch.tensor([10, 19, 40, 50], **torch_kwargs)
    x2 = torch.tensor([10, 21, 40, 50], **torch_kwargs)
    idx, lhs, rhs = constraints["equalities"][0]
    assert x1[idx] @ lhs == rhs
    assert x2[idx] @ lhs == rhs
    idx, lhs, rhs = constraints["inequalities"][0]
    assert x1[idx] @ lhs >= rhs  # satisfied
    assert x2[idx] @ lhs <= rhs  # not satisfied

    # with normalized constraints
    constraints = make_constraints(problem, normalize=True)
    assert np.allclose(constraints["bounds"].numpy()[0], 0)
    assert np.allclose(constraints["bounds"].numpy()[1], 1)
    xlo, xhi = torch.tensor(problem.inputs.bounds.values, **torch_kwargs)
    x1n = (x1 - xlo) / (xhi - xlo)
    x2n = (x2 - xlo) / (xhi - xlo)
    idx, lhs, rhs = constraints["equalities"][0]
    assert x1n[idx] @ lhs == rhs
    assert x2n[idx] @ lhs == rhs
    idx, lhs, rhs = constraints["inequalities"][0]
    assert x1n[idx] @ lhs >= rhs  # satisfied
    assert x2n[idx] @ lhs <= rhs  # not satisfied


def test_make_objective():
    outputs = opti.Parameters(
        [opti.Continuous("y0", [0, 1]), opti.Continuous("y1", [0, 1])]
    )
    Y = torch.tensor([[1, 1], [2, 4], [3, 6]])

    # minimize
    obj = make_objective(opti.objective.Minimize("y0"), outputs)
    assert obj(Y).allclose(Y[:, 0])

    # maxmize
    obj = make_objective(opti.objective.Maximize("y0"), outputs)
    assert obj(Y).allclose(-Y[:, 0])

    # close-to-target
    obj = make_objective(opti.objective.CloseToTarget("y0", 2), outputs)
    assert obj(Y).allclose((Y[:, 0] - 2).abs())
