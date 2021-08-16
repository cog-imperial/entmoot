import opti
import pytest
from plotly.graph_objects import Figure

import bo


@pytest.fixture
def optimizer():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(10)
    optimizer = bo.algorithm.ParEGO(problem, restarts=1)
    optimizer._fit_model()
    return optimizer


def test_residuals_plot(optimizer):
    figs = bo.plot.residuals(optimizer)
    assert isinstance(figs, list)
    assert isinstance(figs[0], Figure)


def test_parallel_data(optimizer):
    fig = bo.plot.parallel_data(optimizer._problem)
    assert isinstance(fig, Figure)


def test_scatter_data(optimizer):
    fig = bo.plot.scatter_data(optimizer._problem)
    assert isinstance(fig, Figure)


def test_correlation_data(optimizer):
    fig = bo.plot.correlation_data(optimizer._problem)
    assert isinstance(fig, Figure)


def test_parallel_model(optimizer):
    fig = bo.plot.parallel_model(optimizer)
    assert isinstance(fig, Figure)


def test_parallel_parameters(optimizer):
    fig = bo.plot.parallel_parameters(optimizer)
    assert isinstance(fig, Figure)
