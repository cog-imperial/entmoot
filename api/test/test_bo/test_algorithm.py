import numpy as np
import opti

from bo.algorithm import Algorithm, ParEGO


def test_init():
    # set up with initial data from problem
    problem = opti.problems.ZDT1(n_inputs=3)
    problem.create_initial_data(6)
    optimizer = Algorithm(problem)
    assert len(optimizer.data) == 6


def test_copy():
    optimizer = ParEGO(opti.problems.FlowReactor384Rf(), mc_samples=128)
    data = optimizer.data
    copy = optimizer.copy()
    assert np.all(optimizer.get_model_parameters() == copy.get_model_parameters())
    assert np.allclose(
        optimizer.predict(data[optimizer.inputs.names]),
        copy.predict(data[copy.inputs.names]),
    )
    small_copy = optimizer.copy(data.iloc[:3])
    assert len(small_copy.data) == 3
    assert optimizer.mc_samples == small_copy.mc_samples
