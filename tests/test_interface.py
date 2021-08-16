import mopti as opti

def test_api():
    problem = opti.problems.Zakharov_categorical(n_inputs=3)
    problem.create_initial_data(5)
    assert 1 == 1
