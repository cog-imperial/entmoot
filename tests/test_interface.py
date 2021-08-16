import mopti as opti
import pandas as pd

# Idea: Create Entmoot class based on Algorithm class from BO


def test_api():

    # Definition of test problem
    problem = opti.problems.Zakharov_categorical(n_inputs=3)
    problem.create_initial_data(5)

    # Declaration of entmoot instanceTrain surrogate model
    # entmoot = Entmoot()

    # Train surrogate model
    X_train, y_train = problem.data[["x0", "x1"]], problem.data["y"]
    # entmoot.add_data_and_fit(X_train, y_train)

    # Optimize acquisition function
    # X_next = entmoot.propose(n_proposals=3)

    # Prediction based on surrogate model
    # y_pred = entmoot.predict(X_next)

    # Run Bayesian Optimization loop
    # entmoot.run()
    assert 1 != 1
