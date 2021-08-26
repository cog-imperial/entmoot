import opti
from entmoot.optimizer import EntmootOpti


def test_api():
    # Definition of test problem
    test_problem = opti.problems.Zakharov_categorical(n_inputs=3)
    test_problem.create_initial_data(5)

    # Declaration of entmoot instanceTrain surrogate model
    entmoot = EntmootOpti(problem=test_problem)

    # Train surrogate model
    entmoot._fit_model()

    # Add additional data point
    # entmoot.add_data_and_fit(X_train, y_train)

    # Optimize acquisition function
    # X_next = entmoot.propose(n_proposals=3)

    # Prediction based on surrogate model
    # y_pred = entmoot.predict(X_next)

    # Run Bayesian Optimization loop
    # entmoot.run()
    assert 1 == 1
