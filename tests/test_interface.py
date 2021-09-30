import opti
import pandas as pd
import lightgbm as lgb

from entmoot.optimizer import EntmootOpti


def test_api():
    # Definition of test problem
    test_problem = opti.problems.Zakharov_categorical(n_inputs=3)
    n_init = 15
    test_problem.create_initial_data(n_init)

    # Declaration of entmoot instanceTrain surrogate model
    surrogat_params = {"verbose": -1, 'min_data_in_leaf': 5}
    entmoot = EntmootOpti(problem=test_problem, surrogat_params=surrogat_params)

    # Train surrogate model
    assert entmoot.model is None
    entmoot._fit_model()
    assert type(entmoot.model) == lgb.Booster

    X_pred = pd.DataFrame([
        {"x0": 5, "x1": 5, "expon_switch": "one"},
        {"x0": 0, "x1": 0, "expon_switch": "two"}
    ])

    # Prediction based on surrogate model
    y_pred = entmoot.predict(X_pred)
    assert len(y_pred) == 2

    # Optimize acquisition function
    X_next = entmoot.propose(n_proposals=10)
    assert len(X_next) == 10

    # Run Bayesian Optimization loop
    n_steps = 3
    n_proposals = 7
    entmoot.run(n_steps=3, n_proposals=7)
    assert len(entmoot.data) == n_init + n_steps*n_proposals
