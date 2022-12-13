Welcome to ENTMOOT's documentation!
===================================

**ENTMOOT** (**EN**\semble **T**\ree **MO**\del **O**\ptimization **T**\ool) is a framework to handle tree-based surrogate
models in Bayesian Optimization applications. Gradient-boosted tree models from `LightGBM <https://lightgbm.readthedocs.io>`__
are combined with a distance-based uncertainty measure in a deterministic global optimization framework to optimize
black-box functions. More details on the method can be found here: https://arxiv.org/abs/2003.04774.

Appetizer
---------
This small example illustrates how to use ENTMOOT.

.. code-block:: python

    from entmoot import Enting, ProblemConfig, PyomoOptimizer
    import numpy as np
    import random

    # This is the function you want to minimize.
    def my_func(x: float) -> float:
        # randomly disturbed function f(x) = x^2 + 1 + eps
        return x**2 + 1 + random.uniform(-0.2, 0.2)

    # Define a one-dimensional minimization problem with one real variable bounded by -2 and 3
    problem_config = ProblemConfig()
    problem_config.add_feature("real", (-2, 3))
    problem_config.add_min_objective()

    # Create training data for the tree model
    X_train = np.reshape(np.linspace(-2, 3, 10), (-1, 1))
    y_train = np.reshape([my_func(x) for x in X_train], (-1, 1))

    # Define Bayesian optimization parameters uding an l1-distance based uncertainty measure and penalizing the distance
    # from well-known areas, i.e. exploitation (instead of exploration)
    params_bo = {"unc_params": {"dist_metric": "l1", "acq_sense": "penalty"}}

    # Define an Enting object which holds information about the problem as well as the parameters...
    enting = Enting(problem_config, params=params_bo)
    # ... and train the underlying tree model.
    enting.fit(X_train, y_train)

    # Create an PyomoOptimizer object that solves the optimization problem using the open source solver "GLPK"
    params_pyo = {"solver_name": "glpk"}
    opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
    res = opt_pyo.solve(enting)

    # Inspect the result. The optimal point should be close to zero.
    print(f"Optimal point: {res.opt_point[0]}")

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation Guide <installation>
   Examples <notebooks>
   API Reference <apidoc/entmoot>

Authors
~~~~~~~
* `Alexander Thebelt <https://optimisation.doc.ic.ac.uk/person/alexander-thebelt>`_ (`ThebTron <https://github.com/ThebTron>`_) - Imperial College London
* `Nathan Sudermann-Merx <https://www.mannheim.dhbw.de/profile/sudermann-merx>`_ (`spiralulam <https://github.com/spiralulam>`_) - Cooperative State University Mannheim

License
~~~~~~~
The ENTMOOT package is released under the BSD 3-Clause License. Please refer to the
`LICENSE <https://github.com/cog-imperial/entmoot/blob/master/LICENSE>`_ file for details.

Acknowledgements
~~~~~~~~~~~~~~~~
The support of BASF SE, Lugwigshafen am Rhein is gratefully acknowledged.