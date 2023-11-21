Welcome to ENTMOOT's documentation!
===================================

**ENTMOOT** (**EN**\semble **T**\ree **MO**\del **O**\ptimization **T**\ool) is a framework to perform Bayesian
Optimization using tree-based surrogate models.

**What does that even mean?**

You define a black-box function :math:`f` and ENTMOOT will try to find an optimum of :math:`f` in an iterative
Bayesian spirit. That means it uses currently available knowledge about good and bad points (*exploitation*) and
explores unseen regions in the search space (*exploration*).

**How does ENTMOOT work?**

In iteration :math:`i`, ENTMOOT approximates :math:`f` using a gradient boosted tree
model :math:`G` from `LightGBM <https://lightgbm.readthedocs.io>`__ which is trained on data points for which
function evaluations of :math:`f` are available. In order to find a good point of :math:`f` a prediction of :math:`G`
is combined with an uncertainty measure that takes into account that we do not trust all data points equally. This
combination of predictions and the corresponding uncertainty measures comprises the *acquisition function* which is
optimized in order to find the best candidate :math:`x^i` for an optimal point of :math:`f` in iteration :math:`i`.
If the true function evaluation :math:`f(x^i)` matches your expectations, you may stop, otherwise include
:math:`x^i` and :math:`f(x^i)` in your training data, retrain your model :math:`G` to get a better approximation
of :math:`f` and start the next iteration.

**How does ENTMOOT optimize the nonsmooth acquisition function?**

Gradient-boosted tree models define step functions, that is piecewise constant functions, which are in particular
discontinuous and cannot be optimized with standard methods from smooth nonlinear optimization since no gradient
information is available. ENTMOOT uses the fact that step functions can be modeled as so called mixed-integer optimization
problem for which very fast solvers are available. More details on the method can be found in the corresponding paper
(https://arxiv.org/abs/2003.04774).

Appetizer
---------
This small example illustrates how to use ENTMOOT.

.. code-block:: python

    from entmoot import Enting, ProblemConfig, PyomoOptimizer
    import numpy as np
    import random

    # This is the function to be minimized. Usually some complicated like a simulation.
    def my_func(x: float) -> float:
        return x**2 + 1 + random.uniform(-0.2, 0.2)

    # Define a one-dimensional minimization problem with one real variable bounded by -2 and 3
    problem_config = ProblemConfig()
    problem_config.add_feature("real", (-2, 3))
    problem_config.add_min_objective()

    # Create training data for the tree model
    X_train = np.reshape(np.linspace(-2, 3, 10), (-1, 1))
    y_train = np.reshape([my_func(x) for x in X_train], (-1, 1))

    # Define Bayesian optimization parameters using an l1-distance based uncertainty measure and
    # penalizing the distance from well-known areas, i.e. exploitation (instead of exploration)
    params_bo = {"unc_params": {"dist_metric": "l1", "acq_sense": "penalty"}}

    # Define an Enting object which holds information about the problem as well as the parameters...
    enting = Enting(problem_config, params=params_bo)
    # ... and train the underlying tree model.
    enting.fit(X_train, y_train)

    # Create an PyomoOptimizer object that solves the optimization problem using the solver "GLPK"
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
* Toby Boyne (`TobyBoyne  <https://github.com/TobyBoyne>`_) - Imperial College London

License
~~~~~~~
The ENTMOOT package is released under the BSD 3-Clause License. Please refer to the
`LICENSE <https://github.com/cog-imperial/entmoot/blob/master/LICENSE>`_ file for details.

Acknowledgements
~~~~~~~~~~~~~~~~
The support of BASF SE, Lugwigshafen am Rhein is gratefully acknowledged.