Welcome to ENTMOOT's documentation!
===================================

**ENTMOOT** (**EN**\semble **T**\ree **MO**\del **O**\ptimization **T**\ool) is a framework to handle tree-based surrogate
models in Bayesian Optimization applications. Gradient-boosted tree models from `lightgbm` are combined with a
distance-based uncertainty measure in a deterministic global optimization framework to optimize black-box functions.
More details on the method can be found here: https://arxiv.org/abs/2003.04774.

Appetizer
---------
This small example illustrates how to use ENTMOOT.

.. code-block:: python

    from entmoot.optimizer.optimizer import Optimizer
    from entmoot.benchmarks import Rosenbrock

    func = Rosenbrock()

    opt = Optimizer(func.get_bounds(10),
                    base_estimator="ENTING",
                    n_initial_points=20,
                    initial_point_generator="random",
                    acq_func="LCB",
                    acq_optimizer="sampling",
                    random_state=100,
                    model_queue_size=None,
                    base_estimator_kwargs={
                        "lgbm_params": {"min_child_samples": 1}
                    },
                    verbose=True,
                    )

    # run optimizer for 20 iterations
    res = opt.run(func, n_iter=20)
    print(f"-> best solution found {res.fun}")

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation <installation>
   Getting Started <quickstart>
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