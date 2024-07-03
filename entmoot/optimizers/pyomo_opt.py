from typing import Optional

import numpy as np
import pyomo.environ as pyo

from entmoot.models.enting import Enting
from entmoot.problem_config import Categorical, ProblemConfig
from entmoot.utils import OptResult

ActiveLeavesT = list[list[tuple[int, str]]]


class PyomoOptimizer:
    """
    This class builds and solves a Pyomo optimization model using available
    information (tree structure, uncertainty measures, ...).

    Example:
        .. code-block:: python

            from entmoot import ProblemConfig
            import numpy as np
            import random

            # Define a one-dimensional minimization problem with one real variable bounded by -2 and 3
            problem_config = ProblemConfig()
            problem_config.add_feature("real", (-2, 3))
            problem_config.add_min_objective()

            # Create training data using the randomly disturbed function f(x) = x^2 + 1 + eps
            X_train = np.linspace(-2, 3, 10)
            y_train = [x**2 + 1 + random.uniform(-0.2, 0.2) for x in X_train]

            # Define enting object and corresponding parameters
            params = {"unc_params": {"dist_metric": "l1"}}
            enting = Enting(problem_config, params=params)
            # Fit tree model
            enting.fit(X_train, y_train)
            # Compute the predictions for training data and see that light gbm fitted a step function
            # with three steps
            enting.predict(X_train)

            # Define parameters needed during optimization
            params_pyo = {"solver_name": "gurobi", "solver_options": {"MIPGap": 0}}
            # Build PyomoOptimizer object. This step will internally call the methods
            # add_to_pyomo_model() or add_to_gurobipy_model(), resp., depending on the choice
            # of your optimizer
            opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
            # As expected, the optimal input of the tree model is near the origin (cf. X_opt_pyo)
            X_opt_pyo, _, _ = opt_pyo.solve(enting)
    """

    def __init__(self, problem_config: ProblemConfig, params: Optional[dict] = None):
        self._params = {} if params is None else params
        self._problem_config = problem_config
        self._curr_sol = None
        self._active_leaves = None

    @property
    def get_curr_sol(self) -> list | np.ndarray:
        """
        Returns current solution (i.e. optimal points) from optimization run
        """
        assert self._curr_sol is not None, "No solution was generated yet."
        return self._curr_sol

    def get_active_leaf_sol(self) -> ActiveLeavesT:
        """
        Returns active leaves in the tree model based on the current solution
        """
        assert self._active_leaves is not None, "No solution was generated yet."
        return self._active_leaves

    def solve(
        self,
        tree_model: Enting,
        model_core: Optional[pyo.ConcreteModel] = None,
        weights: Optional[tuple[float, ...]] = None,
    ) -> OptResult:
        """
        Solves the Pyomo optimization model
        """
        if model_core is None:
            opt_model = self._problem_config.get_pyomo_model_core()
        else:
            # create model copy to not overwrite original one
            opt_model = self._problem_config.copy_pyomo_model_core(model_core)

        # check weights
        if weights is not None:
            assert len(weights) == len(self._problem_config.obj_list), (
                f"Number of 'weights' is '{len(weights)}', number of objectives "
                f"is '{len(self._problem_config.obj_list)}'."
            )
            assert sum(weights) == 1.0, "weights don't add up to 1.0"

        # choose solver
        opt = pyo.SolverFactory(
            self._params["solver_name"],
            manage_env="solver_factory_options" in self._params,
            options=self._params.get("solver_factory_options", {}),
        )

        # set solver parameters
        if "solver_options" in self._params:
            for k, v in self._params["solver_options"].items():
                opt.options[k] = v

        # build pyomo model using information from tree model
        tree_model.add_to_pyomo_model(opt_model)

        # Solve optimization model
        verbose = self._params.get("verbose", True)
        opt.solve(opt_model, tee=verbose)

        # update current solution
        self._curr_sol, self._active_leaves = self._get_sol(opt_model)

        return OptResult(
            self.get_curr_sol,
            pyo.value(opt_model.obj),
            [opt_model._unscaled_mu[k].value for k in opt_model._unscaled_mu],
            pyo.value(opt_model._unc),
            self._active_leaves,
        )

    def _get_sol(
        self, solved_model: pyo.ConcreteModel
    ) -> tuple[list | np.ndarray, ActiveLeavesT]:
        # extract solutions from conti and discrete variables
        res = []
        for idx, feat in enumerate(self._problem_config.feat_list):
            curr_var = solved_model._all_feat[idx]
            if isinstance(feat, Categorical):
                # find active category
                sol_cat = [
                    int(round(pyo.value(curr_var[enc_cat])))
                    for enc_cat in feat.enc_cat_list
                ].index(1)
                res.append(sol_cat)
            else:
                res.append(pyo.value(curr_var))

        # extract active leaves of solution
        def obj_leaf_index(model_obj, obj_name):
            # this function is the same as in 'tree_ensemble.py', TODO: put this in a tree_utils?
            for tree in range(model_obj._num_trees(obj_name)):
                for leaf in model_obj._leaves(obj_name, tree):
                    yield tree, leaf

        act_leaves = []
        for idx, obj in enumerate(self._problem_config.obj_list):
            act_leaves.append(
                [
                    (tree_id, leaf_enc)
                    for tree_id, leaf_enc in obj_leaf_index(solved_model, obj.name)
                    if round(pyo.value(solved_model._z[obj.name, tree_id, leaf_enc]))
                    == 1.0
                ]
            )

        return self._problem_config.decode([res]), act_leaves
