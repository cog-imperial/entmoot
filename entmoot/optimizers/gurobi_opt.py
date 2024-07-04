import os
from typing import Optional

import gurobipy as gur
import numpy as np

from entmoot.models.enting import Enting
from entmoot.problem_config import Categorical, ProblemConfig
from entmoot.typing.optimizer_stubs import GurobiModelT
from entmoot.utils import OptResult

ActiveLeavesT = list[list[tuple[int, str]]]


class GurobiOptimizer:
    """
    This class builds and solves a Gurobi optimization model using available
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
            params_gur = {"MIPGap": 1e-3}
            # Build GurobiOptimizer object.
            opt_gur = GurobiOptimizer(problem_config, params=params_gurobi)
            # As expected, the optimal input of the tree model is near the origin (cf. X_opt_pyo)
            X_opt_pyo, _, _ = opt_gur.solve(enting)
    """

    def __init__(self, problem_config: ProblemConfig, params: Optional[dict] = None):
        self._params = {} if params is None else params
        self._problem_config = problem_config
        self._curr_sol = None
        self._active_leaves: Optional[ActiveLeavesT] = None

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
        model_core: Optional[GurobiModelT] = None,
        weights: Optional[tuple[float, ...]] = None,
        use_env: bool = False,
    ) -> OptResult:
        """
        Solves the Gurobi optimization model
        """

        if model_core is None:
            if use_env:
                env_params = {}
                if "CLOUDACCESSID" in os.environ:
                    # Use Gurobi Cloud
                    env_params = {
                        "CLOUDACCESSID": os.getenv("CLOUDACCESSID", ""),
                        "CLOUDSECRETKEY": os.getenv("CLOUDSECRETKEY", ""),
                        "CLOUDPOOL": os.getenv("CLOUDPOOL", ""),
                    }
                # TODO: Support passing in env params
                env_cld = gur.Env(params=env_params)
                env_cld.start()
                opt_model = self._problem_config.get_gurobi_model_core(env=env_cld)

            else:
                opt_model = self._problem_config.get_gurobi_model_core()
        else:
            # create model copy to not overwrite original one
            opt_model = self._problem_config.copy_gurobi_model_core(model_core)

        # check weights
        if weights is not None:
            assert len(weights) == len(self._problem_config.obj_list), (
                f"Number of 'weights' is '{len(weights)}', number of objectives "
                f"is '{len(self._problem_config.obj_list)}'."
            )
            assert sum(weights) == 1.0, "weights don't add up to 1.0"

        # set solver parameters
        for param, param_val in self._params.items():
            opt_model.setParam(param, param_val)

        # build gurobi model
        tree_model.add_to_gurobipy_model(opt_model, weights=weights)

        # Solve optimization model
        opt_model.optimize()

        # update current solution
        self._curr_sol, self._active_leaves = self._get_sol(opt_model)

        return OptResult(
            self.get_curr_sol(),
            opt_model.ObjVal,
            [x.X for x in opt_model._unscaled_mu],
            opt_model._unc.X,
            self._active_leaves,
        )

    def _get_sol(
        self, solved_model: gur.Model
    ) -> tuple[list | np.ndarray, ActiveLeavesT]:
        # extract solutions from conti and discrete variables
        res = []
        for idx, feat in enumerate(self._problem_config.feat_list):
            curr_var = solved_model._all_feat[idx]
            if isinstance(feat, Categorical):
                # find active category
                sol_cat = [
                    int(round(curr_var[enc_cat].x)) for enc_cat in feat.enc_cat_list
                ].index(1)
                res.append(sol_cat)
            else:
                res.append(curr_var.x)

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
                    if round(solved_model._z[obj.name, tree_id, leaf_enc].x) == 1.0
                ]
            )

        return self._problem_config.decode([res]), act_leaves
