from collections import namedtuple
from entmoot import Enting, ProblemConfig
from entmoot.utils import OptResult
import gurobipy as gur
import os


class GurobiOptimizer:
    def __init__(self, problem_config: ProblemConfig, params: dict = None) -> float:
        self._params = {} if params is None else params
        self._problem_config = problem_config
        self._curr_sol = None
        self._active_leaves = None

    def get_curr_sol(self) -> list:
        """
        returns current solution (i.e. optimal points) from optimization run
        """
        assert self._curr_sol is not None, "No solution was generated yet."
        return self._curr_sol

    def get_active_leaf_sol(self) -> list:
        """
        returns current solution (i.e. optimal points) from optimization based
        """
        assert self._curr_sol is not None, "No solution was generated yet."
        return self._active_leaves

    def solve(
        self,
        tree_model: Enting,
        model_core: gur.Model = None,
        weights: tuple = None,
        use_env: bool = False,
    ) -> namedtuple:
        """
        Solves the Gurobi optimization model
        """

        if model_core is None:
            if use_env:
                if "CLOUDACCESSID" in os.environ:
                    # Use Gurobi Cloud
                    connection_params_cld = {
                        "CLOUDACCESSID": os.getenv("CLOUDACCESSID"),
                        "CLOUDSECRETKEY": os.getenv("CLOUDSECRETKEY"),
                        "CLOUDPOOL": os.getenv("CLOUDPOOL"),
                    }
                    env_cld = gur.Env(params=connection_params_cld)
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

    def _get_sol(self, solved_model: gur.Model) -> list:
        # extract solutions from conti and discrete variables
        res = []
        for idx, feat in enumerate(self._problem_config.feat_list):
            curr_var = solved_model._all_feat[idx]
            if feat.is_cat():
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
