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

    def get_curr_sol(self) -> list:
        """
        returns current solution (i.e. optimal points) from optimization run
        """
        assert self._curr_sol is not None, "No solution was generated yet."
        return self._curr_sol

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
                if "WLSACCESSID" in os.environ:
                    # Use WLS license
                    connection_params_wls = {
                        "WLSACCESSID": os.getenv("WLSACCESSID"),
                        "WLSSECRET": os.getenv("WLSSECRET"),
                        "LICENSEID": os.getenv("LICENSEID"),
                    }
                    env_wls = gur.Env(params=connection_params_wls)
                    env_wls.start()
                    opt_model = self._problem_config.get_gurobi_model_core(env=env_wls)
                elif "CLOUDACCESSID" in os.environ:
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

        # Close Gurobi envs after usage
        if "WLSACCESSID" in os.environ:
            env_wls.close()
        elif "CLOUDACCESSID" in os.environ:
            env_cld.close()

        # update current solution
        self._curr_sol = self._get_sol(opt_model)

        return OptResult(
            self.get_curr_sol(),
            opt_model.ObjVal,
            [x.X for x in opt_model._unscaled_mu],
        )

    def _get_sol(self, solved_model: gur.Model) -> list:
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

        return self._problem_config.decode([res])
