from entmoot.problem_config import ProblemConfig
from entmoot.models.enting import Enting
import gurobipy as gur

class GurobiOptimizer:
    def __init__(self, problem_config: ProblemConfig, params: dict = None) -> float:
        self._params = {} if params is None else params
        self._problem_config = problem_config

    def solve(self, tree_model: Enting, model_core: gur.Model = None, weights: tuple = None):
        if model_core is None:
            opt_model = self._problem_config.get_gurobi_model_core()
        else:
            # create model copy to not overwrite original one
            opt_model = self._problem_config.copy_gurobi_model_core(model_core)

        # check weights
        if weights is not None:
            assert len(weights) == len(self._problem_config.obj_list), \
                f"Number of 'weights' is '{len(weights)}', number of objectives " \
                f"is '{len(self._problem_config.obj_list)}'."
            assert sum(weights) == 1.0, "weights don't add up to 1.0"

        # set solver parameters
        for param, param_val in self._params.items():
            opt_model.setParam(param, param_val)

        # build gurobi model
        tree_model.add_to_gurobipy_model(opt_model, weights=weights)

        # Solve optimization model
        opt_model.optimize()

        return opt_model.ObjVal

    def get_next_x(self):
        raise NotImplementedError()

    def sample_feas(self, model_core: gur.Model, num_points):
        raise NotImplementedError()
