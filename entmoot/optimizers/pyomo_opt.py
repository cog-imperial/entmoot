from entmoot.problem_config import ProblemConfig
from entmoot.models.enting import Enting


class PyomoOptimizer:
    def __init__(self, problem_config: ProblemConfig, params: dict = None):
        self._params = {} if params is None else params
        self._problem_config = problem_config

    def solve(self, tree_model: Enting, model_core=None, weights=None):

        import pyomo.environ as pyo

        if model_core is None:
            opt_model = self._problem_config.get_pyomo_model_core()
        else:
            # create model copy to not overwrite original one
            opt_model = self._problem_config.copy_pyomo_model_core(model_core)

        # check weights
        if weights is not None:
            assert len(weights) == len(self._problem_config.obj_list), \
                f"Number of 'weights' is '{len(weights)}', number of objectives " \
                f"is '{len(self._problem_config.obj_list)}'."
            assert sum(weights) == 1.0, "weights don't add up to 1.0"

        # choose solver
        opt = pyo.SolverFactory(self._params["solver_name"])

        # set solver parameters
        if "solver_options" in self._params:
            for k, v in self._params["solver_options"].items():
                opt.options[k] = v

        # build pyomo model using information from tree model
        tree_model.add_to_pyomo_model(opt_model)

        # Solve optimization model
        opt.solve(opt_model)

        return pyo.value(opt_model.obj)

    def sample_feas(num_points):
        raise NotImplementedError()
