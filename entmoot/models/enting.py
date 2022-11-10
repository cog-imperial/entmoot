from entmoot.models.base_model import BaseModel
from entmoot.models.mean_models.tree_ensemble import TreeEnsemble
from entmoot.models.uncertainty_models.euclidean_squared_distance import EuclideanSquaredDistance
from entmoot.models.uncertainty_models.l1_distance import L1Distance
from entmoot.models.uncertainty_models.l2_distance import L2Distance

from entmoot.models.uncertainty_models.overlap_distance import OverlapDistance
from entmoot.models.uncertainty_models.goodall4_distance import Goodall4Distance
from entmoot.models.uncertainty_models.of_distance import OfDistance

class Enting(BaseModel):
    def __init__(self, problem_config, params=None):

        if params is None:
            params = {}

        self._problem_config = problem_config

        # check params values
        tree_training_params = params.get("tree_train_params", {})

        # initialize mean model
        self.mean_model = TreeEnsemble(problem_config=problem_config, params=tree_training_params)

        # initialize unc model
        unc_params = params.get("unc_params", {})
        dist_metric = unc_params.get("dist_metric", "euclidean_squared")
        dist_trafo = unc_params.get("dist_trafo", "normalize")
        acq_sense = unc_params.get("acq_sense", "exploration")
        cat_metric = unc_params.get("cat_metric", "overlap")
        self._beta = unc_params.get("beta", 1.96)
        self._non_cat_x, self._cat_x = None, None

        assert dist_trafo in ('normal', 'standard'), \
            f"Pick 'dist_trafo' '{dist_trafo}' in '('normalize', 'standardize')'."

        assert acq_sense in ('exploration', 'penalty'), \
            f"Pick 'acq_sense' '{acq_sense}' in '('exploration', 'penalty')'."

        # pick distance metric for non-cat features
        if dist_metric == "euclidean_squared":
            self.non_cat_unc_model = EuclideanSquaredDistance(
                problem_config=self._problem_config, acq_sense=acq_sense, dist_trafo=dist_trafo)
        elif dist_metric == "l1":
            self.non_cat_unc_model = L1Distance(
                problem_config=self._problem_config, acq_sense=acq_sense, dist_trafo=dist_trafo)
        elif dist_metric == "l2":
            self.non_cat_unc_model = L2Distance(
                problem_config=self._problem_config, acq_sense=acq_sense, dist_trafo=dist_trafo)
        else:
            raise IOError(f"Non-categorical uncertainty metric '{dist_metric}' for "
                          f"{self.__class__.__name__} model is not supported. "
                          f"Check 'params['uncertainty_type']'.")

        # pick distance metric for cat features
        if cat_metric == "overlap":
            self.cat_unc_model = OverlapDistance(
                problem_config=self._problem_config, acq_sense=acq_sense)
        elif cat_metric == "of":
            self.cat_unc_model = L1Distance(
                problem_config=self._problem_config, acq_sense=acq_sense)
        elif cat_metric == "goodall4":
            self.cat_unc_model = L2Distance(
                problem_config=self._problem_config, acq_sense=acq_sense)
        else:
            raise IOError(
                f"Categorical uncertainty metric '{cat_metric}' for {self.__class__.__name__} "
                f"model is not supported. Check 'params['uncertainty_type']'.")

    @property
    def non_cat_x(self):
        return self._non_cat_x

    @property
    def cat_x(self):
        return self._cat_x

    def _add_to_gurobipy_model(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_pyomo_model(core_model):
        raise NotImplementedError()

    def _add_gurobipy_mean(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_gurobipy_uncertainty(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_pyomo_mean(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_pyomo_uncertainty(core_model, gurobi_env):
        raise NotImplementedError()

    def update_params(params):
        raise NotImplementedError()
