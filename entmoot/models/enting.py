from entmoot.models.base_model import BaseModel
from entmoot.models.mean_models.tree_ensemble import TreeEnsemble
from entmoot.models.uncertainty_models.euclidean_squared_distance import EuclideanSquaredDistance
from entmoot.models.uncertainty_models.l1_distance import L1Distance
from entmoot.models.uncertainty_models.l2_distance import L2Distance


class Enting(BaseModel):
    def __init__(self, num_obj, params=None):

        if params is None:
            params = {}

        # check params values
        tree_training_params = params.get("tree_training_params", {})

        # initialize mean model
        self.mean_model = TreeEnsemble(params=tree_training_params)

        # initialize unc model
        unc_params = params.get("unc_params", {})
        dist_metric = unc_params.get("distance_metric", "euclidean_squared")
        dist_trafo = unc_params.get("distance_transformation", "normalize")
        acq_sense = unc_params.get("acquisition_sense", "exploration")

        assert dist_trafo in ('normalize', 'standardize'), \
            f"Pick 'distance_transformation' '{dist_trafo}' in '('normalize', 'standardize')'."

        assert acq_sense in ('exploration', 'penalty'), \
            f"Pick 'acquisition_sense' '{acq_sense}' in '('exploration', 'penalty')'."

        # pick distance metric
        if dist_metric == "euclidean_squared":
            self.unc_model = EuclideanSquaredDistance(dist_trafo=dist_trafo, acq_sense=acq_sense)
        elif dist_metric == "l1":
            self.unc_model = L1Distance(dist_trafo=dist_trafo, acq_sense=acq_sense)
        elif dist_metric == "l2":
            self.unc_model = L2Distance(dist_trafo=dist_trafo, acq_sense=acq_sense)
        else:
            raise IOError(f"Uncertainty type '{dist_metric}' for {self.__class__.__name__} model "
                          f"is not supported. Check 'params['uncertainty_type']'.")

    def _add_gurobipy_model(core_model, gurobi_env):
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
