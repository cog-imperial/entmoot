from entmoot.models.base_model import BaseModel
from entmoot.models.mean_models.tree_ensemble import TreeEnsemble
from entmoot.models.uncertainty_models.distance_based_uncertainty import DistanceBasedUncertainty
import numpy as np

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
        self._acq_sense = unc_params.get("acq_sense", "exploration")
        assert self._acq_sense in ('exploration', 'penalty'), \
            f"Pick 'acq_sense' '{self._acq_sense}' in '('exploration', 'penalty')'."

        self._beta = unc_params.get("beta", 1.96)
        assert self._beta >= 0.0, \
            f"Value for 'beta' is {self._beta} but must be '>= 0.0'."

        if self._acq_sense == "exploration":
            self._beta = -self._beta

        self.unc_model = DistanceBasedUncertainty(problem_config=problem_config, params=unc_params)

    def fit(self, X, y):
        # encode categorical features
        X = self._problem_config.encode(X)

        # check dims of X and y
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._problem_config.feat_list), \
            "Argument 'X' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._problem_config.feat_list)})', got '{X.shape}'."

        if y.ndim == 1:
            y = np.atleast_2d(y)

        assert ((y.shape[-1] == 2 and len(self._problem_config.obj_list) == 1) or
                (y.shape[-1] == len(self._problem_config.obj_list))), \
            "Argument 'y' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._problem_config.obj_list)})', got '{y.shape}'."

        self.mean_model.fit(X, y)
        self.unc_model.fit(X, y)

    def predict(self, X):
        # encode categorical features
        X = self._problem_config.encode(X)

        # check dims of X
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._problem_config.feat_list), \
            "Argument 'X' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._problem_config.feat_list)})', got '{X.shape}'."

        mean_pred = self.mean_model.predict(X).tolist()
        unc_pred = self.unc_model.predict(X)

        comb_pred = [(mean, unc)
                     for mean, unc in zip(mean_pred, unc_pred)]
        return comb_pred

    def predict_pareto(self):
        pass

    def predict_acq(self, X):
        acq_pred = []
        comb_pred = self.predict(X)
        for mean, unc in comb_pred:
            acq_pred.append(mean + self._beta * unc)
        return acq_pred


    def add_to_gurobipy_model(self, core_model):
        from gurobipy import GRB

        self.mean_model.add_to_gurobipy_model(core_model, add_mu_var=True)
        self.unc_model.add_to_gurobipy_model(core_model)

        # single objective case
        if len(self._problem_config.obj_list) == 1:
            core_model.setObjective(core_model._mu[0] + self._beta * core_model._unc)

        # TODO: obj trafo build into model mu and returned by tree model

    def add_to_pyomo_model(self, core_model):

        self.mean_model.add_to_pyomo_model(core_model, add_mu_var=True)

    def update_params(params):
        raise NotImplementedError()
