from entmoot.models.base_model import BaseModel
from entmoot.models.uncertainty_models.euclidean_squared_distance import EuclideanSquaredDistance
from entmoot.models.uncertainty_models.l1_distance import L1Distance
from entmoot.models.uncertainty_models.l2_distance import L2Distance

from entmoot.models.uncertainty_models.overlap_distance import OverlapDistance
from entmoot.models.uncertainty_models.goodall4_distance import Goodall4Distance
from entmoot.models.uncertainty_models.of_distance import OfDistance

import numpy as np


class DistanceBasedUncertainty(BaseModel):

    def __init__(self, problem_config, params):
        self._problem_config = problem_config

        dist_metric = params.get("dist_metric", "euclidean_squared")
        dist_trafo = params.get("dist_trafo", "normal")
        acq_sense = params.get("acq_sense", "exploration")
        cat_metric = params.get("cat_metric", "overlap")

        self._beta = params.get("beta", 1.96)
        self._non_cat_x, self._cat_x = None, None
        self._dist_bound = None
        self._acq_sense = acq_sense
        self._num_cache_x = None

        if dist_trafo == "standard":
            assert len(self._problem_config.obj_list) == 1, \
                "Distance transformation 'standard' can only be used for single objective problems."

            assert len(self._problem_config.cat_idx) == 0, \
                "Distance transformation 'standard' can only be used for non-categorical problems."

            self._dist_has_var_bound = True
            self._bound_coeff = params.get("bound_coeff", 0.5)
            self._dist_coeff = 1.0
        elif dist_trafo == "normal":
            self._dist_has_var_bound = False
            self._bound_coeff = None
            self._dist_coeff = 1 / len(self._problem_config.feat_list)
        else:
            raise IOError(f"Pick 'dist_trafo' '{dist_trafo}' in '('normal', 'standard')'.")

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
    def num_cache_x(self):
        assert self._num_cache_x is not None, \
            f"Uncertainty model needs fit function call before it can predict."
        return self._num_cache_x

    def fit(self, X, y):
        if self._dist_has_var_bound:
            self._dist_bound = abs(np.var(y) * self._bound_coeff)

        self._num_cache_x = len(X)

        self.non_cat_unc_model.fit(X)
        self.cat_unc_model.fit(X)

    def predict(self, X):
        comb_pred = []
        for xi in X:
            non_cat_pred = self.non_cat_unc_model.predict(xi)
            cat_pred = self.cat_unc_model.predict(xi)
            dist_pred = np.min(non_cat_pred + cat_pred) * self._dist_coeff

            # the standard trafo case has a bound on the prediction
            if self._dist_has_var_bound:
                if dist_pred > self._dist_bound:
                    dist_pred = self._dist_bound

            comb_pred.append(dist_pred)
        return np.asarray(comb_pred)

    def add_to_gurobipy_model(self, model):
        from gurobipy import GRB

        # define main uncertainty variables
        if self._dist_has_var_bound:
            dist_bound = self._dist_bound
        else:
            dist_bound = GRB.INFINITY

        model._unc = model.addVar(
            lb=0.0, ub=dist_bound, name="uncertainty", vtype="C"
        )

        # get constr terms for non-categorical and categorical contributions
        non_cat_terms = self.non_cat_unc_model.get_gurobipy_model_constr_terms(model)
        cat_terms = self.cat_unc_model.get_gurobipy_model_constr_terms(model)

        for i, (term1, term2) in enumerate(zip(non_cat_terms, cat_terms)):
            model.addQConstr(
                model._unc <= (term1 + term2) * self._dist_coeff,
                name=f"unc_x_{i}"
            )

        model.update()
