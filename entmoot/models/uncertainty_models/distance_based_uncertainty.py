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

        assert dist_trafo in ('normal', 'standard'), \
            f"Pick 'dist_trafo' '{dist_trafo}' in '('normal', 'standard')'."

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

    def fit(self, X, y):
        self.non_cat_unc_model.fit(X, y)
        self.cat_unc_model.fit(X, y)

    def predict(self, X):
        comb_pred = []
        for xi in X:
            non_cat_pred = self.non_cat_unc_model.predict(xi)
            cat_pred = self.cat_unc_model.predict(xi)
            comb_pred.append(
                np.min(non_cat_pred + cat_pred) / len(self._problem_config.feat_list)
            )
        return np.asarray(comb_pred)