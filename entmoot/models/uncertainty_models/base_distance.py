from entmoot.models.base_model import BaseModel
import numpy as np


class BaseDistance(BaseModel):

    def __init__(self, problem_config, acq_sense, dist_trafo):
        self._problem_config = problem_config
        self._acq_sense = acq_sense
        self._dist_trafo = dist_trafo

        self._shift, self._scale = None, None
        self._X = None

    @property
    def shift(self):
        return self._shift

    @property
    def scale(self):
        return self._scale

    def _get_distance(self, x_left, x_right):
        raise NotImplementedError()

    def predict(self, X):
        return self._get_distance(self._X_trafo, X)

    def _array_predict(self, X):
        raise NotImplementedError()

    def fit(self, X, y):
        # define shift and scalar values for non-cat feats
        if self._dist_trafo == "normal":
            self._shift = np.asarray(self._problem_config.non_cat_lb)
            self._scale = np.asarray(self._problem_config.non_cat_bnd_diff)
        elif self._dist_trafo == "standard":
            self._shift = np.mean(np.asarray(X), axis=0)
            self._scale = np.std(np.asarray(X), axis=0)
        else:
            raise IOError("Parameter 'dist_trafo' for uncertainty model needs to be "
                          "in '('normal', 'standard')'.")

        self._X_trafo = (X - self._shift) / self._scale

    def _add_to_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def _add_pyomo_model(self, model_core):
        raise NotImplementedError()
