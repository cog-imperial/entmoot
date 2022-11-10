from entmoot.models.base_model import BaseModel
import numpy as np


class BaseDistance(BaseModel):

    def __init__(self, problem_config, params=None):
        if params is None:
            params = {}

        self._problem_config = problem_config
        self._acq_sense = params.get("acq_sense", "exploration")
        self._dist_trafo = params.get("dist_trafo", "normal")

        self._shift, self._scale = None, None
        self._non_cat_x, self._cat_x = None, None

        assert self._dist_trafo in ('normal', 'standard'), \
            "Parameter 'dist_trafo' for uncertainty model needs to be " \
            "in '('normal', 'standard')'."

        assert self._acq_sense in ('exploration', 'penalty'), \
            "Parameter 'acq_sense' for uncertainty model needs to be " \
            "in '('exploration', 'penalty')'."

    @property
    def shift(self):
        return self._shift

    @property
    def scale(self):
        return self._scale

    @property
    def non_cat_x(self):
        return self._non_cat_x

    @property
    def cat_x(self):
        return self._cat_x

    def _get_distance(self, x_left, x_right):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def _array_predict(self, X):
        raise NotImplementedError()

    def fit(self, X, y):
        # define shift and scalar values for non-cat feats
        if self._dist_trafo == "normal":
            self._shift = np.asarray(self._problem_config.non_cat_lb)
            self._scale = np.asarray(self._problem_config.non_cat_bnd_diff)
        elif self._dist_trafo == "standard":
            self._shift = np.mean(np.asarray(X[:, self._problem_config.non_cat_idx]), axis=0)
            self._scale = np.std(np.asarray(X[:, self._problem_config.non_cat_idx]), axis=0)
        else:
            raise IOError("Parameter 'dist_trafo' for uncertainty model needs to be "
                          "in '('normal', 'standard')'.")

        self._non_cat_x = X[:, self._problem_config.non_cat_idx]
        self._cat_x = X[:, self._problem_config.cat_idx]

    def _add_to_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def _add_pyomo_model(self, model_core):
        raise NotImplementedError()
