from typing import Optional

import numpy as np

from entmoot.models.base_model import BaseModel
from entmoot.problem_config import ProblemConfig, Categorical


class NonCatDistance(BaseModel):
    def __init__(self, problem_config: ProblemConfig, acq_sense, dist_trafo):
        self._problem_config = problem_config
        self._acq_sense = acq_sense
        self._dist_trafo = dist_trafo

        self._shift: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._x_trafo = None

    @property
    def shift(self):
        assert self._shift is not None, "Must first cache shift"
        return self._shift

    @property
    def scale(self):
        assert self._scale is not None, "Must first cache scale"
        return self._scale

    @property
    def x_trafo(self):
        assert (
            self._x_trafo is not None
        ), "Uncertainty model needs fit function call before it can predict."
        return self._x_trafo

    def get_big_m(self):
        lb = self._trafo(self._problem_config.non_cat_lb)
        ub = self._trafo(self._problem_config.non_cat_ub)
        return self._get_distance(lb, ub)

    def _get_distance(self, x_left, x_right):
        raise NotImplementedError()

    def _trafo(self, non_cat_x):
        return (non_cat_x - self._shift) / self._scale

    def predict(self, xi):
        non_cat_x = self._trafo(np.atleast_2d(xi)[:, self._problem_config.non_cat_idx])
        return self._get_distance(self.x_trafo, non_cat_x)

    def _array_predict(self, X):
        raise NotImplementedError()

    def fit(self, X):
        non_cat_x = X[:, self._problem_config.non_cat_idx]

        # define shift and scalar values for non-cat feats
        if self._dist_trafo == "normal":
            self._shift = np.asarray(self._problem_config.non_cat_lb)
            self._scale = np.asarray(self._problem_config.non_cat_bnd_diff)
        elif self._dist_trafo == "standard":
            self._shift = np.mean(np.asarray(non_cat_x), axis=0)
            self._scale = np.std(np.asarray(non_cat_x), axis=0)
        else:
            raise IOError(
                "Parameter 'dist_trafo' for uncertainty model needs to be "
                "in '('normal', 'standard')'."
            )

        self._x_trafo = self._trafo(non_cat_x)

    def get_gurobipy_model_constr(self, model_core):
        raise NotImplementedError()

    def get_pyomo_model_constr(self, model_core):
        raise NotImplementedError()
    
    def get_gurobipy_model_constr_terms(self, model) -> list:
        raise NotImplementedError()

    def get_pyomo_model_constr_terms(self, model) -> list:
        raise NotImplementedError()



class CatDistance(BaseModel):
    def __init__(self, problem_config: ProblemConfig, acq_sense):
        self._problem_config = problem_config
        self._acq_sense = acq_sense

        self._cat_x: Optional[np.ndarray] = None
        self._sim_map: Optional[dict[int, np.ndarray]] = None
        self._cache_x: Optional[np.ndarray] = None

    @property
    def cache_x(self):
        assert (
            self._cache_x is not None
        ), "Uncertainty model needs fit function call before it can predict."
        return self._cache_x

    @property
    def sim_map(self):
        assert (
            self._sim_map is not None
        ), "Uncertainty model needs fit function call before it can predict."
        return self._sim_map

    def get_big_m(self):
        # categorical contribution has maximum of one
        return len(self._problem_config.cat_idx)

    def predict(self, xi):
        # iterate through every evaluation row
        sim_vec = np.zeros(len(self.cache_x))

        for idx in self._problem_config.cat_idx:
            pred_cat = int(xi[idx])
            cached_cats = (np.rint(self.cache_x[:, idx])).astype(int)
            sim_vec += self.sim_map[idx][pred_cat, cached_cats]

        # compute distance based on similarity
        dist_vec = (-1) * sim_vec + len(self._problem_config.cat_idx)
        return dist_vec

    def _sim_mat_rule(self, x_left, x_right, cat_idx):
        raise NotImplementedError()

    def fit(self, X):
        self._cache_x = X

        # generate similarity matrix for all data points
        self._sim_map = {}

        for idx, feat in self._problem_config.get_idx_and_feat_by_type(Categorical):
            all_cats = feat.enc_cat_list

            # creates similarity entries for all categories of all categorical features
            mat = np.fromfunction(
                np.vectorize(self._sim_mat_rule, otypes=[float]),
                (len(all_cats), len(all_cats)),
                dtype=int,
                cat_idx=idx,
            )
            self._sim_map[idx] = mat

    def get_gurobipy_model_constr_terms(self, model):
        features = model._all_feat
        constr_list = []
        for xi in self.cache_x:
            constr = 0

            # iterate through all categories to check which distances are active
            for idx, feat in self._problem_config.get_idx_and_feat_by_type(Categorical):
                for cat in feat.enc_cat_list:
                    sim = self.sim_map[idx][cat, int(xi[idx])]
                    constr += (1 - sim) * features[idx][cat]

            constr_list.append(constr)
        return constr_list

    def get_pyomo_model_constr_terms(self, model):
        features = model._all_feat
        constr_list = []
        for xi in self.cache_x:
            constr = 0

            # iterate through all categories to check which distances are active
            for idx, feat in self._problem_config.get_idx_and_feat_by_type(Categorical):
                for cat in feat.enc_cat_list:
                    sim = self.sim_map[idx][cat, int(xi[idx])]
                    constr += (1 - sim) * features[idx][cat]

            constr_list.append(constr)
        return constr_list
