from entmoot.models.uncertainty_models.base_distance import CatDistance
import numpy as np
from math import log


class OfDistance(CatDistance):
    def _get_of_frac(self, cat_rows, cat_left, cat_right):
        count_cat_left = np.sum(cat_rows == cat_left)
        count_cat_right = np.sum(cat_rows == cat_right)
        n_rows = len(cat_rows)
        return 1 / (1 + log(n_rows / count_cat_left) * log(n_rows / count_cat_right))

    def _sim_mat_rule(self, x_left, x_right, cat_idx):
        return (
            self._get_of_frac(self._cache_x[:, cat_idx], x_left, x_right)
            if x_left != x_right
            else 1.0
        )
