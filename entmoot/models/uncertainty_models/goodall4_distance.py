import numpy as np

from entmoot.models.uncertainty_models.base_distance import CatDistance


class Goodall4Distance(CatDistance):
    def _get_pk2(self, cat_rows, cat):
        count_cat = np.sum(cat_rows == cat)
        n_rows = len(cat_rows)
        return (count_cat * (count_cat - 1)) / (n_rows * (n_rows - 1))

    def _sim_mat_rule(self, x_left, x_right, cat_idx):
        return (
            self._get_pk2(self.cache_x[:, cat_idx], x_left)
            if x_left == x_right
            else 0.0
        )
