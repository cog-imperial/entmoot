from entmoot.models.uncertainty_models.base_distance import CatDistance


class OverlapDistance(CatDistance):
    def _sim_mat_rule(self, x_left, x_right, cat_idx):
        return 1 if x_left == x_right else 0
