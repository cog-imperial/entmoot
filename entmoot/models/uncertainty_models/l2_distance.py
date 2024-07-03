import numpy as np

from entmoot.models.uncertainty_models.base_distance import NonCatDistance


class L2Distance(NonCatDistance):
    def _get_distance(self, x_left, x_right):
        if x_left.ndim == 1:
            dist = np.sqrt(np.sum((x_left - x_right) ** 2))
        else:
            dist = np.sqrt(np.sum((x_left - x_right) ** 2, axis=1))
        return dist

    def _array_predict(self, X):
        raise NotImplementedError()

    def get_gurobipy_model_constr_terms(self, model):
        from gurobipy import quicksum

        features = model._all_feat

        constr_list = []
        for xi in self.x_trafo:
            constr = quicksum(
                (xi[i] - (features[idx] - self.shift[i]) / self.scale[i])
                * (xi[i] - (features[idx] - self.shift[i]) / self.scale[i])
                for i, idx in enumerate(self._problem_config.non_cat_idx)
            )
            constr_list.append(constr)
        return constr_list

    def get_pyomo_model_constr_terms(self, model):
        features = model._all_feat

        constr_list = []
        for xi in self.x_trafo:
            constr = sum(
                (xi[i] - (features[idx] - self.shift[i]) / self.scale[i])
                * (xi[i] - (features[idx] - self.shift[i]) / self.scale[i])
                for i, idx in enumerate(self._problem_config.non_cat_idx)
            )
            constr_list.append(constr)
        return constr_list
