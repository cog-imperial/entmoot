from entmoot.models.uncertainty_models.base_distance import NonCatDistance
import numpy as np

class EuclideanSquaredDistance(NonCatDistance):

    def _get_distance(self, x_left, x_right):
        if x_left.ndim == 1:
            dist = np.sum((x_left - x_right) ** 2)
        else:
            dist = np.sum((x_left - x_right) ** 2, axis=1)
        return dist

    def _array_predict(self, X):
        raise NotImplementedError()

    def get_gurobipy_model_constr_terms(self, model):
        from gurobipy import quicksum

        feat = model._all_feat

        constr_list = []
        for xi in self.x_trafo:
            constr = quicksum(
                (xi[i] - (feat[idx] - self.shift[i]) / self.scale[i]) *
                (xi[i] - (feat[idx] - self.shift[i]) / self.scale[i])
                for i, idx in enumerate(self._problem_config.non_cat_idx)
            )
            constr_list.append(constr)
        return constr_list

    def _add_to_pyomo_model(self, model_core):
        raise NotImplementedError()
