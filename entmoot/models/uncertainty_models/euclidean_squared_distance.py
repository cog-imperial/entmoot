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

    def _add_to_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def _add_pyomo_model(self, model_core):
        raise NotImplementedError()
