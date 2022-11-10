from entmoot.models.uncertainty_models.base_distance import BaseDistance

class OfDistance(BaseDistance):

    def _get_distance(self, x_left, x_right):
        raise NotImplementedError()

    def _array_predict(self, X):
        raise NotImplementedError()

    def _add_to_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def _add_pyomo_model(self, model_core):
        raise NotImplementedError()