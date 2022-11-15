from entmoot.models.uncertainty_models.base_distance import CatDistance

class OfDistance(CatDistance):

    def __init__(self, problem_config, acq_sense):
        pass

    def _get_distance(self, x_left, x_right):
        raise NotImplementedError()

    def _array_predict(self, X):
        raise NotImplementedError()

    def add_to_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def add_to_pyomo_model(self, model_core):
        raise NotImplementedError()