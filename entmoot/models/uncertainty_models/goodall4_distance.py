from entmoot.models.uncertainty_models.base_distance import CatDistance

class Goodall4Distance(CatDistance):

    def __init__(self, problem_config, acq_sense):
        pass

    def _get_distance(self, x_left, x_right):
        raise NotImplementedError()

    def _array_predict(self, X):
        raise NotImplementedError()

    def _add_to_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def _add_pyomo_model(self, model_core):
        raise NotImplementedError()
