class BaseModel:

    def __init__(self, problem_config, num_obj, params):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def _array_predict(self, X):
        raise NotImplementedError()

    def fit(self, X, y):
        raise NotImplementedError()

    def _add_gurobipy_model(self, model_core):
        raise NotImplementedError()

    def _add_pyomo_model(self, model_core):
        raise NotImplementedError()
