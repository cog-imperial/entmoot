from entmoot.models.base_model import BaseModel


class Enting(BaseModel):
    def _add_gurobipy_model(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_pyomo_model(core_model):
        raise NotImplementedError()

    def _add_gurobipy_mean(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_gurobipy_uncertainty(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_pyomo_mean(core_model, gurobi_env):
        raise NotImplementedError()

    def _add_pyomo_uncertainty(core_model, gurobi_env):
        raise NotImplementedError()

    def update_params(params):
        raise NotImplementedError()
