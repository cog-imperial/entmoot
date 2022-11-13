from entmoot.models.base_model import BaseModel


class TreeKernelEntmoot(BaseModel):

    def _add_to_gurobipy_model(core_model, gurobi_env):
        return NotImplementedError()

    def _add_to_pyomo_model(core_model):
        return NotImplementedError()
