
## single objective optimization
from entmoot.benchmarks import Rosenbrock
func = Rosenbrock()

from entmoot.optimizer import Optimizer
opt = Optimizer(func.get_bounds(5),
                n_initial_points=5,
                base_estimator="ENTING",
                acq_func="LCB",
                acq_optimizer="global",
                base_estimator_kwargs={
                    "lgbm_params": {"min_child_samples": 2},
                    "unc_metric": 'exploration',
                    "unc_scaling": "standard",
                    "dist_metric": "manhattan"
                },
                random_state=100)

# initialize the search space manually
from entmoot.space.space import Space
space = Space(func.get_bounds(5))

# define core model to add constraints
from entmoot.optimizer.gurobi_utils import \
    get_core_gurobi_model, get_opt_core_copy

core_model = get_core_gurobi_model(space)
x0 = core_model._cont_var_dict[0]
core_model.addConstr(x0 == 0.5)
core_model.update()

for _ in range(10):
    print(f"iter.: {_}")

    # copying the model is necessary since it will be updated internally
    new_core = get_opt_core_copy(core_model)

    next_x = opt.ask(add_model_core=new_core)
    next_y = func(next_x)

    # print(next_y)
    opt.tell(next_x,next_y)

print(next_x)

print("\n \n")



## multi-objective optimization
from entmoot.optimizer import Optimizer
from entmoot.benchmarks import BenchmarkFunction
import numpy as np

# artificial function that
class RosenbrockMulti(BenchmarkFunction):
    def __init__(self, func_config={}):
        self.name = 'benchmark_function'
        self.func_config = func_config
        self.y_opt = 0.0

    def get_bounds(self, n_dim=2):
        return [(-2.048,2.048) for _ in range(n_dim)]

    def get_X_opt(self, n_dim=2):
        return [[ 1.0 for _ in range(n_dim) ]]

    def _eval_point(self, X):
        X = np.asarray_chkfinite(X)
        X0 = X[:-1]
        X1 = X[1:]

        add1 = sum( (1.0 - X0)**2.0 )
        add2 = 100.0 * sum( (X1 - X0**2.0)**2.0 )
        return add1 + add2, add2*5/add1

funcMulti = RosenbrockMulti()
opt = Optimizer(funcMulti.get_bounds(),
                num_obj=2,
                n_initial_points=5,
                random_state=100)

# define core model to add constraints
space = Space(funcMulti.get_bounds())
core_model = get_core_gurobi_model(space)
x0 = core_model._cont_var_dict[0]
core_model.addConstr(x0 == 0.5)
core_model.update()

for _ in range(10):
    print(f"iter.: {_}")

    # copying the model is necessary since it will be updated internally
    new_core = get_opt_core_copy(core_model)

    next_x = opt.ask(weights=[[0.5,0.5]], add_model_core=new_core)
    next_y = funcMulti(next_x)

    opt.tell(next_x,next_y)

print(next_x)