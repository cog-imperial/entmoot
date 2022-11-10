Examples
========
Here are some examples on how to work with ENTMOOTT using Jupyter Notebooks.

## Example: Single Objective Optimization with Constraints
For a simple example we use Rosenbrock as our black-box function. We import the
`entmoot_minimize` function which automates the optimization procedure. The
black-box function is given as `func` to the solver. For more details on
individual settings we refer to the source code which explains all parameters
in detail.

An example script that minimizes the Rosenbrock function using `ENTMOOT` is
given in the following.
```
from entmoot.optimizer.optimizer import Optimizer
from entmoot.benchmarks import Rosenbrock

func = Rosenbrock()

opt = Optimizer(func.get_bounds(10),
                base_estimator="ENTING",
                n_initial_points=20,
                initial_point_generator="random",
                acq_func="LCB",
                acq_optimizer="sampling",
                random_state=100,
                model_queue_size=None,
                base_estimator_kwargs={
                    "lgbm_params": {"min_child_samples": 1}
                },
                verbose=True,
                )

# run optimizer for 20 iterations
res = opt.run(func, n_iter=20)
print(f"-> best solution found {res.fun}")
```

The following example shows how own constraints can be enforced on the input
variables. This only works when Gurobi is selected as the solution strategy,
 i. e. `acq_optimizer="global"`. Constraints are formulated according to
 [documentation](https://www.gurobi.com/documentation/9.0/refman/py_model_addconstr.html).
```
from entmoot.optimizer.optimizer import Optimizer
from entmoot.benchmarks import SimpleCat

func = SimpleCat()

# initialize the search space manually
from entmoot.space.space import Space
space = Space(func.get_bounds())

# get the core of the gurobi model from helper function 'get_core_gurobi_model'
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model
core_model = get_core_gurobi_model(space)

# ordering of variable indices is dependent on space definition

# cont_var_dict contains all continuous variables
x0 = core_model._cont_var_dict[0]
x1 = core_model._cont_var_dict[1]

# cat_var_dict contains all categorical variables
x2 = core_model._cat_var_dict[2]

# define constraints accordingly
core_model.addConstr(x0 + x1 >= 2)
core_model.addConstr(x1 == 1)
core_model.update()

opt = Optimizer(func.get_bounds(),
                base_estimator="ENTING",
                n_initial_points=0,
                initial_point_generator="random",
                acq_func="LCB",
                acq_optimizer="global",
                random_state=100,
                acq_func_kwargs=None,
                acq_optimizer_kwargs={
                    "add_model_core": core_model
                },
                model_queue_size=None,
                base_estimator_kwargs={
                    "lgbm_params": {"min_child_samples": 1}
                },
                verbose=True,
                )

# add initial points that are feasible
for x in [(1.1, 1.0, 'mult6'), (1.2, 1.0, 'mult6'), (1.3, 1.0, 'pow2')]:
    opt.tell(x, func(x))

# run optimizer for 20 iterations
res = opt.run(func, n_iter=20)
print(f"-> best solution found {res.fun}")
```

## Example: Multiple Objective Optimization with Constraints

`ENTMOOT` also supports multi-objective optimization according to:

```
@article{thebelt2022multi,
  title={Multi-objective constrained optimization for energy applications via tree ensembles},
  author={Thebelt, Alexander and Tsay, Calvin and Lee, Robert M and Sudermann-Merx, Nathan and Walz, David and Tranter, Tom and Misener, Ruth},
  journal={Applied Energy},
  volume={306},
  pages={118061},
  year={2022},
  publisher={Elsevier}
}
```

An example that derives Pareto-optimal points of the
[Fonzeca Freming](https://en.wikipedia.org/wiki/Test_functions_for_optimization) is given in the
following:

```
from entmoot.benchmarks import FonzecaFleming
from entmoot.optimizer import Optimizer

# initialize multi-objective test function
funcMulti = FonzecaFleming()

# define optimizer object and specify num_obj=2
opt = Optimizer(funcMulti.get_bounds(),
                num_obj=2,
                n_initial_points=10,
                random_state=100)

# main BO loop that derives pareto-optimal points
for _ in range(50):
    next_x = opt.ask()
    next_y = funcMulti(next_x)
    opt.tell(next_x,next_y)
```

Using multi-objective functionality in `ENTMOOT` requires the specification of `num_obj` which
informs the solver about the number of objectives that we optimize for. `ENTMOOT` minimizes
objectives which requires the modification of maximization problems, i.e. minimizing the
negative objective.
