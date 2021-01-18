
<img src="media/entmoot_logo.png" width="400">

`ENTMOOT` (**EN**semble **T**ree **MO**del **O**ptimization **T**ool) is a novel framework to handle tree-based models in Bayesian optimization applications. Gradient-boosted
tree models from `lightgbm` are combined with a distance-based uncertainty 
measure in a deterministic global optimization framework to optimize black-box functions. More
details on the method here: https://arxiv.org/abs/2003.04774.

When using any `ENTMOOT` for any publications please reference this software package as:
```
@article{entmoot,
  author  = {Alexander Thebelt, Jan Kronqvist, Miten Mistry, Robert M. Lee, Nathan Sudermann-Merx and Ruth Misener},
  title   = {{ENTMOOT: A Framework for Optimization over Ensemble Tree Models}},
  journal = {ArXiv},
  volume = {2003.04774},
  year    = {2020}
}
```

The "ask-tell" optimizer interface for an intuitive incorporation of own 
black-box functions was included from 
[scikit-optimize](https://github.com/scikit-optimize/scikit-optimize/) 
which is available under the BSD 3-Clause License.

## Installation - Linux & Mac OS
### Requirements
* python >= 3.7
* numpy >= 1.18.4
* scikit-learn >= 0.21.3
* pyyaml >= 5.3.1
* lightgbm >= 2.3.1

On Mac, you will also need to install libomp: `brew install libomp`.

### Optional
* gurobi >= 9.0.1

### Creating a virtual environment
We recommend installing `ENTMOOT` in a [virtual environment](https://docs.python.org/3/library/venv.html). 

To set up a new virtual environment called 'entmoot', run the command
```
python3 -m venv entmoot
```
in the folder where you want to store the virtual environment.
Afterwards, activate the environment using
```
source entmoot/bin/activate
```
It is recommended that you update the pip installation in the virtual environment
```
pip install --upgrade pip
```
### Installing ENTMOOT
Install all required packages by running the command
```
pip install -r requirements.txt
```
Another option is to install all required packages separately using
```
pip install numpy scikit-learn pyyaml lightgbm
```
To install `ENTMOOT`, run the following command in the virtual environment
```
pip install git+https://github.com/cog-imperial/entmoot
```

### Uninstalling ENTMOOT
The ENTMOOT package can be uninstalled by running
```
pip uninstall entmoot
```
Alternatively, the folder containing the virtual environment can be deleted.

## Installation - Windows
The installation on Windows-based systems is similar with some exceptions for
individual commands. Make sure that `python` and `git` is installed and 
accessible in the shell you are using to install `ENTMOOT`. Follow the 
instructions by executing all commands in `cmd.exe`.

### Creating a virtual environment
We recommend installing `ENTMOOT` in a [virtual environment](https://docs.python.org/3/library/venv.html). 

To set up a new virtual environment called 'entmoot', run the command
```
python -m venv entmoot
```
in the folder where you want to store the virtual environment.
Afterwards, activate the environment using
```
.\entmoot\Scripts\activate.bat
```
It is recommended that you update the pip installation in the virtual environment
```
pip install --upgrade pip
```
### Installing ENTMOOT
Install all required packages by running the command
```
pip install -r requirements.txt
```
Another option is to install all required packages separately using
```
pip install numpy scikit-learn pyyaml lightgbm
```
To install `ENTMOOT`, run the following command in the virtual environment
```
pip install git+https://github.com/cog-imperial/entmoot
```

### Uninstalling ENTMOOT
The ENTMOOT package can be uninstalled by running
```
pip uninstall entmoot
```
Alternatively, the folder containing the virtual environment can be deleted.

## Installing Gurobi
To use the `acq_optimizer= 'global'` setting in `ENTMOOT`, the solver 
software [Gurobi v9.0](https://www.gurobi.com/resource/overview-of-gurobi-9-0/) 
or newer is required. Gurobi is a commercial mathematical optimization solver and 
free of charge for academic research. It is available on Linux, Windows and 
Mac OS. 

Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). Once Gurobi 
is installed on your system, follow the steps to setup the Python interface [gurobipy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html) in your virtual environment created for `ENTMOOT`.

## Example
For a simple example we use Rosenbrock as our black-box function. We import the
`entmoot_minimize` function which automates the optimization procedure. The 
black-box function is given as `func` to the solver. For more details on 
individual settings we refer to the source code which explains all parameters
in detail.

An example script that minimizes the Rosenbrock function using `ENTMOOT` is
given in the following.
```
from entmoot.optimizer.entmoot_minimize import entmoot_minimize
from entmoot.benchmarks import Rosenbrock

func = Rosenbrock()

res = entmoot_minimize(
    func,
    func.get_bounds(),
    n_calls=60,
    n_points=10000,
    base_estimator="GBRT",
    std_estimator="L1DDP", 
    n_initial_points=50, 
    initial_point_generator="random",
    acq_func="LCB", 
    acq_optimizer="sampling",
    x0=None,
    y0=None,  
    random_state=100, 
    acq_func_kwargs=None, 
    acq_optimizer_kwargs=None,
    std_estimator_kwargs=None,
    model_queue_size=None,
    base_estimator_kwargs=None,
    verbose = True,
)

```

The following example shows how own constraints can be enforced on the input 
variables. This only works when Gurobi is selected as the solution strategy,
 i. e. `acq_optimizer="global"`. Constraints are formulated according to 
 [documentation](https://www.gurobi.com/documentation/9.0/refman/py_model_addconstr.html).
```
from entmoot.optimizer.entmoot_minimize import entmoot_minimize
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

# specify the model core in `acq_optimizer_kwargs`
res = entmoot_minimize(
    func,
    func.get_bounds(),
    n_calls=15,
    n_points=10000,
    base_estimator="GBRT",
    std_estimator="L1DDP", 
    n_initial_points=5, 
    initial_point_generator="random",
    acq_func="LCB", 
    acq_optimizer="global",
    x0=None,
    y0=None,  
    random_state=100, 
    acq_func_kwargs=None, 
    acq_optimizer_kwargs={
      "add_model_core": core_model
    },
    std_estimator_kwargs=None,
    model_queue_size=None,
    base_estimator_kwargs={
        "min_child_samples": 2
    },
    verbose = True,
)

```

## Authors
* **[Alexander Thebelt](https://optimisation.doc.ic.ac.uk/person/alexander-thebelt/)** ([ThebTron](https://github.com/ThebTron)) - Imperial College London

## License
The ENTMOOT package is released under the BSD 3-Clause License. Please refer to the [LICENSE](https://github.com/cog-imperial/entmoot/blob/master/LICENSE) file for details.

## Acknowledgements
The support of BASF SE, Lugwigshafen am Rhein is gratefully acknowledged.