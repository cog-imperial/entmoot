How to install ENTMOOT
======================
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


### Installing ENTMOOT
Install all required packages by running the command
```
pip install -r requirements.txt
```
To install `ENTMOOT`, run the following command
```
pip install git+https://github.com/cog-imperial/entmoot
```

### Uninstalling ENTMOOT
The ENTMOOT package can be uninstalled by running
```
pip uninstall entmoot
```

## Installation - Windows
The installation on Windows-based systems is similar with some exceptions for
individual commands. Make sure that `python` and `git` is installed and
accessible in the shell you are using to install `ENTMOOT`. Follow the
instructions by executing all commands in `cmd.exe`.

### Installing ENTMOOT
Install all required packages by running the command
```
pip install -r requirements.txt
```
To install `ENTMOOT`, run the following command
```
pip install git+https://github.com/cog-imperial/entmoot
```

### Uninstalling ENTMOOT
The ENTMOOT package can be uninstalled by running
```
pip uninstall entmoot
```

## Installing Gurobi
To use the `acq_optimizer= 'global'` setting in `ENTMOOT`, the solver
software [Gurobi v9.0](https://www.gurobi.com/resource/overview-of-gurobi-9-0/)
or newer is required. Gurobi is a commercial mathematical optimization solver and
free of charge for academic research. It is available on Linux, Windows and
Mac OS.

Please follow the instructions to obtain a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/). Once Gurobi
is installed on your system, follow the steps to setup the Python interface [gurobipy](https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html) in your virtual environment created for `ENTMOOT`.