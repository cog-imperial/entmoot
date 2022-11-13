How to install ENTMOOT
======================
Requirements
-------------
* python >= 3.7
* numpy >= 1.18.4
* scikit-learn >= 0.21.3
* pyyaml >= 5.3.1
* lightgbm >= 2.3.1
In addition, you will need the commercial solver Gurobi or the open source framework Pyomo (which has also APIs to
non-commercial solvers), i.e. one of the following packages should be installed.

* gurobi >= 9.0.1
* pyomo >= 6.4.2

Installing ENTMOOT
------------------
Install all required packages by running the command
.. code-block:: python
    pip install -r requirements.txt

To install ENTMOOT, run the following command
.. code-block:: python
    pip install git+https://github.com/cog-imperial/entmoot

Uninstalling ENTMOOT
--------------------
The ENTMOOT package can be uninstalled by running
.. code-block:: python
pip uninstall entmoot

Installation - Linux & Mac OS
-----------------------------
On Mac, you also need to install libomp:
.. code-block:: python
    brew install libomp
Installing Gurobi
-----------------
To use the :code:`acq_optimizer= 'global'` setting in :code:`ENTMOOT`, the solver
software [Gurobi v9.0](https://www.gurobi.com/resource/overview-of-gurobi-9-0/)
or newer is required. Gurobi is a commercial mathematical optimization solver and
free of charge for academic research. It is available on Linux, Windows and
Mac OS.

Please follow the instructions to obtain a [free academic license]
(https://www.gurobi.com/academia/academic-program-and-licenses/). Once Gurobi is installed on your system, follow the
steps to setup the Python interface [gurobipy]
(https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html) in your virtual environment
created for :code:`ENTMOOT`.