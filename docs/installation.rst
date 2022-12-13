How to install ENTMOOT
======================
ENTMOOT supports the Python (CPython) versions 3.7 - 3.11.

Using PIP
-------------

You can install ENTMOOT with

::

   pip install entmoot

**On Mac**, you also need to install libomp:

::

    brew install libomp

In addition, you need to install **one of the following** Python packages.

* gurobipy >= 10.0.0
* pyomo >= 6.4.4

The package gurobipy belongs to the commercial solver `Gurobi <https://www.gurobi.com/>`__ whereas pyomo refers to the
open source framework `Pyomo <http://www.pyomo.org/>`__ which supports several commercial and noncommercial solvers.


Gurobi or Pyomo?
-----------------
You can work with ENTMOOT using Gurobi or Pyomo.

* `Gurobi <https://www.gurobi.com/>`__ is a commercial mathematical optimization solver and faster than Pyomo. Not only
  in solving the corresponding optimization problem but also in building it. Using Gurobi requires a valid license. Note
  that Gurobi offers `free academic licenses <https://www.gurobi.com/academia/academic-program-and-licenses/>`__.
* `Pyomo <http://www.pyomo.org/>`__ is an open-source optimization modeling language with APIs to several commercial and
  noncommercial solvers. Note that noncommercial solvers are generally (much) slower than commercial ones.

If you decide to work with Pyomo, you have to install additional solvers that can solve mixed-integer optimization
problems. The most common choices are the commercial solvers `Gurobi <https://www.gurobi.com/>`__,
`CPLEX <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__ and
`Xpress <https://www.fico.com/en/products/fico-xpress-optimization>`__ as well as the noncommercial solvers
`GLPK <https://www.gnu.org/software/glpk/>`__,
`CBC <https://github.com/coin-or/Cbc/>`__  and `SCIP <https://www.scipopt.org/>`__.