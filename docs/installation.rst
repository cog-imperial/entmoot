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

* `Gurobi <https://www.gurobi.com/>`__ is a commercial mathematical optimization solver which is very fast. Not only
  in solving the corresponding optimization problem but also in building it. Using Gurobi requires a valid license. Note
  that Gurobi offers `free academic licenses <https://www.gurobi.com/academia/academic-program-and-licenses/>`__.
* `Pyomo <http://www.pyomo.org/>`__ is an open-source optimization modeling language with APIs to several commercial and
  noncommercial solvers. Note that noncommercial solvers are generally (much) slower than commercial ones.

If you decide to work with Pyomo, you have to install additional solvers that can solve mixed-integer optimization
problems. The most common choices are the commercial solvers `Gurobi <https://www.gurobi.com/>`__,
`CPLEX <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__ and
`Xpress <https://www.fico.com/en/products/fico-xpress-optimization>`__ as well as the noncommercial solvers
`GLPK <https://www.gnu.org/software/glpk/>`__,
`CBC <https://github.com/coin-or/Cbc/>`__  and `SCIP <https://www.scipopt.org
/>`__.

Some help for Windows users with CBC and GLPK
-----------------------------------------------------------------

If you are a Windows user and you want to use Pyomo with the free solvers CBC and/or GLPK, here is what you have to do.

1. Install the latest Microsoft Visual C++ Redistributable. You can get them
   `here <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_.
2. Download the binaries. `Here <https://github.com/coin-or/Cbc/releases>`__ are the latest files for CBC and
   `here <https://sourceforge.net/projects/winglpk/>`__ the files for GLPK, respectively.
3. Create one folder for each solver and extract the ZIP files there. These folders might be ``C:\CBC`` and ``C:\GLPK``.
4. Copy the paths containing the executables. In our example, these paths would be ``C:\CBC\bin`` and ``C:\GLPK\glpk-4.65\w64``.
5. Insert these paths into the environment variable named ``PATH``. You can access this variable under System and
   Security>>System>>Advanced system settings>>Environment Variables. Then click on `path` in the top window, click the
   `Edit` button, then `New`.
