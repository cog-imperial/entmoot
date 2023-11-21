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

Gurobi or Pyomo?
-----------------
You can work with ENTMOOT using the Gurobi interface or the modeling framework Pyomo.

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

On the installation of noncommercial solvers
---------------------------------------------
Data scientists are used to install everything they need using

::

   pip install everything_I_need

Unfortunately, installing a free (i.e. noncommercial) solver is usually not that easy, so we provide some help in this
section on how to install the solvers `GLPK <https://www.gnu.org/software/glpk/>`__,
`CBC <https://github.com/coin-or/Cbc/>`__  and `SCIP <https://www.scipopt.org
/>`__. We tested some types of installations (not all, though) for Windows and macOS and hope that this may help you.

**Which of the solvers CBC, GLPK and SCIP is best?**

SCIP is considered to be superior to GLPK and CBC in terms of speed and modeling features. However, CBC and GLPK are
also widely used and there might be other reasons for you to use them, so we will guide you through the installation
of all three solvers.

**Are you a Windows user who likes to download files?**

If that is an option for you, you can follow these steps to install CBC and GLPK:

1. Install the latest Microsoft Visual C++ Redistributable. You can get them
   `here <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>`_.
2. Download the binaries. `Here <https://github.com/coin-or/Cbc/releases>`__ are the latest files for CBC and
   `here <https://sourceforge.net/projects/winglpk/>`__ the files for GLPK, respectively.
3. Create one folder for each solver and extract the ZIP files there. These folders might be ``C:\CBC`` and ``C:\GLPK``.
4. Copy the paths containing the executables. In our example, these paths would be ``C:\CBC\bin`` and ``C:\GLPK\glpk-4.65\w64``.
5. Insert these paths into the environment variable named ``PATH``. You can access this variable under System and
   Security>>System>>Advanced system settings>>Environment Variables. Then click on `path` in the top window, click the
   `Edit` button, then `New`.

**Are you a Windows user who likes to use conda?**

If you like to install your Python packages with conda, just use

::

   conda install -c conda-forge glpk

to install GLPK and

::

    conda install -c conda-forge scip

to install SCIP, respectively.

Two remarks:

1. It should also be possible to install CBC using conda but that did not work on my PC. Might be my fault, though.
2. If you use Windows and Anaconda you might encounter an openSSL error, which occurs due to a bug from Anaconda.
   `Here <https://community.anaconda.cloud/t/getting-openssl-working/51512/>`__ is some advice on how to fix it:

**Are you a macOS user?**

You may install *GLPK* with pip or conda using the commands

::

    pip install glpk

or

::

   conda install -c conda-forge glpk

The solvers *CBC* and *SCIP* can be installed via conda using

::

    conda install -c conda-forge coincbc

or

::

    conda install -c conda-forge scip