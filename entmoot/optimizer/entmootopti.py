from typing import Callable, Optional

import gurobipy
import lightgbm as lgb
import opti
import pandas as pd
from mbo.algorithm import Algorithm

from entmoot.optimizer import Optimizer
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model

from entmoot.space.space import Categorical, Integer, Real, Space


class EntmootOpti(Algorithm):
    """"
    This class serves as connector between the package mopti (https://github.com/basf/mopti) and entmoot.
    Mopti is a Python package for specifying problems in a number of closely related fields, including experimental
    design, multiobjective optimization, decision making and Bayesian optimization.
    EntmootOpti inherits from mbo.algorithm (https://github.com/basf/mbo) and migrates problems specified in mopti to
    entmoot.

    :param problem: opti.Problem
        contains all information about the mopti problem
    : param base_est_params: dict
        base estimator parameters which are handed over to entmoot's Optimizer object
    : param gurobi_env: Optional[Callable]
        calls a function that returns a Gurobi CloudEnv object, if None: use local license instead
    """

    def __init__(self, problem: opti.Problem, base_est_params: dict = None, gurobi_env: Optional[Callable] = None):

        self.problem: opti.Problem = problem
        if base_est_params is None:
            self._base_est_params: dict = {}
        else:
            self._base_est_params: dict = base_est_params
        self.model: lgb.Booster = None

        self.num_obj = len(self.problem.outputs.names)

        # Gurobi environment handling in case you are using the Gurobi Cloud service
        self.gurobi_env = gurobi_env

        self.cat_names: list[str] = None
        self.cat_idx: list[int] = None


        if self.problem.data is None:
            raise ValueError("No initial data points provided.")

        dimensions: list = self._build_dimensions_list()

        self.space = Space(dimensions)

        self.entmoot_optimizer: Optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator="ENTING",
            n_initial_points=0,
            num_obj=self.num_obj,
            random_state=73,
            base_estimator_kwargs=self._base_est_params
        )

        self._fit_model()

    def _build_dimensions_list(self) -> list:
        """
        Builds a list with information (variable bounds and variable type) about input variables (decision variables)
        from mopti. This is then later used by the Optimizer object.
        """
        dimensions = []
        for parameter in self.problem.inputs:
            if isinstance(parameter, opti.Continuous):
                dimensions.append(Real(*parameter.bounds, name=parameter.name))
            elif isinstance(parameter, opti.Categorical):
                dimensions.append(Categorical(parameter.domain, name=parameter.name))
            elif isinstance(parameter, opti.Discrete):
                # skopt only supports integer variables [1, 2, 3, 4], not discrete ones [1, 2, 4]
                # We handle this by rounding the proposals
                dimensions.append(Integer(*parameter.bounds, name=parameter.name))

        return dimensions

    def _fit_model(self) -> None:
        """Fit a probabilistic model to the available data."""
        X = self.problem.data[self.problem.inputs.names]
        if self.num_obj == 1:
            y = self.problem.data[self.problem.outputs.names[0]]
        else:
            y = self.problem.data[self.problem.outputs.names]

        self.entmoot_optimizer.tell(x=X.to_numpy().tolist(), y=y.to_numpy().tolist())

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Yields prediction y from surrogate model(s) for provided X.
        """
        return self.entmoot_optimizer.predict_with_est(X.to_numpy().tolist())

    def _migrate_constraints(self, gurobi_model):
        # Migrate constraints from opti to gurobi
        if self.problem.constraints:
            for c in self.problem.constraints:
                if isinstance(c, opti.constraint.LinearInequality):
                    coef = {x: a for (x, a) in zip(c.names, c.lhs)}
                    gurobi_model.addConstr(
                        (
                            sum(
                                coef[v.varname] * v
                                for v in gurobi_model.getVars()
                                if v.varname in coef
                            )
                            <= c.rhs
                        ),
                        name="LinearInequalityOpti"
                    )
                elif isinstance(c, opti.constraint.LinearEquality):
                    coef = {x: a for (x, a) in zip(c.names, c.lhs)}
                    gurobi_model.addConstr(
                        (
                            sum(
                                coef[v.varname] * v
                                for v in gurobi_model.getVars()
                                if v.varname in coef
                            )
                            == c.rhs
                        ),
                        name="LinearEqualityOpti"
                    )
                elif isinstance(c, opti.constraint.NChooseK):
                    # Big-M implementation of n-choose-k constraint
                    y = gurobi_model.addVars(c.names, vtype=gurobipy.GRB.BINARY)
                    gurobi_model.addConstrs(
                        (
                            y[v.varname] * v.lb <= v
                            for v in gurobi_model.getVars()
                            if v.varname in c.names
                        ),
                        name="n-choose-k-constraint LB",
                    )
                    gurobi_model.addConstrs(
                        (
                            y[v.varname] * v.ub >= v
                            for v in gurobi_model.getVars()
                            if v.varname in c.names
                        ),
                        name="n-choose-k-constraint UB",
                    )
                    gurobi_model.addConstr(
                        y.sum() == c.max_active, name="max active components"
                    )
                else:
                    raise ValueError(f"Constraint of type {type(c)} not supported.")

    def propose(self, n_proposals: int = 1, gurobi_env=None) -> pd.DataFrame:
        """
        Suggests next proposal by optimizing the acquisition function.
        """

        # update gurobi environment if new object is given
        if gurobi_env:
            self.gurobi_env = gurobi_env

        gurobi_model = get_core_gurobi_model(self.space, env=self.gurobi_env)

        # migrate opti constraints into gurobi model
        self._migrate_constraints(gurobi_model)

        X_res = self.entmoot_optimizer.ask(n_points=n_proposals, add_model_core=gurobi_model)

        return pd.DataFrame(X_res, columns=self.problem.inputs.names)

    def predict_pareto_front(
            self, sampling_strategy="random", num_samples=10, num_levels=10,
            gurobi_env=None
    ) -> pd.DataFrame:

        # update gurobi environment if new object is given
        if gurobi_env:
            self.gurobi_env = gurobi_env

        gurobi_model = get_core_gurobi_model(self.space, env=self.gurobi_env)

        # migrate opti constraints into gurobi model
        self._migrate_constraints(gurobi_model)

        pf_res = self.entmoot_optimizer.predict_pareto(
            sampling_strategy=sampling_strategy,
            num_samples=num_samples,
            num_levels=num_levels,
            add_model_core=gurobi_model
        )

        pf_list = [list(x)+y for x, y in pf_res]

        pf_df = pd.DataFrame(pf_list, columns=self.problem.inputs.names + self.problem.outputs.names)

        return pf_df