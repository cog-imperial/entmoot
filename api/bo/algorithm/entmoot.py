import os
from typing import Optional

import numpy as np
import opti
import pandas as pd
from entmoot.optimizer.gurobi_utils import get_core_gurobi_model
from entmoot.optimizer.optimizer import Optimizer as EntmootOptimizer
from entmoot.space.space import Categorical, Integer, Real, Space
from gurobipy import GRB, Env

from bo.algorithm import Algorithm


class Entmoot(Algorithm):
    """
    ENTMOOT (ENsemble Tree MOdel Optimization Tool) is a framework to handle tree-based models in Bayesian optimization
    applications.

    References:
        - Thebelt et al., 2020, ENTMOOT: A Framework for Optimization over Ensemble Tree Models
    """

    def __init__(
        self,
        problem,
        random_state=None,
        acq_func_kwargs=None,
        acq_optimizer_kwargs=None,
        base_estimator_kwargs=None,
        std_estimator_kwargs=None,
        model_queue_size=None,
        verbose=False,
    ):
        super().__init__(problem)

        if self.data is None or len(self.data) < 3:
            raise ValueError("Entmoot requires initial data.")

        if len(problem.objectives) > 1:
            raise ValueError("Entmoot doesn't support multiple objectives.")

        for obj in problem.objectives:
            if not isinstance(obj, (opti.objective.Minimize, opti.objective.Maximize)):
                raise ValueError("Entmoot only supports minimization / maximization.")

        dimensions = []
        for parameter in problem.inputs:
            if isinstance(parameter, opti.Continuous):
                dimensions.append(Real(*parameter.bounds, name=parameter.name))
            elif isinstance(parameter, opti.Categorical):
                dimensions.append(Categorical(parameter.domain, name=parameter.name))
            elif isinstance(parameter, opti.Discrete):
                # skopt only supports integer variables [1, 2, 3, 4], not discrete ones [1, 2, 4]
                # We handle this by rounding the proposals
                dimensions.append(Integer(*parameter.bounds, name=parameter.name))
        entmoot_space = Space(dimensions)

        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = {}

        # Set up Gurobi environment that contains the login credentials
        gurobi_env = Env.CloudEnv(
            logfilename="gurobi.log",
            accessID=os.environ["GRB_CLOUDACCESSID"],
            secretKey=os.environ["GRB_CLOUDKEY"],
            pool=os.environ["GRB_CLOUDPOOL"],
        )
        acq_optimizer_kwargs["env"] = gurobi_env

        # Translate constraints from opti to gurobi
        if self._problem.constraints:
            core_model = get_core_gurobi_model(entmoot_space, env=gurobi_env)
            for c in self._problem.constraints:
                if isinstance(c, opti.constraint.LinearInequality):
                    coef = {x: a for (x, a) in zip(c.names, c.lhs)}
                    core_model.addConstr(
                        sum(
                            coef[v.varname] * v
                            for v in core_model.getVars()
                            if v.varname in coef
                        )
                        <= c.rhs
                    )
                elif isinstance(c, opti.constraint.LinearEquality):
                    coef = {x: a for (x, a) in zip(c.names, c.lhs)}
                    core_model.addConstr(
                        sum(
                            coef[v.varname] * v
                            for v in core_model.getVars()
                            if v.varname in coef
                        )
                        == c.rhs
                    )
                elif isinstance(c, opti.constraint.NChooseK):
                    # Big-M implementation of n-choose-k constraint
                    y = core_model.addVars(c.names, vtype=GRB.BINARY)
                    core_model.addConstrs(
                        (
                            y[v.varname] * v.lb <= v
                            for v in core_model.getVars()
                            if v.varname in c.names
                        ),
                        name="n-choose-k-constraint LB",
                    )
                    core_model.addConstrs(
                        (
                            y[v.varname] * v.ub >= v
                            for v in core_model.getVars()
                            if v.varname in c.names
                        ),
                        name="n-choose-k-constraint UB",
                    )
                    core_model.addConstr(
                        y.sum() == c.max_active, name="max active components"
                    )
                else:
                    raise ValueError(f"Constraint of type {type(c)} not supported.")

            acq_optimizer_kwargs["add_model_core"] = core_model

        self.entmoot_optimizer = EntmootOptimizer(
            dimensions,
            base_estimator="GBRT",
            std_estimator="BDD",
            n_initial_points=len(self.data),
            acq_func="LCB",
            acq_optimizer="global",
            random_state=random_state,
            acq_func_kwargs=acq_func_kwargs,
            acq_optimizer_kwargs=acq_optimizer_kwargs,
            base_estimator_kwargs=base_estimator_kwargs,
            std_estimator_kwargs=std_estimator_kwargs,
            model_queue_size=model_queue_size,
            verbose=verbose,
        )

        # tell Entmoot about the intial data
        self.y_mean = self.data[problem.outputs.names[0]].mean()
        self.y_std = self.data[problem.outputs.names[0]].std()
        self._tell_entmoot(self.data)

    def _fit_model(self, _: Optional[pd.DataFrame] = None) -> None:
        """Fit a probabilistic model to the available data."""
        pass

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        μ, σ = np.array(
            [
                self.entmoot_optimizer.predict_with_est(x)
                for x in X[self.inputs.names].values
            ]
        ).T

        # inverse-transform to output values
        name = self.outputs.names[0]
        obj = self._problem.objectives[0]
        μ = obj(μ)

        # unstandardize
        μ = μ * self.y_std + self.y_mean
        σ = σ * self.y_std

        return pd.DataFrame(index=X.index, data={f"mean_{name}": μ, f"std_{name}": σ})

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        X = self.entmoot_optimizer.ask(n_points=n_proposals)
        df = pd.DataFrame(
            X,
            columns=self.inputs.names,
            index=np.arange(len(X)) + len(self._problem.data),
        )
        return self.inputs.round(df)

    def _tell_entmoot(self, data):
        """Add data to the internal Entmoot"""
        x = data[self.inputs.names].values
        # standardize the outputs
        y = data[self.outputs.names[0]]
        y = (y - self.y_mean) / self.y_std
        # transform output to objective values for Entmoot
        z = self._problem.objectives[0](y).values
        self.entmoot_optimizer.tell(x.tolist(), z.tolist())

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "ENTMOOT",
            "problem": self._problem.to_config(),
            "parameters": {},
        }
