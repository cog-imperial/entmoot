import numpy as np
import opti
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling.samplers import SobolQMCNormalSampler

from bo.algorithm import Algorithm
from bo.torch_tools import (
    fit_gp,
    get_gp_parameters,
    make_constraints,
    make_objective,
    predict,
    torch_kwargs,
)


class SOBO(Algorithm):
    """Single-Objective Bayesian Optimization Algorithm

    Args:
        problem: Optimization problem
        mc_samples: Number of MC samples for estimating the acquisition function
        restarts: Number of optimization restarts. Defaults to 5
        data: Initial Data to use instead of the problem.data
    """

    def __init__(
        self,
        problem: opti.Problem,
        acquisition="EI",
        mc_samples: int = 128,
        restarts: int = 10,
    ):
        super().__init__(problem)
        self.acquisition = acquisition
        self.restarts = restarts
        self.mc_samples = mc_samples
        self.sampler = SobolQMCNormalSampler(num_samples=mc_samples)

        # Require multiple objectives (scalarization needs multiple objectives)
        if len(problem.objectives) > 1:
            raise ValueError("SOBO requires a single objectives.")

        # Require all continuous inputs (would require special GP kernel otherwise)
        for p in problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise ValueError("SOBO can only optimize over continuous inputs")

        # Require initial data
        if self.data is None:
            raise ValueError("SOBO requires initial data.")

        # Require no nonlinear constraint (BoTorch cannot handle them).
        # We'll be working with normalized inputs, thus, transform bounds and constraints as well.
        try:
            constraints = make_constraints(problem, normalize=True)
            self.bounds = constraints["bounds"]
            self.equalities = constraints["equalities"]
            self.inequalities = constraints["inequalities"]
        except ValueError:
            raise ValueError("ParEGO can only handle linear constraints.")

        self._fit_model()

    def _fit_model(self) -> None:
        """Fit a GP model to the data."""

        X = self.data[self.inputs.names].values
        Y = self.data[self.outputs.names].values
        Xn = self._transform_inputs(X)
        self.model = fit_gp(Xn, Y)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the model posterior mean and std."""
        X = data[self.inputs.names].values
        X = self._transform_inputs(X)
        Ymean, Ystd = predict(self.model, X)
        return pd.DataFrame(
            np.concatenate([Ymean, Ystd], axis=1),
            columns=[f"mean_{n}" for n in self.outputs.names]
            + [f"std_{n}" for n in self.outputs.names],
            index=data.index,
        )

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        objective = GenericMCObjective(
            make_objective(self._problem.objectives[0], self.outputs, maximize=True)
        )
        X, Y = self.get_XY()
        Yt = torch.tensor(Y.values, **torch_kwargs)

        # create a list of acquisition functions
        acq_func_list = []
        for i in range(n_proposals):
            acq_func = qExpectedImprovement(
                model=self.model,
                objective=objective,
                best_f=objective(Yt).max(),
                sampler=self.sampler,
            )
            acq_func_list.append(acq_func)

        # sequentially optimize each acquistion, setting previous candidates as pending
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.bounds,
            equality_constraints=self.equalities,
            inequality_constraints=self.inequalities,
            num_restarts=self.restarts,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
        )
        X = self._untransform_inputs(candidates.detach().numpy())
        return pd.DataFrame(
            X,
            columns=self.inputs.names,
            index=np.arange(len(X)) + len(self.data),
        )

    def get_model_parameters(self) -> pd.DataFrame:
        """Get a dataframe of the model parameters."""
        params = get_gp_parameters(self.model)
        return params.rename(
            index={i: name for i, name in enumerate(self.outputs.names)},
            columns={
                f"ls_{i}": f"ls_{name}" for i, name in enumerate(self.inputs.names)
            },
        )

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "SOBO",
            "problem": self._problem.to_config(),
            "parameters": {
                "acquisition": self.acquisition,
                "restarts": self.restarts,
                "mc_samples": self.mc_samples,
            },
        }


if __name__ == "__main__":
    problem = opti.problems.Sphere(n_inputs=3)
    problem.create_initial_data(n_samples=5)
    optimizer = SOBO(problem)
    optimizer.run(n_steps=20)
    print(optimizer.data)
