from typing import Optional

import numpy as np
import opti
import pandas as pd
import torch
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import normalize
from tqdm import tqdm

from bo.algorithm import Algorithm
from bo.torch_tools import (
    fit_gp,
    fit_gp_list,
    get_gp_parameters,
    make_constraints,
    make_objective,
    predict,
    torch_kwargs,
)


class ParEGO(Algorithm):
    """ParEGO algorithm for multi-objective Bayesian optimization.

    Args:
        problem: Optimization problem
        mc_samples: Number of MC samples for estimating the acquisition function
        restarts: Number of optimization restarts.
        data: Initial Data to use instead of the problem.data

    References:
    - Knowles 2006, ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems
    - Dalton 2020, Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization
    """

    def __init__(
        self,
        problem: opti.Problem,
        mc_samples: int = 1024,
        restarts: int = 3,
    ):
        super().__init__(problem)
        self.restarts = restarts
        self.mc_samples = mc_samples
        self.sampler = SobolQMCNormalSampler(num_samples=mc_samples)

        # Require multiple objectives (scalarization needs multiple objectives)
        if len(problem.objectives) == 1:
            raise ValueError("ParEGO requires multiple objectives.")
        self.objective_funcs = [
            make_objective(obj, problem.outputs) for obj in problem.objectives
        ]

        # Check for output constraints
        if problem.output_constraints is not None:
            self.constraint_funcs = [
                make_objective(obj, problem.outputs)
                for obj in problem.output_constraints
            ]

        # Require all continuous inputs (would require special GP kernel otherwise)
        for p in problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise ValueError("ParEGO can only optimize over continuous inputs.")

        # Require initial data
        if self.data is None:
            raise ValueError("ParEGO requires initial data.")

        # Require at least one full observation of all objectives for the EI scalarization
        Y = self.data[problem.outputs.names].values
        if not np.isfinite(Y).all(axis=1).any():
            raise ValueError("ParEGO requires at least one full observation.")

        # Require no nonlinear constraint (BoTorch cannot handle them).
        # We'll be working with normalized inputs, thus transform bounds and constraints as well
        try:
            constraints = make_constraints(problem, normalize=True)
            self.bounds = constraints["bounds"]
            self.equalities = constraints["equalities"]
            self.inequalities = constraints["inequalities"]
        except ValueError as ve:
            raise ValueError(f"ParEGO can only handle linear constraints. {str(ve)}")

        self._fit_model()

    def _get_parego_objective(self, weights):
        # get the observed objective bounds for normalization
        Z = self._problem.objectives.eval(self.data).values
        Z_bounds = torch.tensor(
            [np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)], **torch_kwargs
        )
        weights = torch.tensor(weights, **torch_kwargs)

        def augmented_chebyshef(Y: torch.tensor) -> torch.tensor:
            Z = torch.stack([f(Y) for f in self.objective_funcs], dim=-1)
            wZ = -weights * normalize(Z, Z_bounds)
            return wZ.min(axis=-1).values + 0.01 * wZ.sum(axis=-1)

        if self._problem.output_constraints is None:
            objective = GenericMCObjective(augmented_chebyshef)
        else:
            objective = ConstrainedMCObjective(
                objective=augmented_chebyshef,
                constraints=self.constraint_funcs,
            )

        # best objective value observed so far
        Y = self.data[self.outputs.names].values
        Y = torch.tensor(Y[np.isfinite(Y).all(axis=1)], **torch_kwargs)
        best = objective(Y).max()
        return objective, best

    def _fit_model(self) -> None:
        """Fit a GP model to the data."""

        X = self.data[self.inputs.names].values
        Y = self.data[self.outputs.names].values
        Xn = self._transform_inputs(X)
        if np.all(np.isfinite(Y)):
            self.model = fit_gp(Xn, Y)
        else:  # missing outputs --> use a list of GPs
            self.model = fit_gp_list(Xn, Y)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the model posterior mean and std."""
        X = data[self._problem.inputs.names].values
        X = self._transform_inputs(X)
        Ymean, Ystd = predict(self.model, X)
        return pd.DataFrame(
            np.concatenate([Ymean, Ystd], axis=1),
            columns=[f"mean_{n}" for n in self._problem.outputs.names]
            + [f"std_{n}" for n in self._problem.outputs.names],
            index=data.index,
        )

    def propose(
        self, n_proposals: int = 1, weights: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        if weights is None:
            weights = opti.sampling.simplex.sample(
                len(self._problem.objectives), n_proposals
            )
        else:
            weights = np.atleast_2d(weights)

        # create a list of acquisition functions with random Chebyshef scalarizations
        acq_func_list = []
        for w in weights:
            objective, best = self._get_parego_objective(w)

            acq_func = qExpectedImprovement(
                model=self.model,
                objective=objective,
                best_f=best,
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
            columns=self._problem.inputs.names,
            index=np.arange(len(X)) + len(self.data),
        )

    def predict_pareto_front(
        self, n_levels: int = 5, beta=0, show_progress: bool = True
    ) -> pd.DataFrame:
        """Predict the Pareto front using the model posterior mean."""
        M = len(self._problem.objectives)
        weights = opti.sampling.simplex.grid(M, n_levels)

        front = pd.DataFrame(weights, columns=[f"weight_obj{i+1}" for i in range(M)])

        for i, w in enumerate(tqdm(weights, disable=not (show_progress))):
            objective, _ = self._get_parego_objective(weights=w)

            acq_func = qUpperConfidenceBound(
                model=self.model,
                beta=beta,  # beta = 0 corresponds to the posterior mean
                objective=objective,
                sampler=self.sampler,
            )

            candidate, _ = optimize_acqf(
                acq_func,
                q=1,
                num_restarts=self.restarts,
                bounds=self.bounds,
                equality_constraints=self.equalities,
                inequality_constraints=self.inequalities,
                raw_samples=1024,
                options={"batch_limit": 5, "maxiter": 200},
            )

            X = self._untransform_inputs(candidate.detach().numpy()).squeeze()
            front.loc[i, self._problem.inputs.names] = X

        return pd.concat([front, self.predict(front)], axis=1)

    def get_model_parameters(self) -> pd.DataFrame:
        """Get a dataframe of the model parameters."""
        params = get_gp_parameters(self.model)
        # set parameter names
        return params.rename(
            index={i: name for i, name in enumerate(self._problem.outputs.names)},
            columns={
                f"ls_{i}": f"ls_{name}"
                for i, name in enumerate(self._problem.inputs.names)
            },
        )

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "ParEGO",
            "problem": self._problem.to_config(),
            "parameters": {
                "restarts": self.restarts,
                "mc_samples": self.mc_samples,
            },
        }


if __name__ == "__main__":
    problem = opti.problems.Detergent_OutputConstraint()
    problem.create_initial_data(n_samples=10)
    optimizer = ParEGO(problem)
    optimizer.run(n_steps=40)
    print(optimizer.data)
