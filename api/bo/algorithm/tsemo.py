import numpy as np
import opti
import pandas as pd
import pyrff
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_termination
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

from bo.algorithm import Algorithm
from bo.metric import hypervolume_improvement
from bo.torch_tools import fit_gp, get_gp_parameters, predict


class TSEMO(Algorithm):
    """Thompson-Sampling for Efficient Multiobjective Optimization (TSEMO).

    In TSEMO the Bayesian optimization loop consists of the following steps:
    1. construct a GP model for each objective
    2. Thompson-sample an approximate function from the GP posterior of each objective
       via spectral sampling
    3. generate a set of Pareto-efficient points for the sampled functions using a
       evolutionary multi-objective optimizer
    4. select the point with the highest hypervolume improvement

    TSEMO easily allows for parallel evaluations by repeating steps 2 to 4.
    Due to the spectral sampling procedure, only GPs with with Matern-kernels and no
    mean function are allowed.

    Args:
        problem: Optimization problem
        rff_samplse: Number of spectral samples.
        rff_retries: Maximum number of retries if the spectral sampling fails.
        generations: Number of generations used by NSGA2.
        pop_size: Population size used by NSGA2.
        data: Initial Data to use instead of the problem.data

    References:
    - Bradford 2008, Efficient multiobjective optimization employing Gaussian processes, spectral sampling and a genetic algorithm
    """

    def __init__(
        self,
        problem: opti.Problem,
        rff_samples: int = 500,
        rff_retries: int = 10,
        generations: int = 100,
        pop_size: int = 100,
    ):
        super().__init__(problem)

        self.rff_samples = rff_samples
        self.rff_retries = rff_retries
        self.generations = generations
        self.pop_size = pop_size

        # Require multiple objectives (hypervolume calculation)
        if len(problem.objectives) == 1:
            raise ValueError("TSEMO requires multiple objectives.")

        # Require no output constraint (this could be implemented though)
        if problem.output_constraints:
            raise ValueError("TSEMO cannot handle output constraints.")

        # Require all continuous inputs (GP modeling)
        for p in problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise ValueError("TSEMO can only optimize over continuous inputs.")

        # Require no equality constraints (pymoo cannot handle them)
        if problem.constraints is not None:
            for c in problem.constraints:
                if c.is_equality:
                    raise ValueError("TSEMO cannot handle equality constraints.")

        # Require initial data (GP model and pyrff)
        if self.data is None:
            raise ValueError("TSEMO requires initial data.")

        # Drop experiments with missing objective (hypervolume can only use full observations)
        Z = problem.objectives.eval(self.data)
        notna = Z.notna().all(axis=1)
        self._problem.data = self.data[notna]
        if len(self.data) < 2:
            raise ValueError("TSEMO requires at least one full observation.")

        self._fit_model()

    def _fit_model(self):
        """Fit a GP model to the data."""

        X = self.data[self.inputs.names].values
        Y = self.data[self.outputs.names].values
        # pyrff interpretes the hyperparameters, so we turn off standardization
        # pyrff cannot handle mean functions, so we turn off the standard constant mean
        self.model = fit_gp(X, Y, standardize=False, mean_function=False)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the model posterior mean and std."""
        X = data[self.inputs.names].values
        Ymean, Ystd = predict(self.model, X)
        return pd.DataFrame(
            np.concatenate([Ymean, Ystd], axis=1),
            columns=[f"mean_{n}" for n in self.outputs.names]
            + [f"std_{n}" for n in self.outputs.names],
            index=data.index,
        )

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        """Propose a number of experiments."""
        X = np.row_stack([self._optimize() for _ in range(n_proposals)])
        return pd.DataFrame(
            X,
            columns=self.inputs.names,
            index=np.arange(len(X)) + len(self.data),
        )

    def _sample(self):
        """Sample an approximate function from the GP posterior for each output."""
        X = self.data[self.inputs.names].values
        Y = self.data[self.outputs.names].values
        n_outputs = Y.shape[1]
        rffs = [None] * n_outputs
        params = get_gp_parameters(self.model)
        for m in range(n_outputs):
            for _ in range(self.rff_retries):
                try:
                    rffs[m] = pyrff.sample_rff(
                        lengthscales=params.loc[
                            m, [f"ls_{d}" for d in range(len(self.inputs))]
                        ],
                        scaling=params.loc[m, "scale"],
                        noise=params.loc[m, "noise"],
                        kernel_nu=params.loc[m, "nu"],
                        X=X,
                        Y=Y[:, m],
                        M=self.rff_samples,
                    )
                    break
                except np.linalg.LinAlgError:
                    pass

            if rffs[m] is None:
                raise RuntimeError(f"Spectral sampling failed for output {m}.")

        return rffs

    def _optimize(self):
        """Find the point that best improves the hypervolume for a posterior sample."""
        rffs = self._sample()

        problem = self._problem

        class PymooProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=len(problem.inputs),
                    n_obj=len(problem.outputs),
                    n_constr=0
                    if problem.constraints is None
                    else len(problem.constraints),
                    xl=problem.inputs.bounds.loc["min"].values,
                    xu=problem.inputs.bounds.loc["max"].values,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                X = pd.DataFrame(x, columns=problem.inputs.names)
                Y = pd.DataFrame(
                    np.column_stack([rff(x) for rff in rffs]),
                    columns=problem.outputs.names,
                )
                out["F"] = problem.objectives.eval(Y).values
                if problem.constraints:
                    out["G"] = problem.constraints.satisfied(X)
                return out

        # use NSGA2 to generate a number of Pareto optimal points.
        results = minimize(
            problem=PymooProblem(),
            algorithm=NSGA2(
                pop_size=self.pop_size,
                n_offsprings=10,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                mutation=get_mutation("real_pm", eta=20),
                eliminate_duplicates=True,
            ),
            termination=get_termination("n_gen", self.generations),
        )

        # select the point with the highest hypervolume improvement
        Z = problem.objectives.eval(self.data).values
        Yp = pd.DataFrame(results.F, columns=problem.outputs.names)
        Zp = problem.objectives.eval(Yp).values
        nadir = np.max(np.row_stack([Z, Zp]), axis=0) + 0.05
        hvi = hypervolume_improvement(Zp, Z, nadir)
        return results.X[np.argmax(hvi)]

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "TSEMO",
            "problem": self._problem.to_config(),
            "parameters": {
                "rff_samples": self.rff_samples,
                "rff_retries": self.rff_retries,
                "generations": self.generations,
                "pop_size": self.pop_size,
            },
        }


if __name__ == "__main__":
    problem = opti.problems.Detergent()
    problem.create_initial_data(10)
    optimizer = TSEMO(problem)
    optimizer.run(n_steps=10)
