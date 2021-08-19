from entmoot.learning import EntingRegressor
from tqdm import tqdm
from typing import Optional, Tuple

import numpy as np
import opti
import pandas as pd



class Algorithm:
    """Base class for Bayesian optimization algorithms"""

    def __init__(self, problem: opti.Problem):
        self._problem = problem
        self.model = None

    @property
    def inputs(self):
        return self._problem.inputs

    @property
    def outputs(self):
        return self._problem.outputs

    @property
    def data(self):
        return self._problem.data

    def get_XY(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get the input and output data."""
        X = self.data[self.inputs.names]
        Y = self.data[self.outputs.names]
        return X, Y

    def _fit_model(self) -> None:
        """Fit a probabilistic model to the available data."""
        pass

    def copy(self, data: Optional[pd.DataFrame] = None) -> "Algorithm":
        """Creates a copy of the optimizer where the data is possibly replaced."""
        new_opt = self.from_config(self.to_config())
        if data is not None:
            new_opt._problem.set_data(data)
            new_opt._fit_model()
        return new_opt

    def add_data_and_fit(self, data: pd.DataFrame) -> None:
        """Add new data points and refit the model."""
        self._problem.add_data(data)
        self._fit_model()

    def sample_initial_data(self, n_samples: int):
        """Create an initial data set for problems with known function y=f(x)."""
        self._problem.create_initial_data(n_samples)
        self._fit_model()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the posterior mean and standard deviation."""
        raise NotImplementedError

    def predict_pareto_front(self, n_levels: int = 5) -> pd.DataFrame:
        """Calculate a finite representation the Pareto front of the model posterior."""
        raise NotImplementedError

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        """Propose a set of experiments according to the algorithm."""
        raise NotImplementedError

    def run(
        self, n_proposals: int = 1, n_steps: int = 10, show_progress: bool = True
    ) -> None:
        """Run the BO algorithm to optimize the problem."""
        if self._problem.f is None:
            raise ValueError(
                "The problem has no function defined. For external function evaluations use the propose() method instead"
            )

        for _ in tqdm(range(n_steps), disable=not show_progress):
            X = self.propose(n_proposals)
            Y = self._problem.eval(X)
            self.add_data_and_fit(pd.concat([X, Y], axis=1))

    def get_model_parameters(self) -> pd.DataFrame:
        """Get the parameters of the surrogate model."""
        raise NotImplementedError

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a config dict."""
        raise NotImplementedError

    def _transform_inputs(self, X: np.ndarray) -> np.ndarray:
        """Transform the inputs from the domain bounds to the unit range"""
        xlo, xhi = self.inputs.bounds.values
        return (X - xlo) / (xhi - xlo)

    def _untransform_inputs(self, X: np.ndarray) -> np.ndarray:
        """Untransform the inputs from the unit range to the domain bounds"""
        xlo, xhi = self.inputs.bounds.values
        return X * (xhi - xlo) + xlo

    @classmethod
    def from_config(cls, config: dict):
        """Create an algorithm instance from a configuration dict."""
        problem = opti.Problem.from_config(config["problem"])
        parameters = config.get("parameters", {})
        return cls(problem, **parameters)


class Entmoot(Algorithm):

    """Class for Entmoot objects"""
    def _fit_model(self) -> None:
        """Fit a probabilistic model to the available data."""
        # TODO: Identify categorical variables in problem object
        cat_idx = []
        self.model = EntingRegressor(cat_idx=cat_idx)
        # TODO: Extract labeled training data
        X = []
        y = []
        self.model.fit(X=X, y=y)
        raise NotImplementedError
