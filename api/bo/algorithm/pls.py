from typing import Optional

import numpy as np
import opti
import pandas as pd

try:
    import pyreto
except ModuleNotFoundError:
    pass

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from bo.algorithm import Algorithm
from bo.metric import adjusted_r2_score


class PLS(Algorithm):
    """PLS modeling, Pareto-approximation using sandwiching and D-optimal proposals."""

    def __init__(self, problem: opti.Problem, model_params: Optional[dict] = None):
        super().__init__(problem)

        # Check for output constraints
        if problem.output_constraints is not None:
            raise ValueError("PLS cannot handle output constraints")

        # Require all continuous inputs (would require special GP kernel otherwise)
        for p in problem.inputs:
            if not (isinstance(p, opti.Continuous)):
                raise ValueError("PLS can only handle continuous inputs.")

        self._model = Pipeline(
            steps=[("poly", PolynomialFeatures()), ("pls", PLSRegression())]
        )

        # Tune PLS model first if no model_params provided.
        if model_params is not None:
            self.model_params = model_params
        elif problem.data is not None:
            self._tune_model()
        else:
            self.model_params = {}

        self._fit_model()

    def _fit_model(self) -> None:
        if self._problem.data is None:
            return
        X, Y = self._problem.get_XY()
        self._model.set_params(**self.model_params)
        self._model.fit(X, Y)

    def _tune_model(self) -> None:
        """Tune the model parameters with cross-validation."""
        X, Y = self._problem.get_XY()  # TODO: shuffle

        n, d = X.shape
        n, m = Y.shape

        # maximum number of latent variables for 5-fold CV
        nc_max = n * 4 // 5 - 1

        params = []

        # linear model
        for nc in range(2, min(nc_max, d + 1)):
            params.append(
                {
                    "poly__degree": 1,
                    "pls__n_components": nc,
                }
            )

        # linear + interaction model
        n_features = d + d * (d - 1) // 2 + 1
        for nc in range(d, min(nc_max, n_features)):
            params.append(
                {
                    "poly__degree": 2,
                    "poly__interaction_only": True,
                    "pls__n_components": nc,
                }
            )

        # quadratic model
        n_features = d + d * (d + 1) // 2 + 1
        for nc in range(d, min(nc_max, n_features)):
            params.append(
                {
                    "poly__degree": 2,
                    "poly__interaction_only": False,
                    "pls__n_components": nc,
                }
            )

        R2 = np.empty(len(params))
        for i, param in enumerate(params):
            self._model.set_params(**param)
            Yp = cross_val_predict(self._model, X, Y, cv=5)
            nc = param["pls__n_components"]
            R2[i] = adjusted_r2_score(Y, Yp, n_features=nc)

        best = np.argmax(R2)
        self.model_params = params[best]

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        X = data[self._problem.inputs.names].values
        Y = self._model.predict(X)
        return pd.DataFrame(
            Y,
            columns=[f"mean_{n}" for n in self._problem.outputs.names],
            index=data.index,
        )

    def predict_pareto_front(self) -> pd.DataFrame:
        """Approximate the model Pareto front using the Sandwich service."""
        self._problem.f = lambda x: self._model.predict(np.atleast_2d(x))
        front = pyreto.convex_front(self._problem)
        return front

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        """Generate proposals using a D-optimal design."""
        raise NotImplementedError

    def get_model_parameters(self) -> pd.DataFrame:
        params = pd.DataFrame(index=self.outputs.names, data=self.model_params)
        params.index.name = "output"
        return params

    def to_config(self) -> dict:
        """Serialize the algorithm settings to a dictionary."""
        return {
            "method": "PLS",
            "problem": self._problem.to_config(),
            "parameters": {"model_params": self.model_params},
        }
