from entmoot.optimizer.gurobi_utils import get_core_gurobi_model, add_gbm_to_gurobi_model, get_gbm_obj
from entmoot.utils import cook_std_estimator
from entmoot.learning.lgbm_processing import order_tree_model_dict
from entmoot.space.space import Space, Real, Categorical, Integer
from entmoot.learning.gbm_model import GbmModel

import enum
import gurobipy
from tqdm import tqdm
from typing import Callable, Optional, Tuple
import numpy as np
import opti
import pandas as pd
import lightgbm as lgb


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


class UncertaintyType(enum.Enum):
    # Exploration
    # for bounded - data distance, which uses squared euclidean distance
    BDD = "BDD"
    # for bounded - data distance, which uses manhattan distance
    L1BDD = "L1BDD"

    # Exploitation
    # for data distance, which uses squared euclidean distance
    DDP = "DDP"
    # for data distance, which uses manhattan distance
    L1DDP = "L1DDP"


class EntmootOpti(Algorithm):
    """Class for Entmoot objects in opti interface"""

    def __init__(self, problem: opti.Problem, surrogat_params: dict = None, gurobi_env: Optional[Callable] = None):
        self._problem: opti.Problem = problem
        self._surrogat_params: dict = surrogat_params
        self.model: lgb.Booster = None
        self._space: Space = self._build_space_object()

        # Handling of categorical features
        self.cat_names: list[str] = None
        self.cat_idx: list[int] = None
        self.cat_encode_mapping: dict = {}
        self.cat_decode_mapping: dict = {}
        self.gurobi_env = gurobi_env

        self._fit_model()

    def _build_space_object(self):
        dimensions = []
        for parameter in self._problem.inputs:
            if isinstance(parameter, opti.Continuous):
                dimensions.append(Real(*parameter.bounds, name=parameter.name))
            elif isinstance(parameter, opti.Categorical):
                dimensions.append(Categorical(parameter.domain, name=parameter.name))
            elif isinstance(parameter, opti.Discrete):
                # skopt only supports integer variables [1, 2, 3, 4], not discrete ones [1, 2, 4]
                # We handle this by rounding the proposals
                dimensions.append(Integer(*parameter.bounds, name=parameter.name))

        return Space(dimensions)

    def _encode_cat_vars(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = X.copy()
        X_enc[self.cat_names] = X_enc[self.cat_names].astype("category")
        for cat in self.cat_names:
            X_enc[cat] = X_enc[cat].cat.codes
        return X_enc

    def _fit_model(self) -> None:
        """Fit a probabilistic model to the available data."""

        X = self.data[self.inputs.names]
        y = self.data[self.outputs.names]

        # Extract names of categorical columns and mark them as categorical variables in Pandas.

        self.cat_names = [i.name for i in self.inputs.parameters.values() if type(i) is opti.Categorical]
        self.cat_idx = [i for i, j in enumerate(self.inputs.parameters.values()) if type(j) is opti.Categorical]

        X_enc = self._encode_cat_vars(X)

        for cat in self.cat_names:
            self.cat_encode_mapping[cat] = {var: enc for (var, enc) in set(zip(X[cat], X_enc[cat]))}
            self.cat_decode_mapping[cat] = {enc: var for (enc, var) in set(zip(X_enc[cat], X[cat]))}

        train_data = lgb.Dataset(X_enc, label=y, params=self._surrogat_params)

        self.model = lgb.train(self._surrogat_params, train_data, categorical_feature=self.cat_idx)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        X_enc = self._encode_cat_vars(X)

        return self.model.predict(X_enc)

    def _get_gbm_model(self):
        original_tree_model_dict = self.model.dump_model()
        import json
        with open('tree_dict_next.json', 'w') as f:
            json.dump(
                original_tree_model_dict,
                f,
                indent=4,
                sort_keys=False
            )

        ordered_tree_model_dict = \
            order_tree_model_dict(
                original_tree_model_dict,
                cat_column=self.cat_idx
            )

        gbm_model = GbmModel(ordered_tree_model_dict)
        return gbm_model

    def propose(self, n_proposals: int = 1, uncertainty_type: UncertaintyType = UncertaintyType.DDP) -> pd.DataFrame:

        X_res = pd.DataFrame(columns=self._problem.inputs.names)
        X_std_data = self._encode_cat_vars(self.data[self.inputs.names]).values
        y_std_data = self.data[self.outputs.names].values

        if self.gurobi_env is not None:
            env = self.gurobi_env()
        else:
            env = None

        for _ in range(n_proposals):

            # build gurobi core model
            gurobi_model: gurobipy.Model = get_core_gurobi_model(self._space, env=env)

            # Shut down logging
            gurobi_model.Params.OutputFlag = 0

            # add lgbm core structure to gurobi model
            gbm_model_dict = {}
            gbm_model_dict['first'] = self._get_gbm_model()
            add_gbm_to_gurobi_model(
                self._space, gbm_model_dict, gurobi_model
            )
            gurobi_model.update()

            # add std to gurobi model
            std_estimator_params = {}
            std_est = cook_std_estimator(uncertainty_type.name, self._space, std_estimator_params=std_estimator_params)
            std_est.cat_idx = self.cat_idx

            std_est.update(X_std_data, y_std_data, cat_column=self.cat_idx)
            std_est.add_to_gurobi_model(gurobi_model)
            gurobi_model.update()

            # add obj to gurobi model
            mu = get_gbm_obj(gurobi_model)
            std = std_est.get_gurobi_obj(gurobi_model, scaled=False)
            kappa = 1.96
            ob_expr = gurobipy.quicksum((mu, kappa * std))
            gurobi_model.setObjective(ob_expr, gurobipy.GRB.MINIMIZE)
            gurobi_model.update()

            # Optimize gurobi model
            gurobi_model.optimize()

            # Get optimization results
            x_next = np.empty(len(self._space.dimensions))
            # cont features
            for i in gurobi_model._cont_var_dict:
                x_next[i] = gurobi_model._cont_var_dict[i].x
            # cat features
            for i in gurobi_model._cat_var_dict:
                cat = [k for k in gurobi_model._cat_var_dict[i] if round(gurobi_model._cat_var_dict[i][k].x, 1) == 1]
                x_next[i] = cat[0]

            # Constant liar strategy where new y-value is chosen as minimal observation
            X_std_data = np.vstack([X_std_data, x_next])
            y_std_data = np.vstack([y_std_data, sum(y_std_data)/len(y_std_data)])

            X_next_df_enc = pd.DataFrame(x_next.reshape(1, -1), columns=self._problem.inputs.names)

            X_next_df = X_next_df_enc.copy()
            for cat in self.cat_decode_mapping:
                X_next_df[cat] = X_next_df[cat].replace(self.cat_decode_mapping[cat])

            X_res = X_res.append(X_next_df)

        return X_res
