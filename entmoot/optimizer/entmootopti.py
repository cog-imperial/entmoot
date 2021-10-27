from enum import Enum
from typing import Callable, Optional

import gurobipy
import lightgbm as lgb
import numpy as np
import opti
import pandas as pd
from mbo.algorithm import Algorithm

from entmoot.learning.gbm_model import GbmModel
from entmoot.learning.lgbm_processing import order_tree_model_dict
from entmoot.optimizer.gurobi_utils import (
    add_gbm_to_gurobi_model,
    get_core_gurobi_model,
    get_gbm_obj,
)
from entmoot.space.space import Categorical, Integer, Real, Space
from entmoot.utils import cook_std_estimator


class UncertaintyType(Enum):
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
        if surrogat_params is None:
            self._surrogat_params: dict = {}
        else:
            self._surrogat_params: dict = surrogat_params
        self.model: lgb.Booster = None
        self._space: Space = self._build_space_object()

        # Handling of categorical features
        self.cat_names: list[str] = None
        self.cat_idx: list[int] = None
        self.cat_encode_mapping: dict = {}
        self.cat_decode_mapping: dict = {}
        self.gurobi_env = gurobi_env

        if self.data is None:
            raise ValueError("No initial data points provided.")

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
            gurobi_model.Params.TimeLimit = 60

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

            # Migrate constraints from opti to gurobi
            if self._problem.constraints:
                for c in self._problem.constraints:
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
