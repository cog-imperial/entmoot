from entmoot import ProblemConfig
from entmoot.models.base_model import BaseModel
from entmoot.models.mean_models.tree_ensemble import TreeEnsemble
from entmoot.models.uncertainty_models.distance_based_uncertainty import (
    DistanceBasedUncertainty,
)
from entmoot.models.model_params import EntingParams
from dataclasses import asdict
import numpy as np
from typing import Union


class Enting(BaseModel):
    """
    This class provides a living space for your tree model. You can fit your model, use it for predictions and
    provide information that are needed to build optimization models that incorporate the tree structure of your model.

    Example:
        .. code-block:: python

            from entmoot import ProblemConfig
            import numpy as np
            import random

            # Define a one-dimensional minimization problem with one real variable bounded by -2 and 3
            problem_config = ProblemConfig()
            problem_config.add_feature("real", (-2, 3))
            problem_config.add_min_objective()

            # Create training data using the randomly disturbed function f(x) = x^2 + 1 + eps
            X_train = np.linspace(-2, 3, 10)
            y_train = [x**2 + 1 + random.uniform(-0.2, 0.2) for x in X_train]

            # Define enting object and corresponding parameters
            params = {"unc_params": {"dist_metric": "l1"}}
            enting = Enting(problem_config, params=params)
            # Fit tree model
            enting.fit(X_train, y_train)
            # Compute the predictions for training data and see that light gbm fitted a step function
            # with three steps
            enting.predict(X_train)

            # Define parameters needed during optimization
            params_pyo = {"solver_name": "gurobi", "solver_options": {"MIPGap": 0}}
            # Build PyomoOptimizer object. This step will internally call the methods
            # add_to_pyomo_model() or add_to_gurobipy_model(), resp., depending on the choice
            # of your optimizer
            opt_pyo = PyomoOptimizer(problem_config, params=params_pyo)
            # As expected, the optimal input of the tree model is near the origin (cf. X_opt_pyo)
            X_opt_pyo, _, _ = opt_pyo.solve(enting)
    """

    def __init__(self, problem_config: ProblemConfig, params: Union[EntingParams, dict, None]):
        if params is None:
            params = {}
        if isinstance(params, dict):
            params = EntingParams(**params)

        self._problem_config = problem_config

        # initialize mean model
        self.mean_model = TreeEnsemble(
            problem_config=problem_config, params=params.tree_train_params
        )

        # initialize unc model
        unc_params = params.unc_params
        self._acq_sense = unc_params.acq_sense

        self._beta = unc_params.beta

        if self._acq_sense == "exploration":
            self._beta = -self._beta

        self.unc_model = DistanceBasedUncertainty(
            problem_config=problem_config, params=unc_params
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Performs the training of you tree model using training data and labels
        """
        # encode categorical features
        X = self._problem_config.encode(X)

        # check dims of X and y
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._problem_config.feat_list), (
            "Argument 'X' has wrong dimensions. "
            f"Expected '(num_samples, {len(self._problem_config.feat_list)})', got '{X.shape}'."
        )

        if y.ndim == 1:
            y = np.atleast_2d(y)

        assert (y.shape[-1] == 2 and len(self._problem_config.obj_list) == 1) or (
            y.shape[-1] == len(self._problem_config.obj_list)
        ), (
            "Argument 'y' has wrong dimensions. "
            f"Expected '(num_samples, {len(self._problem_config.obj_list)})', got '{y.shape}'."
        )
        y = self._problem_config.transform_objective(y)
        self.mean_model.fit(X, y)
        self.unc_model.fit(X, y)

    def leaf_bnd_predict(self, obj_name, leaf_enc):
        bnds = self._problem_config.get_enc_bnd()
        return self.mean_model.meta_tree_dict[obj_name].prune_var_bnds(leaf_enc, bnds)

    def predict(self, X: np.ndarray, is_enc=False) -> list:
        """
        Computes prediction value of tree model for X.
        """
        # encode categorical features
        if not is_enc:
            X = self._problem_config.encode(X)

        # check dims of X
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._problem_config.feat_list), (
            "Argument 'X' has wrong dimensions. "
            f"Expected '(num_samples, {len(self._problem_config.feat_list)})', got '{X.shape}'."
        )

        mean_pred = self.mean_model.predict(X) #.tolist()
        unc_pred = self.unc_model.predict(X)
        
        mean_pred = self._problem_config.transform_objective(mean_pred)
        mean_pred = mean_pred.tolist()

        comb_pred = [(mean, unc) for mean, unc in zip(mean_pred, unc_pred)]
        return comb_pred

    def predict_acq(self, X: np.ndarray, is_enc=False) -> list:
        """
        Predicts value of acquisition function (which contains not only the mean value but also the uncertainty)
        """
        acq_pred = []
        comb_pred = self.predict(X, is_enc=is_enc)
        for mean, unc in comb_pred:
            acq_pred.append(mean + self._beta * unc)
        return acq_pred

    def add_to_gurobipy_model(self, core_model, weights: tuple = None) -> None:
        """
        Enriches the core model by adding variables and constraints based on information
        from the tree model.
        """
        from gurobipy import GRB
        from entmoot.utils import sample

        # add uncertainty model part
        self.unc_model.add_to_gurobipy_model(core_model)

        core_model._mu = core_model.addVar(
            lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"mean_obj", vtype="C"
        )

        if len(self._problem_config.obj_list) == 1:
            # single objective case
            self.mean_model.add_to_gurobipy_model(core_model, add_mu_var=True)
            core_model.addConstr(core_model._mu == core_model._aux_mu[0])
        else:
            # multi-objective case
            self.mean_model.add_to_gurobipy_model(
                core_model, add_mu_var=True, normalize_mean=True
            )
            if weights is not None:
                moo_weights = weights
            else:
                moo_weights = sample(len(self._problem_config.obj_list), 1)[0]

            for idx, obj in enumerate(self._problem_config.obj_list):
                core_model.addConstr(
                    core_model._mu >= moo_weights[idx] * core_model._aux_mu[idx],
                    name=f"weighted_mean_obj_{idx}",
                )

        core_model.setObjective(core_model._mu + self._beta * core_model._unc)
        core_model.update()

    def add_to_pyomo_model(self, core_model, weights: tuple = None) -> None:
        """
        Enriches the core model by adding variables and constraints based on information
        from the tree model.
        """
        import pyomo.environ as pyo
        from entmoot.utils import sample

        # add uncertainty model part
        self.unc_model.add_to_pyomo_model(core_model)

        core_model._mu = pyo.Var(domain=pyo.Reals)

        if len(self._problem_config.obj_list) == 1:
            # single objective case
            self.mean_model.add_to_pyomo_model(core_model, add_mu_var=True)
            # Get objective name
            obj_name = self._problem_config.obj_list[0].name
            core_model.constraint_link_mu_auxmu = pyo.Constraint(
                expr=core_model._aux_mu[obj_name] == core_model._mu
            )
        else:
            # multi-objective case
            self.mean_model.add_to_pyomo_model(
                core_model, add_mu_var=True, normalize_mean=True
            )
            if weights is not None:
                moo_weights = weights
            else:
                moo_weights = sample(len(self._problem_config.obj_list), 1)[0]

            objectives_position_name = list(
                enumerate([obj.name for obj in self._problem_config.obj_list])
            )

            def constrs_weights_auxmus(model, pos, objname):
                return model._mu >= moo_weights[pos] * core_model._aux_mu[objname]

            core_model.constr_coupling_mu_auxmu = pyo.Constraint(
                objectives_position_name,
                rule=constrs_weights_auxmus,
            )

        # Define objective function
        core_model.obj = pyo.Objective(
            expr=core_model._mu + self._beta * core_model._unc, sense=pyo.minimize
        )
