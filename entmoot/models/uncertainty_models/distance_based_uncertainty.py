from entmoot.models.base_model import BaseModel
from entmoot.models.uncertainty_models.euclidean_squared_distance import (
    EuclideanSquaredDistance,
)
from entmoot.models.uncertainty_models.l1_distance import L1Distance
from entmoot.models.uncertainty_models.l2_distance import L2Distance

from entmoot.models.uncertainty_models.overlap_distance import OverlapDistance
from entmoot.models.uncertainty_models.goodall4_distance import Goodall4Distance
from entmoot.models.uncertainty_models.of_distance import OfDistance

import numpy as np


class DistanceBasedUncertainty(BaseModel):
    def __init__(self, problem_config, params):
        self._problem_config = problem_config

        dist_metric = params.get("dist_metric", "euclidean_squared")
        dist_trafo = params.get("dist_trafo", "normal")
        cat_metric = params.get("cat_metric", "overlap")
        acq_sense = params.get("acq_sense", "exploration")

        self._non_cat_x, self._cat_x = None, None
        self._dist_bound = None
        self._dist_metric = dist_metric
        self._num_cache_x = None
        self._acq_sense = acq_sense

        if dist_trafo == "standard":
            assert (
                len(self._problem_config.obj_list) == 1
            ), "Distance transformation 'standard' can only be used for single objective problems."

            assert (
                len(self._problem_config.cat_idx) == 0
            ), "Distance transformation 'standard' can only be used for non-categorical problems."

            self._dist_has_var_bound = False if self._acq_sense == "penalty" else True
            self._bound_coeff = params.get("bound_coeff", 0.5)
            self._dist_coeff = 1.0
        elif dist_trafo == "normal":
            self._dist_has_var_bound = False
            self._bound_coeff = None
            self._dist_coeff = 1 / len(self._problem_config.feat_list)
        else:
            raise IOError(
                f"Pick 'dist_trafo' '{dist_trafo}' in '('normal', 'standard')'."
            )

        # pick distance metric for non-cat features
        if dist_metric == "euclidean_squared":
            self.non_cat_unc_model = EuclideanSquaredDistance(
                problem_config=self._problem_config,
                acq_sense=acq_sense,
                dist_trafo=dist_trafo,
            )
        elif dist_metric == "l1":
            self.non_cat_unc_model = L1Distance(
                problem_config=self._problem_config,
                acq_sense=acq_sense,
                dist_trafo=dist_trafo,
            )
        elif dist_metric == "l2":
            self.non_cat_unc_model = L2Distance(
                problem_config=self._problem_config,
                acq_sense=acq_sense,
                dist_trafo=dist_trafo,
            )
        else:
            raise IOError(
                f"Non-categorical uncertainty metric '{dist_metric}' for "
                f"{self.__class__.__name__} model is not supported. "
                f"Check 'params['uncertainty_type']'."
            )

        # pick distance metric for cat features
        if cat_metric == "overlap":
            self.cat_unc_model = OverlapDistance(
                problem_config=self._problem_config, acq_sense=acq_sense
            )
        elif cat_metric == "of":
            self.cat_unc_model = OfDistance(
                problem_config=self._problem_config, acq_sense=acq_sense
            )
        elif cat_metric == "goodall4":
            self.cat_unc_model = Goodall4Distance(
                problem_config=self._problem_config, acq_sense=acq_sense
            )
        else:
            raise IOError(
                f"Categorical uncertainty metric '{cat_metric}' for {self.__class__.__name__} "
                f"model is not supported. Check 'params['uncertainty_type']'."
            )

    @property
    def num_cache_x(self):
        assert (
            self._num_cache_x is not None
        ), f"Uncertainty model needs fit function call before it can predict."
        return self._num_cache_x

    def fit(self, X, y):
        if self._dist_has_var_bound:
            self._dist_bound = abs(np.var(y) * self._bound_coeff)

        self._num_cache_x = len(X)

        self.non_cat_unc_model.fit(X)
        self.cat_unc_model.fit(X)

    def predict(self, X):
        comb_pred = []
        for xi in X:
            non_cat_pred = self.non_cat_unc_model.predict(xi)
            cat_pred = self.cat_unc_model.predict(xi)
            dist_pred = np.min(non_cat_pred + cat_pred) * self._dist_coeff

            # the standard trafo case has a bound on the prediction
            if self._dist_has_var_bound:
                if dist_pred > self._dist_bound:
                    dist_pred = self._dist_bound

            comb_pred.append(dist_pred)
        return np.asarray(comb_pred)

    def add_to_gurobipy_model(self, model):
        from gurobipy import GRB

        # define main uncertainty variables
        if self._dist_has_var_bound:
            dist_bound = self._dist_bound
        else:
            dist_bound = GRB.INFINITY

        model._unc = model.addVar(lb=0.0, ub=dist_bound, name="uncertainty", vtype="C")

        # get constr terms for non-categorical and categorical contributions
        non_cat_term_list = self.non_cat_unc_model.get_gurobipy_model_constr_terms(
            model
        )
        cat_term_list = self.cat_unc_model.get_gurobipy_model_constr_terms(model)

        # add binaries for big_m penalty constraints
        if self._acq_sense == "penalty":
            model._bin_penalty = []
            big_m = self.non_cat_unc_model.get_big_m() + self.cat_unc_model.get_big_m()

        for i, (non_cat_term, cat_term) in enumerate(
            zip(non_cat_term_list, cat_term_list)
        ):
            # check if penalty term is needed
            if self._acq_sense == "penalty":

                model._bin_penalty.append(
                    model.addVar(name=f"bin_penalty_{i}", vtype="B")
                )

                big_m_term = big_m * (1 - model._bin_penalty[-1])

                if self._dist_metric == "l2":
                    # take sqrt for l2 distance
                    aux_non_cat_unc = model.addVar(
                        lb=0.0, ub=dist_bound, name=f"aux_non_cat_unc_x_{i}", vtype="C"
                    )

                    model.addQConstr(
                        aux_non_cat_unc * aux_non_cat_unc >= non_cat_term,
                        name=f"unc_x_{i}",
                    )

                    model.addQConstr(
                        model._unc + big_m_term
                        >= (aux_non_cat_unc + cat_term) * self._dist_coeff,
                        name=f"unc_x_{i}",
                    )
                    model.params.NonConvex = 2
                else:
                    # self._dist_metric == "l1" or "euclidean_squared"
                    model.addQConstr(
                        model._unc + big_m_term
                        >= (non_cat_term + cat_term) * self._dist_coeff,
                        name=f"unc_x_{i}",
                    )
            else:
                #  self._acq_sense =="exploration"
                if self._dist_metric == "l2":
                    # take sqrt for l2 distance
                    aux_non_cat_unc = model.addVar(
                        lb=0.0, ub=dist_bound, name=f"aux_non_cat_unc_x_{i}", vtype="C"
                    )

                    model.addQConstr(
                        aux_non_cat_unc * aux_non_cat_unc <= non_cat_term,
                        name=f"unc_x_{i}",
                    )

                    model.addQConstr(
                        model._unc * model._unc
                        <= (aux_non_cat_unc + cat_term) * self._dist_coeff,
                        name=f"unc_x_{i}",
                    )
                else:
                    model.addQConstr(
                        model._unc <= (non_cat_term + cat_term) * self._dist_coeff,
                        name=f"unc_x_{i}",
                    )

        if self._acq_sense == "exploration":
            model.params.NonConvex = 2

        model.update()

    def add_to_pyomo_model(self, model):
        import pyomo.environ as pyo

        # define main uncertainty variables
        if self._dist_has_var_bound:
            dist_bound = self._dist_bound
        else:
            dist_bound = float("inf")

        model._unc = pyo.Var(bounds=(0, dist_bound), domain=pyo.Reals)

        # get constr terms for non-categorical and categorical contributions
        non_cat_term_list = self.non_cat_unc_model.get_pyomo_model_constr_terms(model)
        cat_term_list = self.cat_unc_model.get_pyomo_model_constr_terms(model)

        model.terms_constrs_cat_noncat_contr = list(
            zip(non_cat_term_list, cat_term_list)
        )
        model.indices_constrs_cat_noncat_contr = list(
            range(len(model.terms_constrs_cat_noncat_contr))
        )

        # add binaries for big_m penalty constraints
        if self._acq_sense == "penalty":
            big_m = self.non_cat_unc_model.get_big_m() + self.cat_unc_model.get_big_m()

            model._bin_penalty = pyo.Var(
                model.indices_constrs_cat_noncat_contr, domain=pyo.Binary
            )

            big_m_term = {
                i: big_m * (1 - model._bin_penalty[i])
                for i in model.indices_constrs_cat_noncat_contr
            }

            if self._dist_metric == "l2":
                # take sqrt for l2 distance
                model.aux_non_cat_unc = pyo.Var(
                    model.indices_constrs_cat_noncat_contr, bounds=(0, dist_bound)
                )

                def constrs_unc_x_aux(model_obj, k):
                    return (
                        model_obj.aux_non_cat_unc[k] * model_obj.aux_non_cat_unc[k]
                        >= non_cat_term_list[k]
                    )

                model.constrs_unc_x_aux = pyo.Constraint(
                    model.indices_constrs_cat_noncat_contr, rule=constrs_unc_x_aux
                )

                def constrs_unc_x(model_obj, k):
                    return (
                        model_obj._unc + big_m_term[k]
                        >= (model_obj.aux_non_cat_unc[k] + cat_term_list[k])
                        * self._dist_coeff
                    )

                model.constrs_unc_x_l2 = pyo.Constraint(
                    model.indices_constrs_cat_noncat_contr, rule=constrs_unc_x
                )

            elif self._dist_metric == "l1" or "euclidean_squared":

                def constrs_unc_x_l1l2squared(model_obj, k):
                    return (
                        model_obj._unc + big_m_term[k]
                        >= (non_cat_term_list[k] + cat_term_list[k]) * self._dist_coeff
                    )

                model.constrs_unc_x_l1l2squared = pyo.Constraint(
                    model.indices_constrs_cat_noncat_contr,
                    rule=constrs_unc_x_l1l2squared,
                )

            else:
                # Illegal value for self._dist_metric
                raise ValueError
        elif self._acq_sense == "exploration":
            if self._dist_metric == "l2":

                # take sqrt for l2 distance
                model.aux_non_cat_unc = pyo.Var(
                    model.indices_constrs_cat_noncat_contr, bounds=(0, dist_bound)
                )

                def constrs_unc_x_aux_expl(model_obj, k):
                    return (
                        model_obj.aux_non_cat_unc[k] * model_obj.aux_non_cat_unc[k]
                        <= non_cat_term_list[k]
                    )

                model.constrs_unc_x_aux_expl = pyo.Constraint(
                    model.indices_constrs_cat_noncat_contr, rule=constrs_unc_x_aux_expl
                )

                def constrs_unc_x_expl(model_obj, k):
                    return (
                        model_obj._unc * model_obj._unc
                        <= (model_obj.aux_non_cat_unc[k] + cat_term_list[k])
                        * self._dist_coeff
                    )

                model.constrs_unc_x_expl = pyo.Constraint(
                    model.indices_constrs_cat_noncat_contr, rule=constrs_unc_x_expl
                )

            elif self._dist_metric == "l1" or "euclidean_squared":

                def constrs_unc_x_l1l2squared_expl(model_obj, k):
                    return (
                        model_obj._unc
                        <= (non_cat_term_list[k] + cat_term_list[k]) * self._dist_coeff
                    )

                model.constrs_unc_x_l1l2squared = pyo.Constraint(
                    model.indices_constrs_cat_noncat_contr,
                    rule=constrs_unc_x_l1l2squared_expl,
                )

            else:
                # Illegal value for self._dist_metric
                raise ValueError

        def rule_constr_cat_noncat_quadr(model_obj, i):
            # take sqrt for l2 distance
            non_cat_term, cat_term = model.terms_constrs_cat_noncat_contr[i]
            return (
                model._unc * model._unc
                <= (non_cat_term + cat_term) * self._dist_coeff**2
            )

        def rule_constr_cat_noncat(model_obj, i):
            non_cat_term, cat_term = model.terms_constrs_cat_noncat_contr[i]
            return model._unc <= (non_cat_term + cat_term) * self._dist_coeff

        if self._dist_metric == "l2":
            model.constrs_cat_noncat_contr = pyo.Constraint(
                model.indices_constrs_cat_noncat_contr,
                rule=rule_constr_cat_noncat_quadr,
            )
        else:
            model.constrs_cat_noncat_contr = pyo.Constraint(
                model.indices_constrs_cat_noncat_contr, rule=rule_constr_cat_noncat
            )
