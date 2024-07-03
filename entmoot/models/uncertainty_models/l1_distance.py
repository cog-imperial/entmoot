import numpy as np

from entmoot.models.uncertainty_models.base_distance import NonCatDistance


class L1Distance(NonCatDistance):
    def _get_distance(self, x_left, x_right):
        if x_left.ndim == 1:
            dist = np.sum(np.abs(x_left - x_right))
        else:
            dist = np.sum(np.abs(x_left - x_right), axis=1)
        return dist

    def _array_predict(self, X):
        raise NotImplementedError()

    def get_gurobipy_model_constr_terms(self, model):
        from gurobipy import GRB, quicksum

        features = model._all_feat

        # define auxiliary variables
        feat_dict = {
            idx: self._problem_config.feat_list[idx]
            for idx in self._problem_config.non_cat_idx
        }

        aux_pos = model.addVars(len(self.x_trafo), feat_dict, name="aux_pos", vtype="C")

        aux_neg = model.addVars(len(self.x_trafo), feat_dict, name="aux_neg", vtype="C")

        # define distance constraints
        constr_list = []
        for data_idx, xi in enumerate(self.x_trafo):
            for i, idx in enumerate(self._problem_config.non_cat_idx):
                # capture positive and negative contributions
                model.addConstr(
                    (xi[i] - (features[idx] - self.shift[i]) / self.scale[i])
                    == aux_pos[data_idx, idx] - aux_neg[data_idx, idx],
                    name=f"unc_aux_({data_idx},{idx})",
                )

                # make sure only one variable is non-zero
                model.addSOS(
                    GRB.SOS_TYPE1, [aux_pos[data_idx, idx], aux_neg[data_idx, idx]]
                )

            # define individual sum terms
            constr = quicksum(
                aux_pos[data_idx, idx] + aux_neg[data_idx, idx]
                for idx in self._problem_config.non_cat_idx
            )

            constr_list.append(constr)

        model.update()
        return constr_list

    def get_pyomo_model_constr_terms(self, model):
        import pyomo.environ as pyo

        features = model._all_feat

        # define auxiliary variables
        feat_dict = {
            idx: self._problem_config.feat_list[idx]
            for idx in self._problem_config.non_cat_idx
        }

        indices_aux_vars = [(i, j) for i in range(len(self.x_trafo)) for j in feat_dict]
        model.aux_pos = pyo.Var(indices_aux_vars, domain=pyo.NonNegativeReals)
        model.aux_neg = pyo.Var(indices_aux_vars, domain=pyo.NonNegativeReals)

        indices_l1_constraints = [
            (data_idx, i, idx)
            for data_idx, _ in enumerate(self.x_trafo)
            for i, idx in enumerate(self._problem_config.non_cat_idx)
        ]

        def rule_contrs_l1_pos_neg_contr(modelobj, data_idx, i, idx):
            xi = self.x_trafo[data_idx]
            return (
                xi[i] - (features[idx] - self.shift[i]) / self.scale[i]
            ) == modelobj.aux_pos[data_idx, idx] - modelobj.aux_neg[data_idx, idx]

        model.contrs_l1_pos_neg_contr = pyo.Constraint(
            indices_l1_constraints, rule=rule_contrs_l1_pos_neg_contr
        )

        model.l1_bigm = {
            (data_idx, idx): max(
                abs(
                    self.x_trafo[data_idx][i]
                    - (features[idx].ub - self.shift[i]) / self.scale[i]
                ),
                abs(
                    self.x_trafo[data_idx][i]
                    - (features[idx].lb - self.shift[i]) / self.scale[i]
                ),
            )
            for (data_idx, i, idx) in indices_l1_constraints
        }

        model.indices_bigm = list(model.l1_bigm.keys())

        model.y_pos_bigm = pyo.Var(model.indices_bigm, domain=pyo.Binary)
        model.y_neg_bigm = pyo.Var(model.indices_bigm, domain=pyo.Binary)

        def bigm_ub_pos(modelobj, data_idx, idx):
            return (
                modelobj.aux_pos[data_idx, idx]
                <= model.l1_bigm[data_idx, idx] * model.y_pos_bigm[data_idx, idx]
            )

        def bigm_ub_neg(modelobj, data_idx, idx):
            return (
                modelobj.aux_neg[data_idx, idx]
                <= model.l1_bigm[data_idx, idx] * model.y_neg_bigm[data_idx, idx]
            )

        model.constrs_l1_bigm_ub_pos = pyo.Constraint(
            model.indices_bigm, rule=bigm_ub_pos
        )
        model.constrs_l1_bigm_ub_neg = pyo.Constraint(
            model.indices_bigm, rule=bigm_ub_neg
        )

        def binaries_mutually_exclusive(modelobj, data_idx, idx):
            return (
                model.y_pos_bigm[data_idx, idx] + model.y_neg_bigm[data_idx, idx] == 1
            )

        model.constrs_l1_bigm_mut_excl = pyo.Constraint(
            model.indices_bigm, rule=binaries_mutually_exclusive
        )

        constr_list = [
            sum(
                model.aux_pos[data_idx, idx] + model.aux_neg[data_idx, idx]
                for idx in self._problem_config.non_cat_idx
            )
            for (data_idx, _) in enumerate(self.x_trafo)
        ]

        return constr_list
