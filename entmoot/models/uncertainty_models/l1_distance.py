from entmoot.models.uncertainty_models.base_distance import NonCatDistance
import numpy as np


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

        feat = model._all_feat

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
                    (xi[i] - (feat[idx] - self.shift[i]) / self.scale[i])
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

        feat = model._all_feat

        # define auxiliary variables
        feat_dict = {
            idx: self._problem_config.feat_list[idx]
            for idx in self._problem_config.non_cat_idx
        }

        indices_aux_vars = [(i, j) for i in range(len(self.x_trafo)) for j in feat_dict]
        model.aux_pos = pyo.Var(indices_aux_vars, domain=pyo.NonNegativeReals)
        model.aux_neg = pyo.Var(indices_aux_vars, domain=pyo.NonNegativeReals)

        indices_l1_constraints = [
            (data_idx, xi, i, idx)
            for data_idx, xi in enumerate(self.x_trafo)
            for i, idx in enumerate(self._problem_config.non_cat_idx)
        ]

        raise NotImplementedError()
