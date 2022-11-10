from typing import Tuple
import numpy as np
import random


class ProblemConfig:

    def __init__(self, rnd_seed: int = None):
        self._feat_list = []
        self._obj_list = []
        self._rnd_seed = rnd_seed

        if rnd_seed is not None:
            assert isinstance(rnd_seed, int), \
                f"Argument 'rnd_seed' needs to be an integer. Got type '{type(rnd_seed)}'."
            np.random.seed(rnd_seed)
            random.seed(rnd_seed)

    @property
    def cat_idx(self):
        return tuple([i for i, feat in enumerate(self.feat_list) if feat.is_cat()])

    @property
    def non_cat_idx(self):
        return tuple([i for i, feat in enumerate(self.feat_list) if not feat.is_cat()])

    @property
    def non_cat_lb(self):
        return tuple([feat.lb for i, feat in enumerate(self.feat_list) if not feat.is_cat()])

    @property
    def non_cat_ub(self):
        return tuple([feat.ub for i, feat in enumerate(self.feat_list) if not feat.is_cat()])

    @property
    def non_cat_bnd_diff(self):
        return tuple(
            [feat.ub - feat.lb for i, feat in enumerate(self.feat_list) if not feat.is_cat()])

    @property
    def feat_list(self):
        return self._feat_list

    @property
    def obj_list(self):
        return self._obj_list

    @property
    def rnd_seed(self):
        return self._rnd_seed

    def add_feature(self, feat_type: str, bounds: Tuple = None, name: str = None):
        if name is None:
            name = f"feat_{len(self.feat_list)}"

        if bounds is None and feat_type in ("real", "integer", "categorical"):
            raise IOError(
                "Please provide bounds for feature types in '(real, integer, categorical)'")

        # perform basic input checks and define features of space
        if feat_type in ("real", "integer"):
            assert len(bounds) == 2, \
                f"Too many values for bounds for feat_type '{feat_type}'. Expected '2', " \
                f"got '{len(bounds)}'. Check feature '{name}'"

            lb, ub = bounds

            if feat_type == "real":
                assert isinstance(lb, float) and isinstance(ub, float), \
                    f"Wrong type for bounds of feat_type '{feat_type}'. Expected '(float, " \
                    f"float)', got '({type(lb)}, {type(ub)})'. Check feature '{name}'"

                assert lb < ub, f"Lower bound '{lb}' of feat_type '{feat_type}' needs to be " \
                                f"smaller than upper bound. Check feature '{name}'."

                self._feat_list.append(Real(lb=lb, ub=ub, name=name))
            else:
                assert isinstance(lb, int) and isinstance(ub, int), \
                    f"Wrong type for bounds of feat_type '{feat_type}'. Expected '(int, int)', " \
                    f"got '({type(lb)}, {type(ub)})'. Check feature '{name}'"

                assert lb < ub, f"Lower bound '{lb}' of feat_type '{feat_type}' needs to be " \
                                f"smaller than upper bound. Check feature '{name}'."

                self._feat_list.append(Integer(lb=lb, ub=ub, name=name))

        elif feat_type == "binary":
            self._feat_list.append(Binary(name=name))

        elif feat_type == "categorical":
            assert len(bounds) > 1, \
                f"Not enough categories specified for bounds of feat_type '{feat_type}'. " \
                f"Expected 'len(bounds) > 1', got 'len(bounds) = {len(bounds)}'. " \
                f"Check feature '{name}'."

            assert all(isinstance(item, str) for item in bounds), \
                f"Wrong type for bounds of feat_type '{feat_type}'. Expected 'str' for all " \
                f"categories."

            assert len(bounds) == len(set(bounds)), \
                f"Categories of feat_type '{feat_type}' are not all unique."

            self._feat_list.append(Categorical(cat_list=bounds, name=name))

        else:
            raise IOError(f"No support for feat_type '{feat_type}'. Check feature '{name}'.")

    def add_min_objective(self, name: str = None):
        if name is None:
            name = f"obj_{len(self.obj_list)}"

        self._obj_list.append(MinObjective(name=name))

    def get_rnd_sample_numpy(self, num_samples):
        # returns np.array for faster processing
        array_list = []
        for feat in self.feat_list:
            if feat.is_real():
                array_list.append(
                    np.random.uniform(low=feat.lb, high=feat.ub, size=num_samples))
            elif feat.is_cat():
                array_list.append(np.random.randint(len(feat.cat_list), size=num_samples))
            elif feat.is_int() or feat.is_bin():
                array_list.append(
                    np.random.random_integers(low=feat.lb, high=feat.ub, size=num_samples))
        return np.squeeze(np.column_stack(array_list))

    def get_rnd_sample_list(self, num_samples=1, cat_enc=False):
        # returns list of tuples
        sample_list = []
        for _ in range(num_samples):
            sample = []
            for feat in self.feat_list:
                if feat.is_real():
                    sample.append(random.uniform(feat.lb, feat.ub))
                elif feat.is_cat():
                    if cat_enc:
                        sample.append(random.randrange(0, len(feat.cat_list)))
                    else:
                        sample.append(random.choice(feat.cat_list))
                elif feat.is_int() or feat.is_bin():
                    sample.append(random.randint(feat.lb, feat.ub))
            sample_list.append(tuple(sample))
        return sample_list if len(sample_list) > 1 else sample_list[0]

    def get_gurobi_model_core(self, env=None):
        import gurobipy as gur

        # initialize gurobi model
        if env is None:
            model = gur.Model()
        else:
            model = gur.Model(env=env)

        # initalize var space
        model._all_feat = []

        for i, feat in enumerate(self.feat_list):
            if feat.is_real():
                model._all_feat.append(
                    model.addVar(lb=feat.lb, ub=feat.ub, name=feat.name, vtype='C'))
            elif feat.is_cat():
                model._all_feat.append(dict())
                for enc, cat in zip(feat.enc_cat_list, feat.cat_list):
                    comb_name = f"{feat.name}_{cat}"
                    model._all_feat[i][enc] = \
                        model.addVar(name=comb_name, vtype='B')
            elif feat.is_int():
                model._all_feat.append(
                    model.addVar(lb=feat.lb, ub=feat.ub, name=feat.name, vtype='I'))
            elif feat.is_bin():
                model._all_feat.append(
                    model.addVar(name=feat.name, vtype='B'))

        model.update()
        return model

    def get_pyomo_model_core(self):
        import pyomo.environ as pyo

        # initialize Pyomo model
        model = pyo.ConcreteModel()

        # Initialize var space
        model._all_feat = []

        # Define some auxiliary dictionaries
        index_to_var_domain = {}
        index_to_var_bounds = {}
        for i, feat in enumerate(self.feat_list):
            if feat.is_real():
                index_to_var_domain[i] = pyo.Reals
                index_to_var_bounds[i] = (feat.lb, feat.ub)
            elif feat.is_cat():
                for enc, cat in zip(feat.enc_cat_list, feat.cat_list):
                    # We encode the index of this variable by (i, enc, cat), where 'i' is the position in the list of
                    # features, 'enc' is the corresponding encoded numerical value and 'cat' is the category
                    index_to_var_domain[i, enc, cat] = pyo.Binary
            elif feat.is_int():
                index_to_var_domain[i] = pyo.Integers
                index_to_var_bounds[i] = (feat.lb, feat.ub)
            elif feat.is_bin():
                index_to_var_domain[i] = pyo.Binary

        # Build Pyomo index set from dictionary keys
        # Why strings?? Well, pyomo doesn't support lists where some indices are multidimensional, but this is handy
        # for encoded categorical features
        model.indices_features = pyo.Set(initialize=[str(x) for x in index_to_var_domain])

        # These auxiliary functions are needed, because Pyomo does not accept the dictionaries instead. The 'model'
        # argument is needed as well, but not explicitly used.
        # We hve the eval-statement eval(i) instead of just i, because the indices are strings for reasons explained
        # above :(
        def i_to_dom(model, i):
            return index_to_var_domain[eval(i)]

        def i_to_bounds(model, i):
            if eval(i) in index_to_var_bounds:
                return index_to_var_bounds[eval(i)]
            else:
                return (None, None)

        # Define decision variables for Pyomo model
        model.x = pyo.Var(model.indices_features, domain=i_to_dom, bounds=i_to_bounds)
        # We store the decision variables corresponding to the features in the list _all_feat in order to follow a
        # similar strategy to the gurobi model. Can be replaced in the future by a more direct approach.
        for i_str in model.x:
            i = eval(i_str)
            if type(i) is int:
                model._all_feat.append(model.x[i_str])
            elif type(i) is tuple:
                if i[1] == 0:
                    model._all_feat.append({})
                    model._all_feat[-1][i[1]] = model.x[i_str]
                elif i[1] > 0:
                    model._all_feat[-1][i[1]] = model.x[i_str]
                else:
                    raise TypeError
            else:
                raise TypeError

        return model

    def __str__(self):
        out_str = list(["\nPROBLEM SUMMARY"])
        out_str.append(len(out_str[-1][:-1]) * "-")
        out_str.append("features:")
        for feat in self.feat_list:
            if feat.is_cat():
                out_str.append(f"{feat.name} :: {feat.__class__.__name__} :: {feat.cat_list} ")
            else:
                out_str.append(
                    f"{feat.name} :: {feat.__class__.__name__} :: ({feat.lb}, {feat.ub}) ")

        out_str.append("\nobjectives:")
        for obj in self.obj_list:
            out_str.append(f"{obj.name} :: {obj.__class__.__name__}")
        return "\n".join(out_str)


class FeatureType:
    def is_real(self):
        return False

    def is_cat(self):
        return False

    def is_int(self):
        return False

    def is_bin(self):
        return False


class Real(FeatureType):
    def __init__(self, lb, ub, name):
        self.lb = lb
        self.ub = ub
        self.name = name

    def is_real(self):
        return True


class Categorical(FeatureType):
    def __init__(self, cat_list, name):
        self._cat_list = cat_list
        self.name = name

        # encode categories
        self._enc2str, self._str2enc = {}, {}
        self._enc_cat_list = []
        for enc, cat in enumerate(cat_list):
            self._enc_cat_list.append(enc)
            self._enc2str[enc] = cat
            self._str2enc[cat] = enc

    @property
    def cat_list(self):
        return self._cat_list

    @property
    def enc_cat_list(self):
        return self._enc_cat_list

    def trafo_str2enc(self, xi):
        return self._str2enc[xi]

    def trafo_enc2str(self, xi):
        return self._enc2str[xi]

    def is_cat(self):
        return True


class Integer(FeatureType):
    def __init__(self, lb, ub, name):
        self.lb = lb
        self.ub = ub
        self.name = name

    def is_int(self):
        return True


class Binary(FeatureType):
    def __init__(self, name):
        self.lb = 0
        self.ub = 1
        self.name = name

    def is_bin(self):
        return True


class MinObjective:
    def __init__(self, name):
        self.name = name
