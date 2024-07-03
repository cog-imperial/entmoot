from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar

import numpy as np
import pyomo.environ as pyo

from entmoot.typing.optimizer_stubs import GurobiModelT, PyomoModelT

BoundsT = tuple[float, float]
CategoriesT = list[str | float | int] | tuple[str | float | int, ...]


class FeatureType(ABC):
    pyomo_domain: pyo.Set
    gurobi_vtype: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_enc_bnds(self):
        pass

    def is_real(self):
        return False

    def is_cat(self):
        return False

    def is_int(self):
        return False

    def is_bin(self):
        return False

    def encode(self, xi):
        return xi

    def decode(self, xi):
        return xi

    def get_pyomo_domain_and_bounds(self, i: int):
        return {i: self.pyomo_domain}, {i: self.get_enc_bnds()}

    def get_gurobi_variable(self, model: GurobiModelT):
        return model.addVar(
            lb=self.lb, ub=self.ub, name=self.name, vtype=self.gurobi_vtype
        )

    def __str__(self):
        return f"{self.name} :: {self.__class__.__name__} :: {self.get_enc_bnds()}"


class Real(FeatureType):
    pyomo_domain = pyo.Reals
    gurobi_vtype = "C"

    def __init__(self, lb: float, ub: float, name: str):
        super().__init__(name)
        self.lb = lb
        self.ub = ub

    def get_enc_bnds(self):
        return (self.lb, self.ub)

    def is_real(self):
        return True


class Categorical(FeatureType):
    def __init__(self, cat_list: CategoriesT, name: str):
        super().__init__(name)
        self._cat_list = cat_list

        # encode categories
        self._enc2str, self._str2enc = {}, {}
        self._enc_cat_list = []
        for enc, cat in enumerate(cat_list):
            self._enc_cat_list.append(enc)
            self._enc2str[enc] = cat
            self._str2enc[cat] = enc

    def get_enc_bnds(self):
        return self._enc_cat_list

    @property
    def cat_list(self):
        return self._cat_list

    @property
    def enc_cat_list(self):
        return self._enc_cat_list

    def encode(self, xi):
        return self._str2enc[xi]

    def decode(self, xi):
        return self._enc2str[xi]

    def is_cat(self):
        return True

    def get_pyomo_domain_and_bounds(self, i: int):
        # We encode the index of this variable by (i, enc, cat), where 'i' is the position in the list of
        # features, 'enc' is the corresponding encoded numerical value and 'cat' is the category
        return {
            (i, enc, cat): pyo.Binary
            for (enc, cat) in zip(self.enc_cat_list, self.cat_list)
        }, {}

    def get_gurobi_variable(self, model: GurobiModelT):
        return {
            enc: model.addVar(name=f"{self.name}_{cat}", vtype="B")
            for (enc, cat) in zip(self.enc_cat_list, self.cat_list)
        }

    def __str__(self):
        return f"{self.name} :: {self.__class__.__name__} :: {self.cat_list}"


class Integer(FeatureType):
    pyomo_domain = pyo.Integers
    gurobi_vtype = "I"

    def __init__(self, lb: int, ub: int, name: str):
        super().__init__(name)
        self.lb = lb
        self.ub = ub

    def get_enc_bnds(self):
        return (self.lb, self.ub)

    def is_int(self):
        return True

    def decode(self, xi):
        return int(xi)


class Binary(FeatureType):
    pyomo_domain = pyo.Binary
    gurobi_vtype = "B"

    def __init__(self, name: str):
        super().__init__(name)
        self.lb = 0
        self.ub = 1

    def get_enc_bnds(self):
        return (self.lb, self.ub)

    def decode(self, xi):
        return abs(int(xi))

    def is_bin(self):
        return True


class Objective:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{self.name} :: {self.__class__.__name__}"


class MinObjective(Objective):
    sign = 1


class MaxObjective(Objective):
    sign = -1


AnyFeatureT = Real | Integer | Categorical | Binary
FeatureT = TypeVar("FeatureT", bound=FeatureType)
AnyObjectiveT = MinObjective | MaxObjective


class ProblemConfig:
    def __init__(self, rnd_seed: Optional[int] = None):
        self._feat_list = []
        self._obj_list = []
        self._rnd_seed = rnd_seed
        self.rng = np.random.default_rng(rnd_seed)

    @property
    def cat_idx(self):
        return tuple(
            [
                i
                for i, feat in enumerate(self.feat_list)
                if isinstance(feat, Categorical)
            ]
        )

    @property
    def non_cat_idx(self):
        return tuple(
            [
                i
                for i, feat in enumerate(self.feat_list)
                if not isinstance(feat, Categorical)
            ]
        )

    @property
    def non_cat_lb(self):
        return tuple(
            [
                feat.lb
                for i, feat in enumerate(self.feat_list)
                if not isinstance(feat, Categorical)
            ]
        )

    @property
    def non_cat_ub(self):
        return tuple(
            [
                feat.ub
                for i, feat in enumerate(self.feat_list)
                if not isinstance(feat, Categorical)
            ]
        )

    @property
    def non_cat_bnd_diff(self):
        return tuple(
            [
                feat.ub - feat.lb
                for i, feat in enumerate(self.feat_list)
                if not isinstance(feat, Categorical)
            ]
        )

    def get_idx_and_feat_by_type(
        self, feature_type: type[FeatureT]
    ) -> tuple[tuple[int, FeatureT], ...]:
        return tuple(
            [
                (i, feat)
                for i, feat in enumerate(self.feat_list)
                if isinstance(feat, feature_type)
            ]
        )

    @property
    def feat_list(self) -> list[AnyFeatureT]:
        return self._feat_list

    @property
    def obj_list(self) -> list[AnyObjectiveT]:
        return self._obj_list

    @property
    def rnd_seed(self):
        return self._rnd_seed

    def get_enc_bnd(self):
        return [feat.get_enc_bnds() for feat in self.feat_list]

    def _encode_xi(self, xi: List):
        return np.asarray(
            [feat.encode(xi[idx]) for idx, feat in enumerate(self.feat_list)]
        )

    def _decode_xi(self, xi: List):
        return [feat.decode(xi[idx]) for idx, feat in enumerate(self.feat_list)]

    def encode(self, X: List):
        if len(X) == 0:
            return []
        elif len(X) == 1:
            return self._encode_xi(X[0])
        else:
            enc = [self._encode_xi(xi) for xi in X]
            return np.asarray(enc)

    def decode(self, X: List):
        if len(X) == 0:
            return []
        elif len(X) == 1:
            return self._decode_xi(X[0])
        else:
            dec = [self._decode_xi(xi) for xi in X]
            return np.asarray(dec)

    def add_feature(
        self,
        feat_type: str,
        bounds: Optional[BoundsT | CategoriesT] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            name = f"feat_{len(self.feat_list)}"

        if bounds is None:
            if feat_type == "binary":
                self._feat_list.append(Binary(name=name))
                return
            raise ValueError(
                "Please provide bounds for feature types in '(real, integer, categorical)'"
            )

        # perform basic input checks and define features of space
        if feat_type in ("real", "integer"):
            assert len(bounds) == 2, (
                f"Too many values for bounds for feat_type '{feat_type}'. Expected '2', "
                f"got '{len(bounds)}'. Check feature '{name}'"
            )

            lb, ub = bounds

            if feat_type == "real":
                if isinstance(lb, int):
                    lb = float(lb)
                if isinstance(ub, int):
                    ub = float(ub)

                assert isinstance(lb, float) and isinstance(ub, float), (
                    f"Wrong type for bounds of feat_type '{feat_type}'. Expected '(float, "
                    f"float)', got '({type(lb)}, {type(ub)})'. Check feature '{name}'"
                )

                assert lb < ub, (
                    f"Lower bound '{lb}' of feat_type '{feat_type}' needs to be "
                    f"smaller than upper bound. Check feature '{name}'."
                )

                self._feat_list.append(Real(lb=lb, ub=ub, name=name))
            else:
                assert isinstance(lb, int) and isinstance(ub, int), (
                    f"Wrong type for bounds of feat_type '{feat_type}'. Expected '(int, int)', "
                    f"got '({type(lb)}, {type(ub)})'. Check feature '{name}'"
                )

                assert lb < ub, (
                    f"Lower bound '{lb}' of feat_type '{feat_type}' needs to be "
                    f"smaller than upper bound. Check feature '{name}'."
                )

                self._feat_list.append(Integer(lb=lb, ub=ub, name=name))

        elif feat_type == "categorical":
            assert len(bounds) > 1, (
                f"Not enough categories specified for bounds of feat_type '{feat_type}'. "
                f"Expected 'len(bounds) > 1', got 'len(bounds) = {len(bounds)}'. "
                f"Check feature '{name}'."
            )

            assert all(isinstance(item, str) for item in bounds), (
                f"Wrong type for bounds of feat_type '{feat_type}'. Expected 'str' for all "
                f"categories."
            )

            assert len(bounds) == len(
                set(bounds)
            ), f"Categories of feat_type '{feat_type}' are not all unique."

            self._feat_list.append(Categorical(cat_list=bounds, name=name))  # type: ignore

        else:
            raise ValueError(
                f"No support for feat_type '{feat_type}'. Check feature '{name}'."
            )

    def add_min_objective(self, name: Optional[str] = None):
        if name is None:
            name = f"obj_{len(self.obj_list)}"

        self._obj_list.append(MinObjective(name=name))

    def add_max_objective(self, name: Optional[str] = None):
        if name is None:
            name = f"obj_{len(self.obj_list)}"

        self._obj_list.append(MaxObjective(name=name))

    def transform_objective(self, y: np.ndarray) -> np.ndarray:
        """Transform data for minimisation/maximisation"""
        # y.shape = (num_samples, num_obj)
        signs = np.array([obj.sign for obj in self.obj_list]).reshape(1, -1)
        return y * signs

    def get_rnd_sample_numpy(self, num_samples):
        # returns np.array for faster processing
        # TODO: defer sample logic to feature
        array_list = []
        for feat in self.feat_list:
            if isinstance(feat, Real):
                array_list.append(
                    self.rng.uniform(low=feat.lb, high=feat.ub, size=num_samples)
                )
            elif isinstance(feat, Categorical):
                array_list.append(
                    self.rng.integers(0, len(feat.cat_list), size=num_samples)
                )
            else:
                array_list.append(
                    self.rng.integers(low=feat.lb, high=feat.ub + 1, size=num_samples)
                )
        return np.squeeze(np.column_stack(array_list))

    def get_rnd_sample_list(self, num_samples=1, cat_enc=False):
        # returns list of tuples
        sample_list = []
        for _ in range(num_samples):
            sample = []
            for feat in self.feat_list:
                if isinstance(feat, Real):
                    sample.append(self.rng.uniform(feat.lb, feat.ub))
                elif isinstance(feat, Categorical):
                    if cat_enc:
                        sample.append(self.rng.integers(0, len(feat.cat_list)))
                    else:
                        sample.append(self.rng.choice(feat.cat_list))
                else:
                    sample.append(self.rng.integers(feat.lb, feat.ub + 1))
            sample_list.append(tuple(sample))
        return sample_list if len(sample_list) > 1 else sample_list[0]

    def get_gurobi_model_core(self, env=None) -> GurobiModelT:
        import gurobipy as gur

        # initialize gurobi model
        model: GurobiModelT = gur.Model(env=env)  # type: ignore

        # initalize var space
        model._all_feat = [feat.get_gurobi_variable(model) for feat in self.feat_list]

        model.update()
        return model

    def copy_gurobi_model_core(self, model_core: GurobiModelT) -> GurobiModelT:
        """
        Computes a copy of a Gurobi model which is decoupled from the original, i.e. you can modify the copy without
        changing the original model.
        :param model_core: The gurobi model we want to copy
        :type model_core: gurobi.Model
        :return:
        :rtype: gurobi.Model
        """
        copy_model_core: GurobiModelT = model_core.copy()
        # Since Gurobi's .copy() method does not copy attributes like ._all_feat we have to copy them manually
        copy_model_core._all_feat = []

        # transfer feature var list to model copy
        for i, feat in enumerate(self.feat_list):
            if isinstance(feat, Categorical):
                copy_model_core._all_feat.append(dict())
                for enc, cat in zip(feat.enc_cat_list, feat.cat_list):
                    var_name = model_core._all_feat[i][enc].VarName
                    copy_model_core._all_feat[i][enc] = copy_model_core.getVarByName(
                        var_name
                    )
            else:
                var_name = model_core._all_feat[i].VarName
                copy_model_core._all_feat.append(copy_model_core.getVarByName(var_name))
        return copy_model_core

    def get_pyomo_model_core(self) -> PyomoModelT:
        import pyomo.environ as pyo

        # initialize Pyomo model
        model: PyomoModelT = pyo.ConcreteModel()

        # Initialize var space
        model._all_feat = []

        # Define some auxiliary dictionaries
        index_to_var_domain = {}
        index_to_var_bounds = {}
        for i, feat in enumerate(self.feat_list):
            domain, bounds = feat.get_pyomo_domain_and_bounds(i)
            index_to_var_domain.update(domain)
            index_to_var_bounds.update(bounds)

        # Build Pyomo index set from dictionary keys
        # Why strings?? Well, pyomo doesn't support lists where some indices are multidimensional, but this is handy
        # for encoded categorical features
        model.indices_features = pyo.Set(
            initialize=[str(x) for x in index_to_var_domain]
        )

        # These auxiliary functions are needed, because Pyomo does not accept the dictionaries instead. The 'model'
        # argument is needed as well, but not explicitly used.
        # We hve the eval-statement eval(i) instead of just i, because the indices are strings for reasons explained
        # above :(
        def i_to_dom(model, i):
            return index_to_var_domain[eval(i)]

        def i_to_bounds(model, i):
            return index_to_var_bounds.get(eval(i), (None, None))

        # Define decision variables for Pyomo model
        model.x = pyo.Var(model.indices_features, domain=i_to_dom, bounds=i_to_bounds)
        # We store the decision variables corresponding to the features in the list _all_feat in order to follow a
        # similar strategy to the gurobi model. Can be replaced in the future by a more direct approach.
        for i_str in model.x:
            i = eval(i_str)
            if isinstance(i, int):
                model._all_feat.append(model.x[i_str])
            elif isinstance(i, tuple):
                if i[1] == 0:
                    model._all_feat.append({})

                model._all_feat[-1][i[1]] = model.x[i_str]
            else:
                raise TypeError

        return model

    def copy_pyomo_model_core(self, model_core: PyomoModelT) -> PyomoModelT:
        """
        Computes a copy of a Pyomo model which is decoupled from the original, i.e. you can modify the copy without
        changing the original model.
        :param model_core: A Pyomo model
        :type model_core: pyomo.environ.ConcreteModel
        :return: A copy of the Pyomo model
        :rtype: pyomo.environ.ConcreteModel
        """
        return model_core.clone()

    def __str__(self):
        return "\n".join(
            (
                "\nPROBLEM SUMMARY",
                "-" * 15,
                "features:",
                *map(str, self.feat_list),
                "\nobjectives:",
                *map(str, self.obj_list),
            )
        )
