from typing import Tuple
import numpy as np
import random


class Space:

    def __init__(self, rnd_seed: int = None):
        self._feat_list = []
        self._rnd_seed = rnd_seed

        if rnd_seed is not None:
            np.random.seed(rnd_seed)
            random.seed(rnd_seed)

    @property
    def cat_idx(self):
        return tuple([i for i, feat in enumerate(self.feat_list) if feat.is_cat()])

    @property
    def feat_list(self):
        return self._feat_list

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

                self.feat_list.append(Real(lb=lb, ub=ub, name=name))
            else:
                assert isinstance(lb, int) and isinstance(ub, int), \
                    f"Wrong type for bounds of feat_type '{feat_type}'. Expected '(int, int)', " \
                    f"got '({type(lb)}, {type(ub)})'. Check feature '{name}'"

                assert lb < ub, f"Lower bound '{lb}' of feat_type '{feat_type}' needs to be " \
                                f"smaller than upper bound. Check feature '{name}'."

                self.feat_list.append(Integer(lb=lb, ub=ub, name=name))

        elif feat_type == "binary":
            self.feat_list.append(Binary(name=name))

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

            self.feat_list.append(Categorical(cat_list=bounds, name=name))

        else:
            raise IOError(f"No support for feat_type '{feat_type}'. Check feature '{name}'.")

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
        return np.column_stack(array_list)

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
        return sample_list

    def __str__(self):
        out_str = ["\nspace summary:"]
        for feat in self.feat_list:
            if feat.is_cat():
                out_str.append(f"{feat.name} :: {feat.__class__.__name__} :: {feat.cat_list} ")
            else:
                out_str.append(
                    f"{feat.name} :: {feat.__class__.__name__} :: ({feat.lb}, {feat.ub}) ")
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
        self.cat_list = cat_list
        self.name = name

        # encode categories
        self._enc2str, self._str2enc = {}, {}
        for enc, cat in enumerate(cat_list):
            self._enc2str[enc] = cat
            self._str2enc[cat] = enc

    def trafo_str2enc(self, X):
        raise NotImplementedError()

    def trafo_enc2str(self, X):
        raise NotImplementedError()

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
