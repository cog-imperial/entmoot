from typing import Tuple, Set


class Space:

    def __init__(self):
        self.feat_list = []

    def add_feature(self, feat_type: str, bounds: Tuple, name: str = None):
        if name is None:
            name = f"feat_{len(self.feat_list)}"

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


class FeatureType:
    pass


class Real(FeatureType):
    def __init__(self, lb, ub, name):
        self.lb = lb
        self.ub = ub
        self.name = name


class Categorical(FeatureType):
    def __init__(self, cat_list, name):
        self.cat_list = cat_list
        self.name = name


class Integer(FeatureType):
    def __init__(self, lb, ub, name):
        self.lb = lb
        self.ub = ub
        self.name = name


class Binary(FeatureType):
    def __init__(self, name):
        self.name = name
