from entmoot.models.base_model import BaseModel
from entmoot.models.mean_models.lgbm_utils import read_lgbm_tree_model_dict
from entmoot.models.mean_models.meta_tree_ensemble import MetaTreeModel
import warnings
import numpy as np


class TreeEnsemble(BaseModel):

    def __init__(self, problem_config, params=None):

        if params is None:
            params = {}

        self._problem_config = problem_config
        self._train_lib = params.get("train_lib", "lgbm")
        self._rnd_seed = problem_config.rnd_seed

        assert self._train_lib in ('lgbm', 'catboost', 'xgboost'), \
            f"Parameter 'train_lib' for tree ensembles needs to be " \
            f"in '('lgbm', 'catboost', 'xgboost')'."

        if "train_params" not in params:
            # default training params
            warnings.warn("No 'train_params' for tree ensemble training specified. "
                          "Switch training to default params!")

            self._train_params = \
                {'objective': 'regression',
                 'metric': 'rmse',
                 'boosting': 'gbdt',
                 'num_boost_round': 100,
                 'max_depth': 3,
                 'min_data_in_leaf': 1,
                 'min_data_per_group': 1,
                 'verbose': -1}

            if self._rnd_seed is not None:
                self._train_params['random_state'] = self._rnd_seed
        else:
            self._train_params = params["train_params"]

        self._tree_dict = None
        self._meta_tree_dict = {}

    @property
    def tree_dict(self):
        assert self._tree_dict is not None, \
            "No tree model is trained yet. Call '.fit(X, y)' first."
        return self._tree_dict

    @property
    def meta_tree_dict(self):
        assert len(self._meta_tree_dict) > 0, \
            "No tree model is trained yet. Call '.fit(X, y)' first."
        return self._meta_tree_dict

    def fit(self, X, y):

        # check dims of X and y
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._problem_config.feat_list), \
            f"Argument 'X' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._problem_config.feat_list)})', got '{X.shape}'."

        if y.ndim == 1:
            y = np.atleast_2d(y)

        assert y.shape[-1] == len(self._problem_config.obj_list), \
            f"Argument 'y' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._problem_config.obj_list)})', got '{y.shape}'."

        # train tree models for every objective
        if self._tree_dict is None:
            self._tree_dict = {}

        for i, obj in enumerate(self._problem_config.obj_list):
            if self._train_lib == "lgbm":
                tree_model = self._train_lgbm(X, y[:, i])
            elif self._train_lib == "catboost":
                raise NotImplementedError()
            elif self._train_lib == "xgboost":
                raise NotImplementedError()
            else:
                raise IOError(f"Parameter 'train_lib' for tree ensembles needs to be "
                              f"in '('lgbm', 'catboost', 'xgboost')'.")
            self._tree_dict[obj.name] = tree_model

    def _train_lgbm(self, X, y):
        import lightgbm as lgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._problem_config.cat_idx:
                # train for categorial vars
                train_data = lgb.Dataset(X, label=y,
                                         categorical_feature=self._problem_config.cat_idx,
                                         free_raw_data=False,
                                         params={'verbose': -1})

                tree_model = lgb.train(self._train_params, train_data,
                                       categorical_feature=self._problem_config.cat_idx,
                                       verbose_eval=False)
            else:
                # train for non-categorical vars
                train_data = lgb.Dataset(X, label=y,
                                         params={'verbose': -1})

                tree_model = lgb.train(self._train_params, train_data,
                                       verbose_eval=False)
        return tree_model

    def predict(self, X):

        # check dims of X
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._problem_config.feat_list), \
            f"Argument 'X' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._problem_config.feat_list)})', got '{X.shape}'."

        # predict vals
        tree_pred = []
        for obj in self._problem_config.obj_list:
            tree_pred.append(self.tree_dict[obj.name].predict(X))

        return np.squeeze(np.column_stack(tree_pred))

    def _update_meta_tree_dict(self):
        self._meta_tree_dict = {}

        # get model information
        for obj in self._problem_config.obj_list:

            if self._train_lib == "lgbm":
                lib_out = self.tree_dict[obj.name].dump_model()
            elif self._train_lib == "catboost":
                raise NotImplementedError()
            elif self._train_lib == "xgboost":
                raise NotImplementedError()
            else:
                raise IOError(f"Parameter 'train_lib' for tree ensembles needs to be "
                              f"in '('lgbm', 'catboost', 'xgboost')'.")

            # order tree_model_dict
            ordered_tree_model_dict = \
                read_lgbm_tree_model_dict(lib_out, cat_idx=self._problem_config.cat_idx)

            # populate meta_tree_model
            self._meta_tree_dict[obj.name] = MetaTreeModel(ordered_tree_model_dict)

    def _add_to_gurobipy_model(self, model):
        from gurobipy import GRB, quicksum
        self._update_meta_tree_dict()

        # attach tree info
        model._num_trees = lambda obj_name: self._meta_tree_dict[obj_name].num_trees

        # attach leaf info
        model._leaves = lambda obj_name, tree: \
            tuple(self._meta_tree_dict[obj_name].get_leaf_encodings(tree))
        model._leaf_weight = lambda obj_name, tree, leaf: \
            self._meta_tree_dict[obj_name].get_leaf_weight(tree, leaf)
        model._leaf_vars = lambda obj_name, tree, leaf: tuple(
            var for var in self._meta_tree_dict[obj_name].get_participating_variables(tree, leaf))

        # attach breakpoint info
        var_break_points = [tree_model.get_var_break_points()
                            for tree_model in self._meta_tree_dict.values()]
        model._breakpoints = {}
        model._breakpoint_index = []

        for idx, feat in enumerate(self._problem_config.feat_list):
            if feat.is_cat():
                continue
            else:
                splits = set()
                for vb in var_break_points:
                    try:
                        splits = splits.union(set(vb[idx]))
                    except KeyError:
                        pass
                if splits:
                    model._breakpoints[idx] = sorted(splits)
                    model._breakpoint_index.append(idx)

        # define indexing helper functions
        def tree_index(model_obj):
            for obj in self._problem_config.obj_list:
                for tree in range(model_obj._num_trees(obj.name)):
                    yield obj.name, tree

        def leaf_index(model_obj):
            for label, tree in tree_index(model_obj):
                for leaf in model_obj._leaves(label, tree):
                    yield label, tree, leaf

        def interval_index(model_obj):
            for var in model_obj._breakpoint_index:
                for j in range(len(model_obj._breakpoints[var])):
                    yield var, j

        def split_index(model_obj, meta_tree_dict):
            for label, tree in tree_index(model_obj):
                for encoding in meta_tree_dict[label].get_branch_encodings(tree):
                    yield label, tree, encoding

        # add leaf variables
        model._z = model.addVars(
            leaf_index(model),
            lb=0,
            ub=GRB.INFINITY,
            name="z", vtype='C'
        )

        # add split variables
        model._nu = model.addVars(
            interval_index(model),
            name="nu",
            vtype=GRB.BINARY
        )

        # add single leaf constraints
        def single_leaf_rule(model_obj, label, tree):
            z, leaves = model_obj._z, model_obj._leaves
            return quicksum(z[label, tree, leaf] for leaf in leaves(label, tree)) == 1

        model.addConstrs(
            (single_leaf_rule(model, label, tree) for (label, tree) in tree_index(model)),
            name="single_leaf"
        )

        # add left split constraints
        def left_split_r(model_obj, meta_tree_dict, label, tree, split_enc):
            split_var, split_val = \
                meta_tree_dict[label].get_branch_partition_pair(tree, split_enc)

            if not isinstance(split_val, list):
                # non-cat vars
                nu_val = model_obj._breakpoints[split_var].index(split_val)
                return quicksum(
                    model_obj._z[label, tree, leaf]
                    for leaf in meta_tree_dict[label].get_left_leaves(tree, split_enc)
                ) <= model_obj._nu[split_var, nu_val]
            else:
                # cat vars
                return quicksum(
                    model_obj._z[label, tree, leaf]
                    for leaf in meta_tree_dict[label].get_left_leaves(tree, split_enc)
                ) <= quicksum(model_obj._all_feat[split_var][cat] for cat in split_val)

        model.addConstrs(
            (left_split_r(model, self.meta_tree_dict, label, tree, encoding)
             for (label, tree, encoding) in split_index(model, self.meta_tree_dict)),
            name="left_split"
        )

        # add right split constraints
        def right_split_r(model_obj, meta_tree_dict, label, tree, split_enc):
            split_var, split_val = \
                meta_tree_dict[label].get_branch_partition_pair(tree, split_enc)

            if not isinstance(split_val, list):
                # non-cat vars
                nu_val = model_obj._breakpoints[split_var].index(split_val)
                return quicksum(
                    model_obj._z[label, tree, leaf]
                    for leaf in meta_tree_dict[label].get_right_leaves(tree, split_enc)
                ) <= 1 - model_obj._nu[split_var, nu_val]
            else:
                # cat vars
                return quicksum(
                    model_obj._z[label, tree, leaf]
                    for leaf in meta_tree_dict[label].get_right_leaves(tree, split_enc)
                ) <= 1 - quicksum(model_obj._all_feat[split_var][cat] for cat in split_val)

        model.addConstrs(
            (right_split_r(model, self.meta_tree_dict, label, tree, encoding)
             for (label, tree, encoding) in split_index(model, self.meta_tree_dict)),
            name="right_split"
        )

        # add split order constraints
        def y_order_r(model_obj, i, j):
            return model_obj._nu[i, j] <= model_obj._nu[i, j + 1]

        model.addConstrs(
            (y_order_r(model, var, j)
             for (var, j) in interval_index(model)
             if j != len(model._breakpoints[var]) - 1),
            name="y_order"
        )

        # add cat var constraints
        def cat_sums(model_obj, i):
            return quicksum(model_obj._all_feat[i][cat] for cat in model_obj._all_feat[i]) == 1

        model.addConstrs(
            (cat_sums(model, var)
             for var in self._problem_config.cat_idx),
            name="cat_sums"
        )

        # add split / space linking constraints
        def var_lower_r(model_obj, lb, i, j):
            j_bound = model_obj._breakpoints[i][j]
            return model_obj._all_feat[i] >= lb + (j_bound - lb) * (1 - model_obj._nu[i, j])

        def var_upper_r(model_obj, ub, i, j):
            j_bound = model_obj._breakpoints[i][j]
            return model_obj._all_feat[i] <= ub + (j_bound - ub) * (model_obj._nu[i, j])

        model.addConstrs(
            (var_lower_r(model, self._problem_config.feat_list[var].lb, var, j)
             for (var, j) in interval_index(model)),
            name="var_lower"
        )

        model.addConstrs(
            (var_upper_r(model, self._problem_config.feat_list[var].ub, var, j)
             for (var, j) in interval_index(model)),
            name="var_upper"
        )

        model.update()
