from entmoot.models.base_model import BaseModel
import warnings
import numpy as np


class TreeEnsemble(BaseModel):

    def __init__(self, space, params=None):

        if params is None:
            params = {}

        self._space = space
        self._train_lib = params.get("train_lib", "lgbm")
        self._rnd_seed = space.rnd_seed

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

        self._tree_list = None

    @property
    def tree_list(self):
        assert self._tree_list is not None, \
            "No tree model is trained yet. Call '.fit(X, y)' first."
        return self._tree_list

    def fit(self, X, y):

        # check dims of X and y
        if X.ndim == 1:
            X = np.atleast_2d(X)

        assert X.shape[-1] == len(self._space.feat_list), \
            f"Argument 'X' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._space.feat_list)})', got '{X.shape}'."

        if y.ndim == 1:
            y = np.atleast_2d(y)

        assert y.shape[-1] == len(self._space.obj_list), \
            f"Argument 'y' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._space.obj_list)})', got '{y.shape}'."

        # train tree models for every objective
        if self._tree_list is None:
            self._tree_list = []

        for i, obj in enumerate(self._space.obj_list):
            if self._train_lib == "lgbm":
                tree_model = self._train_lgbm(X, y[:, i])
            elif self._train_lib == "catboost":
                raise NotImplementedError()
            elif self._train_lib == "xgboost":
                raise NotImplementedError()
            else:
                raise IOError(f"Parameter 'train_lib' for tree ensembles needs to be "
                              f"in '('lgbm', 'catboost', 'xgboost')'.")
            self._tree_list.append(tree_model)

    def _train_lgbm(self, X, y):
        import lightgbm as lgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._space.cat_idx:
                # train for categorial vars
                train_data = lgb.Dataset(X, label=y,
                                         categorical_feature=self._space.cat_idx,
                                         free_raw_data=False,
                                         params={'verbose': -1})

                tree_model = lgb.train(self._train_params, train_data,
                                       categorical_feature=self._space.cat_idx,
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

        assert X.shape[-1] == len(self._space.feat_list), \
            f"Argument 'X' has wrong dimensions. " \
            f"Expected '(num_samples, {len(self._space.feat_list)})', got '{X.shape}'."

        # predict vals
        tree_pred = []
        for tree_model in self.tree_list:
            tree_pred.append(tree_model.predict(X))

        return np.squeeze(np.column_stack(tree_pred))
