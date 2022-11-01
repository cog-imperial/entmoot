from entmoot.models.base_model import BaseModel
import warnings


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

    def fit(self, X, y):

        for i, obj in enumerate(self._space.obj_list):
            if self._train_lib == "lgbm":
                tree_model = self._train_lgbm(X, yi)
            elif self._train_lib == "catboost":
                raise NotImplementedError()
            elif self._train_lib == "xgboost":
                raise NotImplementedError()
            else:
                raise IOError(f"Parameter 'train_lib' for tree ensembles needs to be "
                              f"in '('lgbm', 'catboost', 'xgboost')'.")

    def _train_lgbm(self, X, y):
        import lightgbm as lgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._space.cat_idx:
                train_data = lgb.Dataset(X, label=y,
                                         categorical_feature=self._space.cat_idx,
                                         free_raw_data=False,
                                         params={'verbose': -1})

                tree_model = lgb.train(self._train_params, train_data,
                                       categorical_feature=self._space.cat_idx,
                                       verbose_eval=False)
            else:
                train_data = lgb.Dataset(X, label=y,
                                         params={'verbose': -1})

                tree_model = lgb.train(self._train_params, train_data,
                                       verbose_eval=False)

        return tree_model
