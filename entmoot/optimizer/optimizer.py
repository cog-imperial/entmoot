"""
Copyright (c) 2016-2020 The scikit-optimize developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

NOTE: Changes were made to the scikit-optimize source code included here. 
For the most recent version of scikit-optimize we refer to:
https://github.com/scikit-optimize/scikit-optimize/

Copyright (c) 2019-2020 Alexander Thebelt.
"""
from typing import Optional
import warnings
from numbers import Number
import numpy as np

from entmoot.acquisition import _gaussian_acquisition


class Optimizer(object):
    """
    This class is used to run the main BO loop. 
    Optimizer objects define all BO settings and store the current data set.
    
    The self.ask function provides new input proposals, and resulting black-box
    values can be added through self.tell. This procedure is automated via
    self.run, where a callable function is provided as an input.
    
    Functions self.predict_with_est and self.predict_with_acq use the current
    surrogate model to predict inputs.
    
    :param dimensions: list
        list of search-space variables, i.e.
            If (lower: float, upper: float), gives continuous variable
            If (lower: int, upper: int), gives discrete variable
            If list of categories, gives categorical variable
    :param base_estimator: str
        currently available estimator types are in ["ENTING"]
    :param n_initial_points: int
        number of initial points sampled before surrogate model is trained
    :param initial_point_generator: str
        currently supported sampling generators are in
        ["sobol", "halton", "hammersly", "lhs", "random", "grid"]
    :param num_obj: int
        gives the number of objectives that the black-box is being optimized for
    :param acq_func: str
        acquisition function type that is used for exploitation vs. exploration
        trade-off, i.e. currently supported ["LCB"] and ["LCB", "HLCB"] for
        num_obj == 1
    :param acq_optimizer: str
        optimization method used to minimize the acquisition function, i.e.
        currently supports ["sampling", "global"]
    :param random_state: Optional[int]
        fixed random seed to generate reproducible results
    :param acq_func_kwargs: Optional[dict]
        define additional params for acquisition function, i.e.
            "kappa": int, influences acquisition function "min mu - kappa * alpha"
    :param acq_optimizer_kwargs: Optional[dict]
        define additional params for acqu. optimizer "sampling":
            "n_points": int, number of points to minimize the acquisition function
        define additional params for acqu. optimizer "global":
            "env": GRBEnv, defines Gurobi environment for cluster computations
            "gurobi_timelimit": int, optimization time limit in sec
            "add_model_core": GRBModel, Gurobi optimization model that includes 
                additional constraints
    :param base_estimator_kwargs: Optional[dict]
        defines additional params that influence the base_estimator behavior
            "lgbm_params": dict, additional parameters that are passed to lightgbm
            "ensemble_type": str, options in ['GBRT', 'RF'],
                "GBRT": uses gradient-boosted tree regressor
                "RF": uses random forest
            "unc_metric": str, options in ['exploration', 'penalty'], i.e.
                negative or positive alpha contribution in "min mu \pm kappa * alpha"
            "unc_scaling": str, options in ["standard", "normalize"], i.e.
                scaling used for the distance-based uncertainty metric
            "dist_metric": str, options in ["manhattan",'squared_euclidean'], i.e.
                metric used to define non-categorical distance for uncertainty
            "cat_dist_metric": str, options in ["overlap", "goodall4", "of"], i.e.
                metric used to define categorical distance for uncertainty
    :param model_queue_size: Optional[int]
        defines number of previous models that are stored in self.models
    :param verbose: bool
        defines the verbosity level of the output
    """

    def __init__(
        self,
        dimensions: list,
        base_estimator: str = "ENTING",
        n_initial_points: int = 50,
        initial_point_generator: str = "random",
        num_obj: int = 1,
        acq_func: str = "LCB",
        acq_optimizer: str = "global",
        random_state: Optional[int] = None,
        acq_func_kwargs: Optional[dict] = None,
        acq_optimizer_kwargs: Optional[dict] = None,
        base_estimator_kwargs: Optional[dict] = None,
        model_queue_size: Optional[int] = None,
        verbose: bool = False
    ):

        from entmoot.utils import is_supported
        from entmoot.utils import cook_estimator
        from entmoot.utils import cook_initial_point_generator
        from sklearn.utils import check_random_state

        # define random state
        self.rng = check_random_state(random_state)

        # store and create acquisition function set
        self.acq_func = acq_func
        if acq_func_kwargs is None:
            self.acq_func_kwargs = dict()
        else:
            self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ["LCB","HLCB"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), self.acq_func))

        # configure counter of points
        if n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` >= 0, got %d" % n_initial_points)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points
        
        # initialize search space and output
        from entmoot.space.space import Space

        self.space = Space(dimensions)

        self._initial_samples = None
        self._initial_point_generator = \
            cook_initial_point_generator(initial_point_generator)

        if self._initial_point_generator is not None:
            transformer = self.space.get_transformer()
            self._initial_samples = self._initial_point_generator.generate(
                self.space.dimensions, n_initial_points,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max))
            self.space.set_transformer(transformer)

        self.num_obj = num_obj

        # create base_estimator
        self.base_estimator_kwargs = {} if base_estimator_kwargs is None else base_estimator_kwargs

        from entmoot.learning.tree_model import EntingRegressor,MisicRegressor

        if type(base_estimator) not in [EntingRegressor,MisicRegressor]:
            if type(base_estimator) in [str]:
                # define random_state of estimator
                est_random_state = self.rng.randint(0, np.iinfo(np.int32).max)

                # check support of base_estimator if exists
                if not is_supported(base_estimator):
                    raise ValueError("Estimator type: %s is not supported." % base_estimator)

                # build base_estimator
                base_estimator = cook_estimator(
                    self.space,
                    base_estimator,
                    self.base_estimator_kwargs,
                    num_obj=self.num_obj,
                    random_state=est_random_state)
            else:
                raise ValueError("Estimator type: %s is not supported." % base_estimator)

        self.base_estimator_ = base_estimator

        # Configure Optimizer
        self.acq_optimizer = acq_optimizer

        # record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.acq_optimizer_kwargs = acq_optimizer_kwargs
        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.gurobi_env = acq_optimizer_kwargs.get("env", None)
        self.gurobi_timelimit = acq_optimizer_kwargs.get("gurobi_timelimit", None)

        ## Initialize storage for optimization
        if not isinstance(model_queue_size, (int, type(None))):
            raise TypeError("model_queue_size should be an int or None, "
                            "got {}".format(type(model_queue_size)))

        # model cache
        self.max_model_queue_size = model_queue_size
        self.models = []

        # data set cache
        self.Xi = []
        self.yi = []

        # model_mu and model_std cache
        self.model_mu = []
        self.model_std = []

        # global opti metrics
        self.gurobi_mipgap = []

        # Initialize cache for `ask` method responses
        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points. Reset to {} at every call to `tell`.
        self.cache_ = {}

        # Handle solver print output
        if not isinstance(verbose, (int, type(None))):
            raise TypeError("verbose should be an int of [0,1,2] or bool, "
                            "got {}".format(type(verbose)))

        if isinstance(verbose, bool):
            if verbose:
                self.verbose = 1
            else:
                self.verbose = 0
        elif isinstance(verbose, int):
            if verbose not in [0,1,2]:
                raise TypeError("if verbose is int, it should in [0,1,2], "
                            "got {}".format(verbose))
            else:
                self.verbose = verbose

        # printed_switch_to_model defines if notification of switch from 
        # intitial point generation to model-based point generation has been
        # printed yet
        if self.verbose > 0:
            self.printed_switch_to_model = False
        else:
            self.printed_switch_to_model = True

    def copy(self, random_state: Optional[int] = None):
        """Create a shallow copy of an instance of the optimizer.

        :param random_state: Optional[int]
        :return optimizer: Optimizer
        """

        optimizer = Optimizer(
            dimensions=self.space.dimensions,
            base_estimator=self.base_estimator_,
            n_initial_points=self.n_initial_points_,
            initial_point_generator=self._initial_point_generator,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state,
            verbose=self.verbose
        )

        optimizer._initial_samples = self._initial_samples
        optimizer.printed_switch_to_model = self.printed_switch_to_model

        if self.Xi:
            optimizer._tell(self.Xi, self.yi)

        return optimizer

    def ask(
        self,
        n_points: int = None,
        strategy: str = "cl_min",
        weights: Optional[list] = None,
        add_model_core=None,
        gurobi_env=None,
    ) -> list:
        """
        Computes the next point (or multiple points) at which the objective
        should be evaluated.

        :param n_points: int = None,
            gives the number of points that should be returned
        :param strategy: str = "cl_min",
            determines the liar strategy, i.e.
            (https://hal.archives-ouvertes.fr/hal-00732512/document)
            used for single-objective batch proposals, i.e.
                "cl_min": str, uses minimum of observations as lie
                "cl_mean": str, uses mean of observations as lie
                "cl_max": str, uses maximum of observations as lie
        :param weights: Optional[list] = None,
            1D list of weights of size num_obj for single multi-objective proposal;
            2D list of weights of size shape(n_points,num_obj) for
                batch multi-objectiveproposals
        :param add_model_core: GRBModel = None,
            Gurobi optimization model that includes additional constraints
        :param gurobi_env: Gurobi Env = None,
            Gurobi environment used for computation of the next proposal

        :return next_x: list, next proposal of shape(n_points, n_dims)
        """

        # update gurobi_env attribute
        if gurobi_env:
            self.gurobi_env = gurobi_env

        # check if single point or batch of point is returned
        if n_points is None and weights is None:
            return self._ask(add_model_core=add_model_core)
        elif self.num_obj > 1:
            if weights is not None:
                n_points = len(weights)

            X = []
            for i in range(n_points):
                self._next_x = None

                # use the weights if provided
                if weights:
                    w = weights[i]
                    assert len(w) == self.num_obj, \
                        f"The {i}'th provided weight has dim '{len(w)}' but " \
                        f"number of objectives is '{self.num_obj}'."
                    next_x = self._ask(weight=w, add_model_core=add_model_core)
                else:
                    next_x = self._ask(add_model_core=add_model_core)
                X.append(next_x)
            return X if len(X) > 1 else X[0]

        supported_strategies = ["cl_min", "cl_mean", "cl_max"]

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of " +
                str(supported_strategies) + ", " + "got %s" % strategy
            )

        # caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # optimizer is simply discarded)
        opt = self.copy(
            random_state=self.rng.randint(0,np.iinfo(np.int32).max))

        X = []
        for i in range(n_points):
            
            x = opt.ask()
            X.append(x)

            if strategy == "cl_min":
                y_lie = np.min(opt.yi) if opt.yi else 0.0  # CL-min lie
            elif strategy == "cl_mean":
                y_lie = np.mean(opt.yi) if opt.yi else 0.0  # CL-mean lie
            else:
                y_lie = np.max(opt.yi) if opt.yi else 0.0  # CL-max lie

            opt._tell(x, y_lie)


        self.printed_switch_to_model = opt.printed_switch_to_model

        self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _ask(
        self,
        weight: Optional[list] = None,
        add_model_core = None
    ):
        """
        Computes the next point at which the objective should be evaluated.

        :param weight: Optional[list] = None,
            list of weights of size num_obj to trade-off between different
            objective contributions
        :param add_model_core: GRBModel = None,
            Gurobi optimization model that includes additional constraints

        :return next_x: list, next proposal of shape(n_points, n_dims)
        """

        if self._n_initial_points > 0 or self.base_estimator_ is None:
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.
            if self._initial_samples is None:
                return self.space.rvs(random_state=self.rng)[0]
            else:
                # The samples are evaluated starting form initial_samples[0]
                return self._initial_samples[len(self._initial_samples) - self._n_initial_points]

        elif self._next_x is not None:
            # return self._next_x if optimizer hasn't learned anything new
            return self._next_x

        else:
            # after being "told" n_initial_points we switch from sampling
            # random points to using a surrogate model

            # create fresh copy of base_estimator
            est = self.base_estimator_.copy()
            self.base_estimator_ = est

            # esimator is fitted using a generic fit function
            est.fit(self.space.transform(self.Xi), self.yi)

            # we cache the estimator in model_queue
            if self.max_model_queue_size is None:
                self.models.append(est)
            elif len(self.models) < self.max_model_queue_size:
                self.models.append(est)
            else:
                # Maximum list size obtained, remove oldest model.
                self.models.pop(0)
                self.models.append(est)

            if not self.printed_switch_to_model:
                print("")
                print("SOLVER: initial points exhausted")
                print("   -> switch to model-based optimization")
                self.printed_switch_to_model = True

            # this code provides a heuristic solution that uses sampling as the optimization strategy
            if self.acq_optimizer == "sampling":
                # sample a large number of points and then pick the best ones as
                # starting points
                X = self.space.transform(self.space.rvs(
                    n_samples=self.n_points, random_state=self.rng))

                values = _gaussian_acquisition(
                    X=X, model=est,
                    y_opt=np.min(self.yi),
                    acq_func=self.acq_func,
                    acq_func_kwargs=self.acq_func_kwargs)
                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                next_x = X[np.argmin(values)]

                # derive model mu and std
                # next_xt = self.space.transform([next_x])[0]
                next_model_mu, next_model_std = \
                    self.models[-1].predict(
                        X=np.asarray(next_x).reshape(1, -1),
                        return_std=True)

                model_mu = next_model_mu[0]
                model_std = next_model_std[0]

            # acquisition function is optimized globally
            elif self.acq_optimizer == "global":
                try:
                    import gurobipy as gp
                except ModuleNotFoundError:
                    ImportError("GurobiNotFoundError: "
                          "To run `aqu_optimizer='global'` "
                          "please install the Gurobi solver "
                          "(https://www.gurobi.com/) and its interface "
                          "gurobipy. "
                          "Alternatively, change `aqu_optimizer='sampling'`.")

                if add_model_core is None:
                    add_model_core = \
                        self.acq_optimizer_kwargs.get("add_model_core", None)

                next_x, model_mu, model_std, gurobi_mipgap = \
                    self.models[-1].get_global_next_x(acq_func=self.acq_func,
                                                      acq_func_kwargs=self.acq_func_kwargs,
                                                      acq_optimizer_kwargs=self.acq_optimizer_kwargs,
                                                      add_model_core=add_model_core,
                                                      weight=weight,
                                                      verbose=self.verbose,
                                                      gurobi_env=self.gurobi_env,
                                                      gurobi_timelimit=self.gurobi_timelimit)

                self.gurobi_mipgap.append(gurobi_mipgap)

            # note the need for [0] at the end
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

            from entmoot.utils import get_cat_idx

            for idx, xi in enumerate(self._next_x):
                if idx not in get_cat_idx(self.space):
                    self._next_x[idx] = xi

                    # enforce variable bounds
                    if self._next_x[idx] > self.space.bounds[idx][1]:
                        self._next_x[idx] = self.space.bounds[idx][1]
                    elif self._next_x[idx] < self.space.bounds[idx][0]:
                        self._next_x[idx] = self.space.bounds[idx][0]

            self._model_mu = model_mu
            self._model_std = model_std

            if self.models:
                self.model_mu.append(self._model_mu)
                self.model_std.append(self._model_std)


            # check how far the new next_x is away from existing data
            next_x = self._next_x
            min_delta_x = min([self.space.distance(next_x, xi)
                               for xi in self.Xi])
            if abs(min_delta_x) <= 1e-8:
                warnings.warn("The objective has been evaluated "
                              "at this point before.")

            # return point computed from last call to tell()
            return next_x

    def tell(self, x: list, y: list):
        """
        Checks that both x and y are valid points for the given search space.

        :param x: list, locations of new data points
        :param y: list, target value of new data points
        """

        from entmoot.utils import check_x_in_space
        check_x_in_space(x, self.space)
        self._check_y_is_valid(x, y)
        self._tell(x, y)

    def _tell(self, x, y):
        """
        Adds the new data points to the data set.

        :param x: list, locations of new data points
        :param y: list, target value of new data points
        """

        from entmoot.utils import is_listlike
        from entmoot.utils import is_2Dlistlike

        # if y isn't a scalar it means we have been handed a batch of points
        if is_listlike(y) and is_2Dlistlike(x):
            self.Xi.extend(x)
            self.yi.extend(y)
            self._n_initial_points -= len(y)
        elif is_listlike(x):
            self.Xi.append(x)
            self.yi.append(y)
            self._n_initial_points -= 1
        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible." % (type(x), type(y)))
            
        # optimizer learned something new - discard cache
        self.cache_ = {}

        # set self._next_x to None to indicate that the solver has learned something new
        self._next_x = None

    def _check_y_is_valid(self, x, y):
        """check if the shape and types of x and y are consistent."""

        from entmoot.utils import is_listlike
        from entmoot.utils import is_2Dlistlike

        # single objective checks for scalar values
        if self.num_obj == 1:
            # if y isn't a scalar it means we have been handed a batch of points
            if is_listlike(y) and is_2Dlistlike(x):
                for y_value in y:
                    if not isinstance(y_value, Number):
                        raise ValueError("expected y to be a list of scalars")

            elif is_listlike(x):
                if not isinstance(y, Number):
                    raise ValueError("`func` should return a scalar")

            else:
                raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                                 "not compatible." % (type(x), type(y)))
        else:
            # if y isn't a scalar it means we have been handed a batch of points
            if is_listlike(y[0]) and is_2Dlistlike(x):
                for y_value in y:
                    if len(y_value) != self.num_obj:
                        raise ValueError(f"expected y to be of size {self.num_obj}")
                    for yi in y_value:
                        if not isinstance(yi, Number):
                            raise ValueError(f"expected y to be a list of list-like items of length {self.num_obj}")
            elif is_listlike(x):
                if len(y) != self.num_obj:
                    raise ValueError(
                        f"`func` should return a list-like item of length {self.num_obj}")
                for yi in y:
                    if not isinstance(yi, Number):
                        raise ValueError(f"`func` should return a list-like item of length {self.num_obj}")
            else:
                raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                                 "not compatible." % (type(x), type(y)))


    def run(self, func, n_iter=1, no_progress_bar=False, update_min=False):
        """Execute ask() + tell() `n_iter` times"""
        from tqdm import tqdm

        for itr in tqdm(range(n_iter),disable=no_progress_bar):
            x = self.ask()
            self.tell(x, func(x))

            if no_progress_bar and update_min:
                print(f"Min. obj.: {round(min(self.yi),2)} at itr.: {itr+1} / {n_iter}", end="\r")

        from entmoot.utils import create_result

        result = create_result(self.Xi, self.yi, self.space, self.rng,
                               models=self.models, 
                               model_mu=self.model_mu,
                               model_std=self.model_std,
                               gurobi_mipgap=self.gurobi_mipgap)

        if no_progress_bar and update_min:
            print(f"Min. obj.: {round(min(self.yi),2)} at itr.: {n_iter} / {n_iter}")
        return result

    def update_next(self):
        """Updates the value returned by opt.ask(). Useful if a parameter
        was updated after ask was called."""
        self.cache_ = {}
        # Ask for a new next_x.
        # We only need to overwrite _next_x if it exists.
        if hasattr(self, '_next_x'):
            opt = self.copy(random_state=self.rng)
            self._next_x = opt._next_x

    def get_result(self):
        """Returns the same result that would be returned by opt.tell()
        but without calling tell

        Parameters
        ----------
        -

        Returns
        -------
        res : `OptimizeResult`, scipy object
            OptimizeResult instance with the required information.

        """
        from entmoot.utils import create_result

        result = create_result(self.Xi, self.yi, self.space, self.rng,
                               models=self.models, 
                               model_mu=self.model_mu,
                               model_std=self.model_std,
                               gurobi_mipgap=self.gurobi_mipgap)
        return result

    def predict_with_est(self, x, return_std=True):
        from entmoot.utils import is_2Dlistlike

        if is_2Dlistlike(x):
            next_x = np.asarray(
                self.space.transform(x)
            )
        else:
            next_x = np.asarray(
                self.space.transform([x])[0]
            ).reshape(1, -1)

        est = self.base_estimator_
        est.fit(self.space.transform(self.Xi), self.yi)
        temp_mu, temp_std = \
            est.predict(
                X=next_x,
                return_std=True)
        
        if is_2Dlistlike(x):
            if return_std:
                return temp_mu, temp_std
            else:
                return temp_mu
        else:
            if return_std:
                return temp_mu[0], temp_std[0]
            else:
                return temp_mu[0]

    def predict_with_acq(self, x):
        from entmoot.utils import is_2Dlistlike

        if is_2Dlistlike(x):
            next_x = np.asarray(
                self.space.transform(x)
            )
        else:
            next_x = np.asarray(
                self.space.transform([x])[0]
            ).reshape(1, -1)

        if self.models:
            temp_val = _gaussian_acquisition(
                X=next_x, model=self.models[-1], 
                y_opt=np.min(self.yi),
                acq_func=self.acq_func,
                acq_func_kwargs=self.acq_func_kwargs
            )
        else:
            est = self.base_estimator_
            est.fit(self.space.transform(self.Xi), self.yi)

            temp_val = _gaussian_acquisition(
                X=next_x, model=est, 
                y_opt=np.min(self.yi),
                acq_func=self.acq_func,
                acq_func_kwargs=self.acq_func_kwargs
            )
            
        if is_2Dlistlike(x):
            return temp_val
        else:
            return temp_val[0]

    def predict_pareto(
        self,
        sampling_strategy: str = 'random',
        num_samples: int = 10,
        num_levels: int = 10,
        add_model_core = None,
        gurobi_env=None,
    ):
        """
        Computes the next point at which the objective should be evaluated.

        :param sampling_strategy: str = 'grid'
            picks the strategy to sample weights for the muli-objective function
                'random': gives 'num_samples' randomly drawn weights that sum to 1
                'grid': defines ordered grid of samples depending on 'num_levels'
        :param num_samples: int = 10,
            defines number of samples for 'random' sampling strategy
        :param num_levels: int = 10,
            defines levels per dimension for 'grid' sampling strategy
        :param add_model_core: GRBModel = None,
            Gurobi optimization model that includes additional constraints
        :param gurobi_env: Gurobi Env = None,
            Gurobi environment used for computation of the next proposal

        :return pareto_x: list, next pareto point predictions of
            shape(n_points, n_dims)
        """

        # update gurobi_env attribute
        if gurobi_env:
            self.gurobi_env = gurobi_env

        assert self.num_obj > 1, \
            f"Number of objectives needs to be > 1 to" \
            f" compute Pareto frontiers."

        from opti.sampling.simplex import sample, grid

        # pick the sampling strategy
        if sampling_strategy == 'random':
            weights = sample(self.num_obj, num_samples)
        elif sampling_strategy == 'grid':
            weights = grid(self.num_obj, num_levels)
        else:
            raise ValueError("'sampling_type' must be in ['random', 'grid'")

        # fit current model
        est = self.base_estimator_
        est.fit(self.space.transform(self.Xi), self.yi)

        # add model constraints if necessary
        if add_model_core is None:
            add_model_core = \
                self.acq_optimizer_kwargs.get("add_model_core", None)

        # compute pareto points based on weight vector
        pareto = []
        for w in weights:
            temp_x, temp_mu, model_std, gurobi_mipgap = \
                est.get_global_next_x(acq_func=self.acq_func,
                                      acq_func_kwargs=self.acq_func_kwargs,
                                      acq_optimizer_kwargs=self.acq_optimizer_kwargs,
                                      add_model_core=add_model_core,
                                      weight=w,
                                      verbose=self.verbose,
                                      gurobi_env=self.gurobi_env,
                                      gurobi_timelimit=self.gurobi_timelimit)
            pareto.append((temp_x, temp_mu))
        return pareto
