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

import warnings
import copy
import inspect
from numbers import Number
import numpy as np
from sklearn.base import clone

from entmoot.acquisition import _gaussian_acquisition


class Optimizer(object):
    """Run bayesian optimisation loop.

    An `Optimizer` represents the steps of a bayesian optimisation loop. To
    use it you need to provide your own loop mechanism. The various
    optimisers provided by `skopt` use this class under the hood.

    Use this class directly if you want to control the iterations of your
    bayesian optimisation loop.

    Parameters
    ----------
    dimensions : list, shape (n_dims,)
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    base_estimator : string, default: "GBRT",
        A default LightGBM surrogate model of the corresponding type is used  
        minimize `func`.
        The following model types are available:
        
        - "GBRT" for gradient-boosted trees

        - "RF" for random forests

        *NOTE: More model types will be added in the future

    n_initial_points : int, default: 50
        Number of evaluations of `func` with initialization points
        before approximating it with `base_estimator`. Initial point
        generator can be changed by setting `initial_point_generator`. For 
        `n_initial_points` <= 20 we need to set `min_child_samples` <= 20 in 
        `base_estimator_kwargs` so LightGBM can train tree models based on small
        data sets.

    initial_point_generator : str, InitialPointGenerator instance, \
            default: "random"
        Sets a initial points generator. Can be either

        - "random" for uniform random numbers,
        - "sobol" for a Sobol sequence,
        - "halton" for a Halton sequence,
        - "hammersly" for a Hammersly sequence,
        - "lhs" for a latin hypercube sequence,
        - "grid" for a uniform grid sequence

    acq_func : string, default: "LCB"
        Function to minimize over the posterior distribution. Can be either

        - "LCB" for lower confidence bound.

    acq_optimizer : string, default: "sampling"
        Method to minimize the acquisition function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to "sampling", then `acq_func` is optimized by computing
          `acq_func` at `n_points` randomly sampled points.
        - If set to "global", then `acq_optimizer` is optimized by using
          global solver to find minimum of `acq_func`.

    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    acq_func_kwargs : dict
        Additional arguments to be passed to the acquisition function.

    acq_optimizer_kwargs : dict
        Additional arguments to be passed to the acquisition optimizer.

    base_estimator_kwargs : dict
        Additional arguments to be passed to the base_estimator.

    model_queue_size : int or None, default: None
        Keeps list of models only as long as the argument given. In the
        case of None, the list has no capped length.

    verbose : bool or int:
        - If it is `True`, general solver information is printed at every 
          iteration
        - If it is `False`, no output is printed

        - If it is 0, same as if `False`
        - If it is 1, same as if `True`
        - If it is 2, general solver information is printed at every iteration
          as well as detailed solver information, i.e. gurobi log

    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    models : list
        Regression models used to fit observations and compute acquisition
        function.
    space : Space
        An instance of :class:`skopt.space.Space`. Stores parameter search
        space used to sample points, bounds, and type of parameters.
    """
    def __init__(self, 
                dimensions, 
                base_estimator="ENTING",
                n_initial_points=50, 
                initial_point_generator="random",
                num_obj=1,
                acq_func="LCB", 
                acq_optimizer="global", 
                random_state=None, 
                acq_func_kwargs=None, 
                acq_optimizer_kwargs=None,
                base_estimator_kwargs=None,
                model_queue_size=None,
                verbose=False
    ):

        from entmoot.utils import is_supported
        from entmoot.utils import cook_estimator
        from entmoot.utils import cook_initial_point_generator
        

        self.specs = {"args": copy.copy(inspect.currentframe().f_locals),
                      "function": "Optimizer"}

        from sklearn.utils import check_random_state

        self.rng = check_random_state(random_state)

        # Store and create acquisition function set
        self.acq_func = acq_func
        if acq_func_kwargs is None:
            self.acq_func_kwargs = dict()
        else:
            self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ["LCB","HLCB"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), self.acq_func))

        # Configure counter of points
        if n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` >= 0, got %d" % n_initial_points)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points
        
        # initialize search space
        import numpy as np
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

        # Initialize storage for optimization
        if not isinstance(model_queue_size, (int, type(None))):
            raise TypeError("model_queue_size should be an int or None, "
                            "got {}".format(type(model_queue_size)))

        self.max_model_queue_size = model_queue_size
        self.models = []
        self.Xi = []
        self.yi = []
        self.model_mu = []
        self.model_std = []
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

    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer.

        Parameters
        ----------
        random_state : int, RandomState instance, or None (default)
            Set the random state of the copy.

        Returns
        -------
        optimizer : Optimizer instance
            Shallow copy of optimizer instance
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

    def ask(self, n_points=None, strategy="cl_min", weights=None, add_model_core=None):
        """Query point or multiple points at which objective should be evaluated.

        Parameters
        ----------
        n_points : int or None, default: None
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.

        strategy : string, default: "cl_min"
            Method to use to sample multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.

            - If set to `"cl_min"`, then constant liar strategy is used
               with lie objective value being minimum of observed objective
               values. `"cl_mean"` and `"cl_max"` means mean and max of values
               respectively. For details on this strategy see:

               https://hal.archives-ouvertes.fr/hal-00732512/document

               With this strategy a copy of optimizer is created, which is
               then asked for a point, and the point is told to the copy of
               optimizer with some fake objective (lie), the next point is
               asked from copy, it is also told to the copy with fake
               objective and so on. The type of lie defines different
               flavours of `cl_x` strategies.

        Returns
        -------
        next_x : array-like, shape(n_points, n_dims)
            If `n_points` is None, only a single array-like object is returned
        """
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

        # Caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # optimizer is simply discarded)
        import numpy as np

        opt = self.copy(random_state=self.rng.randint(0,
                                                      np.iinfo(np.int32).max))

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

            fit_model = not i == n_points-1
            opt._tell(x, y_lie, fit=fit_model)


        self.printed_switch_to_model = opt.printed_switch_to_model

        self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _ask(self, weight=None, add_model_core=None):
        """Suggest next point at which to evaluate the objective.

        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_estimator` is used
        to determine the next point.

        Parameters
        ----------
        -

        Returns
        -------
        next_x : array-like, shape(n_dims,)
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
                    print("GurobiNotFoundError: "
                          "To run `aqu_optimizer='global'` "
                          "please install the Gurobi solver "
                          "(https://www.gurobi.com/) and its interface "
                          "gurobipy. "
                          "Alternatively, change `aqu_optimizer='sampling'`.")
                    import sys
                    sys.exit(1)

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
                    self._next_x[idx] = round(xi, 5)

                    # enforce variable bounds
                    if self._next_x[idx] > self.space.transformed_bounds[idx][1]:
                        self._next_x[idx] = self.space.transformed_bounds[idx][1]
                    elif self._next_x[idx] < self.space.transformed_bounds[idx][0]:
                        self._next_x[idx] = self.space.transformed_bounds[idx][0]

            self._model_mu = round(model_mu, 5)
            self._model_std = round(model_std, 5)

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

    def tell(self, x, y, fit=True):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by
        `ask()` or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
        `ask()`.

        To add observations without fitting a new model set `fit` to False.

        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.

        Parameters
        ----------
        x : list or list-of-lists
            Point at which objective was evaluated.

        y : scalar or list
            Value of objective at `x`.

        fit : bool, default: True
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.

        Returns
        -------
        res : `OptimizeResult`, scipy object
               OptimizeResult instance with the required information.
        """
        from entmoot.utils import check_x_in_space
        check_x_in_space(x, self.space)
        self._check_y_is_valid(x, y)

        return self._tell(x, y, fit=fit)

    def _tell(self, x, y, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.

        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation.

        Parameters
        ----------
        x : list or list-of-lists
            Point at which objective was evaluated.

        y : scalar or list
            Value of objective at `x`.

        fit : bool, default: True
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.

        Returns
        -------
        res : `OptimizeResult`, scipy object
               OptimizeResult instance with the required information.
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
        """Check if the shape and types of x and y are consistent."""

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
            self.tell(x, func(x), fit= not itr == n_iter-1)

            if no_progress_bar and update_min:
                print(f"Min. obj.: {round(min(self.yi),2)} at itr.: {itr+1} / {n_iter}", end="\r")

        from entmoot.utils import create_result

        result = create_result(self.Xi, self.yi, self.space, self.rng,
                               models=self.models, 
                               model_mu=self.model_mu,
                               model_std=self.model_std,
                               gurobi_mipgap=self.gurobi_mipgap)
        result.specs = self.specs

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
        result.specs = self.specs
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
