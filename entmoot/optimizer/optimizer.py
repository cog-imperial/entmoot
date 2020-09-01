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

from entmoot.acquisition import _gaussian_acquisition

import sys

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

        *NOTE: `Integer` and `Categorical` variables are not yet supported and
        will be added in the next release.

    base_estimator : string, default: "GBRT",
        A default LightGBM surrogate model of the corresponding type is used  
        minimize `func`.
        The following model types are available:
        
        - "GBRT" for gradient-boosted trees

        - "RF" for random forests

        *NOTE: More model types will be added in the future

    std_estimator : string, default: "BDD",
        A model is used to estimate uncertainty of `base_estimator`.
        Different types can be classified as exploration measures, i.e. move
        as far as possible from reference points, and penalty measures, i.e.
        stay as close as possible to reference points. Within these types, the 
        following uncertainty estimators are available:
        
        - exploration:
            - "BDD" for bounded-data distance, which uses squared euclidean
              distance to standardized data points
            - "L1BDD" for bounded-data distance, which uses manhattan
              distance to standardized data points
        
        - penalty:
            - "DDP" for data distance, which uses squared euclidean
              distance to standardized data points
            - "L1DDP" for data distance, which uses manhattan
              distance to standardized data points

        *NOTE: More model uncertainty estimators will be added in the future

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

    std_estimator_kwargs : dict
        Additional arguments to be passed to the std_estimator.

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
                base_estimator="GBRT",
                std_estimator="BDD", 
                n_initial_points=50, 
                initial_point_generator="random",
                acq_func="LCB", 
                acq_optimizer="global", 
                random_state=None, 
                acq_func_kwargs=None, 
                acq_optimizer_kwargs=None,
                base_estimator_kwargs=None,
                std_estimator_kwargs=None,
                model_queue_size=None,
                verbose=False
    ):

        from entmoot.utils import is_supported
        from entmoot.utils import cook_estimator
        from entmoot.utils import cook_initial_point_generator
        from entmoot.utils import cook_std_estimator
        

        self.specs = {"args": copy.copy(inspect.currentframe().f_locals),
                      "function": "Optimizer"}

        from sklearn.utils import check_random_state

        self.rng = check_random_state(random_state)
        
        # Configure acquisition function

        # Store and create acquisition function set
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        allowed_acq_funcs = ["LCB"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), self.acq_func))

        # Configure counter of points

        if n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` >= 0, got %d" % n_initial_points)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # Configure std estimator

        self.std_estimator = std_estimator

        if std_estimator_kwargs is None:
            std_estimator_kwargs = dict()
        
        allowed_std_est = ["BDD","BCD","DDP","L1BDD","L1DDP"]
        if self.std_estimator not in allowed_std_est:
            raise ValueError("expected std_estimator to be in %s, got %s" %
                             (",".join(allowed_std_est), self.std_estimator))

        # build std_estimator
        import numpy as np
        est_random_state = self.rng.randint(0, np.iinfo(np.int32).max)
        self.std_estimator = cook_std_estimator(
            std_estimator,
            space=dimensions,
            random_state=est_random_state,
            std_estimator_params=std_estimator_kwargs)

        # Configure estimator

        # check support of base_estimator if exists
        if not is_supported(base_estimator):
            raise ValueError(
                "Estimator type: %s is not supported." % base_estimator)

        if base_estimator_kwargs is None:
            base_estimator_kwargs = dict()

        # build base_estimator
        base_estimator = cook_estimator(
            base_estimator, 
            self.std_estimator, 
            space=dimensions,
            random_state=est_random_state,
            base_estimator_params=base_estimator_kwargs)

        self.base_estimator_ = base_estimator

        # Configure Optimizer

        self.acq_optimizer = acq_optimizer

        # record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        if std_estimator_kwargs is None:
            std_estimator_kwargs = dict()

        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)

        self.gurobi_timelimit = acq_optimizer_kwargs.get("gurobi_timelimit", None)

        # Configure search space
        
        # initialize search space
        import numpy as np
        from entmoot.space.space import Space
        from entmoot.space.space import Categorical

        self.space = Space(dimensions)

        self._initial_samples = None
        self._initial_point_generator = cook_initial_point_generator(
            initial_point_generator)

        if self._initial_point_generator is not None:
            transformer = self.space.get_transformer()
            self._initial_samples = self._initial_point_generator.generate(
                self.space.dimensions, n_initial_points,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max))
            self.space.set_transformer(transformer)

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

    def ask(self, n_points=None, strategy="cl_min"):
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
        if n_points is None:
            return self._ask()

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

    def _ask(self):
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
                return self._initial_samples[
                    len(self._initial_samples) - self._n_initial_points]

        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no "
                                   "model has been fit.")

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
        import numpy as np

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

        if self.models:
            temp_mu, temp_std = \
                self.models[-1].predict(
                    X=np.asarray(self._next_x).reshape(1, -1), 
                    return_std=True)
            self.model_mu.append(temp_mu)
            self.model_std.append(temp_std)
            
        # optimizer learned something new - discard cache
        self.cache_ = {}

        from sklearn.base import clone

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if (fit and self._n_initial_points <= 0 and
                self.base_estimator_ is not None):

            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator_)

            # esimator is fitted using a generic fit function
            est.fit(self.space.transform(self.Xi), self.yi)

            if self.max_model_queue_size is None:
                self.models.append(est)
            elif len(self.models) < self.max_model_queue_size:
                self.models.append(est)
            else:
                # Maximum list size obtained, remove oldest model.
                self.models.pop(0)
                self.models.append(est)

            # sample a large number of points and then pick the best ones as 
            # starting points
            X = self.space.transform(self.space.rvs(
                n_samples=self.n_points, random_state=self.rng))

            if not self.printed_switch_to_model:
                print("")
                print("SOLVER: initial points exhausted")
                print("   -> switch to model-based optimization")
                self.printed_switch_to_model = True

            if self.acq_optimizer == "sampling":
                values = _gaussian_acquisition(
                    X=X, model=est, 
                    y_opt=np.min(self.yi),
                    acq_func=self.acq_func,
                    acq_func_kwargs=self.acq_func_kwargs)
                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                next_x = X[np.argmin(values)]

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

                from entmoot.optimizer.gurobi_utils import \
                    add_core_to_gurobi_model, add_gbm_to_gurobi_model, \
                    add_std_to_gurobi_model, add_acq_to_gurobi_model, \
                    set_gurobi_init_to_ref

                # suppress  output to command window
                import logging
                logger = logging.getLogger()
                logger.setLevel(logging.CRITICAL)

                # start building model
                gurobi_model = gp.Model('seq_opt')

                if self.verbose == 2:
                    print("")
                    print("")
                    print("")
                    print("SOLVER: *** start gurobi solve ***")
                    gurobi_model.Params.LogToConsole=1
                else:
                    gurobi_model.Params.LogToConsole=0
                add_core_to_gurobi_model(self.space, gurobi_model)

                # convert into gbm_model format
                # and add to gurobi model
                gbm_model_dict = {}
                gbm_model_dict['first'] = est.get_gbm_model()
                add_gbm_to_gurobi_model(gbm_model_dict, gurobi_model)

                # add std estimator to gurobi model
                add_std_to_gurobi_model(est, gurobi_model)

                # add obj to gurobi model
                add_acq_to_gurobi_model(gurobi_model, est,
                    acq_func_kwargs=self.acq_func_kwargs)

                # set initial gurobi model vars to std_est reference points
                set_gurobi_init_to_ref(est, gurobi_model)

                # set gurobi time limit
                if self.gurobi_timelimit is not None:
                    gurobi_model.Params.TimeLimit= self.gurobi_timelimit
                gurobi_model.Params.OutputFlag=1
                
                gurobi_model.optimize()

                # store optimality gap of gurobi computation
                self.gurobi_mipgap.append(gurobi_model.mipgap)

                next_x = np.asarray(
                    [gurobi_model._c_x[i_x].x 
                    for i_x in range(len(gurobi_model._c_x))])      

            # note the need for [0] at the end
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

        # Pack results
        from entmoot.utils import create_result

        result = create_result(self.Xi, self.yi, self.space, self.rng,
                               models=self.models,
                               model_mu=self.model_mu,
                               model_std=self.model_std,
                               gurobi_mipgap=self.gurobi_mipgap)
    
        result.specs = self.specs
        return result

    def _check_y_is_valid(self, x, y):
        """Check if the shape and types of x and y are consistent."""

        from entmoot.utils import is_listlike
        from entmoot.utils import is_2Dlistlike

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

    def run(self, func, n_iter=1, no_progress_bar=False):
        """Execute ask() + tell() `n_iter` times"""
        from tqdm import tqdm

        for itr in tqdm(range(n_iter),disable=no_progress_bar):
            x = self.ask()
            self.tell(x, func(x), fit= not itr == n_iter-1)

        from entmoot.utils import create_result

        result = create_result(self.Xi, self.yi, self.space, self.rng,
                               models=self.models, 
                               model_mu=self.model_mu,
                               model_std=self.model_std,
                               gurobi_mipgap=self.gurobi_mipgap)
        result.specs = self.specs
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