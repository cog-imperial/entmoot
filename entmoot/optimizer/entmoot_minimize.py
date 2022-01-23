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

from entmoot.optimizer.optimizer import Optimizer

import copy
import inspect
import numbers

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

def entmoot_minimize(
    func,
    dimensions,
    n_calls=60,
    batch_size=None,
    batch_strategy="cl_mean",
    n_points=10000,
    base_estimator="GBRT",
    n_initial_points=50, 
    initial_point_generator="random",
    acq_func="LCB", 
    acq_optimizer="global",
    x0=None,
    y0=None,  
    random_state=None, 
    acq_func_kwargs=None, 
    acq_optimizer_kwargs=None,
    base_estimator_kwargs=None,
    model_queue_size=None,
    verbose=False,
):
    """Run bayesian optimisation loop.

    An `Optimizer` is initialized which represents the steps of a bayesian 
    optimisation loop. To use it you need to provide your own loop mechanism. 

    Parameters specify how `Optimizer` is initialized, which function is 
    optimized and how each iteration is conducted.

    Parameters
    ----------
    func : BenchmarkFunction instance,
        Use class entmoot.benchmarks.BenchmarkFunction as a base to incorporate
        own black-box functions

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

    n_calls : int, default: 60
        Budget of func calls that shall be used. `n_calls` includes initial
        points as well as suggested model-based points.

    batch_size : int, default: None
        Number of suggestions given per sequential iteration. For `None` only
        one suggestion per iteration is given.

    batch_strategy : string, default: "cl_mean"
        In case `batch_size` is not None, `batch_strategy` specifies the 
        strategy used to suggest multiple points per iteration. Available
        strategies are ["cl_min", "cl_mean", "cl_max"]

    n_points : int, default: 10,000
        For `acq_optimizer` = "sampling", `n_points` specifies the number of
        random points used to sample the acquisition function to derive its
        minimum.

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

    x0 : list, list of lists or `None`
        Initial input points.
        - If it is a list of lists, use it as a list of input points.
        - If it is a list, use it as a single initial input point.
        - If it is `None`, no initial input points are used.

    y0 : list, scalar or `None`
        Evaluation of initial input points.
        - If it is a list, then it corresponds to evaluations of the function
          at each element of `x0` : the i-th element of `y0` corresponds
          to the function evaluated at the i-th element of `x0`.
        - If it is a scalar, then it corresponds to the evaluation of the
          function at `x0`.
        - If it is None and `x0` is provided, then the function is evaluated
          at each element of `x0`.

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

    Returns
    -------
    result : `OptimizeResult`, scipy object
        OptimizeResult instance with the required information.
    """

    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    if acq_optimizer_kwargs is None:
        acq_optimizer_kwargs = {}

    acq_optimizer_kwargs["n_points"] = n_points

    # Initialize optimization
    # Suppose there are points provided (x0 and y0), record them

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    if n_initial_points <= 0 and not x0:
        raise ValueError("Either set `n_initial_points` > 0,"
                         " or provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    required_calls = n_initial_points + (len(x0) if not y0 else 0)
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))
    # calculate the total number of initial points
    n_initial_points = n_initial_points + len(x0)

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(
        dimensions, 
        base_estimator=base_estimator,
        n_initial_points=n_initial_points, 
        initial_point_generator=initial_point_generator,
        acq_func=acq_func, 
        acq_optimizer=acq_optimizer,
        random_state=random_state, 
        acq_func_kwargs=acq_func_kwargs, 
        acq_optimizer_kwargs=acq_optimizer_kwargs,
        base_estimator_kwargs=base_estimator_kwargs,
        model_queue_size=model_queue_size,
        verbose=verbose
    )

    # Record provided points

    # create return object
    result = None
    # evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)
    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        result = optimizer.tell(x0, y0)
        result.specs = specs

    # Handle solver output
    if not isinstance(verbose, (int, type(None))):
        raise TypeError("verbose should be an int of [0,1,2] or bool, "
                        "got {}".format(type(verbose)))

    if isinstance(verbose, bool):
        if verbose:
            verbose = 1
        else:
            verbose = 0
    elif isinstance(verbose, int):
        if verbose not in [0,1,2]:
            raise TypeError("if verbose is int, it should in [0,1,2], "
                        "got {}".format(verbose))

    # Optimize
    _n_calls = n_calls

    itr = 1

    if verbose > 0:
        print("")
        print("SOLVER: start solution process...")
        print("")
        print(f"SOLVER: generate \033[1m {n_initial_points}\033[0m initial points...")


    while _n_calls>0:

        # check if optimization is performed in batches
        if batch_size is not None:
            _batch_size = min([_n_calls,batch_size])
            _n_calls -= _batch_size
            next_x = optimizer.ask(_batch_size,strategy=batch_strategy)
        else:
            _n_calls -= 1
            next_x = optimizer.ask(strategy=batch_strategy)
            
        next_y = func(next_x)

        # first iteration uses next_y as best point instead of min of next_y
        if itr == 1:
            if batch_size is None:
                best_fun = next_y
            else:
                best_fun = min(next_y)

        # handle output print at every iteration
        if verbose > 0:
            print("")
            print(f"\033[1m itr_{itr}\033[0m")

            if isinstance(next_y,Iterable):
                # in case of batch optimization, print all new proposals and 
                # mark improvements of objectives with (*)
                print_str = []
                for y in next_y:
                    if y <= best_fun:
                        print_str.append(f"\033[1m{round(y,5)}\033[0m (*)")
                    else:
                        print_str.append(str(round(y,5)))
                print(f"   new points obj.: {print_str[0]}")
                for y_str in print_str[1:]:
                    print(f"                    {y_str}")    
            else:
                # in case of single point sequential optimization, print new 
                # point proposal
                if next_y <= best_fun:
                    print_str = f"\033[1m itr_{round(next_y,5)}\033[0m"
                else:
                    print_str = str(round(next_y,5))
                print(f"   new point obj.: {round(next_y,5)}")

            # print best obj until (not including) current iteration
            print(f"   best obj.:       {round(best_fun,5)}")

        itr += 1

        optimizer.tell(
            next_x, next_y,
            fit= batch_size is None and not _n_calls <=0
        )

        result = optimizer.get_result()

        best_fun = result.fun  

        result.specs = specs
    
    # print end of solve once convergence criteria is met
    if verbose>0:
        print("")
        print("SOLVER: finished run!")
        print(f"SOLVER: best obj.: {round(result.fun,5)}")
        print("")

    return result