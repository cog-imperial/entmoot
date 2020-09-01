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

import numpy as np
from scipy.optimize import OptimizeResult
from joblib import dump as dump_
from joblib import load as load_

def is_supported(base_estimator):
    from entmoot.learning.tree_model import EntingRegressor
    return isinstance(base_estimator, (EntingRegressor, str, type(None)))


def cook_estimator(base_estimator, std_estimator=None, space=None, random_state=None, 
        base_estimator_params=None):
    """Cook a default estimator.

    For the special base_estimator called "DUMMY" the return value is None.
    This corresponds to sampling points at random, hence there is no need
    for an estimator.

    Parameters
    ----------
    base_estimator : "GBRT", creates LightGBM tree model based on base_estimator 

    std_estimator : DistandBasedStd instance, 
        Estimates model uncertainty of base_estimator

    space : Space instance

    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    base_estimator_params : dict
        Extra parameters provided to the base_estimator at init time.
    """

    from lightgbm import LGBMRegressor
    from entmoot.learning.tree_model import EntingRegressor

    if isinstance(base_estimator, str):
        base_estimator = base_estimator.upper()
        if base_estimator not in ["GBRT", "RF", "DUMMY"]:
            raise ValueError("Valid strings for the base_estimator parameter "
                             " are: 'GBRT', 'RF', or 'DUMMY' not "
                             "%s." % base_estimator)
    elif is_supported(base_estimator):
        base_estimator = \
            EntingRegressor(
                base_estimator=base_estimator,
                random_state=random_state
            )
    else:
        raise ValueError("base_estimator is not supported.")

    if base_estimator == "GBRT":
        gbrt = LGBMRegressor(boosting_type='gbdt',
                            objective='regression',
                            verbose=-1,
                            )
        base_estimator = EntingRegressor(base_estimator=gbrt,
                                        std_estimator=std_estimator,
                                        random_state=random_state)
    elif base_estimator == "RF":
        rf = LGBMRegressor(boosting_type='random_forest',
                            objective='regression',
                            verbose=0,
                            subsample_freq=1,
                            subsample=0.9,
                            bagging_seed= random_state
                            )
        base_estimator = EntingRegressor(base_estimator=rf,
                                        std_estimator=std_estimator,
                                        random_state=random_state)

    elif base_estimator == "DUMMY":
        return None

    if base_estimator_params is not None:
        base_estimator.set_params(**base_estimator_params)
        
    return base_estimator

def cook_std_estimator(std_estimator, 
                    space=None, 
                    random_state=None, 
                    std_estimator_params=None):
    """Cook a default uncertainty estimator.

    Parameters
    ----------
    std_estimator : string.
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

    space : Space instance

    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    std_estimator_params : dict
        Extra parameters provided to the std_estimator at init time.
    """
    from entmoot.learning.distance_based_std import \
        DistanceBasedExploration, DistanceBasedPenalty

    if std_estimator == "BDD":
        std_estimator = \
            DistanceBasedExploration(
                metric="sq_euclidean",
            )

    elif std_estimator == "DDP":
        std_estimator = \
            DistanceBasedPenalty(
                metric="sq_euclidean",
            )

    elif std_estimator == "L1BDD":
        std_estimator = \
            DistanceBasedExploration(
                metric="manhattan",
            )

    elif std_estimator == "L1DDP":
        std_estimator = \
            DistanceBasedPenalty(
                metric="manhattan",
            )

    std_estimator.set_params(**std_estimator_params)
    return std_estimator

def cook_initial_point_generator(generator, **kwargs):
    """Cook a default initial point generator.

    For the special generator called "random" the return value is None.

    Parameters
    ----------
    generator : "lhs", "sobol", "halton", "hammersly", "grid", "random" \
            or InitialPointGenerator instance"
        Should inherit from `skopt.sampler.InitialPointGenerator`.

    kwargs : dict
        Extra parameters provided to the generator at init time.
    """
    from entmoot.sampler import Sobol, Lhs, Hammersly, Halton, Grid
    from entmoot.sampler import InitialPointGenerator

    if generator is None:
        generator = "random"
    elif isinstance(generator, str):
        generator = generator.lower()
        if generator not in ["sobol", "halton", "hammersly", "lhs", "random",
                             "grid"]:
            raise ValueError("Valid strings for the generator parameter "
                             " are: 'sobol', 'lhs', 'halton', 'hammersly',"
                             "'random', or 'grid' not "
                             "%s." % generator)
    elif not isinstance(generator, InitialPointGenerator):
        raise ValueError("generator has to be an InitialPointGenerator."
                         "Got %s" % (str(type(generator))))

    if isinstance(generator, str):
        if generator == "sobol":
            generator = Sobol()
        elif generator == "halton":
            generator = Halton()
        elif generator == "hammersly":
            generator = Hammersly()
        elif generator == "lhs":
            generator = Lhs()
        elif generator == "grid":
            generator = Grid()
        elif generator == "random":
            return None
    generator.set_params(**kwargs)
    return generator

def is_listlike(x):
    return isinstance(x, (list, tuple))


def is_2Dlistlike(x):
    return np.all([is_listlike(xi) for xi in x])


def check_x_in_space(x, space):
    if is_2Dlistlike(x):
        if not np.all([p in space for p in x]):
            raise ValueError("Not all points are within the bounds of"
                             " the space.")
        if any([len(p) != len(space.dimensions) for p in x]):
            raise ValueError("Not all points have the same dimensions as"
                             " the space.")
    elif is_listlike(x):
        if x not in space:
            raise ValueError("Point (%s) is not within the bounds of"
                             " the space (%s)."
                             % (x, space.bounds))
        if len(x) != len(space.dimensions):
            raise ValueError("Dimensions of point (%s) and space (%s) do not match"
                             % (x, space.bounds))

def create_result(Xi, yi, space=None, rng=None, specs=None, models=None,
                    model_mu=None, model_std=None, gurobi_mipgap=None):
    """
    Initialize an `OptimizeResult` object.

    Parameters
    ----------
    Xi : list of lists, shape (n_iters, n_features)
        Location of the minimum at every iteration.

    yi : array-like, shape (n_iters,)
        Minimum value obtained at every iteration.

    space : Space instance, optional
        Search space.

    rng : RandomState instance, optional
        State of the random state.

    specs : dict, optional
        Call specifications.

    models : list, optional
        List of fit surrogate models.

    Returns
    -------
    res : `OptimizeResult`, scipy object
        OptimizeResult instance with the required information.
    """
    res = OptimizeResult()
    yi = np.asarray(yi)
    if np.ndim(yi) == 2:
        res.log_time = np.ravel(yi[:, 1])
        yi = np.ravel(yi[:, 0])
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]
    res.func_vals = yi
    res.x_iters = Xi
    res.models = models
    res.model_mu = model_mu
    res.model_std = model_std
    res.gurobi_mipgap = gurobi_mipgap
    res.space = space
    res.random_state = rng
    res.specs = specs
    return res

def dump(res, filename, store_objective=True, **kwargs):
    """
    Store an skopt optimization result into a file.

    Parameters
    ----------
    res : `OptimizeResult`, scipy object
        Optimization result object to be stored.

    filename : string or `pathlib.Path`
        The path of the file in which it is to be stored. The compression
        method corresponding to one of the supported filename extensions ('.z',
        '.gz', '.bz2', '.xz' or '.lzma') will be used automatically.

    store_objective : boolean, default=True
        Whether the objective function should be stored. Set `store_objective`
        to `False` if your objective function (`.specs['args']['func']`) is
        unserializable (i.e. if an exception is raised when trying to serialize
        the optimization result).

        Notice that if `store_objective` is set to `False`, a deep copy of the
        optimization result is created, potentially leading to performance
        problems if `res` is very large. If the objective function is not
        critical, one can delete it before calling `skopt.dump()` and thus
        avoid deep copying of `res`.

    **kwargs : other keyword arguments
        All other keyword arguments will be passed to `joblib.dump`.
    """
    if store_objective:
        dump_(res, filename, **kwargs)

    elif 'func' in res.specs['args']:
        # If the user does not want to store the objective and it is indeed
        # present in the provided object, then create a deep copy of it and
        # remove the objective function before dumping it with joblib.dump.
        res_without_func = deepcopy(res)
        del res_without_func.specs['args']['func']
        dump_(res_without_func, filename, **kwargs)

    else:
        # If the user does not want to store the objective and it is already
        # missing in the provided object, dump it without copying.
        dump_(res, filename, **kwargs)


def load(filename, **kwargs):
    """
    Reconstruct a skopt optimization result from a file
    persisted with skopt.dump.

    .. note::
        Notice that the loaded optimization result can be missing
        the objective function (`.specs['args']['func']`) if `skopt.dump`
        was called with `store_objective=False`.

    Parameters
    ----------
    filename : string or `pathlib.Path`
        The path of the file from which to load the optimization result.

    **kwargs : other keyword arguments
        All other keyword arguments will be passed to `joblib.load`.

    Returns
    -------
    res : `OptimizeResult`, scipy object
        Reconstructed OptimizeResult instance.
    """
    return load_(filename, **kwargs)