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

import os
from typing import Optional
from entmoot.space.space import Space
from gurobipy import Env
import numpy as np
from scipy.optimize import OptimizeResult
from joblib import dump as dump_
from joblib import load as load_
from .space import Space, Dimension


def get_gurobi_env():
    """Return a Gurobi CloudEnv if environment variables are set, else None."""
    if "GRB_CLOUDPOOL" in os.environ:
        return Env.CloudEnv(
            logfilename="gurobi.log",
            accessID=os.environ["GRB_CLOUDACCESSID"],
            secretKey=os.environ["GRB_CLOUDKEY"],
            pool=os.environ["GRB_CLOUDPOOL"],
        )
    return None


def is_supported(base_estimator):
    return base_estimator in ['ENTING']


def get_cat_idx(space):
    from entmoot.space.space import Space, Categorical, Integer, Real, Dimension

    if space is None:
        return []

    # collect indices of categorical features
    cat_idx = []
    for idx,dim in enumerate(space):
        if isinstance(dim, Categorical):
            cat_idx.append(idx)
    return cat_idx


def cook_estimator(
        space: Space,
        base_estimator: str,
        base_estimator_kwargs: Optional[dict] = None,
        num_obj: int = 1,
        random_state: Optional[int] = None):
    """Cook an estimator that used for mean and uncertainty prediction.

    :param space: Space, defines the search space of variables
    :param base_estimator: str
        currently available estimator types are in ["ENTING"]
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
    :param num_obj: int
        gives the number of objectives that the black-box is being optimized for
    :param random_state: Optional[int]
        fixed random seed to generate reproducible results
    """

    from lightgbm import LGBMRegressor
    from entmoot.learning.tree_model import EntingRegressor, MisicRegressor

    # collect indices of categorical features
    cat_idx = get_cat_idx(space)

    if not is_supported(base_estimator):
        raise ValueError("base_estimator is not supported.")

    # build enting estimator with distance-based uncertainty
    if base_estimator == "ENTING":

        # check uncertainty metric parameters
        unc_metric = base_estimator_kwargs.get("unc_metric", "exploration")
        unc_scaling = base_estimator_kwargs.get("unc_scaling", "standard")
        dist_metric = base_estimator_kwargs.get("dist_metric", "squared_euclidean")
        cat_dist_metric = base_estimator_kwargs.get("cat_dist_metric", "goodall4")

        unc_estimator = cook_unc_estimator(space,
                                           unc_metric=unc_metric,
                                           unc_scaling=unc_scaling,
                                           dist_metric=dist_metric,
                                           cat_dist_metric=cat_dist_metric,
                                           num_obj=num_obj,
                                           random_state=random_state)

        ensemble_type = base_estimator_kwargs.get("ensemble_type", "GBRT")
        lgbm_params = base_estimator_kwargs.get("lgbm_params", {})

        if unc_estimator.std_type == 'distance':
            tree_reg = EntingRegressor
        elif unc_estimator.std_type == 'proximity':
            tree_reg = MisicRegressor

        if ensemble_type == "GBRT":
            gbrt = []
            for _ in range(num_obj):
                gbrt.append(
                    LGBMRegressor(boosting_type='gbdt',
                                  objective='regression',
                                  verbose=-1))
            base_estimator = tree_reg(space,
                                    gbrt,
                                    unc_estimator,
                                    random_state=random_state,
                                    cat_idx=cat_idx)
        elif ensemble_type == "RF":
            rf = []
            for _ in range(num_obj):
                rf.append(
                    LGBMRegressor(boosting_type='random_forest',
                                  objective='regression',
                                  verbose=0,
                                  subsample_freq=1,
                                  subsample=0.9,
                                  bagging_seed=random_state))
            base_estimator = tree_reg(space,
                                    rf,
                                    unc_estimator,
                                    random_state=random_state,
                                    cat_idx=cat_idx)

    if base_estimator_kwargs is not None:
        base_estimator.set_params(**lgbm_params)

    return base_estimator


def cook_unc_estimator(space: Space,
                    unc_metric: str,
                    unc_scaling: str,
                    dist_metric: str,
                    cat_dist_metric: str,
                    num_obj: int = 1,
                    random_state: Optional[int] = None):
    """Cook a distance-based uncertainty estimator.

    :param space: Space, defines the search space of variables
    :param unc_metric: str, options in ['exploration', 'penalty'], i.e.
        negative or positive alpha contribution in "min mu \pm kappa * alpha"
    :param unc_scaling: str, options in ["standard", "normalize"], i.e.
        scaling used for the distance-based uncertainty metric
    :param dist_metric: str, options in ["manhattan",'squared_euclidean'], i.e.
        metric used to define non-categorical distance for uncertainty
    :param cat_dist_metric: str, options in ["overlap", "goodall4", "of"], i.e.
        metric used to define categorical distance for uncertainty
    :param num_obj: int
        gives the number of objectives that the black-box is being optimized for
    :param random_state: Optional[int]
        fixed random seed to generate reproducible results
    """

    from entmoot.learning.distance_based_std import \
        DistanceBasedExploration, DistanceBasedPenalty

    from entmoot.learning.proximit_based_std import \
        ProximityMetric

    # normalize distances if multi-obj
    if num_obj>1: unc_scaling="normalize"

    if unc_metric == "exploration":
        unc_estimator = DistanceBasedExploration(
                space,
                unc_scaling=unc_scaling,
                dist_metric=dist_metric,
                cat_dist_metric=cat_dist_metric)

    elif unc_metric == "penalty":
        unc_estimator = DistanceBasedPenalty(
                space,
                unc_scaling=unc_scaling,
                dist_metric=dist_metric,
                cat_dist_metric=cat_dist_metric)

    elif unc_metric == "prox":
        if num_obj > 1:
            raise NotImplementedError(
                "unc_metric 'prox' is currently unavailable and will be adapted for "
                "multi-objective usage.")
        unc_estimator = \
            ProximityMetric(space)

    return unc_estimator

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

def normalize_dimensions(dimensions):
    """Create a ``Space`` where all dimensions are normalized to unit range.
    This is particularly useful for Gaussian process based regressors and is
    used internally by ``gp_minimize``.
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
         NOTE: The upper and lower bounds are inclusive for `Integer`
         dimensions.
    """
    space = Space(dimensions)
    transformed_dimensions = []
    for dimension in space.dimensions:
        # check if dimension is of a Dimension instance
        if isinstance(dimension, Dimension):
            # Change the transformer to normalize
            # and add it to the new transformed dimensions
            dimension.set_transformer("normalize")
            transformed_dimensions.append(
                dimension
            )
        else:
            raise RuntimeError("Unknown dimension type "
                               "(%s)" % type(dimension))

    return Space(transformed_dimensions)