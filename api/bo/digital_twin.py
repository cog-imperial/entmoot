import ast
import warnings
from importlib import import_module
from typing import Any, Callable, List, Optional, Union

import click
import numpy as np
import pandas as pd
from opti import Problem

from bo.algorithm import Algorithm
from bo.metric import hypervolume


class _DummyLogger:
    def set_experiment(self, _):
        pass

    def start_run(self, run_name):
        pass

    def log_params(self, _):
        pass

    def log_metric(self, _, __, step):
        pass

    def end_run(self):
        pass


def _check_args(problem: Problem, algorithm: Algorithm):
    if not isinstance(algorithm, Algorithm):
        warnings.warn("Algorithm type seems wrong")
    if not isinstance(problem, Problem):
        warnings.warn("Problem type seems wrong")


def _compute_metrics(metrics, data, problem, i, n_proposals):
    outputs_values = data[problem.outputs.names].values
    objective_values = problem.objectives(outputs_values)
    n_objectives = objective_values.shape[1]
    if metrics is None:
        if n_objectives > 1:
            nadir_point = (
                problem.objectives.bounds(problem.outputs).loc["max"].values.copy()
            )
            # Pareto
            return [
                ("hypervolume", hypervolume(objective_values, ref_point=nadir_point))
            ]
        else:
            # Single objective
            return [
                (
                    problem.objectives.names[0],
                    np.amin(objective_values[i : i + n_proposals]),
                )
            ]

    return [(m.__name__, m(problem, objective_values, i, n_proposals)) for m in metrics]


def run(
    problem: Problem,
    method: Algorithm,
    param_log_str: str = "",
    max_experiments: int = 20,
    n_proposals: int = 1,
    n_initial: int = 1,
    metrics: Optional[List[Callable[[np.ndarray], Any]]] = None,
    experiment_name: Optional[str] = "Digital Twin Benchmark",
    logger: Optional[Union[str, Any]] = "mlflow",
):
    """Run an algorithm on a problem and log results to mlflow

    Args:
        problem: optimization problem to be solved
        method: optimization algorithm
        param_log_str: free string for, e.g., parameters of the algorithm
        max_experiments: maximum number of experiments including n_initial
        n_proposals: same as batch size
        n_initial: number of initial proposals, if > 0 all data from problem will be replaced by newly generated data
        metrics: list of callables that represent metrics. Defaults to None.
        experiment_name: name of the experiment where the results are logged to.
        logger: module name or instance to be used for logging. must fulfill the mlflow interface

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): all X, all Y, all metrics
    """
    _check_args(problem, method)
    problem_name = problem.name
    method_name = type(method).__name__

    logger = _DummyLogger() if logger is None else logger

    if isinstance(logger, str):
        logger = import_module(logger)

    logger.set_experiment(experiment_name)
    logger.start_run(run_name=f"{method_name} vs. {problem_name}")
    logger.log_params(
        {
            "problem": problem_name,
            "method": method_name,
            "param_log_str": param_log_str,
            "max_experiments": max_experiments,
            "n_proposals": n_proposals,
            "n_initial": n_initial,
        }
    )

    if n_initial == 0 and len(problem.get_XY()[0]) == 0:
        raise ValueError(
            "No initial data available. Pass either n_initial > 0 or make sure your "
            "problem contains data."
        )

    if n_initial > 0:
        method.sample_initial_data(n_initial)

    all_metrics = []
    for i in range(0, max_experiments - n_initial, n_proposals):
        print(
            f"Running experiments {[i + n_initial for i in range(i, i + n_proposals)]}",
            flush=True,
        )
        X = method.propose(n_proposals=n_proposals)
        Y = problem.eval(X)
        method.add_data_and_fit(pd.concat([X, Y], axis=1))

        new_metrics = _compute_metrics(metrics, method.data, problem, i, n_proposals)
        print(new_metrics)
        all_metrics.append({"name": name, "val": val} for name, val in new_metrics)
        for name, val in new_metrics:
            logger.log_metric(name, val, step=i)
    logger.end_run()
    return method.data, all_metrics


def _try_convert_value_via_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def _parse_inputs(problem_factory, method_factory, method_kwargs):
    def get_factory(factory_path):
        factory_path_parts = factory_path.split(".")
        module, factory = ".".join(factory_path_parts[:-1]), factory_path_parts[-1]
        return getattr(import_module(module), factory), factory_path_parts

    problem_factory, problem_path_parts = get_factory(problem_factory)
    problem_name = problem_path_parts[-2]
    problem = problem_factory()
    if problem.name == "Problem":
        problem.name = problem_name
    if method_kwargs is not None:
        splitted = (kwarg.split(":") for kwarg in method_kwargs.split(","))
        method_kwargs = {
            name: _try_convert_value_via_eval(val) for name, val in splitted
        }
    else:
        method_kwargs = {}
    method_factory, _ = get_factory(method_factory)
    method = method_factory(problem, **method_kwargs)
    return problem, method


@click.command()
@click.argument("problem-factory")
@click.argument("method-factory")
@click.option(
    "--method-kwargs", help="keyword arguments of the algorithm's constructor"
)
@click.option(
    "--max-experiments",
    default=20,
    help="number of max. experiments including n-initial",
)
@click.option("--n-proposals", default=1, help="also called batch size")
@click.option(
    "--n-initial",
    default=1,
    help="number of initial proposals drawn randomly, if > 0 existing historic data is ignored",
)
@click.option(
    "--mlflow-experiment",
    default="Digital Twin Benchmark",
    help="name of the mlflow experiment where results are logged to",
)
@click.option(
    "--no-mlflow", is_flag=True, default=False, help="Avoid logging to mlflow"
)
def main(
    problem_factory,
    method_factory,
    method_kwargs,
    max_experiments,
    n_proposals,
    n_initial,
    mlflow_experiment,
    no_mlflow,
):
    """
    Benchmarks an algorithm on a problem defined by its importable path on the problem defined by
    its importable paths, i.e.,

    PROBLEM_FACTORY, e.g., opti.problems.flow_reactor_unconstrained.get_problem_pareto,

    METHOD_FACTORY, e.g., bo.algorithm.parego.ParEGO,

    respectively.

    """
    problem, method = _parse_inputs(problem_factory, method_factory, method_kwargs)

    run(
        problem,
        method,
        method_kwargs,
        max_experiments,
        n_proposals,
        n_initial,
        metrics=None,
        experiment_name=mlflow_experiment,
        logger=None if no_mlflow else "mlflow",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
