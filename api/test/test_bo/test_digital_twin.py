import numpy as np
import pandas as pd
from click.testing import CliRunner
from opti.problems import (
    ZDT1,
    Cake,
    Detergent,
    FlowReactor384Rf,
    FlowReactorUnconstrainedInterp,
    FlowReactorUnconstrainedRf,
    noisify_problem_with_gaussian,
)

from bo.algorithm import Algorithm, RandomSearch
from bo.digital_twin import (
    _check_args,
    _DummyLogger,
    _parse_inputs,
    _try_convert_value_via_eval,
    main,
    run,
)


class _ZeroProposer(Algorithm):
    def __init__(self, problem, X=None, Y=None):
        super().__init__(problem)
        self.X = X
        self.Y = Y
        self.name = "_ZeroProposer"
        self.problem = problem

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        return pd.DataFrame(data=np.zeros((n_proposals, self.X.shape[1])))


def test_check_inputs():
    problem = Cake()
    _check_args(problem, _ZeroProposer(problem, np.zeros((5, 5)), np.zeros((5, 5))))


def test_try_convert_val():
    assert _try_convert_value_via_eval("alpha") == "alpha"
    assert _try_convert_value_via_eval("12") == 12
    assert np.abs(_try_convert_value_via_eval("12.5") - 12.5) < 1e-6
    assert _try_convert_value_via_eval('["alpha", 1]') == ["alpha", 1]
    assert (
        _try_convert_value_via_eval("import os; os.remove('x')")
        == "import os; os.remove('x')"
    )


def test_parse_inputs():
    args = ["opti.problems.Cake", "test.test_bo.test_digital_twin._ZeroProposer"]
    problem, method = _parse_inputs(*args, method_kwargs=None)
    assert problem.data.equals(Cake().data)
    assert method._problem.data.equals(Cake().data)
    assert type(method).__name__ == _ZeroProposer.__name__
    assert problem.name.lower() == "cake"
    assert method.name == "_ZeroProposer"


def test_initial_sampling():
    def test(prb, ref_data_rows):
        method = RandomSearch(prb)
        assert prb.data.shape[0] == ref_data_rows
        assert np.allclose(prb.data.values, method.data.values)
        method.sample_initial_data(5)
        assert prb.data.shape[0] == 5 == method.data.shape[0]

    test(FlowReactor384Rf(), ref_data_rows=384)
    test(FlowReactorUnconstrainedRf(), ref_data_rows=112)


def test_run():
    zdt1_gaussian = noisify_problem_with_gaussian(ZDT1())
    problems = [
        FlowReactorUnconstrainedInterp(),
        Detergent(),
        zdt1_gaussian,
    ]

    for problem in problems:
        problem.data = (
            None
            if problem.data is None
            else pd.DataFrame(data=None, columns=problem.data.columns)
        )
        algorithm = RandomSearch(problem)
        data, metrics = run(
            problem, algorithm, max_experiments=5, logger=_DummyLogger()
        )
        assert data.shape[0] == 5
        assert len(metrics) == 4
        data, metrics = run(problem, algorithm, max_experiments=5, logger=None)
        assert data.shape[0] == 5
        assert len(metrics) == 4

    runner = CliRunner()
    res = runner.invoke(
        main,
        [
            "opti.problems.detergent.Detergent",
            "bo.algorithm.random.RandomSearch",
            "--no-mlflow",
        ],
    )
    assert res.exit_code == 0
