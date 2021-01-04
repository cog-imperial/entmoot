
import numpy as np
import pytest

from numpy.testing import assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from entmoot import entmoot_minimize
from entmoot.benchmarks import Rosenbrock

from entmoot.learning import EntingRegressor
from entmoot.optimizer import Optimizer
from scipy.optimize import OptimizeResult

ESTIMATOR_STRINGS = ("GBRT", "RF")

STD_ESTIMATORS = ("BDD","L1BDD","DDP","L1DDP")
ACQ_FUNCS = ["LCB"]
ACQ_OPTIMIZER = ["sampling", "global"]

@pytest.mark.fast_test
def test_multiple_asks():
    # calling ask() multiple times without a tell() inbetween should
    # be a "no op"
    bench1 = Rosenbrock()

    opt = Optimizer(bench1.get_bounds(2), "GBRT", n_initial_points=10,
                    acq_optimizer="sampling",
                    base_estimator_kwargs={"min_child_samples":2})

    opt.run(bench1, n_iter=13)
    # tell() computes the next point ready for the next call to ask()
    # hence there are three after three iterations
    assert_equal(len(opt.models), 3)
    assert_equal(len(opt.Xi), 13)
    opt.ask()
    assert_equal(len(opt.models), 3)
    assert_equal(len(opt.Xi), 13)
    assert_equal(opt.ask(), opt.ask())
    opt.update_next()
    assert_equal(opt.ask(), opt.ask())

@pytest.mark.fast_test
def test_model_queue_size():
    # Check if model_queue_size limits the model queue size
    bench1 = Rosenbrock()

    opt = Optimizer(bench1.get_bounds(2), "GBRT", n_initial_points=10,
                    acq_optimizer="sampling", model_queue_size=2,
                    base_estimator_kwargs={"min_child_samples":2})

    opt.run(bench1, n_iter=13)
    # tell() computes the next point ready for the next call to ask()
    # hence there are three after three iterations
    assert_equal(len(opt.models), 2)
    assert_equal(len(opt.Xi), 13)
    opt.ask()
    assert_equal(len(opt.models), 2)
    assert_equal(len(opt.Xi), 13)
    assert_equal(opt.ask(), opt.ask())
    opt.update_next()
    assert_equal(opt.ask(), opt.ask())

@pytest.mark.fast_test
def test_invalid_tell_arguments():
    bench1 = Rosenbrock()

    opt = Optimizer(bench1.get_bounds(2), "GBRT", n_initial_points=10,
                    acq_optimizer="sampling", model_queue_size=2,
                    base_estimator_kwargs={"min_child_samples":2})

    # can't have single point and multiple values for y
    assert_raises(ValueError, opt.tell, [1.], [1., 1.])

@pytest.mark.fast_test
def test_invalid_tell_arguments_list():
    bench1 = Rosenbrock()

    opt = Optimizer(bench1.get_bounds(2), "GBRT", n_initial_points=10,
                    acq_optimizer="sampling", model_queue_size=2,
                    base_estimator_kwargs={"min_child_samples":2})

    assert_raises(ValueError, opt.tell, [[1.], [2.]], [1., None])

@pytest.mark.fast_test
def test_bounds_checking_1D():
    low = -2.
    high = 2.

    opt = Optimizer([(low, high)], "GBRT", n_initial_points=1,
                    acq_optimizer="sampling")

    assert_raises(ValueError, opt.tell, [high + 0.5], 2.)
    assert_raises(ValueError, opt.tell, [low - 0.5], 2.)
    # feed two points to tell() at once
    assert_raises(ValueError, opt.tell, [high + 0.5, high], (2., 3.))
    assert_raises(ValueError, opt.tell, [low - 0.5, high], (2., 3.))

@pytest.mark.fast_test
def test_bounds_checking_2D():
    low = -2.
    high = 2.

    opt = Optimizer([(low, high)], "GBRT", n_initial_points=1,
                    acq_optimizer="sampling")

    assert_raises(ValueError, opt.tell, [high + 0.5, high + 4.5], 2.)
    assert_raises(ValueError, opt.tell, [low - 0.5, low - 4.5], 2.)

    # first out, second in
    assert_raises(ValueError, opt.tell, [high + 0.5, high + 0.5], 2.)
    assert_raises(ValueError, opt.tell, [low - 0.5, high + 0.5], 2.)


@pytest.mark.fast_test
def test_bounds_checking_2D_multiple_points():
    low = -2.
    high = 2.

    opt = Optimizer([(low, high)], "GBRT", n_initial_points=1,
                    acq_optimizer="sampling")

    # first component out, second in
    assert_raises(ValueError, opt.tell,
                  [(high + 0.5, high + 0.5), (high + 0.5, high + 0.5)],
                  [2., 3.])
    assert_raises(ValueError, opt.tell,
                  [(low - 0.5, high + 0.5), (low - 0.5, high + 0.5)],
                  [2., 3.])


@pytest.mark.fast_test
def test_dimension_checking_1D():
    low = -2
    high = 2
    opt = Optimizer([(low, high)], "GBRT", n_initial_points=1,
                    acq_optimizer="sampling")

    with pytest.raises(ValueError) as e:
        # within bounds but one dimension too high
        opt.tell([low+1, low+1], 2.)
    assert "Dimensions of point " in str(e.value)


@pytest.mark.fast_test
def test_dimension_checking_2D():
    low = -2
    high = 2
    opt = Optimizer([(low, high), (low, high)], "GBRT", n_initial_points=10,
                    acq_optimizer="sampling")

    # within bounds but one dimension too little
    with pytest.raises(ValueError) as e:
        opt.tell([low+1, ], 2.)
    assert "Dimensions of point " in str(e.value)
    # within bounds but one dimension too much
    with pytest.raises(ValueError) as e:
        opt.tell([low+1, low+1, low+1], 2.)
    assert "Dimensions of point " in str(e.value)


@pytest.mark.fast_test
def test_dimension_checking_2D_multiple_points():
    low = -2
    high = 2
    opt = Optimizer([(low, high), (low, high)])
    # within bounds but one dimension too little
    with pytest.raises(ValueError) as e:
        opt.tell([[low+1, ], [low+1, low+2], [low+1, low+3]], 2.)
    assert "dimensions as the space" in str(e.value)
    # within bounds but one dimension too much
    with pytest.raises(ValueError) as e:
        opt.tell([[low + 1, low + 1, low + 1], [low + 1, low + 2],
                  [low + 1, low + 3]], 2.)
    assert "dimensions as the space" in str(e.value)

@pytest.mark.fast_test
@pytest.mark.parametrize("acq_optimizer", ACQ_OPTIMIZER)
def test_result_rosenbrock(acq_optimizer):
    bench1 = Rosenbrock()

    opt = Optimizer(bench1.get_bounds(2), "GBRT", n_initial_points=10,
                    acq_optimizer=acq_optimizer,
                    base_estimator_kwargs={"min_child_samples":2},
                    random_state=100)

    opt.run(bench1, n_iter=20, no_progress_bar=True)

    if acq_optimizer == "sampling":
        res_known = np.array(
            [[0.701053475292035, -0.36025795213264167],
            [0.10806258681567815, -1.4393145750513043],
            [-1.4061114159677395, -1.2842304177627843],
            [-1.1873986869551771, -0.19357735788859287],
            [1.5161042350797533, -1.7871624483849504],
            [0.5091814944214517, 0.09563236039209677],
            [0.25516550013357264, -2.0241727838227086],
            [-0.7887945257988347, 1.843954950307714],
            [-1.5292242379168393, -1.7244656733329853],
            [-0.7726975624367143, 0.5422431772370664],
            [-0.26851521090965313, -0.27168208839378893], # 11
            [0.18559023118643525, -0.30091277104839054], # 12

            [2.037449372031385, -0.003908949645985427], # 13
            [0.3618262062035891, -0.886575853445807],
            [-0.70099499312817, -0.013753379624784401],
            [0.3074119029928717, 1.3227888228213858], # 16
            [0.43370695792140035, -0.5401762255031577],
            [0.060723595879015324, 2.0360116729103517],
            [-0.8891589013732526, -0.2756841022715071],
            [-0.5803197996462619, -0.6720516369715639]]
        )
    elif acq_optimizer == "global":
        res_known = np.array(
            [[0.701053475292035, -0.36025795213264167],
            [0.10806258681567815, -1.4393145750513043],
            [-1.4061114159677395, -1.2842304177627843],
            [-1.1873986869551771, -0.19357735788859287],
            [1.5161042350797533, -1.7871624483849504],
            [0.5091814944214517, 0.09563236039209677],
            [0.25516550013357264, -2.0241727838227086],
            [-0.7887945257988347, 1.843954950307714],
            [-1.5292242379168393, -1.7244656733329853],
            [-0.7726975624367143, 0.5422431772370664], #10
            [-0.27264229248767846, -0.2769176550000001],
            [0.19137509090861582, -0.31858780399999986],

            [2.048, 0.0], # 13
            [0.38217349700000014, -0.8962288678056055],
            [-0.7085115407036777, 0.0],
            [-0.8912067038309598, -0.23524750599999988],
            [0.0, 1.2022767981974225],
            [0.31866949899999986, 2.048],
            [0.43534523214208437, -0.5351559776571816],
            [-0.05614667929761541, 0.4412071169912909]]
        )

    res = opt.get_result()
    assert_array_equal(res.x_iters, res_known)