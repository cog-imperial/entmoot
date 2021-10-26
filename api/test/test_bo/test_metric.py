import numpy as np
import pandas as pd

from bo.metric import goodness_of_fit, hypervolume, hypervolume_improvement


def test_hypervolume():
    A = np.array([[0.1, 0.9], [0.4, 0.4], [0.9, 0.1]])

    vol = hypervolume(A, ref_point=[1, 1])
    assert np.isclose(vol, 0.6 * 0.6 + 2 * (0.9 * 0.1 - 0.6 * 0.1))

    # when no ref_point is given the nadir should be used, here (0.9, 0.9)
    vol1 = hypervolume(A, ref_point=[0.9, 0.9])
    vol2 = hypervolume(A, ref_point=None)
    assert np.isclose(vol1, vol2)

    # points that are worse than the ref_point should be clipped to it
    vol = hypervolume(A, ref_point=[0.8, 0.8])
    assert np.isclose(vol, 0.4 ** 2)


def test_hypervolume_improvement():
    A = np.array([[0.4, 0.4], [0.5, 0.5], [0.15, 0.95]])
    R = np.array([[0.1, 0.9], [0.9, 0.1]])
    hvi = hypervolume_improvement(A, R, ref_point=[1, 1])
    np.testing.assert_almost_equal(hvi, [0.25, 0.16, 0])


def test_goodness_of_fit():
    predictions = pd.DataFrame(
        {
            "y1": [1, 2, 3, 4],
            "y2": [1, 2, 3, 4],
            "mean_y1": [2.5, 2.5, 2.5, 2.5],
            "mean_y2": [1, 2, 3, 4],
            "std_y1": [0.5, 0.5, 0.5, 0.5],
            "std_y2": [0, 0, 0, 0],
        }
    )
    gof = goodness_of_fit(predictions)
    assert np.isclose(gof.loc["y1", "RMSE"], 1.25 ** 0.5)
    assert np.isclose(gof.loc["y2", "RMSE"], 0)
    assert np.isclose(gof.loc["y1", "MAE"], 1)
    assert np.isclose(gof.loc["y2", "MAE"], 0)
    assert np.isclose(gof.loc["y1", "R2"], 0)
    assert np.isclose(gof.loc["y2", "R2"], 1)
