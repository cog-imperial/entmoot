import pytest

from entmoot.models.model_params import (
    EntingParams,
    ParamValidationError,
    TrainParams,
    TreeTrainParams,
    UncParams,
)


def test_model_params_creation():
    """Check EntingParams is instantiated correctly, and check default values."""
    params = EntingParams(**{
        "unc_params": {"beta": 2},
        "tree_train_params" : {
            "train_params": {"max_depth": 5}
            }
    })

    assert params.unc_params.beta == 2
    assert params.tree_train_params.train_params.max_depth == 5

    # check a selection of defaults
    assert params.unc_params.acq_sense in ("exploration", "penalty")
    assert params.tree_train_params.train_params.min_data_in_leaf == 1

    # check alternate initialisation method
    params_other = EntingParams(
        unc_params=UncParams(beta=2),
        tree_train_params=TreeTrainParams(
            train_params=TrainParams(max_depth=5)
        )
    )
    assert params == params_other


def test_model_params_invalid_values():
    """Check EntingParams raises an error for invalid values."""
    with pytest.raises(ParamValidationError):
        _ = EntingParams(**{"unc_params": {"beta": -1}})

    with pytest.raises(ParamValidationError):
        _ = EntingParams(**{"tree_train_params": {"train_lib": "notimplementedlib"}})

    with pytest.raises(ParamValidationError):
        _ = EntingParams(**{"unc_params": {"acq_sense": "notimplementedsense"}})