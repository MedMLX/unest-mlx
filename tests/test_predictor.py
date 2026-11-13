"""Volume prediction against the upstream Torch graph and MONAI inferer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from medmlx_core.typing import ArrayFactory

from tests._contracts import PredictorMetadata
from tests.fixture_cases import (
    DeterministicNetwork,
    SeedRuntime,
    fixture_metadata,
    predictor_input,
    predictor_weights,
    recording,
)
from tests.numerical import assert_stable_labels, assert_upstream_close
from unest_mlx import UNesT, UNesTPredictor, required_shapes
from unest_mlx._typing import NativeRuntime

if TYPE_CHECKING:
    from mlx.core import Dtype, array


@pytest.mark.parametrize("batch_size", [4, 1], ids=["bundle-config", "single-window"])
@pytest.mark.mlx
def test_stand_in_matches_upstream_predictor_path_bitwise(
    batch_size: int, mx: NativeRuntime
) -> None:
    """Compare the stand-in sliding-window predictor with upstream recordings."""
    with np.load(recording(f"unest_predictor_stand_in_batch{batch_size}")) as fixture:
        info = cast(PredictorMetadata, fixture_metadata(fixture))
        assert info["window_batches"] == ([4, 2] if batch_size == 4 else [1] * 6)
        seed_mx = cast("SeedRuntime[array, Dtype]", mx)
        network: DeterministicNetwork[array, Dtype] = DeterministicNetwork(
            info["weights_seed"], mx=seed_mx
        )
        inputs = predictor_input(info["input_shape"])
        array = cast(ArrayFactory, mx.array)
        actual = UNesTPredictor(network, mx=mx, sw_batch_size=batch_size)(array(inputs))
        assert actual.dtype == np.float32
        assert np.array_equal(actual, fixture["output"])
        # Cross-image batching and the final short batch are part of the bundle contract.
        assert network.window_batches == info["window_batches"]


@pytest.mark.mlx
def test_full_unest_matches_upstream_predictor_path(mx: NativeRuntime) -> None:
    """Compare complete native UNesT prediction with the upstream volume."""
    with np.load(recording("unest_predictor_full_batch4")) as fixture:
        info = cast(PredictorMetadata, fixture_metadata(fixture))
        weights = predictor_weights(required_shapes(), info["weights_seed"])
        array = cast(ArrayFactory, mx.array)
        network = UNesT({key: array(value) for key, value in weights.items()}, mx=mx)
        inputs = predictor_input(info["input_shape"])
        prediction = UNesTPredictor(network, mx=mx)(inputs)
        # Retain the existing absolute bound for the padded graph.
        assert_upstream_close(prediction, fixture["output"], "predictor")
        assert_stable_labels(prediction, fixture["output"], "predictor")
