"""Seeded graph comparison against the pinned upstream Torch implementation.

Uses reduced-width seeded weights and an HCC-TACE CT input.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from tests._contracts import LevelMetadata, WindowMetadata
from tests.fixture_cases import (
    fixture_metadata,
    level_case,
    recording,
    small_weights,
    window_input,
)
from tests.numerical import assert_upstream_close
from unest_mlx import UNesT
from unest_mlx._typing import FloatArray, NativeRuntime

pytestmark = pytest.mark.mlx


@pytest.fixture(scope="module")
def window_output(mx: NativeRuntime) -> FloatArray:
    """Run the seeded native UNesT window fixture."""
    with np.load(recording("unest_window_0"), allow_pickle=False) as fixture:
        info = cast(WindowMetadata, fixture_metadata(fixture))
    shapes = {key: tuple(shape) for key, shape in info["required_shapes"].items()}
    weights = small_weights(shapes, info["weights_seed"])
    network = UNesT({key: mx.array(value) for key, value in weights.items()}, mx=mx)
    output = network(mx.array(window_input()))
    mx.eval(output)
    return cast(FloatArray, np.array(output))


def test_full_graph_matches_upstream(window_output: FloatArray) -> None:
    """Compare the complete native graph with recorded upstream output tiles."""
    for part in range(8):
        with np.load(recording(f"unest_window_{part}"), allow_pickle=False) as fixture:
            info = cast(WindowMetadata, fixture_metadata(fixture))
            actual = window_output[:, :, info["depth_start"] : info["depth_stop"]]
            # Retain the pre-existing absolute bound across all eight tiles.
            assert_upstream_close(actual, fixture["output"], "window")


@pytest.mark.parametrize("level", range(3))
def test_transformer_levels_match_upstream(level: int, mx: NativeRuntime) -> None:
    """Compare each native transformer level with the upstream recording."""
    with np.load(recording(f"unest_level_{level}"), allow_pickle=False) as fixture:
        info = cast(LevelMetadata, fixture_metadata(fixture))
        weights, inputs = level_case(level, info["seed"])
        network = UNesT({key: mx.array(value) for key, value in weights.items()}, mx=mx)
        actual = network.transformer_level(mx.array(inputs), level)
        mx.eval(actual)
        # Absolute bounds from tests/fixtures/numerical_contract.json; native Metal passed them.
        assert_upstream_close(
            cast(FloatArray, np.array(actual)), fixture["output"], f"level_{level}"
        )
