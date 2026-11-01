"""Released key inventory and strict checkpoint admission against it."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from tests._contracts import InventoryMetadata
from tests.fixture_cases import fixture_metadata, recording
from unest_mlx import mlx_weights, required_shapes
from unest_mlx._typing import HostArray


def test_released_inventory_matches_the_upstream_network() -> None:
    """Compare the released tensor inventory, in registration order, with the upstream graph."""
    with np.load(recording("unest_inventory"), allow_pickle=False) as fixture:
        expected = cast(InventoryMetadata, fixture_metadata(fixture))["required_shapes"]
    # Seeded values are consumed in registration order, as in the upstream recorder.
    assert list(required_shapes().items()) == [(k, tuple(v)) for k, v in expected.items()]


@pytest.mark.parametrize("defect", ["missing", "shape", "dtype", "nan"])
def test_rejects_incompatible_checkpoints(defect: str) -> None:
    """Reject a checkpoint whose names, shapes, dtypes or values break the released inventory."""
    state: dict[str, HostArray] = {
        key: np.zeros(shape, np.float32) for key, shape in required_shapes().items()
    }
    key = next(iter(state))
    if defect == "missing":
        del state[key]
    elif defect == "shape":
        state[key] = np.zeros(1, np.float32)
    elif defect == "dtype":
        state[key] = state[key].astype(np.float16)
    else:
        state[key].flat[0] = np.nan
    with pytest.raises(InvalidInputError):
        mlx_weights.validate_weights(state)
