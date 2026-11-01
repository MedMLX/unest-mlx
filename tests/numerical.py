"""Declared, finite float32 comparisons against the original upstream recordings."""

import json
from pathlib import Path
from typing import cast

import numpy as np

from tests._contracts import NumericalContract, NumericalRule
from unest_mlx._typing import FloatArray

CONTRACT = Path(__file__).resolve().parents[1] / "tests/fixtures/numerical_contract.json"


def _rule(name: str) -> NumericalRule:
    """Load the numerical acceptance rule for a named reference boundary."""
    contract = cast(NumericalContract, json.loads(CONTRACT.read_text()))
    assert contract["schema_version"] == 2
    return contract["arrays"][name]


def assert_upstream_close(actual: FloatArray, expected: FloatArray, name: str) -> None:
    """Compare finite output with the declared upstream numerical tolerance."""
    rule = _rule(name)
    assert rule["allow_nan"] is rule["allow_inf"] is False
    assert actual.dtype == expected.dtype == np.dtype(rule["dtype"])
    assert actual.shape == expected.shape == tuple(rule["shape"])
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(
        actual, expected, rtol=rule["rtol"], atol=rule["atol"], equal_nan=False, err_msg=name
    )


def assert_stable_labels(actual: FloatArray, expected: FloatArray, name: str) -> None:
    """Check winners whose reference margin exceeds both logits' error budgets."""
    bound = _rule(name)["atol"]
    ordered = np.sort(expected, axis=1)
    stable = ordered[:, -1] - ordered[:, -2] > 2 * bound
    assert stable.any(), "Reference must contain decisions outside the rounding budget"
    np.testing.assert_array_equal(actual.argmax(axis=1)[stable], expected.argmax(axis=1)[stable])
