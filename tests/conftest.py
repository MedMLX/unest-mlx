"""Load the numerical backend once; pure contract tests remain independently runnable."""

from typing import cast

import pytest

from unest_mlx._typing import NativeRuntime


@pytest.fixture(scope="session")
def mx() -> NativeRuntime:
    """Provide the explicitly admitted native MLX runtime."""
    from medmlx_core.runtime import import_mlx

    return cast(NativeRuntime, import_mlx())
