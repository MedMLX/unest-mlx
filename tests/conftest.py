"""Load the numerical backend once; pure contract tests remain independently runnable."""

from typing import cast

import pytest
from medmlx_core.typing import MlxRuntime


@pytest.fixture(scope="session")
def mx() -> MlxRuntime:
    """Provide the explicitly admitted native MLX runtime."""
    from medmlx_core.runtime import import_mlx

    return cast(MlxRuntime, import_mlx())
