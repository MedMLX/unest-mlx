"""Native MLX implementation of UNesT.

Public names load on first access, so importing the package (or the medmlx
declaration in ``unest_mlx.medmlx``) does not import MLX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unest_mlx.mlx_model import UNesT as UNesT
    from unest_mlx.mlx_predictor import UNesTPredictor as UNesTPredictor
    from unest_mlx.mlx_weights import required_shapes as required_shapes
    from unest_mlx.mlx_weights import validate_weights as validate_weights

__version__ = "0.2.8"

_EXPORTS = {
    "UNesT": "unest_mlx.mlx_model",
    "UNesTPredictor": "unest_mlx.mlx_predictor",
    "validate_weights": "unest_mlx.mlx_weights",
    "required_shapes": "unest_mlx.mlx_weights",
}

__all__ = [
    "UNesT",
    "UNesTPredictor",
    "required_shapes",
    "validate_weights",
]


def __getattr__(name: str) -> object:
    """Load a public export from its module on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'unest_mlx' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List eager and lazy package exports."""
    return sorted({*globals(), *_EXPORTS})
