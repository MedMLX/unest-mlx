"""Native tensor, image geometry and result contracts for the renal bundle."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

import numpy as np
from medmlx_core.outputs import RunResult as _RunResult
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import array


type RealArray = NDArray[np.bool_ | np.integer | np.floating]
type FloatArray = NDArray[np.float32]
type DoubleArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type NativeNetwork = Callable[[array], array]


class Orientations(Protocol):
    def aff2axcodes(self, affine: DoubleArray) -> tuple[str, ...]:
        """Derive axis codes from an affine."""
        ...
    def axcodes2ornt(self, axcodes: str | tuple[str, ...]) -> DoubleArray:
        """Convert axis codes to an orientation transform."""
        ...
    def ornt_transform(self, start: DoubleArray, end: DoubleArray) -> DoubleArray:
        """Compute the transform between two orientations."""
        ...
    def inv_ornt_aff(self, ornt: DoubleArray, shape: tuple[int, ...]) -> DoubleArray:
        """Return the inverse voxel affine for an orientation and shape."""
        ...


class Nibabel(Protocol):
    """The nibabel orientation helpers the inverse transforms call."""

    orientations: Orientations

    def io_orientation(self, affine: DoubleArray) -> DoubleArray:
        """Derive voxel-axis orientation from an affine."""
        ...


class NumpyFlip(Protocol):
    def __call__[T: np.generic](self, values: NDArray[T], *, axis: list[int]) -> NDArray[T]:
        """Flip a host array along the requested axes."""
        ...


class Label(TypedDict):
    """One entry of the mask's label table: integer voxel value and structure name."""

    value: int
    name: str


class SlidingWindowDetails(TypedDict):
    roi_size: list[int]
    sw_batch_size: int
    overlap: float


class RunDetails(TypedDict):
    """JSON-compatible run metadata returned under ``details`` by ``medmlx.run``.

    ``voxel_counts`` is keyed by label name. ``shape`` and ``dtype`` describe the
    written mask (the input grid, float32). ``input_orientation`` is the axis-code string
    of the reader affine before reorientation to RAS (for example ``"LAS"``).
    ``network_shape`` is the voxel grid the network ran on after RAS reorientation and
    1 mm resampling. ``mlx_device`` is ``str(mx.default_device())`` and
    ``duration_seconds`` is wall time from the start of the call until the mask has been
    written, so it includes image reading and weight loading.
    """

    weights_revision: NotRequired[str | None]
    labels: list[Label]
    voxel_counts: dict[str, int]
    shape: list[int]
    dtype: str
    input_orientation: str
    network_shape: list[int]
    sliding_window: SlidingWindowDetails
    mlx_device: str
    duration_seconds: float


class RunOutputs(TypedDict):
    """Files written by a run; ``mask`` is the absolute path of ``mask.nii.gz``."""

    mask: str


# Return value of ``unest_mlx.medmlx.run``.
type RunResult = _RunResult[RunDetails, RunOutputs]
