"""Native tensor, image geometry and result contracts for the renal bundle."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict, overload

import numpy as np
from medmlx_core.outputs import Artifact
from medmlx_core.typing import MlxRuntime
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from mlx.core import array


type HostArray = NDArray[np.generic]
type RealArray = NDArray[np.bool_ | np.integer | np.floating]
type FloatArray = NDArray[np.float32]
type DoubleArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type NativeNetwork = Callable[[array], array]


class NativeRuntime(MlxRuntime, Protocol):
    """The subset of ``mlx.core`` this package calls, typed structurally."""

    def eval(self, *args: array | Mapping[str, array] | Sequence[array]) -> None:
        """Evaluate native arrays and array collections."""
        ...
    def synchronize(self) -> None:
        """Wait for outstanding native work."""
        ...
    def load(self, file: str) -> array | dict[str, array]:
        """Load an array or named tensor mapping from a file."""
        ...
    def save_safetensors(self, file: str, arrays: dict[str, array]) -> None:
        """Write named arrays to a safetensors file."""
        ...
    def softmax(self, x: array, *, axis: int) -> array:
        """Normalize scores along the requested axis."""
        ...
    def isfinite(self, x: array) -> array:
        """Return the elementwise finiteness mask."""
        ...
    def all(self, x: array) -> array:
        """Reduce a native Boolean array to one scalar."""
        ...


class NativeOperation(Protocol):
    def __call__(self, x: array) -> array:
        """Apply a native neural-network operation."""
        ...


class NeuralRuntime(Protocol):
    def MaxPool3d(self, *, kernel_size: int, stride: int, padding: int) -> NativeOperation:
        """Construct a 3D max-pooling operation."""
        ...
    def leaky_relu(self, x: array, *, negative_slope: float) -> array:
        """Apply leaky ReLU with the requested negative slope."""
        ...
    def gelu(self, x: array) -> array:
        """Apply GELU activation."""
        ...


class SlidingWindow(Protocol):
    def __call__(
        self,
        inputs: FloatArray,
        *,
        roi_size: tuple[int, int, int],
        sw_batch_size: int,
        predictor: NativeNetwork,
        overlap: float,
    ) -> FloatArray:
        """Run sliding-window inference and return host output scores."""
        ...


class SlidingWindowNamespace(Protocol):
    sliding_window_inference: SlidingWindow


class NiftiHeader(Protocol):
    def __getitem__(self, key: str) -> HostArray:
        """Read a named NIfTI header field."""
        ...
    def __setitem__(self, key: str, value: int | HostArray) -> None:
        """Set a named NIfTI header field."""
        ...
    def get(self, key: str) -> HostArray | None:
        """Return a named header field when present."""
        ...
    def get_zooms(self) -> tuple[float, ...]:
        """Return voxel spacing from the header."""
        ...
    def set_zooms(self, zooms: ArrayLike) -> None:
        """Set voxel spacing in the header."""
        ...
    def set_slope_inter(self, slope: float, inter: float) -> None:
        """Set stored-data scaling slope and intercept."""
        ...


class NiftiImage(Protocol):
    affine: DoubleArray
    shape: tuple[int, ...]
    dataobj: ArrayLike
    header: NiftiHeader

    @overload
    def get_sform(self, coded: Literal[True]) -> tuple[DoubleArray | None, int]:
        """Return the sform affine and its code."""
        ...
    @overload
    def get_sform(self, coded: Literal[False] = False) -> DoubleArray:
        """Return the sform affine."""
        ...
    @overload
    def get_qform(self, coded: Literal[True]) -> tuple[DoubleArray | None, int]:
        """Return the qform affine and its code."""
        ...
    @overload
    def get_qform(self, coded: Literal[False] = False) -> DoubleArray:
        """Return the qform affine."""
        ...
    def set_sform(self, affine: DoubleArray | None, code: int | None = None) -> None:
        """Set the sform affine and optional code."""
        ...
    def set_qform(self, affine: DoubleArray | None, code: int | None = None) -> None:
        """Set the qform affine and optional code."""
        ...


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


class NiftiIo(Protocol):
    orientations: Orientations

    def load(self, filename: str | PathLike[str]) -> NiftiImage:
        """Load a NIfTI image from a path."""
        ...
    def save(self, image: NiftiImage, filename: str | PathLike[str]) -> None:
        """Save a NIfTI image to a path."""
        ...
    def Nifti1Image(self, data: ArrayLike, affine: ArrayLike | None) -> NiftiImage:
        """Construct a NIfTI-1 image from data and affine."""
        ...
    def aff2axcodes(self, affine: DoubleArray) -> tuple[str, ...]:
        """Derive axis codes from an image affine."""
        ...
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


class RunResult(TypedDict):
    """Return value of ``unest_mlx.medmlx.run``."""

    outputs: RunOutputs
    artifacts: dict[str, Artifact]
    details: RunDetails
