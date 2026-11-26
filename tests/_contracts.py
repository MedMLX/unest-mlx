"""Schemas from retained upstream fixture roles, not expected numerical values."""

from os import PathLike
from typing import Literal, Protocol, TypedDict, overload

from medmlx_core.typing import HostArray
from numpy.typing import ArrayLike

from unest_mlx._typing import DoubleArray, FloatArray, IntArray


class SourceManifest(TypedDict):
    schema_version: int
    recording_platform: str
    revision: str
    bundle_version: str
    monai_version: str
    torch_version: str
    files: list[str]


class CommonMetadata(TypedDict):
    schema_version: int
    recording_platform: str
    reference_framework: str
    upstream_revision: str
    bundle_version: str
    monai_version: str
    torch_version: str
    torch_threads: int


class InventoryMetadata(CommonMetadata):
    required_shapes: dict[str, list[int]]


class WindowMetadata(InventoryMetadata):
    weights_seed: int
    depth_start: int
    depth_stop: int


class LevelMetadata(CommonMetadata):
    level: int
    seed: int


class PredictorMetadata(InventoryMetadata):
    weights_seed: int
    input_shape: list[int]
    window_batches: list[int]


class RunnerMetadata(InventoryMetadata):
    weights_seed: int
    windows: int
    sw_batch_size: int
    window_batches: list[int]


class RunnerCase(TypedDict):
    seed: int
    shape: tuple[int, int, int]
    dtype: str
    scaling: tuple[float, float] | None
    linear: DoubleArray


class NumericalRule(TypedDict):
    allow_nan: bool
    allow_inf: bool
    dtype: str
    shape: list[int]
    rtol: float
    atol: float


class NumericalContract(TypedDict):
    schema_version: int
    arrays: dict[str, NumericalRule]


class RecordedTransforms(TypedDict):
    input_data: HostArray
    input_affine: DoubleArray
    reader_affine: DoubleArray
    image: FloatArray
    network_affine: DoubleArray
    logits: FloatArray
    probabilities: FloatArray
    inverted: FloatArray
    inverted_affine: DoubleArray
    labels: FloatArray
    saved: FloatArray
    saved_affine: DoubleArray
    saved_codes: IntArray


# nibabel calls the tests make to write and read NIfTI fixtures.
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
    def set_xyzt_units(self, xyz: str) -> None:
        """Set the spatial unit by name."""
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


class NiftiIo(Protocol):
    def load(self, filename: str | PathLike[str]) -> NiftiImage:
        """Load a NIfTI image from a path."""
        ...
    def save(self, image: NiftiImage, filename: str | PathLike[str]) -> None:
        """Save a NIfTI image to a path."""
        ...
    def Nifti1Image(self, data: ArrayLike, affine: ArrayLike | None) -> NiftiImage:
        """Construct a NIfTI-1 image from data and affine."""
        ...
