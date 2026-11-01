"""Schemas from retained upstream fixture roles, not expected numerical values."""

from typing import TypedDict

from unest_mlx._typing import DoubleArray, FloatArray, HostArray, IntArray


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
    input_seed: int
    depth_start: int
    depth_stop: int


class LevelMetadata(CommonMetadata):
    level: int
    seed: int


class PredictorMetadata(InventoryMetadata):
    weights_seed: int
    input_seed: int
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
