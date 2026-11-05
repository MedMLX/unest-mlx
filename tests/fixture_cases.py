"""Seeded synthetic inputs shared by fixture generation and equivalence tests.

This module generates inputs only; expected values come from the pinned upstream bundle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from numpy.lib.npyio import NpzFile

from tests._contracts import CommonMetadata, RunnerCase, SourceManifest
from unest_mlx._typing import DoubleArray, FloatArray, HostArray, NiftiIo

if TYPE_CHECKING:
    from mlx.core import array

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"


def recording(name: str) -> Path:
    """Return the recorded fixture path or report that it is missing."""
    path = FIXTURES / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream fixture {path}"
        )
    return path


def fixture_metadata(fixture: NpzFile) -> CommonMetadata:
    """Check fixture provenance against the single upstream source manifest."""
    pins = cast(
        SourceManifest, json.loads((FIXTURES / "upstream.json").read_text())
    )
    metadata = cast(CommonMetadata, json.loads(str(fixture["metadata"])))
    assert metadata["schema_version"] == pins["schema_version"] == 3
    assert metadata["recording_platform"] == pins["recording_platform"] == "darwin-arm64"
    assert metadata["reference_framework"] == "torch"
    assert metadata["upstream_revision"] == pins["revision"]
    assert metadata["bundle_version"] == pins["bundle_version"]
    assert metadata["monai_version"] == pins["monai_version"]
    assert metadata["torch_version"].partition("+")[0] == pins["torch_version"]
    return metadata


def small_weights(shapes: Mapping[str, tuple[int, ...]], seed: int) -> dict[str, FloatArray]:
    """Reduce channel widths without changing the UNesT graph or spatial sizes."""
    # All transformer channel widths remain divisible by the fixed head counts.
    widths = {
        1: 1,
        4: 1,
        32: 1,
        64: 2,
        128: 4,
        256: 8,
        512: 16,
        1024: 32,
        384: 12,
        768: 24,
        1536: 48,
        2048: 64,
    }
    rng = np.random.default_rng(seed)
    weights: dict[str, FloatArray] = {}
    for name, shape in shapes.items():
        if name.endswith("pos_embed"):
            shape = (*shape[:-1], widths[shape[-1]])
        elif len(shape) == 5:
            shape = (widths[shape[0]], widths[shape[1]], *shape[2:])
        else:
            shape = tuple(widths[size] for size in shape)
        values = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if ".norm" in name and name.endswith(".weight"):
            values += np.float32(1)
        weights[name] = values
    return weights


def window_input(seed: int) -> FloatArray:
    """Create a seeded float32 UNesT window input."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 1, 96, 96, 96)).astype(np.float32)


def level_case(level: int, seed: int) -> tuple[dict[str, FloatArray], FloatArray]:
    """Exercise all transformer depths with one 6-cubed block per level."""
    channels = 8 * 2**level
    prefix = f"nestViT.levels.{level}"
    shapes: dict[str, tuple[int, ...]] = {f"{prefix}.pos_embed": (1, 1, 216, channels)}
    if level:
        shapes.update(
            {
                f"{prefix}.pool.conv.weight": (channels, channels // 2, 3, 3, 3),
                f"{prefix}.pool.conv.bias": (channels,),
                f"{prefix}.pool.norm.weight": (channels,),
                f"{prefix}.pool.norm.bias": (channels,),
            }
        )
    for layer in range((2, 2, 8)[level]):
        block = f"{prefix}.transformer_encoder.{layer}"
        for norm in ("norm1", "norm2"):
            shapes[f"{block}.{norm}.weight"] = (channels,)
            shapes[f"{block}.{norm}.bias"] = (channels,)
        for name, out_channels, in_channels in (
            ("attn.qkv", channels * 3, channels),
            ("attn.proj", channels, channels),
            ("mlp.fc1", channels * 4, channels),
            ("mlp.fc2", channels, channels * 4),
        ):
            shapes[f"{block}.{name}.weight"] = (out_channels, in_channels)
            shapes[f"{block}.{name}.bias"] = (out_channels,)
    rng = np.random.default_rng(seed)
    weights: dict[str, FloatArray] = {}
    for name, shape in shapes.items():
        weights[name] = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if ".norm" in name and name.endswith(".weight"):
            weights[name] += np.float32(1)
    shape = (1, channels, 6, 6, 6) if level == 0 else (1, channels // 2, 12, 12, 12)
    return weights, rng.standard_normal(shape).astype(np.float32)


class Float32ArrayConstructor(Protocol):
    def __call__(self, values: list[float]) -> FloatArray:
        """Construct a float32 host array from values."""
        ...


class SeedRuntime[SeedArray, SeedDtype](Protocol):
    float32: SeedDtype

    def array(self, values: FloatArray, /) -> SeedArray:
        """Create a seeded fixture array from host values."""
        ...
    def arange(self, stop: int, *, dtype: SeedDtype) -> SeedArray:
        """Create a seeded fixture integer range."""
        ...
    def reshape(self, values: SeedArray, shape: Sequence[int], /) -> SeedArray:
        """Reshape a seeded fixture array."""
        ...


class DeterministicNetwork[SeedArray: array | FloatArray, SeedDtype]:
    """Cheap MLX network with window-local coordinates to constrain stitching."""

    def __init__(self, seed: int, *, mx: SeedRuntime[SeedArray, SeedDtype]) -> None:
        """Initialize the native stand-in predictor and its batch recorder."""
        self.mx = mx
        rng = np.random.default_rng(seed)
        self.gain = mx.array(rng.standard_normal((1, 4, 1, 1, 1)).astype(np.float32))
        self.bias = mx.array(rng.standard_normal((1, 4, 1, 1, 1)).astype(np.float32))
        self.window_batches: list[int] = []

    def __call__(self, values: SeedArray) -> SeedArray:
        """Run the stand-in predictor and record its window batch size."""
        self.window_batches.append(values.shape[0])
        depth = self.mx.arange(values.shape[2], dtype=self.mx.float32)
        return cast(
            "SeedArray",
            values * self.gain + self.bias + self.mx.reshape(depth, (1, 1, -1, 1, 1)) / 128,
        )


def predictor_input(seed: int, shape: Sequence[int]) -> FloatArray:
    """Create a seeded float32 predictor volume."""
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def predictor_weights(shapes: Mapping[str, tuple[int, ...]], seed: int) -> dict[str, FloatArray]:
    """Use all UNesT layers with reduced widths and the bundle's four classes."""
    weights = small_weights(shapes, seed)
    rng = np.random.default_rng(seed + 1)
    weights["out.conv.conv.weight"] = (rng.standard_normal((4, 1, 1, 1, 1)) * 0.05).astype(
        np.float32
    )
    weights["out.conv.conv.bias"] = (rng.standard_normal(4) * 0.05).astype(np.float32)
    return weights


def runner_weights(
    shapes: Mapping[str, tuple[int, ...]], seed: int = 3200
) -> dict[str, FloatArray]:
    """Reduced-width UNesT whose one-channel decoder feature selects all four labels."""
    weights = predictor_weights(shapes, seed)
    # On the CT phantom each class wins one interval of the decoder feature.
    weights["out.conv.conv.weight"] = cast(Float32ArrayConstructor, np.float32)(
        [-30, -10, 10, 30]
    ).reshape(4, 1, 1, 1, 1)
    weights["out.conv.conv.bias"] = cast(Float32ArrayConstructor, np.float32)([-0.084, 0, -3, -13])
    return weights


def _rotation(x: float, y: float, z: float) -> DoubleArray:
    """Build a rotation matrix from three angles."""
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rx @ ry @ rz


# Seeded NIfTI inputs for the bundle transforms: flipped anisotropic CT, an oblique
# permuted grid with intensity scaling, an unchanged 1 mm grid, and a float32 grid
# whose first spacing is kept because it lies within MONAI's 1e-3 tolerance.
RUNNER_CASES: dict[str, RunnerCase] = {
    "las_int16": RunnerCase(
        seed=3001,
        shape=(13, 11, 7),
        dtype="int16",
        scaling=None,
        linear=np.diag([-0.94, 0.94, 3.0]),
    ),
    "oblique_scaled": RunnerCase(
        seed=3002,
        shape=(9, 12, 8),
        dtype="int16",
        scaling=(0.5, -10.0),
        linear=_rotation(0.3, -0.2, 0.4) @ (np.eye(3)[:, [2, 0, 1]] * [1.3, -0.7, 2.2]),
    ),
    "ras_1mm": RunnerCase(
        seed=3003, shape=(10, 9, 8), dtype="int16", scaling=None, linear=np.eye(3)
    ),
    "near_1mm_float32": RunnerCase(
        seed=3004,
        shape=(8, 10, 9),
        dtype="float32",
        scaling=None,
        linear=np.diag([1.0004, -2.2, 0.6]),
    ),
}

HEADER_CASES = ("no_forms", "disagreeing_forms", "corrected_affine")


def write_header_case(name: str, path: Path) -> None:
    """Seeded header cases admitted by the upstream NibabelReader."""
    import nibabel as nib_runtime

    nib = cast(NiftiIo, nib_runtime)

    write_runner_case("ras_1mm", path)
    image = nib.load(str(path))
    if name == "no_forms":
        image.set_qform(None, code=0)
        image.set_sform(None, code=0)
    elif name == "disagreeing_forms":
        qform = image.affine.copy()
        qform[0, 3] += 2.0
        image.set_qform(qform, code=1)
    elif name == "corrected_affine":
        image.set_sform(np.diag([2.0, 2.0, 2.0, 1.0]), code=1)
        # qform and header zooms are still 1 mm; the reader copies qform to sform.
    else:
        raise ValueError(f"Unknown header case: {name}")
    nib.save(image, str(path))


def write_runner_case(name: str, path: Path) -> None:
    """Write a named synthetic runner image case."""
    case = RUNNER_CASES[name]
    rng = np.random.default_rng(case["seed"])
    data = (rng.standard_normal(case["shape"]) * 300).astype(case["dtype"])
    affine = np.eye(4)
    affine[:3, :3] = case["linear"]
    affine[:3, 3] = rng.uniform(-200, 200, 3)
    _save_nifti(path, data, affine, case["scaling"])


def write_ct_phantom(path: Path, seed: int = 3100) -> None:
    """Flipped anisotropic CT phantom whose 1 mm grid needs two sliding windows."""
    shape, spacing = (75, 50, 20), np.array([1.6, 1.6, 4.0])
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), -1) * spacing
    centre = (np.array(shape) - 1) * spacing / 2
    data = np.full(shape, -1000.0)
    body = (((grid - centre) / [55, 35, 60]) ** 2).sum(-1) <= 1
    data[body] = 40.0
    for side in (-1, 1):
        kidney = (((grid - centre - [side * 25, 5, 0]) / [9, 14, 20]) ** 2).sum(-1)
        data[kidney <= 1] = 180.0
        data[kidney <= 0.3] = 20.0
    data += np.random.default_rng(seed).normal(0, 15, shape)
    affine = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])
    affine[:3, 3] = [60.0, -40.0, 300.0]
    _save_nifti(path, np.round(data).astype(np.int16), affine, None)


def _save_nifti(
    path: Path, data: HostArray, affine: DoubleArray, scaling: tuple[float, float] | None
) -> None:
    """Write synthetic image values with their reference affine."""
    import nibabel as nib_runtime

    nib = cast(NiftiIo, nib_runtime)

    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    if scaling is not None:
        image.header.set_slope_inter(*scaling)
    nib.save(image, str(path))
