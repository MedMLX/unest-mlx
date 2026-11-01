"""medmlx declaration and end-to-end runner for renal UNesT segmentation.

Follows ``renalStructures_UNEST_segmentation`` 0.2.7's inference configuration:
the NIfTI reader, RAS/1 mm preprocessing, batches of four 96-cubed windows,
softmax, inverse transforms, argmax, and float32 NIfTI writing.

The model is intended for contrast-enhanced CT in the arterial or portal venous phase,
as in the upstream bundle's training data. It has failed on the non-contrast scans tried
so far (the kidneys were missed), so check enhancement before running it. This is research
software, not a diagnostic tool.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import cast

import nibabel as nib_runtime
import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.runtime import import_mlx
from medmlx_core.weights import weights_dir, weights_revision
from nibabel.filebasedimages import ImageFileError

from unest_mlx._typing import (
    DoubleArray,
    FloatArray,
    HostArray,
    Label,
    NativeNetwork,
    NativeRuntime,
    NiftiImage,
    NiftiIo,
    RunResult,
)

nib = cast(NiftiIo, nib_runtime)

OUTPUT_NAME = "mask.nii.gz"
# The bundle's configs/metadata.json network_data_format.outputs.pred.channel_def.
LABELS: tuple[Label, ...] = (
    {"value": 0, "name": "background"},
    {"value": 1, "name": "kidney cortex"},
    {"value": 2, "name": "medulla"},
    {"value": 3, "name": "pelvicalyceal system"},
)

MODEL = {
    "id": "unest",
    "label": "Renal structures UNesT segmentation",
    "task": "segmentation",
    "run": "unest_mlx.medmlx:run",
    "summary": (
        "Kidney cortex, medulla, and pelvicalyceal system segmentation for one CT "
        "volume with the MONAI renalStructures_UNEST_segmentation 0.2.7 bundle."
    ),
    "labels": {str(label["value"]): label["name"] for label in LABELS},
}
# configs/inference.json: SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=4,
# overlap=0.5). The roi size and overlap are fixed in UNesTPredictor.
SW_BATCH_SIZE = 4
_WEIGHTS_HINT = (
    "Download unest.safetensors from https://huggingface.co/MedMLX/unest-renal-ct-mlx"
)


def run(
    *,
    image_path: str | os.PathLike[str],
    weights_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str],
    overwrite: bool = False,
) -> RunResult:
    """Segment kidney cortex, medulla, and pelvicalyceal system in one CT volume.

    Preprocessing follows the bundle: reorient to RAS, resample to 1 mm spacing
    (bilinear), clip -175 to 250 HU and scale to 0-1. Windows of 96x96x96 voxels with 0.5
    overlap are run through the network, softmaxed, resampled back to the input grid and
    reduced with argmax. The model targets contrast-enhanced CT (arterial or portal venous
    phase) and has failed on non-contrast scans.

    Args:
        image_path: 3D CT NIfTI (``.nii`` or ``.nii.gz``) in Hounsfield units, with
            contrast enhancement. Header slope and intercept are applied on read.
            Spatial units must be mm or unspecified (assumed mm).
        weights_path: Optional model folder containing ``unest.safetensors``.
            Omit to download and reuse ``MedMLX/unest-renal-ct-mlx`` automatically. Explicit
            native ``.safetensors`` checkpoint paths are also accepted.
        output_dir: Directory for ``mask.nii.gz``; created when the mask is written.
        overwrite: Replace an existing mask in ``output_dir`` (default ``False``);
            input aliases are refused.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}``. The mask keeps the input's
        shape and reader affine; upstream header correction may replace a conflicting
        affine with the form matching the header zooms. Like ``SaveImaged`` it stores
        float32 label values 0-3 with sform and qform code 1 and mm spatial units.

    Raises:
        InvalidInputError: The image is missing, not a finite 3D NIfTI with a valid
            affine, ``overwrite`` is not a bool, ``output_dir`` is not a directory, or the
            mask exists and ``overwrite`` is false.
        AssetNotReadyError: The weights are missing, unreadable, not ``.safetensors``,
            or do not match the released tensor inventory.
        ModelExecutionError: The network produced non-finite logits or invalid labels.
        MissingDependencyError: MLX or its Metal GPU is unavailable (no CPU fallback).
    """

    started = time.perf_counter()
    # Check every argument before downloading weights or loading MLX.
    source = _require_nifti(image_path)
    if not isinstance(cast(object, overwrite), bool):
        raise InvalidInputError("overwrite must be true or false")
    directory = Path(output_dir).expanduser().absolute()
    if directory.exists() and not directory.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {directory}")
    mask_path = directory / OUTPUT_NAME
    if mask_path.resolve() == source.resolve() or (
        mask_path.exists() and mask_path.samefile(source)
    ):
        raise InvalidInputError("UNesT output must differ from the input image")
    if mask_path.exists() and not overwrite:
        raise InvalidInputError(f"{mask_path} already exists; pass overwrite=true to replace it")
    bundle_dir = (
        weights_dir("unest") if weights_path is None
        else Path(weights_path).expanduser()
    )
    # Preserve explicit checkpoint-file callers and named cache symlinks.
    checkpoint = bundle_dir / "unest.safetensors" if bundle_dir.is_dir() else bundle_dir
    weights = _require_weights(str(checkpoint))
    from unest_mlx import transforms

    image, data = read_volume(source)
    reader_affine = transforms.correct_nifti_header_if_necessary(image).affine
    network_input, geometry = transforms.preprocess(data, reader_affine)
    mx = cast(NativeRuntime, import_mlx())
    from unest_mlx.mlx_predictor import UNesTPredictor

    network = _load_network(weights, mx=mx)
    predictor = UNesTPredictor(network, mx=mx, sw_batch_size=SW_BATCH_SIZE)
    try:
        logits = predictor(network_input[None])
    except RuntimeError as error:
        raise ModelExecutionError(str(error)) from error
    labels = transforms.postprocess(logits[0], geometry)
    if labels.shape != geometry.original_shape:
        raise ModelExecutionError("UNesT mask does not match the input grid")
    if not set(np.unique(labels).tolist()) <= {label["value"] for label in LABELS}:
        raise ModelExecutionError("UNesT exported labels outside the bundle contract")
    write_mask(mask_path, labels, reader_affine)
    return {
        "outputs": {"mask": mask_path.resolve().as_posix()},
        "details": {
            "weights_revision": weights_revision(weights.parent),
            "labels": [cast(Label, dict(label)) for label in LABELS],
            "voxel_counts": {
                str(label["name"]): int(np.count_nonzero(labels == label["value"]))
                for label in LABELS
            },
            "shape": [int(value) for value in labels.shape],
            "dtype": "float32",
            "input_orientation": "".join(nib.aff2axcodes(reader_affine)),
            "network_shape": [int(value) for value in geometry.network_shape],
            "sliding_window": {
                "roi_size": [96, 96, 96],
                "sw_batch_size": SW_BATCH_SIZE,
                "overlap": 0.5,
            },
            "mlx_device": str(mx.default_device()),
            "duration_seconds": round(time.perf_counter() - started, 3),
        },
    }


def _require_nifti(path: str | os.PathLike[str]) -> Path:
    """Validate the input NIfTI path without resolving its symlinks."""
    source = Path(path).expanduser().absolute()
    if not source.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"image_path must be a .nii or .nii.gz file: {source}")
    if not source.is_file():
        raise InvalidInputError(f"image_path does not exist: {source}")
    return source


def _require_weights(path: str | os.PathLike[str]) -> Path:
    """Validate the native safetensors checkpoint path without resolving symlinks."""
    weights = Path(path).expanduser().absolute()
    if not weights.is_file():
        raise AssetNotReadyError(
            f"UNesT weights not found: {weights}; expected a .safetensors file",
            reason=f"missing {weights}",
            hint=_WEIGHTS_HINT,
        )
    if weights.suffix != ".safetensors":
        raise AssetNotReadyError(
            f"UNesT weights must be a .safetensors file, got {weights.name}",
            reason=f"unconverted {weights.name}",
            hint=_WEIGHTS_HINT,
        )
    return weights


def read_volume(path: Path) -> tuple[NiftiImage, HostArray]:
    """Read real finite 3D NIfTI voxels in the float32 range on a millimetre grid.

    The array is ``np.asanyarray(image.dataobj)`` in file axis order, with the header's
    slope and intercept applied. Which affine (sform or qform) defines the grid is decided
    later by ``transforms.correct_nifti_header_if_necessary``, as in the upstream reader.

    Raises:
        InvalidInputError: The file or spatial units are invalid, voxels are not real
            finite float32-range values, the affine is non-finite or singular, or header
            spacing is not finite and positive.
    """

    try:
        image = nib.load(path.as_posix())
        data = np.asanyarray(image.dataobj)
    except (OSError, ValueError, ImageFileError) as error:
        raise InvalidInputError(f"Invalid UNesT input NIfTI: {path}") from error
    if not isinstance(image, nib_runtime.Nifti1Image | nib_runtime.Nifti2Image):
        raise InvalidInputError(f"UNesT input must be NIfTI: {path}")
    # The bundle treats affine coordinates as mm and does not convert NIfTI units.
    if (int(image.header["xyzt_units"]) & 0x07) not in (0, 2):
        raise InvalidInputError("UNesT input spatial units must be mm or unspecified")
    if data.ndim != 3:
        raise InvalidInputError(f"UNesT input must be a 3D NIfTI; got shape {data.shape}")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise InvalidInputError("UNesT input must contain real-valued voxels")
    if not np.all(np.isfinite(data)):
        raise InvalidInputError("UNesT input contains non-finite voxels")
    limit = np.finfo(np.float32).max
    if np.any(data > limit) or np.any(data < -limit):
        raise InvalidInputError("UNesT input contains voxels outside the finite float32 range")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise InvalidInputError("UNesT input affine must be a finite 4x4 matrix")
    if np.linalg.det(affine[:3, :3]) == 0:
        raise InvalidInputError("UNesT input affine must be invertible")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise InvalidInputError("UNesT input spacing_mm values must be finite and positive")
    return cast(NiftiImage, image), data


def _load_network(path: Path, *, mx: NativeRuntime) -> NativeNetwork:
    """Load ``path``, validate it against the released inventory and build the MLX network.

    Any read or inventory failure is reported as ``AssetNotReadyError``.
    """

    from unest_mlx import mlx_weights
    from unest_mlx.mlx_model import UNesT

    try:
        loaded = mx.load(str(path))
    except (OSError, RuntimeError, ValueError) as error:
        raise AssetNotReadyError(
            f"Cannot read UNesT weights {path}: {error}",
            reason=f"unreadable {path.name}",
            hint=_WEIGHTS_HINT,
        ) from error
    if not isinstance(loaded, dict):
        raise AssetNotReadyError(
            f"UNesT weights are not a tensor mapping: {path}",
            reason=f"unreadable {path.name}",
            hint=_WEIGHTS_HINT,
        )
    try:
        state = {str(key): cast(HostArray, np.array(value)) for key, value in loaded.items()}
        weights = mlx_weights.validate_weights(state)
    except (InvalidInputError, RuntimeError, TypeError, ValueError) as error:
        raise AssetNotReadyError(
            f"UNesT weights do not match the released 0.2.7 checkpoint: {error}",
            reason=f"mismatched {path.name}",
            hint=_WEIGHTS_HINT,
        ) from error
    arrays = {key: mx.array(value) for key, value in weights.items()}
    mx.eval(arrays)
    return UNesT(arrays, mx=mx)


def write_mask(path: Path, labels: FloatArray, affine: DoubleArray) -> None:
    """Write a label map like MONAI's ``NibabelWriter``, replacing ``path`` atomically.

    Args:
        path: Destination for the gzip-compressed NIfTI. Parent directories are created.
        labels: Label values on the reader grid, stored as float32.
        affine: 4x4 voxel-to-world affine in mm, written as sform and qform with code 1.
    """

    image = nib.Nifti1Image(np.asarray(labels, dtype=np.float32), affine)
    image.header["xyzt_units"] = 2  # NIfTI millimetres; the mask has no time axis.
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so a failed run never leaves a truncated mask.
    partial = path.with_name(f".{path.name[: -len('.nii.gz')]}.{os.getpid()}.partial.nii.gz")
    try:
        nib.save(image, partial)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
