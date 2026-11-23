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

import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError
from medmlx_core.files import output_targets, publish
from medmlx_core.outputs import artifact
from medmlx_core.runtime import import_mlx
from medmlx_core.transforms import axis_codes
from medmlx_core.typing import MlxRuntime
from medmlx_core.volumes import read_volume, write_volume
from medmlx_core.weights import load_tensors, weights_file, weights_revision

from unest_mlx._typing import Label, NativeNetwork, RunResult

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
    # Measured MLX peak for a 478x277x161 contrast CT; larger inputs need more.
    "peak_memory_gb": 7.3,
    "summary": (
        "Kidney cortex, medulla, and pelvicalyceal system segmentation for one CT "
        "volume with the MONAI renalStructures_UNEST_segmentation 0.2.7 bundle."
    ),
    "labels": {str(label["value"]): label["name"] for label in LABELS},
}
# configs/inference.json: SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=4,
# overlap=0.5). The roi size and overlap are fixed in UNesTPredictor.
SW_BATCH_SIZE = 4


def run(
    *,
    image_path: str | os.PathLike[str],
    weights_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str],
    overwrite: bool = False,
    sw_batch_size: int = SW_BATCH_SIZE,
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
            Omit to download and reuse ``MedMLX/unest-renal-ct-mlx`` automatically.
        output_dir: Directory for ``mask.nii.gz``; created when the mask is written.
        overwrite: Replace an existing mask in ``output_dir`` (default ``False``);
            input aliases are refused.
        sw_batch_size: Positive number of windows evaluated together; defaults to
            the source batch of four. Smaller batches reduce activation residency.

    Returns:
        ``{"outputs": {"mask": path}, "artifacts": {"mask": ...}, "details": {...}}``. The
        mask artifact is a ``label_map`` in ``image_physical`` space on ``image_path``'s grid,
        labelled with the four bundle classes (``LABELS``). The mask keeps the input's
        shape and reader affine; upstream header correction may replace a conflicting
        affine with the form matching the header zooms. Like ``SaveImaged`` it stores
        float32 label values 0-3 with sform and qform code 1 and mm spatial units.

    Raises:
        InvalidInputError: The image is missing, not a finite 3D NIfTI with a valid
            affine and float32-range voxels (``medmlx_core.volumes.read_volume``), has a
            spatial unit other than mm or unspecified, ``overwrite`` is not a bool,
            ``output_dir`` is not a directory, or the mask exists and ``overwrite`` is
            false.
        AssetNotReadyError: ``weights_path`` is not a folder, or its
            ``unest.safetensors`` is missing, unreadable or does not match the released
            tensor inventory.
        ModelExecutionError: The network failed, or produced logits of the wrong shape,
            non-finite logits or invalid labels.
        MissingDependencyError: MLX or its Metal GPU is unavailable (no CPU fallback).
    """

    started = time.perf_counter()
    # Check every argument before downloading weights or loading MLX.
    source = _require_nifti(image_path)
    if type(sw_batch_size) is not int or sw_batch_size <= 0:
        raise InvalidInputError("sw_batch_size must be a positive integer")
    mask_path = output_targets(
        output_dir, {"mask": OUTPUT_NAME}, overwrite=overwrite, inputs=[source]
    )["mask"]
    weights = weights_file("unest", weights_path, "unest.safetensors")
    from unest_mlx import transforms

    volume = read_volume(source)
    # The bundle treats affine coordinates as mm and does not convert NIfTI units.
    if (int(volume.header["xyzt_units"]) & 0x07) not in (0, 2):
        raise InvalidInputError("image_path spatial units must be mm or unspecified")
    reader_affine = transforms.reader_affine(volume)
    network_input, geometry = transforms.preprocess(volume.data, reader_affine)
    del volume
    mx = cast(MlxRuntime, import_mlx())
    from unest_mlx.mlx_predictor import UNesTPredictor

    network = _load_network(weights, mx=mx)
    predictor = UNesTPredictor(network, mx=mx, sw_batch_size=sw_batch_size)
    try:
        logits = predictor(network_input[None])
    except RuntimeError as error:
        raise ModelExecutionError(str(error)) from error
    logits = logits[0]
    if logits.ndim != 4 or tuple(logits.shape[1:]) != geometry.network_shape:
        raise ModelExecutionError(
            f"UNesT logits {logits.shape} do not match {geometry.network_shape}"
        )
    # Softmax into the logits buffer when it is contiguous, then drop the logits so
    # they are not held alongside the probabilities during resampling.
    probabilities = transforms.softmax(
        logits, out=logits if logits.flags.c_contiguous else None
    )
    del logits
    labels = transforms.labels_from_probabilities(probabilities, geometry)
    del probabilities
    if labels.shape != geometry.original_shape:
        raise ModelExecutionError("UNesT mask does not match the input grid")
    if not set(np.unique(labels).tolist()) <= {label["value"] for label in LABELS}:
        raise ModelExecutionError("UNesT exported labels outside the bundle contract")
    # Like MONAI's NibabelWriter: float32 labels, sform and qform code 1, mm units.
    publish(
        {
            mask_path: lambda partial: write_volume(
                partial, labels, reader_affine, dtype=np.float32
            )
        }
    )
    return {
        "outputs": {"mask": mask_path.resolve().as_posix()},
        "artifacts": {
            "mask": artifact(
                kind="label_map",
                coordinate_space="image_physical",
                grid_of="image_path",
                labels=[{"value": label["value"], "name": label["name"]} for label in LABELS],
            )
        },
        "details": {
            "weights_revision": weights_revision(weights.parent),
            "labels": [cast(Label, dict(label)) for label in LABELS],
            "voxel_counts": {
                str(label["name"]): int(np.count_nonzero(labels == label["value"]))
                for label in LABELS
            },
            "shape": [int(value) for value in labels.shape],
            "dtype": "float32",
            "input_orientation": axis_codes(reader_affine),
            "network_shape": [int(value) for value in geometry.network_shape],
            "sliding_window": {
                "roi_size": [96, 96, 96],
                "sw_batch_size": sw_batch_size,
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


def _load_network(path: Path, *, mx: MlxRuntime) -> NativeNetwork:
    """Load ``path``, check it against the released inventory and build the MLX network."""

    from unest_mlx import mlx_weights
    from unest_mlx.mlx_model import UNesT

    weights = load_tensors(
        path, model_id="unest", label="UNesT", shapes=mlx_weights.required_shapes()
    )
    return UNesT(weights, mx=mx)
