"""Public runner checked against the pinned upstream Torch/MONAI pipeline.

The end-to-end case uses an HCC-TACE CT crop and seeded reduced-width weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn, cast

import medmlx_core.weights as core_weights
import nibabel as nib_runtime
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.outputs import validate_outputs
from medmlx_core.typing import MlxRuntime

from tests._contracts import NiftiIo, RunnerMetadata
from tests.fixture_cases import ct_path, fixture_metadata, recording, runner_weights, write_ct_crop
from unest_mlx import medmlx, mlx_weights

nib = cast(NiftiIo, nib_runtime)


@pytest.mark.mlx
@pytest.mark.parametrize("batch_size", [1, 4])
def test_run_matches_upstream_pipeline_bitwise(
    tmp_path: Path, mx: MlxRuntime, monkeypatch: pytest.MonkeyPatch, batch_size: int,
) -> None:
    """Compare the real CT's saved mask with the recorded upstream pipeline."""
    weights = runner_weights(mlx_weights.required_shapes())
    checkpoint = tmp_path / "weights" / "unest.safetensors"
    checkpoint.parent.mkdir()
    mx.save_safetensors(str(checkpoint), {key: mx.array(value) for key, value in weights.items()})
    image = tmp_path / "ct_crop.nii.gz"
    write_ct_crop(image)
    # Only the expected inventory is reduced; loading, admission, and the graph are real.
    shapes = {key: value.shape for key, value in weights.items()}
    monkeypatch.setattr(mlx_weights, "required_shapes", lambda: dict(shapes))
    result = medmlx.run(
        image_path=image, weights_path=checkpoint.parent, output_dir=tmp_path / "out",
        sw_batch_size=batch_size,
    )
    assert result["details"]["sliding_window"]["sw_batch_size"] == batch_size
    mask_artifact = validate_outputs(result).artifacts["mask"]
    assert (mask_artifact["kind"], mask_artifact.get("grid_of")) == ("label_map", "image_path")
    assert mask_artifact.get("labels") == [
        {"value": label["value"], "name": label["name"]} for label in medmlx.LABELS
    ]
    written = nib.load(result["outputs"]["mask"])
    with np.load(recording("runner_end_to_end"), allow_pickle=False) as fixture:
        info = cast(RunnerMetadata, fixture_metadata(fixture))
        assert (info["windows"], info["sw_batch_size"], info["window_batches"]) == (2, 4, [2])
        assert np.array_equal(np.asanyarray(nib.load(image).dataobj), fixture["input_data"])
        assert np.array_equal(np.asanyarray(written.dataobj), fixture["saved"])
        assert np.array_equal(written.affine, fixture["saved_affine"])
        assert fixture["saved_codes"].tolist() == [1, 1]


@pytest.mark.mlx
@pytest.mark.parametrize(("unit", "scale"), [("meter", 1e-3), ("micron", 1e3)])
def test_run_converts_meter_and_micron_headers_to_millimeters(
    tmp_path: Path, mx: MlxRuntime, monkeypatch: pytest.MonkeyPatch, unit: str, scale: float,
) -> None:
    """A copy of the CT in meters or microns gives the millimeter mask and its mm geometry."""
    weights = runner_weights(mlx_weights.required_shapes())
    checkpoint = tmp_path / "weights" / "unest.safetensors"
    checkpoint.parent.mkdir()
    mx.save_safetensors(str(checkpoint), {key: mx.array(value) for key, value in weights.items()})
    shapes = {key: value.shape for key, value in weights.items()}
    monkeypatch.setattr(mlx_weights, "required_shapes", lambda: dict(shapes))
    millimeter = tmp_path / "ct_mm.nii.gz"
    write_ct_crop(millimeter)
    source = nib.load(millimeter)
    affine = source.affine.copy()
    affine[:3] *= scale
    converted = nib.Nifti1Image(np.asanyarray(source.dataobj), affine)
    converted.set_qform(affine, code=1)
    converted.set_sform(affine, code=1)
    converted.header.set_xyzt_units(unit)
    other = tmp_path / f"ct_{unit}.nii.gz"
    nib.save(converted, other)
    masks = [
        nib.load(
            medmlx.run(
                image_path=path, weights_path=checkpoint.parent, output_dir=tmp_path / name
            )["outputs"]["mask"]
        )
        for path, name in ((millimeter, "mm"), (other, unit))
    ]
    assert np.array_equal(np.asanyarray(masks[0].dataobj), np.asanyarray(masks[1].dataobj))
    assert np.allclose(masks[0].affine, masks[1].affine, rtol=0.0, atol=1e-4)
    assert np.allclose(masks[1].affine, source.affine, rtol=0.0, atol=1e-4)


@pytest.mark.parametrize(
    "problem", ["bad_suffix", "non_bool_overwrite", "existing_output", "dangling_output",
                "zero_batch", "bool_batch"]
)
def test_run_rejects_bad_arguments_before_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Each rejection category fails before the weights download starts."""

    def download(model_id: str) -> NoReturn:
        """Fail if the download starts before the refusal."""
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr(core_weights, "weights_dir", download)
    source = tmp_path / ("scan.txt" if problem == "bad_suffix" else "scan.nii.gz")
    source.write_bytes(b"unread")
    output_dir = tmp_path / "out"
    if problem == "existing_output":
        output_dir.mkdir()
        (output_dir / medmlx.OUTPUT_NAME).write_bytes(b"kept")
    elif problem == "dangling_output":
        output_dir.mkdir()
        (output_dir / medmlx.OUTPUT_NAME).symlink_to(tmp_path / "missing.nii.gz")
    overwrite = cast(bool, "false") if problem == "non_bool_overwrite" else False
    with pytest.raises(InvalidInputError):
        medmlx.run(
            image_path=source, output_dir=output_dir, overwrite=overwrite,
            sw_batch_size=0 if problem == "zero_batch" else cast(int, True)
            if problem == "bool_batch" else 4,
        )


@pytest.mark.parametrize(
    "defect", ["missing", "torch", "file", pytest.param("mismatched", marks=pytest.mark.mlx)]
)
def test_rejects_missing_or_unconverted_weights(
    tmp_path: Path, defect: str, request: pytest.FixtureRequest
) -> None:
    """Reject a checkpoint file, a folder without the native checkpoint, or bad tensors."""
    image = tmp_path / "image.nii.gz"
    if defect == "mismatched":
        volume = nib.load(ct_path())
        crop = np.asanyarray(volume.dataobj)[200:216, 200:216, 40:48]
        affine = volume.affine.copy()
        affine[:3, 3] += affine[:3, :3] @ np.array([200, 200, 40])
        nib.save(nib.Nifti1Image(crop, affine), image)
    else:
        # These cases reject the checkpoint before reading the input file.
        image.touch()
    folder = tmp_path / "weights"
    folder.mkdir()
    weights = folder / ("model.pt" if defect == "torch" else "unest.safetensors")
    if defect in ("torch", "file"):
        weights.write_bytes(b"")
    elif defect == "mismatched":
        # A checkpoint holding only one of the released tensors fails the strict inventory.
        mx = request.getfixturevalue("mx")
        mx.save_safetensors(str(weights), {"out.conv.conv.bias": mx.zeros((4,))})
    given = weights if defect == "file" else folder
    with pytest.raises(AssetNotReadyError):
        medmlx.run(image_path=image, weights_path=given, output_dir=tmp_path / "out")
