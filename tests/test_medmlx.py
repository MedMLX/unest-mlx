"""Public runner checked against the pinned upstream Torch/MONAI pipeline.

The end-to-end case uses an HCC-TACE CT crop and seeded reduced-width weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn, cast

import nibabel as nib_runtime
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from tests._contracts import RunnerMetadata
from tests.fixture_cases import ct_path, fixture_metadata, recording, runner_weights, write_ct_crop
from unest_mlx import medmlx, mlx_weights
from unest_mlx._typing import NativeRuntime, NiftiIo

nib = cast(NiftiIo, nib_runtime)


@pytest.mark.mlx
@pytest.mark.parametrize("batch_size", [1, 4])
def test_run_matches_upstream_pipeline_bitwise(
    tmp_path: Path, mx: NativeRuntime, monkeypatch: pytest.MonkeyPatch, batch_size: int,
) -> None:
    """Compare the real CT's saved mask with the recorded upstream pipeline."""
    weights = runner_weights(mlx_weights.required_shapes())
    checkpoint = tmp_path / "unest.safetensors"
    mx.save_safetensors(str(checkpoint), {key: mx.array(value) for key, value in weights.items()})
    image = tmp_path / "ct_crop.nii.gz"
    write_ct_crop(image)
    # Only the expected inventory is reduced; loading, admission, and the graph are real.
    shapes = {key: value.shape for key, value in weights.items()}
    monkeypatch.setattr(mlx_weights, "required_shapes", lambda: dict(shapes))
    result = medmlx.run(
        image_path=image, weights_path=str(checkpoint), output_dir=tmp_path / "out",
        sw_batch_size=batch_size,
    )
    assert result["details"]["sliding_window"]["sw_batch_size"] == batch_size
    written = nib.load(result["outputs"]["mask"])
    with np.load(recording("runner_end_to_end"), allow_pickle=False) as fixture:
        info = cast(RunnerMetadata, fixture_metadata(fixture))
        assert (info["windows"], info["sw_batch_size"], info["window_batches"]) == (2, 4, [2])
        assert np.array_equal(np.asanyarray(nib.load(image).dataobj), fixture["input_data"])
        assert np.array_equal(np.asanyarray(written.dataobj), fixture["saved"])
        assert np.array_equal(written.affine, fixture["saved_affine"])
        assert fixture["saved_codes"].tolist() == [1, 1]


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

    monkeypatch.setattr(medmlx, "weights_dir", download)
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
    "defect", ["missing", "torch", pytest.param("mismatched", marks=pytest.mark.mlx)]
)
def test_rejects_missing_or_unconverted_weights(
    tmp_path: Path, defect: str, request: pytest.FixtureRequest
) -> None:
    """Reject missing, non-safetensors or inventory-mismatched weights."""
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
    weights = tmp_path / ("model.pt" if defect == "torch" else f"{defect}.safetensors")
    if defect == "torch":
        weights.write_bytes(b"")
    elif defect == "mismatched":
        # A checkpoint holding only one of the released tensors fails the strict inventory.
        mx = request.getfixturevalue("mx")
        mx.save_safetensors(str(weights), {"out.conv.conv.bias": mx.zeros((4,))})
    with pytest.raises(AssetNotReadyError):
        medmlx.run(image_path=image, weights_path=weights, output_dir=tmp_path / "out")
