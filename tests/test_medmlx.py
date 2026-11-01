"""Public runner checked against the pinned upstream Torch/MONAI pipeline.

The end-to-end case uses a synthetic CT phantom and seeded reduced-width weights, not the
released checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn, cast

import nibabel as nib_runtime
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from tests._contracts import RunnerMetadata
from tests.fixture_cases import fixture_metadata, recording, runner_weights, write_ct_phantom
from unest_mlx import medmlx, mlx_weights
from unest_mlx._typing import NativeRuntime, NiftiIo

nib = cast(NiftiIo, nib_runtime)


@pytest.mark.mlx
def test_run_matches_upstream_pipeline_bitwise(
    tmp_path: Path, mx: NativeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the seeded phantom and compare the saved mask with the recorded upstream pipeline."""
    weights = runner_weights(mlx_weights.required_shapes())
    checkpoint = tmp_path / "unest.safetensors"
    mx.save_safetensors(str(checkpoint), {key: mx.array(value) for key, value in weights.items()})
    image = tmp_path / "ct_phantom.nii.gz"
    write_ct_phantom(image)
    # Only the expected inventory is reduced; loading, admission, and the graph are real.
    shapes = {key: value.shape for key, value in weights.items()}
    monkeypatch.setattr(mlx_weights, "required_shapes", lambda: dict(shapes))
    result = medmlx.run(image_path=image, weights_path=str(checkpoint), output_dir=tmp_path / "out")
    written = nib.load(result["outputs"]["mask"])
    with np.load(recording("runner_end_to_end"), allow_pickle=False) as fixture:
        info = cast(RunnerMetadata, fixture_metadata(fixture))
        assert (info["windows"], info["sw_batch_size"], info["window_batches"]) == (2, 4, [2])
        assert np.array_equal(np.asanyarray(nib.load(image).dataobj), fixture["input_data"])
        assert np.array_equal(np.asanyarray(written.dataobj), fixture["saved"])
        assert np.array_equal(written.affine, fixture["saved_affine"])
        assert fixture["saved_codes"].tolist() == [1, 1]


@pytest.mark.parametrize("problem", ["bad_suffix", "non_bool_overwrite", "existing_output"])
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
    overwrite = cast(bool, "false") if problem == "non_bool_overwrite" else False
    with pytest.raises(InvalidInputError):
        medmlx.run(image_path=source, output_dir=output_dir, overwrite=overwrite)


@pytest.mark.parametrize(
    "defect", ["missing", "torch", pytest.param("mismatched", marks=pytest.mark.mlx)]
)
def test_rejects_missing_or_unconverted_weights(
    tmp_path: Path, defect: str, request: pytest.FixtureRequest
) -> None:
    """Reject missing, non-safetensors or inventory-mismatched weights."""
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.float32), np.eye(4)), str(image))
    weights = tmp_path / ("model.pt" if defect == "torch" else f"{defect}.safetensors")
    if defect == "torch":
        weights.write_bytes(b"")
    elif defect == "mismatched":
        # A checkpoint holding only one of the released tensors fails the strict inventory.
        mx = request.getfixturevalue("mx")
        mx.save_safetensors(str(weights), {"out.conv.conv.bias": mx.zeros((4,))})
    with pytest.raises(AssetNotReadyError):
        medmlx.run(image_path=image, weights_path=weights, output_dir=tmp_path / "out")
