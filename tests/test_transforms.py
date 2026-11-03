"""Bitwise preprocessing, inversion, and writing against seeded upstream MONAI recordings."""

import os
from pathlib import Path
from typing import cast

import nibabel as nib_runtime
import numpy as np
import pytest

from tests._contracts import RecordedTransforms
from tests.fixture_cases import (
    HEADER_CASES,
    RUNNER_CASES,
    fixture_metadata,
    recording,
    write_header_case,
    write_runner_case,
)
from unest_mlx import medmlx, transforms
from unest_mlx._typing import DoubleArray, HostArray, NiftiIo

nib = cast(NiftiIo, nib_runtime)

Case = tuple[DoubleArray, HostArray, RecordedTransforms]

# --- bundle transforms ---------------------------------------------------------------


@pytest.fixture(params=sorted((*RUNNER_CASES, *HEADER_CASES)))
def case(request: pytest.FixtureRequest, tmp_path: Path) -> Case:
    """Load an upstream transform fixture case."""
    path = recording(f"runner_{request.param}")
    with np.load(path, allow_pickle=False) as fixture:
        expected = {key: fixture[key] for key in fixture.files}
        info = fixture_metadata(fixture)
    expected.pop("metadata")
    assert info["torch_threads"] == 1
    path = tmp_path / f"{request.param}.nii.gz"
    if request.param in RUNNER_CASES:
        write_runner_case(request.param, path)
    else:
        write_header_case(request.param, path)
    image, data = medmlx.read_volume(path)
    assert np.array_equal(data, expected["input_data"])
    assert np.array_equal(image.affine, expected["input_affine"])
    affine = transforms.correct_nifti_header_if_necessary(image).affine
    assert np.array_equal(affine, expected["reader_affine"])
    return affine, data, cast(RecordedTransforms, expected)


def test_preprocessing_matches_monai_bitwise(case: Case) -> None:
    """Compare preprocessing with the MONAI recording bitwise."""
    affine, data, expected = case
    image, geometry = transforms.preprocess(data, affine)
    assert image.dtype == expected["image"].dtype == np.float32
    assert np.array_equal(image, expected["image"])
    assert np.array_equal(geometry.network_affine, expected["network_affine"])
    assert geometry.network_shape == expected["image"].shape[1:]


def test_postprocessing_matches_monai_bitwise(case: Case) -> None:
    """Compare postprocessing with the MONAI recording bitwise."""
    affine, data, expected = case
    _, geometry = transforms.preprocess(data, affine)
    probabilities = transforms.softmax(expected["logits"])
    assert np.array_equal(probabilities, expected["probabilities"])
    inverted, inverted_affine = transforms.invert(probabilities, geometry)
    assert np.array_equal(inverted, expected["inverted"])
    assert np.array_equal(inverted_affine, expected["inverted_affine"])
    labels = transforms.postprocess(expected["logits"], geometry)
    assert labels.dtype == expected["saved"].dtype == np.float32
    assert np.array_equal(labels, expected["labels"])
    assert np.array_equal(labels, expected["saved"])
    # Header correction defines the reader grid preserved by upstream inversion/writing.
    assert np.array_equal(expected["saved_affine"], expected["reader_affine"])
    assert expected["saved_codes"].tolist() == [1, 1]


def test_writer_matches_upstream_saved_data_and_grid(case: Case, tmp_path: Path) -> None:
    """Compare saved image data and geometry with the upstream recording."""
    _, _, expected = case
    path = tmp_path / "mask.nii.gz"
    medmlx.write_mask(path, expected["labels"], expected["reader_affine"])
    image = nib.load(str(path))
    assert np.array_equal(np.asanyarray(image.dataobj), expected["saved"])
    assert np.array_equal(image.affine, expected["saved_affine"])
    assert [int(image.header[key]) for key in ("qform_code", "sform_code")] == [1, 1]


def test_writer_does_not_follow_a_staging_symlink(tmp_path: Path) -> None:
    """A writable output folder cannot redirect the mask write to another file."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    (tmp_path / f".mask.{os.getpid()}.partial.nii.gz").symlink_to(victim)
    path = tmp_path / "mask.nii.gz"

    medmlx.write_mask(path, np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    assert victim.read_bytes() == b"keep"
    assert nib.load(str(path)).shape == (2, 2, 2)
