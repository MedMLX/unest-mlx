"""NumPy port of the renal UNesT bundle's MONAI 1.6.0 image transforms.

The pinned bundle configuration (``configs/inference.json``, no ``_disabled_`` steps)
preprocesses with ``LoadImaged`` (nibabel reader) -> ``EnsureChannelFirstd`` ->
``Orientationd("RAS")`` -> ``Spacingd(1 mm, bilinear)`` -> ``ScaleIntensityRanged``
(-175..250 HU -> 0..1, clipped), and postprocesses with ``Activationsd(softmax)`` ->
``Invertd`` (bilinear inverse of the spacing and orientation) ->
``AsDiscreted(argmax)``. Reference recordings use the upstream MONAI 1.6.0 package.

Conventions: volumes are channel first, ``(C, X, Y, Z)``, where the three spatial axes
are the array's own storage axes (after ``preprocess`` they are the RAS axes). Affines
are float64 4x4 voxel-to-world matrices in millimeters, as in nibabel.

MONAI resamples through Torch ``affine_grid`` and ``grid_sample`` in float64.
The affine arithmetic and vectorized softmax follow the pinned Torch reference
recorded on macOS arm64 with one thread. The NumPy implementation preserves the
reference's multiply-add grouping and float32 exponential rounding. Preprocessing,
probabilities, inversion, labels, and written grids are compared exactly with those
upstream recordings (synthetic inputs only); other recording platforms are outside the
package's scope.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

import nibabel as nib_runtime
import numpy as np
from numpy.typing import ArrayLike, NDArray

from unest_mlx._typing import (
    DoubleArray,
    FloatArray,
    HostArray,
    IntArray,
    NiftiImage,
    NiftiIo,
    NumpyFlip,
    RealArray,
)

nib = cast(NiftiIo, nib_runtime)

AFFINE_TOL = 1e-3
PIXDIM = (1.0, 1.0, 1.0)
AXCODES = "RAS"
# ScaleIntensityRanged window: -175..250 HU is mapped linearly to 0..1 and clipped.
A_MIN, A_MAX, B_MIN, B_MAX = -175, 250, 0.0, 1.0
# Resampling and softmax blocks run concurrently, each with its own temporaries.
_THREADS = min(8, os.cpu_count() or 1)
# Output voxels per block. The concurrent blocks together span the 2**21 voxels
# that bounded the serial loop's float64 temporaries, so threading does not
# raise peak memory. Every voxel's value is independent of the block size.
_BLOCK_VOXELS = max(1, (1 << 21) // _THREADS)
# Float32 lanes used by the pinned Torch reference's vectorized softmax.
_VECTOR_LANES = 8
_SPLIT = 134217729.0  # 2**27 + 1, Veltkamp splitting constant for float64


@dataclass(frozen=True)
class Geometry:
    """Grids and affines tracked by the forward transforms for the inversion.

    Shapes are voxel counts per spatial axis; affines are 4x4 voxel-to-world matrices
    interpreted in mm. These transforms do not inspect NIfTI unit metadata.
    ``original_*`` is the reader grid of the input volume, ``oriented_*`` the grid after
    reorientation to RAS, and ``network_*`` the 1 mm grid the network runs on.
    """

    original_affine: DoubleArray
    original_shape: tuple[int, int, int]
    oriented_affine: DoubleArray
    oriented_shape: tuple[int, int, int]
    network_affine: DoubleArray
    network_shape: tuple[int, int, int]


# --- exact float64 arithmetic used by Torch's CPU kernels -------------------------
# These helpers reproduce Torch's CPU rounding (fused multiply-add, summation order) so
# that sampling coordinates match the reference recordings exactly.


def _two_sum(a: DoubleArray, b: DoubleArray) -> tuple[DoubleArray, DoubleArray]:
    """Error-free sum (Knuth): ``(s, e)`` with ``s = fl(a + b)`` and ``a + b = s + e``."""

    total = a + b
    shifted = total - a
    return total, (a - (total - shifted)) + (b - shifted)


def _two_product(a: DoubleArray, b: DoubleArray) -> tuple[DoubleArray, DoubleArray]:
    """Error-free product (Veltkamp-Dekker split): ``(p, e)`` with ``a * b = p + e`` exactly."""

    product = a * b
    scaled_a = _SPLIT * a
    high_a = scaled_a - (scaled_a - a)
    low_a = a - high_a
    scaled_b = _SPLIT * b
    high_b = scaled_b - (scaled_b - b)
    low_b = b - high_b
    error = ((high_a * high_b - product) + high_a * low_b + low_a * high_b) + low_a * low_b
    return product, error


def fma(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> DoubleArray:
    """Correctly rounded float64 ``a * b + c`` (Boldo-Melquiond, rounding to odd)."""

    a, b, c = np.broadcast_arrays(
        np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(c, np.float64)
    )
    product_high, product_low = _two_product(a, b)
    sum_high, sum_low = _two_sum(c, product_high)
    tail, error = _two_sum(sum_low, product_low)
    even = (tail.view(np.int64) & 1) == 0
    toward = np.where(error > 0, np.inf, -np.inf)
    tail = np.where((error != 0) & even, np.nextafter(tail, toward), tail)
    return sum_high + tail


def _torch_mm(left: DoubleArray, right: DoubleArray) -> DoubleArray:
    """Reference multiply-add grouping for float64 4x4 affine products."""

    first = fma(left[:, 2:3], right[2:3, :], left[:, 0:1] * right[0:1, :])
    second = fma(left[:, 3:4], right[3:4, :], left[:, 1:2] * right[1:2, :])
    return first + second


def _torch_bmm_small(left: DoubleArray, right: DoubleArray) -> DoubleArray:
    """Torch's naive CPU ``bmm`` kernel, used below 400 multiply-adds."""

    total = np.zeros((left.shape[0], right.shape[1]), dtype=np.float64)
    for index in range(left.shape[1]):
        total = total + left[:, index : index + 1] * right[index : index + 1, :]
    return total


# --- MONAI affine utilities ---------------------------------------------------------


def affine_to_spacing(affine: DoubleArray, rank: int = 3) -> DoubleArray:
    """Voxel size (mm) per axis: the column norms of the affine's linear block.

    A zero-length column is reported as 1.0, as in MONAI.
    """
    block = np.asarray(affine[:rank, :rank], dtype=np.float64)
    spacing = np.sqrt(np.sum(block * block, axis=0))
    spacing[spacing == 0] = 1.0
    return spacing


def correct_nifti_header_if_necessary(image: NiftiImage) -> NiftiImage:
    """MONAI's reader fix for NIfTI zooms that disagree with the affine (in place)."""

    if image.header.get("dim") is None:
        return image
    rank = int(image.header["dim"][0])
    if rank >= 5:
        return image
    pixdim = np.asarray(image.header.get_zooms())[:rank]
    if np.allclose(pixdim, affine_to_spacing(image.affine, rank)):
        return image
    if not hasattr(image, "get_sform"):
        return image
    sform, qform = image.get_sform(), image.get_qform()
    sform_mismatch = not np.allclose(affine_to_spacing(sform, rank), pixdim)
    qform_mismatch = not np.allclose(affine_to_spacing(qform, rank), pixdim)
    if image.header["sform_code"] != 0:
        if not sform_mismatch:
            return image
        if not qform_mismatch:
            image.set_sform(image.get_qform())
            return image
    if image.header["qform_code"] != 0:
        if not qform_mismatch:
            return image
        if not sform_mismatch:
            image.set_qform(image.get_sform())
            return image
    image.header.set_zooms(affine_to_spacing(image.affine, rank))
    return image


def zoom_affine(affine: DoubleArray, scale: DoubleArray) -> DoubleArray:
    """MONAI ``zoom_affine(..., diagonal=False)`` for a 4x4 affine."""

    affine = np.array(affine, dtype=float, copy=True)
    scale_np = np.array(scale, dtype=float, copy=True)
    norm = affine_to_spacing(affine, rank=len(affine) - 1)
    scale_np = np.asarray(
        [value if value > 0 else default for value, default in zip(scale_np, norm, strict=True)]
    )
    scale_np[scale_np == 0] = 1.0
    rzs = affine[:-1, :-1]
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    signed = np.sign(np.diag(zs)) * np.abs(scale_np)
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(signed)
    return new_affine


def compute_shape_offset(
    spatial_shape: tuple[int, ...], in_affine: DoubleArray, out_affine: DoubleArray
) -> tuple[IntArray, DoubleArray]:
    """MONAI ``compute_shape_offset(..., scale_extent=False)``."""

    shape = np.array(tuple(spatial_shape), copy=True, dtype=float)
    in_affine_ = np.array(in_affine, dtype=np.float64, copy=True)
    out_affine_ = np.array(out_affine, dtype=np.float64, copy=True)
    in_coords = [(0.0, dim - 1.0) for dim in shape]
    corners = cast(DoubleArray, np.asarray(np.meshgrid(*in_coords, indexing="ij"))).reshape(
        (len(shape), -1)
    )
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    corners_out = np.linalg.solve(out_affine_, in_affine_) @ corners
    corners = in_affine_ @ corners
    all_dist = corners_out[:-1].copy()
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(cast(DoubleArray, np.ptp(corners_out, axis=1)) + 1.0)
    offset = None
    for index in range(corners.shape[1]):
        min_corner = np.min(all_dist - all_dist[:, index : index + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            offset = corners[:-1, index]
            break
    if offset is None:
        offset = (
            in_affine_[:-1, :-1] @ (shape / 2.0)
            + in_affine_[:-1, -1]
            - out_affine_[:-1, :-1] @ (out_shape / 2.0)
        )
    return out_shape.astype(int, copy=False), offset


# --- resampling ---------------------------------------------------------------------


def _normalize_transform(shape: tuple[int, ...]) -> DoubleArray:
    """Build an align-corners-false voxel-to-normalized coordinate transform."""
    size = np.asarray(shape, dtype=np.float64)
    norm = size.copy()
    norm[norm <= 0.0] = 2.0
    norm = 2.0 / norm
    matrix = np.diag(np.concatenate((norm, np.ones(1))))
    matrix[:-1, -1] = 1.0 / size - 1.0
    return matrix


def _base_axis(steps: int) -> DoubleArray:
    """Torch's float64 ``linspace(-1, 1)`` scaled for ``align_corners=False``."""

    if steps <= 1:
        return np.zeros(1)
    step = (1.0 - -1.0) / (steps - 1)
    index = np.arange(steps, dtype=np.float64)
    values = np.where(
        index < steps // 2, fma(step, index, -1.0), fma(-step, (steps - 1) - index, 1.0)
    )
    return (values * (steps - 1)) / steps


def _source_index(coordinate: DoubleArray, size: int) -> DoubleArray:
    # Normalized [-1, 1] coordinate -> source voxel index (align_corners=False),
    # clamped to the border (padding_mode="border").
    """Convert normalized coordinates to border-clamped source voxel indices."""
    coordinate = ((coordinate + 1) * size - 1) / 2
    return np.minimum(float(size - 1), np.maximum(coordinate, 0.0))


def _sample(values: RealArray, size: tuple[int, ...], grid: list[DoubleArray]) -> DoubleArray:
    """Torch 3D ``grid_sample`` (bilinear, border, ``align_corners=False``).

    ``values`` is the source flattened to ``(C, D*H*W)`` and ``size`` its ``(D, H, W)``.
    ``grid`` holds the normalized x, y, z coordinates (x runs along the last array axis)
    of each output voxel in a block. Returns float64 ``(C, *grid[0].shape)``.
    """

    depth, height, width = size
    ix = _source_index(grid[0], width)
    iy = _source_index(grid[1], height)
    iz = _source_index(grid[2], depth)
    x0 = np.floor(ix).astype(np.int64)
    y0 = np.floor(iy).astype(np.int64)
    z0 = np.floor(iz).astype(np.int64)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    corners = (
        (z0, y0, x0, (x1 - ix) * (y1 - iy) * (z1 - iz)),
        (z0, y0, x1, (ix - x0) * (y1 - iy) * (z1 - iz)),
        (z0, y1, x0, (x1 - ix) * (iy - y0) * (z1 - iz)),
        (z0, y1, x1, (ix - x0) * (iy - y0) * (z1 - iz)),
        (z1, y0, x0, (x1 - ix) * (y1 - iy) * (iz - z0)),
        (z1, y0, x1, (ix - x0) * (y1 - iy) * (iz - z0)),
        (z1, y1, x0, (x1 - ix) * (iy - y0) * (iz - z0)),
        (z1, y1, x1, (ix - x0) * (iy - y0) * (iz - z0)),
    )
    output = np.zeros((values.shape[0], *ix.shape), dtype=np.float64)
    for z, y, x, weight in corners:
        inside = (x < width) & (y < height) & (z < depth)
        flat = (np.minimum(z, depth - 1) * height + np.minimum(y, height - 1)) * width
        flat = flat + np.minimum(x, width - 1)
        # Gathering float32 then widening equals Torch's float64 copy of the input.
        gathered = np.take(values, flat, axis=1).astype(np.float64)
        output = output + np.where(inside, gathered * weight, 0.0)
    return output


def _affine_resample(
    image: RealArray,
    xform: DoubleArray,
    spatial_size: tuple[int, ...],
    *,
    channel_argmax: bool = False,
) -> FloatArray:
    """MONAI ``AffineTransform(normalized=False, reverse_indexing=True)`` in float64.

    ``image`` is ``(C, D, H, W)``, ``xform`` the 4x4 map from output voxel indices to
    input voxel indices, and ``spatial_size`` the output ``(D, H, W)``. Returns float32
    ``(C, *spatial_size)`` sampled bilinearly, with border values extended.
    With ``channel_argmax`` each block is rounded to float32 and reduced to its first
    maximal channel, returning float32 labels ``(1, *spatial_size)`` without storing
    every resampled channel.
    """

    source_size = tuple(int(value) for value in image.shape[1:])
    theta = _torch_bmm_small(_normalize_transform(source_size), xform)
    theta = _torch_bmm_small(theta, np.linalg.inv(_normalize_transform(spatial_size)))
    # grid_sample orders coordinates (x, y, z) = (last axis ... first axis), the reverse of
    # array order, so swap the axes of theta on both sides.
    reverse = [2, 1, 0]
    theta[:3] = theta[reverse]
    theta[:, :3] = theta[:, reverse]
    depth, height, width = spatial_size
    base_w, base_h, base_d = _base_axis(width), _base_axis(height), _base_axis(depth)
    # Below 400 multiply-adds Torch's bmm uses its naive kernel, which accumulates the
    # coordinates in a different order than the fused path below.
    small = 4 * depth * height * width * 3 < 400
    values = np.ascontiguousarray(image, dtype=np.float32).reshape(image.shape[0], -1)
    channels = 1 if channel_argmax else image.shape[0]
    output = np.empty((channels, depth, height, width), dtype=np.float32)
    rows = max(1, _BLOCK_VOXELS // max(1, height * width))

    def resample_block(start: int) -> None:
        """Resample one disjoint block of output voxels."""
        block = base_d[start : start + rows]
        grid: list[DoubleArray] = []
        for row in theta[:3]:
            if small:
                coordinate = (
                    (base_w[None, None, :] * row[0] + base_h[None, :, None] * row[1])
                    + block[:, None, None] * row[2]
                ) + row[3]
            else:
                first = fma(block[:, None], row[2], base_w[None, :] * row[0])
                second = fma(1.0, row[3], base_h * row[1])
                coordinate = first[:, None, :] + second[None, :, None]
            grid.append(np.broadcast_to(coordinate, (len(block), height, width)))
        sampled = _sample(values, source_size, grid).astype(np.float32)
        output[:, start : start + len(block)] = (
            np.argmax(sampled, axis=0)[None] if channel_argmax else sampled
        )

    _each_block(range(0, depth, rows), resample_block)
    return output


def _each_block(starts: range, work: Callable[[int], None]) -> None:
    """Run ``work`` for each block start on a thread pool, re-raising its first error.

    Blocks write disjoint outputs and NumPy releases the GIL in their arithmetic
    and gathers, so the threads change only the elapsed time, not any value.
    """

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        for task in [pool.submit(work, start) for start in starts]:
            task.result()


def spatial_resample(
    image: RealArray,
    src_affine: DoubleArray,
    dst_affine: DoubleArray,
    spatial_size: tuple[int, ...],
    *,
    channel_argmax: bool = False,
) -> tuple[FloatArray, DoubleArray]:
    """MONAI ``SpatialResample`` (bilinear, border, float64); returns float32 data and affine.

    ``channel_argmax`` returns float32 ``argmax`` labels ``(1, *spatial_size)`` of the
    resampled float32 data instead, computed per block (see ``_affine_resample``).

    ``image`` is ``(C, X, Y, Z)`` on the grid ``src_affine``; the result is sampled onto
    ``dst_affine`` with ``spatial_size`` voxels. The data is returned unresampled
    (cast to float32) when the shapes are ``np.allclose`` and either the affines or
    their relative transform and identity are ``np.allclose`` with ``atol=AFFINE_TOL``.
    These comparisons retain NumPy's default relative tolerance as well.
    """

    in_size = tuple(int(value) for value in image.shape[1:])
    src = np.array(src_affine, dtype=np.float64, copy=True)
    dst = np.array(dst_affine, dtype=np.float64, copy=True)
    xform = np.linalg.solve(src, dst)
    same_size = np.allclose(np.asarray(spatial_size), np.asarray(in_size))
    unchanged = (np.allclose(src, dst, atol=AFFINE_TOL) and same_size) or (
        np.allclose(xform, np.eye(len(xform)), atol=AFFINE_TOL) and same_size
    )
    if unchanged:
        if channel_argmax:
            return np.argmax(image.astype(np.float32), axis=0)[None].astype(np.float32), src
        return image.astype(np.float32), src
    resampled = _affine_resample(image, xform, spatial_size, channel_argmax=channel_argmax)
    return resampled, _torch_mm(src, xform)


# --- orientation --------------------------------------------------------------------


def _orientation[T: np.bool_ | np.integer | np.floating](
    image: NDArray[T], affine: DoubleArray, axcodes: str | tuple[str, ...]
) -> tuple[NDArray[T], DoubleArray]:
    """Flip and permute the spatial axes of a ``(C, X, Y, Z)`` array to follow ``axcodes``.

    No interpolation: only axis flips and transposes. Returns a contiguous array and the
    affine of the reordered grid.
    """
    shape = tuple(int(value) for value in image.shape[1:])
    source = nib.io_orientation(np.array(affine, dtype=np.float64, copy=True))
    target = nib.orientations.axcodes2ornt(axcodes)
    ornt = nib.orientations.ornt_transform(source, target)
    xform = nib.orientations.inv_ornt_aff(ornt, shape)
    ornt[:, 0] += 1
    ornt = np.concatenate([np.array([[0, 1]]), ornt])
    axes = [axis for axis, flip in enumerate(ornt[:, 1]) if flip == -1]
    transpose = np.arange(len(shape) + 1)
    transpose[: len(ornt)] = np.argsort(ornt[:, 0])
    if axes:
        image = cast(NumpyFlip, np.flip)(image, axis=axes)
    if not np.all(transpose == np.arange(image.ndim)):
        image = np.transpose(image, transpose.tolist())
    return np.ascontiguousarray(image), np.asarray(affine, dtype=np.float64) @ xform


# --- bundle pipeline ----------------------------------------------------------------


def scale_intensity_range(image: RealArray) -> FloatArray:
    """Map -175..250 HU linearly to 0..1 in float32; values outside saturate at 0 and 1."""
    scaled = (image - np.float32(A_MIN)) / np.float32(A_MAX - A_MIN)
    scaled = scaled * np.float32(B_MAX - B_MIN) + np.float32(B_MIN)
    return np.clip(scaled, np.float32(B_MIN), np.float32(B_MAX)).astype(np.float32)


def preprocess(data: HostArray, affine: DoubleArray) -> tuple[FloatArray, Geometry]:
    """Bundle preprocessing of one 3D volume; returns ``1xXxYxZ`` float32 network input.

    Args:
        data: Voxel values in Hounsfield units, in the reader's storage order ``(X, Y, Z)``.
        affine: The reader's 4x4 voxel-to-world affine (after header correction).

    Returns:
        The volume reoriented to RAS, bilinearly resampled to 1 mm spacing and scaled to
        [0, 1], plus the ``Geometry`` needed by ``labels_from_probabilities``. An axis
        whose spacing is already within ``AFFINE_TOL`` of 1 mm keeps its spacing rather
        than being snapped, as in MONAI's ``Spacing``.
    """

    original_affine = np.array(affine, dtype=np.float64, copy=True)
    image = np.asarray(data, dtype=np.float32)[None]
    original_shape = cast(tuple[int, int, int], tuple(int(value) for value in image.shape[1:]))
    image, oriented_affine = _orientation(image, original_affine, AXCODES)
    oriented_shape = cast(tuple[int, int, int], tuple(int(value) for value in image.shape[1:]))
    out_d = np.array(PIXDIM, dtype=np.float64)
    for index, current in enumerate(affine_to_spacing(oriented_affine)):
        target = out_d[index]
        if not (target - AFFINE_TOL) <= current <= (target + AFFINE_TOL):
            current = target
        out_d[index] = current
    target_affine = zoom_affine(oriented_affine, out_d)
    shape, offset = compute_shape_offset(oriented_shape, oriented_affine, target_affine)
    target_affine[:3, -1] = offset[:3]
    network_shape = cast(tuple[int, int, int], tuple(int(value) for value in shape))
    image, network_affine = spatial_resample(image, oriented_affine, target_affine, network_shape)
    geometry = Geometry(
        original_affine=original_affine,
        original_shape=original_shape,
        oriented_affine=oriented_affine,
        oriented_shape=oriented_shape,
        network_affine=network_affine,
        network_shape=network_shape,
    )
    return scale_intensity_range(image), geometry


def _fma32(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> FloatArray:
    """Correctly rounded float32 ``a * b + c`` (exact product, sum rounded to odd)."""

    product = np.asarray(a, np.float32).astype(np.float64) * np.asarray(b, np.float32)
    total, error = _two_sum(product, np.asarray(c, np.float32).astype(np.float64))
    even = (total.view(np.int64) & 1) == 0
    toward = np.where(error > 0, np.inf, -np.inf)
    total = np.where((error != 0) & even, np.nextafter(total, toward), total)
    return total.astype(np.float32)


def _sleef_expf(values: FloatArray) -> FloatArray:
    """SLEEF ``expf_u10`` with fused multiply-adds, Torch's ``Vectorized<float>::exp``."""

    q = np.rint(values * np.float32(1.442695040888963407359924681001892137426645954152985934))
    q = q.astype(np.int32)
    qf = q.astype(np.float32)
    s = _fma32(qf, np.float32(-0.693145751953125), values)
    s = _fma32(qf, np.float32(-1.428606765330187045e-06), s)
    u = np.float32(0.000198527617612853646278381)
    for coefficient in (
        0.00139304355252534151077271,
        0.00833336077630519866943359,
        0.0416664853692054748535156,
        0.166666671633720397949219,
        0.5,
    ):
        u = _fma32(u, s, np.float32(coefficient))
    u = np.float32(1.0) + _fma32(s * s, u, s)
    underflow = values < np.float32(-104)
    q = np.where(underflow, 0, q)  # Those lanes return zero; keep their scaling finite.
    half = q >> 1
    u = u * ((half + 127) << 23).astype(np.int32).view(np.float32)
    u = u * ((q - half + 127) << 23).astype(np.int32).view(np.float32)
    return np.where(underflow, np.float32(0), u).astype(np.float32)


def softmax(logits: FloatArray, *, out: FloatArray | None = None) -> FloatArray:
    """Torch's float32 CPU softmax over the channel axis (axis 0 of ``(C, X, Y, Z)``).

    Torch exponentiates eight spatial positions at a time with SLEEF ``exp``; the
    remaining positions use the C library ``expf``. Both divide by the channel sum.
    """

    logits = np.asarray(logits, dtype=np.float32)
    flat = logits.reshape(logits.shape[0], -1)
    if out is None:
        output = np.empty_like(flat)
    else:
        # Each block reads its own voxels before writing them, so ``out`` may be
        # ``logits``; a contiguous buffer keeps the reshape a view of it.
        if out.shape != logits.shape or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError("softmax out must be a C-contiguous float32 array like logits")
        output = out.reshape(flat.shape)
    vectorized = flat.shape[1] - flat.shape[1] % _VECTOR_LANES

    def softmax_block(start: int) -> None:
        """Compute source-ordered softmax for one block of output voxels."""
        stop = min(start + _BLOCK_VOXELS, flat.shape[1])
        for begin, end, scalar in (
            (start, min(stop, vectorized), False),
            (max(start, vectorized), stop, True),
        ):
            if begin >= end:
                continue
            block = flat[:, begin:end]
            maximum = block[0]
            for channel in block[1:]:
                maximum = np.maximum(maximum, channel)
            shifted = block - maximum
            if scalar:  # C library expf, approximated by a correctly rounded exp
                exponent = np.exp(shifted.astype(np.float64)).astype(np.float32)
            else:
                exponent = _sleef_expf(shifted)
            total = np.float32(0) + exponent[0]
            for channel in exponent[1:]:
                total = total + channel
            output[:, begin:end] = exponent / total

    _each_block(range(0, flat.shape[1], _BLOCK_VOXELS), softmax_block)
    return output.reshape(logits.shape)


def labels_from_probabilities(probabilities: FloatArray, geometry: Geometry) -> FloatArray:
    """Invert spacing then orientation and take the argmax: ``(X, Y, Z)`` float32 labels.

    ``probabilities`` is ``(C, *geometry.network_shape)``; the result is on the reader
    grid. The argmax runs per resampling block, and orientation, a spatial
    flip/transpose, then moves one label channel, so the resampled probabilities are
    never stored on the reader grid.
    """

    labels, affine = spatial_resample(
        probabilities,
        geometry.network_affine,
        geometry.oriented_affine,
        geometry.oriented_shape,
        channel_argmax=True,
    )
    axcodes = nib.orientations.aff2axcodes(geometry.original_affine)
    return _orientation(labels, affine, axcodes)[0][0]

