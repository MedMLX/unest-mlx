"""Strict admission of the released renal UNesT 0.2.7 checkpoint."""

from collections.abc import Mapping
from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from numpy.typing import NDArray


def required_shapes() -> dict[str, tuple[int, ...]]:
    """Released parameter inventory in upstream ``state_dict`` registration order.

    Maps each parameter name to its shape in the original Torch layout: convolution
    weights are (out, in, D, H, W), transposed-convolution weights are (in, out, D, H, W)
    and linear weights are (out, in). The order matters because seeded reference weights
    are drawn in registration order.
    """
    shapes: dict[str, tuple[int, ...]] = {}

    def affine(prefix: str, channels: int) -> None:
        """Add one normalization affine to the expected weight shapes."""
        shapes[f"{prefix}.weight"] = (channels,)
        shapes[f"{prefix}.bias"] = (channels,)

    def conv(prefix: str, out: int, inp: int, kernel: int, bias: bool = False) -> None:
        """Add convolution weight and optional bias shapes."""
        shapes[f"{prefix}.weight"] = (out, inp, kernel, kernel, kernel)
        if bias:
            shapes[f"{prefix}.bias"] = (out,)

    def residual(prefix: str, out: int, inp: int) -> None:
        """Add the expected convolution shapes for one residual block."""
        conv(f"{prefix}.conv1.conv", out, inp, 3)
        conv(f"{prefix}.conv2.conv", out, out, 3)
        if inp != out:
            conv(f"{prefix}.conv3.conv", out, inp, 1)

    conv("nestViT.patch_embed.proj", 128, 1, 4, True)
    for level, (channels, depth, blocks) in enumerate(((128, 2, 64), (256, 2, 8), (512, 8, 1))):
        prefix = f"nestViT.levels.{level}"
        shapes[f"{prefix}.pos_embed"] = (1, blocks, 216, channels)
        if level:
            conv(f"{prefix}.pool.conv", channels, channels // 2, 3, True)
            affine(f"{prefix}.pool.norm", channels)
        for layer in range(depth):
            block = f"{prefix}.transformer_encoder.{layer}"
            affine(f"{block}.norm1", channels)
            for name, out, inp in (
                ("attn.qkv", channels * 3, channels),
                ("attn.proj", channels, channels),
                ("mlp.fc1", channels * 4, channels),
                ("mlp.fc2", channels, channels * 4),
            ):
                if name == "mlp.fc1":
                    affine(f"{block}.norm2", channels)
                shapes[f"{block}.{name}.weight"] = (out, inp)
                shapes[f"{block}.{name}.bias"] = (out,)
    affine("nestViT.norm", 512)
    residual("encoder1.layer", 32, 1)
    # Transposed-convolution weights retain the source IO(DHW) layout.
    conv("encoder2.transp_conv_init.conv", 128, 64, 2)
    conv("encoder2.blocks.0.conv", 64, 64, 1)
    residual("encoder3.layer", 128, 128)
    residual("encoder4.layer", 256, 256)
    for stage in range(5, 0, -1):
        channels = 16 * 2**stage
        conv(f"decoder{stage}.transp_conv.conv", channels * 2, channels, 2)
        residual(f"decoder{stage}.conv_block", channels, channels * 2)
    conv("encoder10.conv", 1024, 512, 3, True)
    shapes["encoder10.adn.A.weight"] = (1,)
    conv("out.conv.conv", 4, 32, 1, True)
    return shapes


def validate_weights(
    payload: Mapping[str, object] | Mapping[object, object],
) -> dict[str, NDArray[np.float32]]:
    """Admit tensors matching the released graph as contiguous float32 arrays.

    The key, shape, dtype and finiteness checks do not verify checkpoint identity
    or origin; a different checkpoint with the same inventory can pass.

    Args:
        payload: A flat mapping of parameter names to NumPy arrays loaded from
            native safetensors weights.

    Returns:
        Parameter name to C-contiguous float32 NumPy array, in ``required_shapes()``
        order, with the original Torch layout and unchanged values.

    Raises:
        InvalidInputError: The key set differs from the released graph, or a tensor
            is not a finite float32 array of the released shape.
    """
    shapes = required_shapes()
    if set(payload) != set(shapes):
        raise InvalidInputError("UNesT checkpoint parameter names differ from the released graph")
    validated: dict[str, NDArray[np.float32]] = {}
    for name, shape in shapes.items():
        value = payload[name]
        if not isinstance(value, np.ndarray):
            raise InvalidInputError(f"UNesT requires finite float32 {name} with shape {shape}")
        raw = cast(NDArray[np.generic], value)
        if (raw.shape != shape or raw.dtype != np.float32
                or not np.isfinite(raw).all()):
            raise InvalidInputError(f"UNesT requires finite float32 {name} with shape {shape}")
        validated[name] = np.ascontiguousarray(raw, dtype=np.float32)
    return validated
