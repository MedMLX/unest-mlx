"""Inference-only MLX graph for the MONAI renal UNesT 0.2.7 release.

Source: the bundle's scripts/networks/{unest,nest_transformer_3D}.py.
Weights stay in the original Torch layout. The public interface is channels-first
(NCDHW), as in Torch; ``medmlx_core`` converts to MLX's channels-last convolution
layout internally. Network arithmetic runs on the supplied MLX runtime; the runner
admits Metal before executing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import TYPE_CHECKING, cast

from unest_mlx._typing import NativeRuntime, NeuralRuntime

if TYPE_CHECKING:
    from mlx.core import array

from medmlx_core.ops import conv3d_ncdhw
from medmlx_core.upsample import deconv2x_ncdhw


class UNesT:
    """Inference-only UNesT network: float32 NCDHW windows in, float32 NCDHW logits out.

    Calling the instance maps a ``(B, 1, 96, 96, 96)`` batch of preprocessed windows
    (CT scaled to [0, 1], see ``unest_mlx.transforms``) to ``(B, C, 96, 96, 96)`` logits.
    ``C`` is the first dimension of ``out.conv.conv.weight``: 4 for the released
    checkpoint (background, kidney cortex, medulla, pelvicalyceal system). Softmax is
    left to the caller. Window stitching lives in ``UNesTPredictor``.

    Args:
        weights: Parameter name to ``mx.array``; convert the NumPy arrays returned by
            ``validate_weights`` to MLX arrays first. They retain the original Torch
            layout: convolution weights (out, in, D, H, W),
            transposed-convolution weights (in, out, D, H, W), linear weights
            (out, in). Keys are looked up when the network runs, so a missing
            parameter raises ``KeyError`` at call time, not here.
        mx: MLX runtime from ``medmlx_core.runtime.import_mlx()``.
    """

    def __init__(self, weights: Mapping[str, array], *, mx: NativeRuntime) -> None:
        """Initialize the native UNesT graph with admitted weights."""
        self.weights = weights
        self.mx = mx
        self.nn = cast(NeuralRuntime, import_module("mlx.nn"))
        self.pool = self.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def _conv(self, x: array, prefix: str, *, padding: int, stride: int = 1) -> array:
        """Apply a named NCDHW convolution with optional bias."""
        return conv3d_ncdhw(
            x,
            self.weights[f"{prefix}.weight"],
            self.weights.get(f"{prefix}.bias"),
            padding=padding,
            stride=stride,
            mx=self.mx,
        )

    def _linear(self, x: array, prefix: str) -> array:
        """Apply a named linear layer and bias."""
        return x @ self.weights[f"{prefix}.weight"].T + self.weights[f"{prefix}.bias"]

    def _layer_norm(self, x: array, prefix: str) -> array:
        """Apply a named layer normalization on the last axis."""
        return self.mx.fast.layer_norm(
            x,
            self.weights[f"{prefix}.weight"],
            self.weights[f"{prefix}.bias"],
            1e-6,
        )

    def _instance_norm(self, x: array) -> array:
        # Over (D, H, W) per sample and channel, with no learned scale or shift: the
        # released checkpoint has no instance-norm parameters.
        """Normalize each sample and channel over its spatial dimensions."""
        mean = self.mx.mean(x, axis=(2, 3, 4), keepdims=True)
        variance = self.mx.var(x, axis=(2, 3, 4), keepdims=True)
        return (x - mean) * self.mx.rsqrt(variance + 1e-5)

    def _residual(self, x: array, prefix: str) -> array:
        """Conv-norm-leaky ReLU, conv-norm, add the skip, then leaky ReLU (slope 0.01).

        Both convolutions are 3x3x3. The skip is the identity unless the block changes the
        channel count, in which case the checkpoint carries ``conv3`` and the skip is a
        1x1x1 conv followed by instance norm.
        """
        y = self._instance_norm(self._conv(x, f"{prefix}.conv1.conv", padding=1))
        y = self.nn.leaky_relu(y, negative_slope=0.01)
        y = self._instance_norm(self._conv(y, f"{prefix}.conv2.conv", padding=1))
        if f"{prefix}.conv3.conv.weight" in self.weights:
            x = self._instance_norm(self._conv(x, f"{prefix}.conv3.conv", padding=0))
        return self.nn.leaky_relu(y + x, negative_slope=0.01)

    def _transpose(self, x: array, prefix: str) -> array:
        """Apply the learned 2x transposed convolution."""
        return deconv2x_ncdhw(x, self.weights[f"{prefix}.weight"], None, mx=self.mx)

    def transformer_level(self, x: array, level: int) -> array:
        """Run one NesT hierarchy level on a channels-first feature map.

        Levels 1 and 2 first aggregate the previous level (3x3x3 conv, LayerNorm over
        channels, 3x3x3 max pool with stride 2), halving each spatial edge. The map is then
        split into non-overlapping 6x6x6 blocks of 216 tokens, given a learned position
        embedding and passed through pre-norm transformer layers whose attention is
        confined to each block, before being merged back to a feature map.

        Args:
            x: float32 ``(B, C_in, E_in, E_in, E_in)``. Level 0 takes the patch
                embedding; levels 1 and 2 take the previous level's output.
            level: 0, 1 or 2 (depth 2, 2, 8 layers; 4, 8, 16 heads).

        Returns:
            ``(B, C, E, E, E)`` with ``E`` divisible by 6 after any pooling. For
            96-cubed windows ``E`` is 24, 12, 6 and ``C`` is 128, 256, 512.
        """
        prefix = f"nestViT.levels.{level}"
        if level:
            # LayerNorm normalizes the last axis and MLX pooling expects channels-last
            # input, so this stretch runs in NDHWC.
            x = self._conv(x, f"{prefix}.pool.conv", padding=1)
            x = self._layer_norm(self.mx.transpose(x, (0, 2, 3, 4, 1)), f"{prefix}.pool.norm")
            x = self.mx.transpose(self.pool(x), (0, 4, 1, 2, 3))
        batch, channels, edge, _, _ = x.shape
        grid = edge // 6
        blocks = grid**3
        x = self.mx.reshape(
            self.mx.transpose(x, (0, 2, 3, 4, 1)), (batch, grid, 6, grid, 6, grid, 6, channels)
        )
        x = self.mx.reshape(
            self.mx.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7)), (batch, blocks, 216, channels)
        )
        x = x + self.weights[f"{prefix}.pos_embed"]
        heads = 4 * 2**level
        for layer in range((2, 2, 8)[level]):
            block = f"{prefix}.transformer_encoder.{layer}"
            y = self._layer_norm(x, f"{block}.norm1")
            qkv = self._linear(y, f"{block}.attn.qkv")
            qkv = self.mx.reshape(qkv, (batch, blocks, 216, 3, heads, channels // heads))
            qkv = self.mx.transpose(qkv, (3, 0, 4, 1, 2, 5))
            q, k, v = qkv[0], qkv[1], qkv[2]
            # Keep the source matmul/scale/softmax arithmetic while evaluating
            # at most eight independent blocks' score matrices at a time.
            parts: list[array] = []
            for start in range(0, blocks, 8):
                stop = min(start + 8, blocks)
                attention = self.mx.softmax(
                    (q[:, :, start:stop] @ k[:, :, start:stop].swapaxes(-2, -1))
                    * (channels // heads) ** -0.5, axis=-1,
                )
                part = attention @ v[:, :, start:stop]
                self.mx.eval(part)
                parts.append(part)
            attended = self.mx.concatenate(parts, axis=2) if len(parts) > 1 else parts[0]
            # The released graph merges heads with the head index varying fastest
            # (head_dim, heads), not the usual (heads, head_dim); trained weights depend
            # on it.
            y = self.mx.reshape(
                self.mx.transpose(attended, (0, 2, 3, 4, 1)), (batch, blocks, 216, channels),
            )
            x = x + self._linear(y, f"{block}.attn.proj")
            y = self._linear(self._layer_norm(x, f"{block}.norm2"), f"{block}.mlp.fc1")
            x = x + self._linear(self.nn.gelu(y), f"{block}.mlp.fc2")
        x = self.mx.reshape(x, (batch, grid, grid, grid, 6, 6, 6, channels))
        return self.mx.reshape(
            self.mx.transpose(x, (0, 7, 1, 4, 2, 5, 3, 6)), (batch, channels, edge, edge, edge)
        )

    def __call__(self, x: array) -> array:
        """Predict segmentation logits for admitted 96-cubed NCDHW windows."""
        if x.ndim != 5 or tuple(x.shape[1:]) != (1, 96, 96, 96) or x.dtype != self.mx.float32:
            raise ValueError("UNesT expects float32 Bx1x96x96x96 windows")
        # Patch embedding takes 96^3 to 24^3; each transformer level keeps or halves that.
        hidden = self._conv(x, "nestViT.patch_embed.proj", padding=0, stride=4)
        skips = [hidden]
        for level in range(3):
            hidden = self.transformer_level(hidden, level)
            skips.append(hidden)
        hidden = self._layer_norm(self.mx.transpose(hidden, (0, 2, 3, 4, 1)), "nestViT.norm")
        hidden = self._conv(
            self.mx.transpose(hidden, (0, 4, 1, 2, 3)), "encoder10.conv", padding=1, stride=2
        )
        # PReLU with the single shared slope stored in encoder10.adn.A.weight.
        hidden = self.mx.where(hidden >= 0, hidden, hidden * self.weights["encoder10.adn.A.weight"])
        hidden = self._instance_norm(hidden)
        encoders = [
            self._residual(x, "encoder1.layer"),
            # encoder2.blocks.0 is a 1x1x1 transposed convolution with no upsampling, so
            # its (in, out) weight is applied as x @ W on channels-last features.
            self.mx.transpose(
                self.mx.transpose(
                    self._transpose(skips[0], "encoder2.transp_conv_init.conv"), (0, 2, 3, 4, 1)
                )
                @ self.weights["encoder2.blocks.0.conv.weight"][:, :, 0, 0, 0],
                (0, 4, 1, 2, 3),
            ),
            self._residual(skips[1], "encoder3.layer"),
            self._residual(skips[2], "encoder4.layer"),
            skips[3],
        ]
        # Decoder stages 5..1 each upsample 2x and concatenate the skip, coarsest first.
        for stage, skip in zip(range(5, 0, -1), reversed(encoders), strict=True):
            hidden = self._transpose(hidden, f"decoder{stage}.transp_conv.conv")
            hidden = self._residual(
                self.mx.concatenate([hidden, skip], axis=1),
                f"decoder{stage}.conv_block",
            )
        return self._conv(hidden, "out.conv.conv", padding=0)
