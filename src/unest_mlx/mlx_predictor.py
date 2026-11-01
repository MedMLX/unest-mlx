"""Pure MLX/NumPy sliding-window prediction for the renal UNesT bundle."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from unest_mlx._typing import (
    FloatArray,
    HostArray,
    NativeNetwork,
    NativeRuntime,
    SlidingWindowNamespace,
)

if TYPE_CHECKING:
    from mlx.core import array

import numpy as np
from medmlx_core.errors import InvalidInputError


class UNesTPredictor:
    """Predict preprocessed NCDHW volumes with an already loaded MLX network.

    The bundle uses 96-cubed windows, batches of four, 0.5 overlap, and all
    other MONAI defaults: uniform (constant) blending and zero padding. Checkpoint
    loading belongs to the caller. Logits are returned as float32 NumPy arrays on the
    preprocessed input grid.

    Args:
        network: Callable mapping a float32 MLX ``(N, 1, 96, 96, 96)`` window batch to
            ``(N, C, 96, 96, 96)`` logits, normally a ``UNesT``.
        mx: MLX runtime from ``medmlx_core.runtime.import_mlx()``.
        sw_batch_size: Windows per network call. The bundle uses 4; the value does not
            change the window layout.

    Raises:
        InvalidInputError: ``sw_batch_size`` is not a positive integer.
    """

    network_label = "UNesT"

    def __init__(
        self, network: NativeNetwork, *, mx: NativeRuntime, sw_batch_size: int = 4
    ) -> None:
        """Validate sliding-window controls and initialize the native predictor."""
        if (
            isinstance(sw_batch_size, bool)
            or not isinstance(cast(object, sw_batch_size), int)
            or (sw_batch_size < 1)
        ):
            raise InvalidInputError("sw_batch_size must be a positive integer")
        self.network = network
        self.mx = mx
        self.sw_batch_size = sw_batch_size

    def _forward_window(self, values: array) -> array:
        """Run the network on one window batch and reject non-finite logits."""

        self.mx.synchronize()
        result = self.network(values)
        # Evaluate eagerly so a non-finite window fails here, before it is stitched in.
        self.mx.eval(result)
        self.mx.synchronize()
        if not np.isfinite(cast(FloatArray, np.asarray(result))).all():
            raise RuntimeError(f"{self.network_label} produced non-finite logits")
        return result

    def forward(self, values: HostArray | array) -> FloatArray:
        """Return source-bundle sliding-window logits on the preprocessed grid.

        Require finite float32 input shaped ``(B, 1, D, H, W)``: a preprocessed CT
        expected to be scaled to [0, 1] on the 1 mm RAS grid. Range and geometry
        are caller responsibilities and are not checked here. Windows are 96x96x96 with 0.5 overlap;
        volumes smaller than a window are zero padded. Return channel-first float32
        NumPy logits ``(B, C, D, H, W)`` (not softmaxed), leaving source-grid
        restoration and label conversion to the caller.

        Raises:
            InvalidInputError: The input is not 5D with one channel, or is not finite
                float32.
            RuntimeError: The network produced non-finite logits.
        """
        host = np.asarray(values)
        if host.ndim != 5 or host.shape[1] != 1:
            raise InvalidInputError("UNesT input must have shape Bx1xDxHxW")
        if host.dtype != np.float32 or not np.isfinite(host).all():
            raise InvalidInputError(f"{self.network_label} input must be finite float32")
        sliding = cast(SlidingWindowNamespace, import_module("medmlx_core.sliding_window"))

        return sliding.sliding_window_inference(
            cast(FloatArray, host),
            roi_size=(96, 96, 96),
            sw_batch_size=self.sw_batch_size,
            predictor=self._forward_window,
            overlap=0.5,
        )

    def __call__(self, values: HostArray | array) -> FloatArray:
        """Alias for ``forward``."""
        return self.forward(values)
