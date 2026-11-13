"""Record the pinned UNesT bundle transforms on HCC-TACE CT crops."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from importlib import import_module, metadata
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from tests.fixture_cases import (
    HEADER_CASES,
    RUNNER_CASES,
    predictor_input,
    predictor_weights,
    runner_weights,
    small_weights,
    window_input,
    write_ct_crop,
    write_header_case,
    write_runner_case,
)
from unest_mlx._typing import HostArray, NiftiIo
from unest_mlx.mlx_weights import required_shapes

REVISION = "55fc1da849a70ed35d2900561429c042a86d65de"
type Payload = dict[str, HostArray]


class ReferenceTensor(Protocol):
    """Tensor and MONAI metadata consumed by the recorder."""

    shape: tuple[int, ...]
    affine: ReferenceTensor
    meta: Mapping[str, object]

    def numpy(self) -> HostArray: ...
    def __getitem__(self, key: int | None) -> ReferenceTensor: ...
    def __mul__(self, other: ReferenceTensor) -> ReferenceTensor: ...
    def __add__(self, other: ReferenceTensor) -> ReferenceTensor: ...
    def __truediv__(self, other: int) -> ReferenceTensor: ...
    def reshape(self, shape: tuple[int, ...]) -> ReferenceTensor: ...


class Transform(Protocol):
    def __call__(self, value: dict[str, object]) -> dict[str, object]: ...


class Compose(Transform, Protocol):
    transforms: list[Transform]


class ConfigParser(Protocol):
    def read_config(self, path: str) -> None: ...
    def __setitem__(self, key: str, value: object) -> None: ...
    def get_parsed_content(self, key: str) -> object: ...


class TorchRuntime(Protocol):
    float32: object

    def set_num_threads(self, value: int) -> None: ...
    def from_numpy(self, value: HostArray) -> ReferenceTensor: ...
    def arange(self, stop: int, *, dtype: object) -> ReferenceTensor: ...
    def inference_mode(self) -> AbstractContextManager[None]: ...


class Network(Protocol):
    def __call__(self, values: ReferenceTensor, /) -> ReferenceTensor: ...


class SourceModel(Network, Protocol):
    def state_dict(self) -> Mapping[str, ReferenceTensor]: ...
    def load_state_dict(self, values: Mapping[str, ReferenceTensor], *, strict: bool) -> object: ...
    def eval(self) -> object: ...


class Inferer(Protocol):
    def __call__(self, inputs: ReferenceTensor, model: Network) -> ReferenceTensor: ...


class Factory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


class ProjectionLayer(Protocol):
    conv3: object
    norm3: object


class ResidualBlock(Protocol):
    layer: ProjectionLayer


class ArchiveWriter(Protocol):
    def __call__(self, path: Path, *, metadata: HostArray, **arrays: HostArray) -> None: ...


class PatchSlices(Protocol):
    def __call__(
        self, image_size: tuple[int, ...], patch_size: tuple[int, ...],
        scan_interval: tuple[int, ...],
    ) -> list[tuple[slice, ...]]: ...


def reduced_model(bundle: Path, *, output_channels: int = 4) -> SourceModel:
    """Construct the upstream graph at the widths used by the runner parity test.

    The upstream constructor hard-codes widths. Its component constructors are
    temporarily adapted while it builds the same graph; a residual projection
    remains present when reduced input and output widths happen to coincide.
    """
    sys.path.insert(0, str(bundle))
    module = import_module("scripts.networks.unest")
    blocks = import_module("monai.networks.blocks.dynunet_block")
    conv = cast(Factory, blocks.get_conv_layer)
    norm = cast(Factory, blocks.get_norm_layer)
    widths = {1: 1, 32: 1, 64: 2, 128: 4, 256: 8, 512: 16, 1024: 32, 2048: 64}
    names = ("NestTransformer3D", "UNesTConvBlock", "UNestUpBlock", "UNesTBlock",
             "Convolution", "UnetOutBlock")
    originals = {name: cast(Factory, getattr(module, name)) for name in names}

    def adapted(name: str, original: Factory) -> Factory:
        def construct(*args: object, **kwargs: object) -> object:
            source_in, source_out = kwargs.get("in_channels"), kwargs.get("out_channels")
            if name == "NestTransformer3D":
                kwargs["embed_dims"] = (4, 8, 16)
            else:
                for key in ("in_channels", "out_channels"):
                    if key in kwargs and not (name == "UnetOutBlock" and key == "out_channels"):
                        value = kwargs[key]
                        if not isinstance(value, int):
                            raise TypeError(f"Expected integer {key}, got {value}")
                        kwargs[key] = widths[value]
            block = original(*args, **kwargs)
            if (name == "UNesTConvBlock" and source_in != source_out
                    and kwargs["in_channels"] == kwargs["out_channels"]):
                layer = cast(ResidualBlock, block).layer
                channels = kwargs["out_channels"]
                layer.conv3 = conv(
                    3, channels, channels, kernel_size=1, stride=1, dropout=None,
                    act=None, norm=None, conv_only=False,
                )
                layer.norm3 = norm(name=kwargs["norm_name"], spatial_dims=3, channels=channels)
            return block
        return construct

    try:
        for name, original in originals.items():
            setattr(module, name, adapted(name, original))
        constructor = cast(Factory, module.UNesT)
        return cast(SourceModel, constructor(in_channels=1, out_channels=output_channels))
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def record_runner(
    bundle: Path, output: Path, parser_factory: Callable[[], ConfigParser],
    torch: TorchRuntime, nib: NiftiIo, revision: str,
) -> None:
    """Run the reduced upstream network and bundle on a two-window CT crop."""
    model = reduced_model(bundle)
    shapes = required_shapes()
    weights = runner_weights(shapes)
    inventory = model.state_dict()
    if set(inventory) != set(weights) or any(
        inventory[name].shape != value.shape for name, value in weights.items()
    ):
        raise ValueError("Reduced upstream parameter shapes differ from the native model")
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()},
                          strict=True)
    model.eval()
    with tempfile.TemporaryDirectory(prefix="unest-runner-reference-") as temporary:
        root = Path(temporary)
        source = root / "ct.nii.gz"
        write_ct_crop(source)
        original = nib.load(source)
        parser = parser_factory()
        parser.read_config(str(bundle / "configs" / "inference.json"))
        parser["bundle_root"] = str(bundle)
        parser["output_dir"] = str(root / "out")
        preprocessing = cast(Transform, parser.get_parsed_content("preprocessing"))
        postprocessing = cast(Compose, parser.get_parsed_content("postprocessing"))
        inferer = cast(Inferer, parser.get_parsed_content("inferer"))
        with torch.inference_mode():
            data = preprocessing({"image": str(source)})
            image = cast(ReferenceTensor, data["image"])
            padded = tuple(max(size, 96) for size in image.shape[1:])
            interval = tuple(96 if size == 96 else 48 for size in padded)
            patches = cast(PatchSlices, import_module("monai.data.utils").dense_patch_slices)
            windows = len(patches(padded, (96, 96, 96), interval))
            data["pred"] = inferer(image[None], model)[0]
            postprocessing(data)
        saved_paths = list((root / "out").rglob("*.nii.gz"))
        if len(saved_paths) != 1:
            raise ValueError(f"Expected one saved segmentation, found {saved_paths}")
        saved = nib.load(saved_paths[0])
        info = {
            "schema_version": 3, "upstream_revision": revision,
            "upstream_repository": "https://github.com/Project-MONAI/model-zoo",
            "bundle_version": "0.2.7", "reference_framework": "torch",
            "recording_platform": "darwin-arm64", "monai_version": metadata.version("monai"),
            "torch_version": metadata.version("torch"), "torch_threads": 1,
            "ct_source": "HCC-TACE-Seg HCC_063 contrast CT", "weights_seed": 3200,
            "required_shapes": shapes, "windows": windows, "sw_batch_size": 4,
            "window_batches": [min(4, windows - start) for start in range(0, windows, 4)],
            "source": "configs/inference.json and reduced scripts/networks/unest.py graph",
        }
        cast(ArchiveWriter, np.savez_compressed)(
            output / "runner_end_to_end.npz", metadata=np.array(json.dumps(info)),
            input_data=np.asanyarray(original.dataobj), input_affine=original.affine,
            saved=np.asanyarray(saved.dataobj), saved_affine=saved.affine,
            saved_codes=np.array([int(saved.header[key]) for key in ("qform_code", "sform_code")]),
        )
        print(f"Recorded runner with {windows} windows", flush=True)


def record_graph(
    bundle: Path, output: Path, parser_factory: Callable[[], ConfigParser],
    torch: TorchRuntime, revision: str,
) -> None:
    """Record full-window and sliding-window graph outputs on CT crops."""
    shapes = required_shapes()
    common: dict[str, object] = {
        "schema_version": 3, "upstream_revision": revision,
        "upstream_repository": "https://github.com/Project-MONAI/model-zoo",
        "bundle_version": "0.2.7", "reference_framework": "torch",
        "recording_platform": "darwin-arm64", "monai_version": metadata.version("monai"),
        "torch_version": metadata.version("torch"), "torch_threads": 1,
        "ct_source": "HCC-TACE-Seg HCC_063 contrast CT", "required_shapes": shapes,
    }
    save = cast(ArchiveWriter, np.savez_compressed)
    model = reduced_model(bundle, output_channels=1)
    weights = small_weights(shapes, 1729)
    if set(model.state_dict()) != set(weights) or any(
        model.state_dict()[name].shape != value.shape for name, value in weights.items()
    ):
        raise ValueError("Reduced upstream window model has unexpected parameter shapes")
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()},
                          strict=True)
    model.eval()
    with torch.inference_mode():
        values = model(torch.from_numpy(window_input())).numpy()
    for part in range(8):
        start, stop = 12 * part, 12 * (part + 1)
        info = {**common, "weights_seed": 1729, "depth_start": start, "depth_stop": stop}
        save(output / f"unest_window_{part}.npz", metadata=np.array(json.dumps(info)),
             output=values[:, :, start:stop])
    print("Recorded eight full-graph CT output tiles", flush=True)

    class StandIn:
        """Use source tensor operations to exercise MONAI's sliding-window stitching."""

        def __init__(self, seed: int) -> None:
            rng = np.random.default_rng(seed)
            self.gain = torch.from_numpy(rng.standard_normal((1, 4, 1, 1, 1)).astype(np.float32))
            self.bias = torch.from_numpy(rng.standard_normal((1, 4, 1, 1, 1)).astype(np.float32))
            self.batches: list[int] = []

        def __call__(self, inputs: ReferenceTensor) -> ReferenceTensor:
            self.batches.append(inputs.shape[0])
            depth = torch.arange(inputs.shape[2], dtype=torch.float32).reshape((1, 1, -1, 1, 1))
            return inputs * self.gain + self.bias + depth / 128

    def inferer(batch: int) -> Inferer:
        parser = parser_factory()
        parser.read_config(str(bundle / "configs" / "inference.json"))
        parser["inferer#sw_batch_size"] = batch
        return cast(Inferer, parser.get_parsed_content("inferer"))

    for batch in (1, 4):
        input_shape = (2, 1, 145, 3, 5)
        network = StandIn(2000)
        with torch.inference_mode():
            values = inferer(batch)(torch.from_numpy(predictor_input(input_shape)), network).numpy()
        info = {**common, "weights_seed": 2000, "input_shape": input_shape,
                "window_batches": network.batches}
        save(output / f"unest_predictor_stand_in_batch{batch}.npz",
             metadata=np.array(json.dumps(info)), output=values)
    model = reduced_model(bundle)
    weights = predictor_weights(shapes, 2100)
    if set(model.state_dict()) != set(weights) or any(
        model.state_dict()[name].shape != value.shape for name, value in weights.items()
    ):
        raise ValueError("Reduced upstream predictor model has unexpected parameter shapes")
    model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()},
                          strict=True)
    model.eval()
    input_shape = (1, 1, 9, 11, 13)
    with torch.inference_mode():
        values = inferer(4)(torch.from_numpy(predictor_input(input_shape)), model).numpy()
    info = {**common, "weights_seed": 2100, "input_shape": input_shape,
            "window_batches": [1]}
    save(output / "unest_predictor_full_batch4.npz", metadata=np.array(json.dumps(info)),
         output=values)
    print("Recorded sliding-window CT predictor outputs", flush=True)


def record(model_zoo: Path, output: Path) -> None:
    """Run the bundle's own transforms and save their output outside Git."""
    revision = subprocess.run(
        ["git", "-C", str(model_zoo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError(f"Expected model-zoo revision {REVISION}, found {revision}")
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise ValueError("The recorded reference requires macOS arm64")
    if metadata.version("monai") != "1.6.0" or metadata.version("torch") != "2.13.0":
        raise ValueError("The recorded reference requires MONAI 1.6.0 and Torch 2.13.0")
    bundle = model_zoo / "models" / "renalStructures_UNEST_segmentation"
    parser_factory = cast(Callable[[], ConfigParser], import_module("monai.bundle").ConfigParser)
    torch = cast(TorchRuntime, import_module("torch"))
    nib = cast(NiftiIo, import_module("nibabel"))
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(sorted((*RUNNER_CASES, *HEADER_CASES))):
        with tempfile.TemporaryDirectory(prefix="unest-reference-") as temporary:
            root = Path(temporary)
            source = root / f"{name}.nii.gz"
            (write_runner_case if name in RUNNER_CASES else write_header_case)(name, source)
            original = nib.load(source)
            parser = parser_factory()
            parser.read_config(str(bundle / "configs" / "inference.json"))
            parser["bundle_root"] = str(bundle)
            parser["output_dir"] = str(root / "out")
            preprocessing = cast(Transform, parser.get_parsed_content("preprocessing"))
            postprocessing = cast(Compose, parser.get_parsed_content("postprocessing"))
            data = preprocessing({"image": str(source)})
            image = cast(ReferenceTensor, data["image"])
            logits = np.random.default_rng(8100 + index).standard_normal(
                (4, *image.shape[1:])
            ).astype(np.float32)
            payload: Payload = {
                "input_data": np.asanyarray(original.dataobj),
                "input_affine": original.affine,
                "reader_affine": np.asarray(cast(npt.ArrayLike, image.meta["original_affine"])),
                "image": image.numpy().copy(),
                "network_affine": image.affine.numpy().copy(),
                "logits": logits,
            }
            data["pred"] = torch.from_numpy(logits)
            data = postprocessing.transforms[0](data)
            payload["probabilities"] = cast(ReferenceTensor, data["pred"]).numpy().copy()
            data = postprocessing.transforms[1](data)
            inverted = cast(ReferenceTensor, data["pred"])
            payload["inverted"] = inverted.numpy().copy()
            payload["inverted_affine"] = inverted.affine.numpy().copy()
            data = postprocessing.transforms[2](data)
            payload["labels"] = cast(ReferenceTensor, data["pred"])[0].numpy().copy()
            postprocessing.transforms[3](data)
            saved_paths = list((root / "out").rglob("*.nii.gz"))
            if len(saved_paths) != 1:
                raise ValueError(f"Expected one saved segmentation, found {saved_paths}")
            saved = nib.load(saved_paths[0])
            payload["saved"] = np.asanyarray(saved.dataobj)
            payload["saved_affine"] = saved.affine
            payload["saved_codes"] = np.array([
                int(saved.header[key]) for key in ("qform_code", "sform_code")
            ])
            info = {
                "schema_version": 3, "upstream_revision": revision,
                "upstream_repository": "https://github.com/Project-MONAI/model-zoo",
                "bundle_version": "0.2.7", "reference_framework": "torch",
                "recording_platform": "darwin-arm64", "monai_version": metadata.version("monai"),
                "torch_version": metadata.version("torch"), "torch_threads": 1,
                "case": name, "ct_source": "HCC-TACE-Seg HCC_063 contrast CT",
                "source": "configs/inference.json preprocessing and postprocessing",
            }
            cast(ArchiveWriter, np.savez_compressed)(
                output / f"runner_{name}.npz", metadata=np.array(json.dumps(info)), **payload,
            )
            print(f"Recorded {name}", flush=True)
    record_runner(bundle, output, parser_factory, torch, nib, revision)
    record_graph(bundle, output, parser_factory, torch, revision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zoo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    record(arguments.model_zoo, arguments.output_dir)


if __name__ == "__main__":
    main()
