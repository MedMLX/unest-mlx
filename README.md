<div align="center">

# unest-mlx

Segment kidney cortex, medulla and collecting system in contrast-enhanced CT with UNesT and native MLX.

[![PyPI](https://img.shields.io/pypi/v/unest-mlx)](https://pypi.org/project/unest-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/unest-mlx)](https://pypi.org/project/unest-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/unest-mlx/blob/v0.2.8/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/unest-renal-ct-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/unest-mlx/main/docs/showcase.png" alt="Contrast CT → kidney cortex, medulla and collecting system, with a detailed renal view." width="100%">

*Contrast CT → kidney cortex, medulla and collecting system, with a detailed renal view.*

</div>

| Input | Output |
| :--- | :--- |
| Contrast-enhanced CT · NIfTI | Renal labels on the input grid |

| Model | Weights download | Peak MLX memory | Measured on |
| :--- | ---: | ---: | :--- |
| `unest` | 349 MB | 7.2 GB | 478×277×161 contrast CT |

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "unest-mlx>=0.2.8"
```

## Run

```python
from unest_mlx.medmlx import run

result = run(image_path="ct.nii.gz", output_dir="out")
print(result["outputs"]["mask"])
```

Weights download from `MedMLX/unest-renal-ct-mlx` on first use. Use `medmlx setup unest` to download ahead of time. To use your own copy,
pass a folder containing `unest.safetensors` as `weights_path`. A checkpoint file is also accepted.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | One 3D CT in NIfTI (`.nii` or `.nii.gz`), in Hounsfield units |
| `output_dir` | Directory for `mask.nii.gz` |
| `weights_path` | Optional `.safetensors` file or a folder containing it |
| `overwrite` | `False` by default; an existing mask is never replaced |

- **Input:** arterial or portal-venous contrast-enhanced CT. On the non-contrast scans
  we tried, the kidneys were missed. Spatial units must be millimetres or unspecified;
  metres and microns are not converted.
- **Mask:** float32 labels on the input grid as the NIfTI reader corrects it:

  | Label | Structure |
  | ---: | :--- |
  | 0 | Background |
  | 1 | Kidney cortex |
  | 2 | Medulla |
  | 3 | Pelvicalyceal system |

- **Return value:** `{"outputs": {"mask": path}, "details": {...}}`, with label counts,
  shapes, orientation, sliding-window settings, MLX device and elapsed time.
- **Errors:** invalid inputs raise `InvalidInputError`; missing or mismatched weights
  raise `AssetNotReadyError` with the download location.

### Python API

```python
import mlx.core as mx
import numpy as np
from unest_mlx import UNesT, UNesTPredictor, validate_weights

loaded = mx.load("unest.safetensors")
weights = validate_weights({key: np.array(value) for key, value in loaded.items()})
network = UNesT({key: mx.array(value) for key, value in weights.items()}, mx=mx)
predictor = UNesTPredictor(network, mx=mx)

volume = np.zeros((1, 1, 96, 96, 96), dtype=np.float32)  # preprocessed CT, Bx1xDxHxW
logits = predictor(volume)  # NumPy float32, Bx4xDxHxW
```

- `UNesTPredictor` uses the upstream 96³ windows with overlap 0.5 and constant blending;
  `sw_batch_size` defaults to four.
- Inference requires Apple Silicon with Metal.

## Use with MedMLX

Use the [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) to run `unest` from an MCP client. Weights download on first use.

```sh
uv tool install --python 3.13 "medmlx[segment,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe unest`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## License

[Apache-2.0](https://github.com/MedMLX/unest-mlx/blob/v0.2.8/LICENSE). See [third-party notices](https://github.com/MedMLX/unest-mlx/blob/v0.2.8/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.
