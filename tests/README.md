# Test input

The checkpoint-rejection and transform tests use the converted HCC-TACE-Seg
HCC_063 contrast CT at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz` by
default. `MEDMLX_TEST_CT` overrides the path; the image remains outside Git.

The transform, runner, and full-graph tests also use `MEDMLX_TEST_REFERENCES`
for their recorded arrays, defaulting to `work/real-reference`. Generate them
outside Git using MONAI 1.6.0, Torch
2.13.0, and the model-zoo revision recorded in `fixtures/upstream.json`:

```sh
python -m tests.record_transform_reference --model-zoo "$MONAI_MODEL_ZOO" \
  --output-dir "${MEDMLX_TEST_REFERENCES:-work/real-reference}"
```

Use the same CT while recording and testing. The recorded arrays include CT crops.
