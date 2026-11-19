# Performance

For axis-aligned grids in the fused affine path, the CPU resampler uses
one-dimensional coordinates. It preserves the eight interpolation corners,
the order of the float64 weight products and corner sums, and the float32
rounding before label selection.

## Paired measurements

Measured on an Apple M1 Max with 64 GiB of memory, macOS 27.2.1, Python 3.13.17,
MLX 0.32.4, and NumPy 2.5.4. The baseline was `24228e8`. Both variants used
the released `unest.safetensors` file at revision `42ac739`, with batches of
four 96³ windows and 50% overlap.

Perfkit 0.3.0.dev0 ran six counted rounds per scan, rotating four variants with
partially balanced order. Each variant had a cache-priming call before its
first counted call. Each counted call ran in a fresh process from NIfTI
loading through mask publication, including checkpoint loading. The table
gives medians for the baseline and separable resampler. MLX peak comes from its
allocator, and process footprint comes from macOS `time -l`. The 96.9% paired
intervals are ratios of candidate to baseline elapsed time.

| HCC-TACE-Seg input | Source grid | Inference grid | Elapsed (s) | Paired time ratio | Peak MLX (GiB) | Peak process footprint (GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| HCC_004 | 512×512×79 | 320×320×196 | 31.39 → 30.51 | 0.955–0.995 | 6.72 → 6.72 | 13.08 → 12.76 |
| HCC_063 | 512×512×99 | 440×440×246 | 84.22 → 82.66 | 0.945–0.985 | 6.72 → 6.72 | 14.51 → 14.20 |
| HCC_004 oblique copy | 512×512×79 | 328×320×196 | 31.49 → 31.48 | 0.965–1.022 | 6.72 → 6.72 | 13.08 → 13.07 |

The end-to-end time intervals reach Perfkit's 2% minimum-effect band. In
separate paired calls on HCC_004, median warm preprocessing fell from 587 to
293 ms (0.477–0.517), and mask restoration fell from 884 to 367 ms
(0.411–0.419). Each stage sample used two measured calls after a cold call.
Restoration used the baseline softmax probabilities and the original CT grid.

The oblique copy preserves the HCC_004 voxels and adds a centered shear of
0.15 x voxels per z voxel, which exercises the general resampler. Across all
three inputs, every float32 logit bit before softmax and every byte of the
written `mask.nii.gz` matched the baseline. [TCIA HCC-TACE-Seg collection](https://www.cancerimagingarchive.net/collection/hcc-tace-seg/)

## Candidates not adopted

Two further changes were measured in the same runs and left out. Both kept
every pre-softmax logit bit and every byte of the written mask.

Reusing the centered values inside instance normalization cut elapsed time to
29.60 s on HCC_004 (ratio 0.925–0.948) and 79.11 s on HCC_063 (0.918–0.941),
and raised peak MLX memory from 6.72 GiB to 7.56 GiB on every input. The 12.6%
increase exceeds the 2% peak-memory limit.

A float32 fused multiply-add that takes the round-to-odd path only for float32
midpoints and sums outside the normal float32 range cut softmax from 1,322 ms
to 845 ms (ratio 0.633–0.647) and elapsed time on HCC_004 to 30.86 s
(0.966–0.987). It matched the existing kernel on 62,405,224 array lanes,
including ties and non-finite values, but raised `TypeError` for a scalar
negative zero and for the smallest scalar subnormal.

## MLX buffer cache

The cache sweep used the separable resampler above on an Apple M1 Max with
64 GiB of memory, macOS 27.2.1, Python 3.13.17, MLX 0.32.4, and NumPy 2.5.4.
The HCC-TACE-Seg scans were HCC_004 (512×512×79, inference grid 320×320×196)
and HCC_063 (512×512×99, inference grid 440×440×246). Both used the released
`unest.safetensors` from revision `42ac739`, with batches of four 96³ windows
and 50% overlap in float32.

### Uncapped memory attribution

One instrumented run per scan sampled macOS physical footprint
(`proc_pid_rusage`) and the MLX allocator counters at the stage boundaries
and on a 10 ms sampling loop during the run. The process value in each row is the
footprint at the sample. The MLX peak is cumulative. All values are measured
in GiB.

| Input | Stage | MLX active | MLX cache | MLX peak | Process footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| HCC_004 | After loading | 0.32 | 0.18 | 0.43 | 1.19 |
| HCC_004 | After inference | 0.32 | 10.90 | 6.72 | 12.18 |
| HCC_004 | Peak during softmax | 0.32 | 10.90 | 6.72 | 12.77 |
| HCC_004 | After mask publication | <0.01 | 11.22 | 6.72 | 12.38 |
| HCC_063 | After loading | 0.32 | 0.19 | 0.43 | 1.78 |
| HCC_063 | After inference | 0.32 | 11.31 | 6.72 | 13.64 |
| HCC_063 | Peak during softmax | 0.32 | 11.31 | 6.72 | 14.22 |
| HCC_063 | After mask publication | <0.01 | 11.64 | 6.72 | 13.38 |

The free cache accounted for 85.3% of the HCC_004 peak and 79.5% of the HCC_063
peak. Loaded weights accounted for another 0.32 GiB on each scan. Subtracting
the two allocator counters leaves 1.55 and 2.58 GiB, respectively, for host
arrays, allocator and driver overhead, and other process memory. The live
host volumes included the CT (39.5 and 99.0 MiB), the preprocessed input
(76.6 and 181.7 MiB), and the stitched logits reused as probabilities
(306.3 and 726.7 MiB). Inference also uses a one-channel stitch count map
and window batches. By the softmax peak, the process footprint had grown
by a further 0.59 and 0.58 GiB from the end of inference.

### Paired cap sweep

Perfkit 0.3.0.dev0 rotated the six variants separately for each scan, with a
fresh-process warmup per variant and six counted rounds. Each invocation ran
the public runner from image and checkpoint loading through mask publication.
The full-run measurement also includes Python startup.
The trial changed the cache limit only while the predictor ran sliding-window
inference. Full-run time and peak footprint came from macOS `time -l`, and
inference time spans the predictor through host stitching. The table reports
medians and the 96.9% paired interval for full-run time relative to that scan's
uncapped baseline. Each written mask was compared directly with the uncapped
`mask.nii.gz` using `cmp`.

| Input | Cache limit | Full run (s) | Inference (s) | Peak MLX (GiB) | Peak process footprint (GiB) | Paired full-run change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HCC_004 | Uncapped | 30.45 | 27.84 | 6.72 | 12.76 | — |
| HCC_004 | 0 | 41.22 | 38.63 | 6.72 | 7.69 | +32.2% to +35.5% |
| HCC_004 | 256 MiB | 37.21 | 34.59 | 6.72 | 7.81 | +19.9% to +22.9% |
| HCC_004 | 512 MiB | 34.59 | 31.97 | 6.72 | 7.83 | +11.8% to +14.2% |
| HCC_004 | 1 GiB | 34.16 | 31.55 | 6.72 | 8.19 | +9.6% to +12.6% |
| HCC_004 | 2 GiB | 32.67 | 30.02 | 6.72 | 8.89 | +4.9% to +15.2% |
| HCC_063 | Uncapped | 82.61 | 77.75 | 6.72 | 14.24 | — |
| HCC_063 | 0 | 115.73 | 110.87 | 6.72 | 8.74 | +38.1% to +46.4% |
| HCC_063 | 256 MiB | 101.89 | 97.02 | 6.72 | 8.88 | +23.2% to +27.8% |
| HCC_063 | 512 MiB | 95.16 | 90.33 | 6.72 | 8.87 | +14.9% to +19.6% |
| HCC_063 | 1 GiB | 94.03 | 89.19 | 6.72 | 9.25 | +13.2% to +18.2% |
| HCC_063 | 2 GiB | 89.13 | 84.28 | 6.72 | 10.02 | +7.4% to +12.2% |

Each cap lowered median peak process footprint by 29.6% to 39.7% across the
two scans. The counted MLX peaks varied by at most 1 KiB per scan, and every
median rounds to 6.72 GiB. All 72 counted masks and 12 warmup masks matched
the uncapped bytes. The full-run interval for every cap was above the 3%
slowdown allowance on both scans, so UNesT keeps the default cache policy.

### Footprint during the 2 GiB trial

The process peak moved to inference in the instrumented 2 GiB runs. At the
highest footprint samples, MLX accounted for about 74% of the footprint.
HCC_004 used 3.77 GiB active and 2.74 GiB cached at 8.83 GiB process footprint.
HCC_063 used 5.45 GiB active and 1.90 GiB cached at 10.00 GiB process footprint.
MLX can exceed the requested limit between freeing a buffer and trimming the
cache on a later allocation. [MLX cache limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html#mlx-core-set-cache-limit)

Restoration resamples four probability channels in blocks on up to eight
threads. It topped out at 3.57 and 4.17 GiB of process footprint, an increase
of 148 and 194 MiB from the start of restoration. Those increases were 1.6%
and 1.9% of the respective full-run peaks. The existing 306.3 and 726.7 MiB
probability buffers are still live during restoration. The resampler could
release the two gathered depth slabs once it has built the column buffers,
preserving the corner products, sum order, and rounding. Their shapes account
for about 92 and 132 MiB of live arrays with eight full blocks in flight.
The footprint effect of that change remains unmeasured.
