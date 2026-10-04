# Qwen3.8 Flash Next ROCm validation on gfx1201

Target: Radeon RX 9070 XT (`gfx1201`, 16 GiB VRAM). The Qwen main text graph
is compiled into the ROCm build. ROCm now has dense, HC, GDN, indexer,
attention, PLE, router, IQ2_XXS/Q2_K and IQ3_XXS MoE, and MTP entry points. The new
attention and MoE kernels favor clear FP32 math over speed. Vision encoding is
not available on ROCm and returns an explicit error.
Input embedding rows and BF16 PLE n-gram rows are staged by the host; the
trunk projections, routing, expert arithmetic, state updates and logits run
through the GPU graph. There is no CPU routed-expert fallback in that graph.

**Q2 and IQ3 full-model parity and generation passed on the RX 9070 XT.** The sandbox
hides `/dev/kfd` and `/dev/dri`; the commands were run with host device
access. The DS4 Q2 GGUF is `gguf/Qwen3.8-Flash-Next-Q2.gguf`, 137.10 GiB on
disk. Its 95.37 GiB BF16 n-gram table stays on disk, while its 41.73 GiB
main/MTP tensor set exceeds the card's 16 GiB VRAM. The graph copies dense
weights and the union of selected routed experts for each layer, synchronizes,
then releases the layer's ranges. Expert staging reuses device scratch.
Host mapping of the file-backed weights failed with HIP's `Memobj map
does not have ptr` error on this host. The existing `--ssd-streaming`
selected-expert cache is not connected to the graph; leave it off.

## Results on the RX 9070 XT

ROCm HIP 7.2.53211, HSA runtime 1.18, `gfx1201`, wavefront 32, 64 compute
units, 16 GiB VRAM. The non-WMMA path is active. `make rocm
ROCM_ARCH=gfx1201 -j2` and all three test binaries build successfully.

The initial index test found a block-key normalization bug: the warp
reduction returned the full sum only in lane 0, but every lane used its own
partial sum. Broadcasting the full sum fixed it. The rerun passes score,
select, expand, and block key (block-key max absolute error 0).

`test_qwen4_rocm_iq3` passes with the same values over four runs:

| Operation | Max absolute | Mean absolute | Max relative |
| --- | ---: | ---: | ---: |
| Q2 baseline expert forward | 2.61e-08 | 9.31e-09 | 7.07e-07 |
| IQ3 gate/up, K=2560, 17 rows | 4.77e-06 | 2.07e-06 | 1.85e-05 |
| IQ3 down, logical 640/physical 768, 17 rows | 1.43e-06 | 5.79e-07 | 1.70e-06 |
| IQ3 two-expert forward | 1.16e-06 | 3.34e-07 | 4.01e-05 |

`DS4_TEST_QWEN4_ROCM_CORE=1` passes HC, GDN, PLE, router, attention,
IQ2_XXS/Q2_K MoE, Q4_K/MXFP4 MTP experts, dense and MTP comparisons. The largest reported MoE mid
absolute error is 3.26e-04 against output scale 1.91e+02. The 641-token
prefill test passes; its IQ2 mid maximum error against exact CPU is 9.68e-05
and Q2 down maximum is 4.51e-06. These are synthetic parity results, not a
substitute for a real model pass.

The Q2 GGUF was downloaded and verified by `download_model.sh`. The imatrix
for the IQ3 converter is at `gguf/source-qwen/imatrix_unsloth.gguf_file` and
its SHA-256 matches the hash pinned in `qwen4_pack.py`. All 131 official BF16
safetensors shards at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
are in `gguf/source-qwen`. The IQ3 gate/up and padded down payloads were
converted from those shards using the standard GGML IQ3_XXS encoder.

## Q2 full-model results

With `DS4_QWEN4_FT_TOKENS=760`, CPU and ROCm evaluated the same one-token
input. The largest logit difference was `0.000650048`; both chose token
`2614`. The layer stream comparator accepted all 48 trunk layers, from
`1.19e-7` maximum absolute error at layer 0 to `0.00034523` at layer 47.
Using two input tokens, the MTP draft had maximum logit difference
`0.00405693` and the same top token (`369`) on CPU and GPU.

The 59-token `Hello` chat-template prompt generated four greedy tokens on
the GPU. Ten additional repetitions exited successfully with identical
output. Prefill measured 17.58–18.28 tokens/s and generation 1.13–1.20
tokens/s. No crash or out-of-memory error was observed. The sampled VRAM
usage is quantified in the measurements below. Longer contexts remain open.

## IQ3 conversion and full-model results

The completed model is `gguf/Qwen3.8-Flash-Next-IQ3.gguf`,
159,916,531,712 bytes (148.93 GiB), SHA-256
`f2d5c0d0e5d229e3230eea8e2464dd51e291637d65d742994f1b7570587581e7`.
It uses 144 IQ3_XXS tensors: 48 each for routed gate, up, and down.
The final assembly changed exactly the 48 down tensors relative to the
verified gate/up intermediate. MTP and every other tensor retain their
payload hashes. The BF16 n-gram table is the final tensor, with a 4096-byte
GGUF alignment and page-aligned absolute offset.

| Storage | Bytes | GiB |
| --- | ---: | ---: |
| Routed gate, including MTP | 15,885,926,400 | 14.79 |
| Routed up, including MTP | 15,885,926,400 | 14.79 |
| Routed down, including MTP | 18,942,525,440 | 17.64 |
| Routed total | 50,714,378,240 | 47.23 |
| Main/MTP tensor payloads | 57,504,328,192 | 53.56 |
| BF16 n-grams, disk only | 102,400,491,520 | 95.37 |

The 48 trunk down tensors alone occupy 18,496,880,640 bytes. Their logical
640-column weights cost exactly **3.675 bpw** after padding to 768 columns.
The converter used the pinned official BF16 checkpoint and the pinned Unsloth
imatrix. Its 24 zero-count expert entries used the converter's documented
per-expert weight-energy fallback.

On one token (`DS4_QWEN4_FT_TOKENS=760`), the largest CPU/GPU logit
difference was `0.000711918`; both selected token `20438`. The layer-stream
comparator accepted all 48 trunk layers, with no divergent layer. Its
largest absolute difference was `0.00141716` at layer 47. On two tokens,
both main positions selected the same top token and the MTP draft's largest
logit difference was `0.00307989`, with top token `369` on both CPU and GPU.

The 59-token `Hello` chat-template prompt generated four greedy tokens:
`We need to respond`. Ten additional runs all exited successfully with the
same output. Across those ten runs, prefill measured 12.81–16.38 tokens/s
and generation 0.86–1.09 tokens/s. A cold-disk run measured 1.93 prefill
tokens/s and 0.84 generation tokens/s, so performance is strongly dependent
on cache state. No NaNs, crash, or growing process footprint were observed
in these short runs.

## Short-run memory and device measurements

The same 59-token prompt and four-token decode were sampled every 100 ms
through sysfs and `/proc`. These are sampled maxima, not guaranteed hardware
peaks. The Q2 and IQ3 resident main/MTP sets exceed 16 GiB; the graph copies
weights to VRAM one layer at a time and frees them after synchronization.

| Model | Max sampled VRAM | Max process RSS | Max process swap | GPU busy average |
| --- | ---: | ---: | ---: | ---: |
| Q2 | 4.56 GiB | 39.31 GiB | 46 MiB | 82% |
| IQ3 | 4.90 GiB | 51.21 GiB | 40 MiB | 82–95% |

The PCIe link reported 32.0 GT/s ×16. PCIe byte traffic and expert kernel
time were not measured. The observed process CPU use in two cold runs was
about 0.6–0.7 of one core on average. The GPU usage percentage is sampled
and includes idle time while loading the model.

## Build and model-free device tests

Run these on the RX 9070 XT machine with `rocminfo` reporting `gfx1201`:

```sh
make rocm ROCM_ARCH=gfx1201 -j2
make tests/test_qwen4_rocm_idx tests/test_qwen4_rocm_iq3 tests/test_qwen4_rocm ROCM_ARCH=gfx1201 -j2
./tests/test_qwen4_rocm_idx
./tests/test_qwen4_rocm_iq3
DS4_TEST_QWEN4_ROCM_CORE=1 ./tests/test_qwen4_rocm
DS4_TEST_QWEN4_ROCM_PREFILL=1 ./tests/test_qwen4_rocm
```

The IQ3 test prints max absolute, mean absolute, and max relative errors for
17 gate/up rows (`K=2560`), 17 down rows (logical `K=640`, encoded `K=768`),
and a two-expert gate/up/down forward. Its device input allocation contains
only the logical activation width. The same binary runs a Q2_K expert
forward first. The staged core test compares HC, GDN, PLE, router, attention,
IQ2_XXS gate/up with Q2_K down output, MTP, and a dense projection against CPU references. The
pre-existing index test checks score/select/expand/block-key behavior.
The prefill test checks the IQ2_XXS/Q2_K expert-list matmul path at a 641-token
test shape and prints its errors against an exact CPU reference.

The converter/CPU reference checks can be run on a machine without ROCm:

```sh
make cpu -j2
DS4_TEST_GGML_LIBRARY=/path/to/libggml-base.so \
  python3 -m unittest discover -s gguf-tools/tests -p 'test_qwen4*.py' -q
```

## Model pass

Set `QWEN_Q2_GGUF` to `gguf/Qwen3.8-Flash-Next-Q2.gguf`, or use
`gguf/Qwen3.8-Flash-Next-IQ3.gguf` for the IQ3 pass. Use a short context and
a single token first:

```sh
DS4_QWEN4_GPU=1 DS4_QWEN4_GPU_CHUNK=1 DS4_QWEN4_FT_TOKENS=760 \
  ./ds4 --rocm -m "$QWEN_Q2_GGUF" --first-token-test -p 'Hello' -c 128
./ds4 --rocm -m "$QWEN_Q2_GGUF" -p 'Hello' -c 128 --prefill-chunk 1 -n 4 --temp 0
```

The first command runs CPU and ROCm over the same token IDs and prints each
position's largest logit difference and top-token agreement. If it fails,
report the first failing operation/diagnostic output before trying a longer
prompt. Once Q2 passes, repeat both commands with the IQ3 recipe GGUF.
For per-layer parity on a short prompt, use the built-in CPU stream dump and
the matching ROCm stream dump. One-token chunks make each GPU file one row:

```sh
DS4_QWEN4_GPU=1 DS4_QWEN4_GPU_CHUNK=1 \
DS4_QWEN4_FT_TOKENS=760 \
DS4_QWEN4_FT_HIDDEN=/tmp/qwen-cpu-hidden.bin \
DS4_ROCM_GRAPH_DUMP_PREFIX=/tmp/qwen-gpu \
DS4_ROCM_GRAPH_DUMP_NAME=qwen_layer_stream \
  ./ds4 --rocm -m "$QWEN_Q2_GGUF" --first-token-test -p 'Hello' -c 128
python3 gguf-tools/compare_qwen4_layer_streams.py \
  /tmp/qwen-cpu-hidden.bin /tmp/qwen-gpu
```

The comparator reports max/mean absolute error and the first layer outside
its configurable absolute/relative tolerance. The ROCm dump is optional and
synchronizes every layer, so use it only for diagnostics. The model-free core
test isolates GDN/attention, routing, routed/shared expert, residual, and MTP
stages if a layer diverges.
To check MTP after the main model passes, set `DS4_QWEN4_MTP_OUT` to a
writable output path and use a prompt with multiple tokens in the first
command.

The ROCm graph currently uses per-layer device copies of weights. It
synchronizes before releasing each layer's allocations, keeping VRAM bounded
at the expense of host-to-device traffic. Long-context stability, true peak
VRAM, PCIe byte traffic, and expert kernel timings remain to be measured.

## Correctness fixes and remaining limits

The first index test found that block-key normalization used a wave sum that
was complete only in lane 0; the ROCm kernel now broadcasts the full sum.
The Q2 model then exposed Q4_K gate/up and MXFP4 down tensors in the MTP
layer; the ROCm expert decoder and graph validation now accept those types.
Host mapping of the large file-backed weight ranges failed under HIP on this
host, so the graph stages bounded device copies and synchronizes before
release. The IQ3 assembler was corrected to preserve the embedded BF16
n-gram tensor and give the whole GGUF a 4096-byte alignment, which keeps
that final disk-only tensor page-aligned as required by the runtime.

GGML IQ3 quantization intermittently aborted inside its neighbor search at
24 converter threads. The same source tensor passed when quantized
sequentially; conversion of all 144 trunk tensors completed with eight
threads and explicit IQ3 grid initialization. The high-thread-count
converter instability remains unresolved. Vision encoding on ROCm remains
unsupported. The current results cover short text prompts and MTP; they do
not establish long-context stability or throughput on cold SSD data.

## October 4: ARC integrity investigation and transfer optimizations

The Q2 teacher-forced ARC run stopped at question 272 after 271 completed
questions. A fresh process reproduced NaN logits. Layer tracing located the
first failure at layer 46's FFN hyper-connection: its F16 up matrix contained
six NaNs, with 256 values differing from both the original BF16 source
converted to F16 and the independently stored IQ3 dense tensor. An HTTP range
read of the original public Q2 agreed exactly with those references. The
damaged tensor was backed up and restored. A full-file read also encountered
Btrfs checksum failures in the n-gram region; those unreadable blocks were
restored from the original file. The repaired Q2's complete SHA-256 is
`b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9`,
matching the published file. Its entire main/MTP prefix was compared against
the original and its n-gram rows against all 128 original BF16 source shards.
The old ARC scores are retained for diagnosis
and marked invalid for model-quality reporting. The cause of corruption has
not been established.

ROCm now stages only the union of experts selected by the batch, in ascending
expert ID order, and remaps IDs into the compact tables. A single token uses
10 of 512 experts: routed transfer volume is 10/512 of the former volume.
Adjacent expert spans share a copy, mid/down reuse the existing temporary
buffer, and the dot products retain their original accumulation order.
Already mapped or resident model ranges retain their existing path.
`DS4_QWEN4_FULL_EXPERT_COPIES=1` selects the old transfers for diagnostics.

The ROCm `DS4_QWEN4_FT_LIST` diagnostic also snapshots the question's GDN and
PLE histories, n-gram context, and position. Consecutive choices with identical
prefix token IDs restore that snapshot and reuse the first logit row. KV and
index rows below the prefix remain intact; rows beyond it are invisible until
overwritten. Every continuation token still uses the same single-token
teacher-forcing path. The snapshot storage is allocated through the existing
verifier graph allocation; no predictor steps run.
`DS4_QWEN4_FT_NO_PREFIX_CACHE=1` disables reuse for diagnostics. CUDA and Metal
diagnostics retain their previous behavior.

| Local comparison | Reference | Optimized | Ratio | Logits |
| --- | ---: | ---: | ---: | --- |
| Q2 selected transfers, 12 ARC choices | 113.201 s | 49.873 s | 2.27x | bitwise identical |
| IQ3 prefix reuse, 12 ARC choices including a 177-token prefix | 114.489 s | 33.148 s | 3.45x | bitwise identical |

These are separate comparisons on different question sets; their ratios
must not be multiplied. Q2 timings were collected during the integrity
investigation and establish transfer parity, not a validated accuracy score.
The final ROCm build, IQ3/Q2 expert tests, staged HC/GDN/PLE/router/MoE/MTP
tests, and indexer tests pass. Expert tests now exercise non-contiguous IDs,
multi-token unions, the shared expert, and explicitly reject nonfinite output.
The host ARC suite has 37 passing tests, including corruption beyond the
first 8 MiB. The runner accepts `--expected-model-sha256` to verify the entire
model before and after inference and record both hashes in the run provenance.
