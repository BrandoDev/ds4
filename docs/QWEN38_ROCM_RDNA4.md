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
weights and selected routed experts for row-kernel batches, synchronizes,
then releases the layer's ranges. Expert staging reuses device scratch;
the grouped Q2 prefill path retains whole-table transfers.
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

ROCm's expert row kernels now stage only the union selected by the batch, in ascending
expert ID order, and remaps IDs into the compact tables. A single token uses
10 of 512 experts: routed transfer volume is 10/512 of the former volume.
Adjacent expert spans share a copy, mid/down reuse the existing temporary
buffer, and the dot products retain their original accumulation order.
Already mapped or resident model ranges retain their existing path.
Q2 grouped prefill still stages the whole tables; IQ3 uses the row kernels
for prefill as well as decode.
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
| Q2 first 20 ARC questions, both optimizations | 545.852 s | 195.831 s | 2.79x | identical choice log-likelihoods and predictions |

These are separate comparisons on different question sets; their ratios
must not be multiplied. Q2 timings were collected during the integrity
investigation and establish transfer parity, not a validated accuracy score.
The first-20-question comparison uses the repaired, fully checksum-verified
Q2 and the same IDs as the earlier teacher-forced pilot. Both score 13/20
for acc and acc_norm. The new 300-question run is independent of the old
partial run and continues in `gguf/benchmarks/arc-challenge/q2-verified-fast-300`;
its postflight hash check is required before reporting success.
The final ROCm build, IQ3/Q2 expert tests, staged HC/GDN/PLE/router/MoE/MTP
tests, and indexer tests pass. Expert tests now exercise non-contiguous IDs,
multi-token unions, the shared expert, and explicitly reject nonfinite output.
The host ARC suite has 37 passing tests, including corruption beyond the
first 8 MiB. The runner accepts `--expected-model-sha256` to verify the entire
model before and after inference and record both hashes in the run provenance.

## Mixed CPU/GPU expert placement

`--cpu-experts` keeps the complete text graph while placing routed expert
projections on the host. CPU inputs and intermediates remain float; no
additional activation quantization is introduced. The 640 logical down
inputs exclude physical Q2_K/IQ3_XXS padding. Shared experts and reduction
remain on GPU. Original n-gram rows are read from SSD as before. The default
GPU expert path is retained when the placement option is absent.

Build and check actual Q2/IQ3 MoE layers with:

```sh
make rocm ROCM_ARCH=gfx1201 -j2
make tests/test_qwen4_cpu_experts ROCM_ARCH=gfx1201 -j2
./tests/test_qwen4_cpu_experts gguf/Qwen3.8-Flash-Next-Q2.gguf
./tests/test_qwen4_cpu_experts gguf/Qwen3.8-Flash-Next-IQ3.gguf
```

Layers 0 and 46 pass for batches 1, 3 and 9; both the MoE output and updated
wide residual are checked against GPU placement. Batch 9 exercises the
Q2 grouped prefill route. Full IQ3 graph checks cover three prompt rows and
two two-token teacher-forced continuations with prefix reuse: all seven
rows are finite, top-1 agrees, and worst vocabulary logit error is 0.00058413.
The greedy generation API is also tested with a CPU-expert call counter,
so a GPU-only fallback cannot pass unnoticed. Qwen greedy session decode
now dispatches through its own graph rather than the DeepSeek graph helper.
Three chat requests produce `4`, ARC answer `A`, and recalled number `17`,
with rendered prompts byte-identical to the official checkpoint template.
Artifacts are in `gguf/benchmarks/hybrid-iq3-validation/` (ignored local
results). The sampled chat RSS was 28.1 GiB, file-backed RSS 27.9 GiB,
no swap, and total sampled GPU VRAM usage 3.33 GiB. These are snapshots,
not peak limits. Dense weight transfers per layer remain in place.

The short full-graph diagnostic took 6.25 seconds with GPU experts and
3.65 seconds with CPU experts, including process startup. GPU was run first,
so cache warmth and the tiny workload prevent treating this as a general
speedup claim. On longer prompts CPU prefill can be slower. Placement has
not yet been validated with a new full ARC benchmark or independent runtime.

A complete greedy CLI chat also produced `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`.
This used CPU expert placement for both the 40-token prefill and decoding,
with 8 host threads: 4.02 prompt tokens/s and 1.46 generated tokens/s.
The earlier GPU-only helper generated the same answer at 12.78 prompt
tokens/s and 2.59 generated tokens/s. These are individual short runs with
different transient capacities, not controlled throughput estimates; mixed
placement is functional but is not established as a speed optimization.

## Budgeted resident CPU/GPU placement

`--gpu-memory-budget 12GB --gpu-expert-layers 8` enables a resident hybrid
placement on ROCm. The model remains complete: dense/shared/router weights
are copied once to VRAM, followed by MTP experts when MTP is enabled, then
as many complete trunk expert layers from layer zero as fit (up to eight).
The remaining routed experts execute on CPU. Token embedding rows remain
host gathers; original BF16 n-gram rows remain SSD reads. Quantized weights
are copied byte-for-byte; no new quantization or activation rounding is added.

The budget is a ceiling on total device usage, including the desktop and
other applications. Admission uses the larger usage from HIP and the same
PCI device's amdgpu `mem_info_vram_used` counter. Even oversized budgets leave
at least 2 GiB outside the ceiling, and residency leaves another 1 GiB inside
it for temporary kernel buffers. Graph allocation precedes expert fitting,
so context and prefill capacity reduce the number of GPU expert layers.
An allocation that cannot fit is refused. Concurrent allocations by other
applications can still change available memory after a check.

Resident weights have a separate lifetime from per-layer staging. Resident
layers skip the old synchronization/release boundary when there are no
temporary weight copies; full model teardown frees both caches. This policy
is confined to Qwen ROCm; other model/backend defaults are unchanged.

```sh
./ds4 --rocm -m gguf/Qwen3.8-Flash-Next-IQ3.gguf \
  --gpu-memory-budget 12GB --gpu-expert-layers 8 \
  --ctx 2048 --prefill-chunk 128 --threads 16 \
  --mtp --mtp-draft 2 --temp 0 --nothink \
  -p 'Count from 1 to 10, separated by commas. Reply with only the numbers.'
```

The same placement flags work in `ds4-server`. With only a memory budget,
the trunk prefix is fitted automatically up to all trunk layers. With only
`--gpu-expert-layers`, the budget defaults to 12 GiB. With `--cpu-experts`
and a budget but no explicit GPU layer count, all trunk experts stay on CPU;
MTP is still fitted with priority. Without a budget or GPU layer request,
`--cpu-experts` retains its original all-CPU expert placement.

Validation artifacts and the sequential reproduction script are in
`gguf/benchmarks/hybrid-resident-validation/`. They include actual Q2/IQ3
layer parity, full-vocabulary IQ3 comparisons against the archived GPU
reference, short MTP chats, 100 ms VRAM/RSS samples and source/binary hashes.
Earlier failed accounting/diagnostic attempts are retained in subdirectories.

On the 41-token counting chat (context 2048, chunk 128, greedy MTP), IQ3
measured 1.88 generated tokens/s with CPU experts and per-layer dense copies
at eight threads, 2.38 with budgeted residency at eight threads, and
3.16–3.36 with residency at sixteen threads. Resident IQ3 chat VRAM peaked
at 10.94 GiB total, including other device users. The fitted chat placement
was one trunk GPU expert layer plus GPU MTP; eight is a requested maximum.
Across all resident validation runs the largest sampled VRAM was 11.30 GiB.
MTP timing confirmed 13 verification cycles and 13 accepted drafts.

The same IQ3 chat immediately following Q2 measured only 0.63 tokens/s;
repeating it recovered to 3.16. Cache warmth is a likely contributor, but
this is not an isolated I/O experiment. Keep the slow run when interpreting
these short measurements. Full IQ3 vocabulary comparisons retain top-1
agreement on all seven rows; maximum logit difference is 0.00060726.

## IQ3 CPU expert optimization

The ROCm hybrid path uses an AVX2 IQ3_XXS dot product on compatible host
builds. A 16 KiB signed-grid lookup replaces per-value bit extraction and
sign branches. Activations remain float; weights retain their original
encoding. Logical down rows stop at 640 values even when physical storage
is padded to 768. The scalar reference remains unchanged, and builds
without AVX2 retain it. This optimization is automatic and adds no CLI flag.

For multi-token batches, tokens selecting the same CPU expert share weight
decoding in groups of up to four. Independent accumulators and the original
output slots preserve routing and reduction order. This applies only when
all three routed projections are IQ3_XXS; Q2 and other formats keep their
existing path. Grouping improves prefill beyond the single-row SIMD kernel.

Sequential real-model comparisons on Ryzen 9 9950X / RX 9070 XT used IQ3,
16 threads, context 4096, chunk 128, greedy MTP, a 14 GiB requested device
budget and a maximum of ten GPU expert layers. The actual placement was
three trunk GPU expert layers plus GPU MTP in every paired chat run.

| Short prompt | Scalar generation | Optimized generation | Scalar prefill | Optimized prefill |
| --- | ---: | ---: | ---: | ---: |
| Counting, warm repeat | 3.39 t/s | 5.59 t/s | 8.14 t/s | 34.48 t/s |
| Date answer | 2.71 t/s | 5.14 t/s | 7.45 t/s | 29.71 t/s |
| Explanation, capped at 32 generated tokens | 2.73 t/s | 4.84 t/s | 7.56 t/s | 30.37 t/s |

The SIMD-only counting run measured 25.63 prefill t/s and 5.53 generation
t/s: grouping primarily helped prefill in this comparison. These are short
prompt measurements, not a general throughput guarantee. The first scalar
counting run was slower and had many more major page faults; it is retained
in the report rather than used as the headline baseline. Total sampled
chat VRAM peaked at 12.90 GiB, including the desktop.

All three greedy answers were identical between old and new binaries.
Actual IQ3 and Q2 expert parity passed for batches 1, 3 and 9, including
repeated expert selections. Full-vocabulary IQ3 comparisons retained top-1
agreement on seven rows, with maximum absolute logit difference 0.00058031
against the archived GPU reference. Small numerical differences are expected
from the SIMD float reduction order. `make test-qwen4-iq3-dot` additionally
checks independent double-precision decoding, grouped projections and guard
pages at activation boundaries. Its cached-row microbenchmark is separate
from the model throughput measurements.

Reproduction commands, outputs, VRAM/RSS samples, page-fault counts and
source/binary hashes are preserved in
`gguf/benchmarks/hybrid-iq3-optimized/summary.json`; `validate.py` executes
the comparisons sequentially. Archived scalar and SIMD-only binaries keep
the performance baseline reproducible. CUDA host and CPU builds compile;
CUDA/Metal hardware execution was unavailable for this change.

The matching synchronized 16-token stage profile is preserved in
`hybrid-iq3-optimized/profile-summary.json`. Mean MoE time for a two-token
verification trunk fell from 427.86 to 155.69 ms; the other timed trunk
groups remained approximately 133–137 ms combined. MoE still takes about
53% of those groups. Output-head and MTP predictor work are outside that
breakdown, and explicit stage synchronization changes throughput, so use
the paired chats above for speed comparisons.

## Further float-preserving hybrid optimizations

The next optimization pass retains model bytes, float activations, F16 KV
and full target-vocabulary verification. No activation quantization or SSD
table conversion is introduced. The ROCm Qwen path now provides:

- Cooperative 32-lane routed expert dots for IQ3/IQ2/Q2_K and the Q4_K/MXFP4
  MTP experts. Adjacent lanes read adjacent K values. The float reduction
  tree changes; numerical comparisons are required rather than bit identity.
- One reusable pinned host allocation for expert inputs, IDs and partial
  outputs. Both readbacks are enqueued before one stream wait; output upload
  is asynchronous and ordered before GPU reduction. The next readback waits
  before reuse, and graph teardown waits before freeing pinned pages. If
  pinned allocation fails, the synchronous pageable path remains available.
- Dynamic 64-row CPU slices for groups containing unequal token counts,
  confined to the hybrid expert job. The general worker pool is unchanged.
- Joint IQ3 CPU gate/up decoding with separate accumulators and shared
  activation loads. Unit tests compare the gate result bit-for-bit against
  the separate SIMD kernel and check the other projection against an
  independent double decoder. Unused row scratch allocations are removed
  for formats with direct float dots.
- One-launch Q8 dense projection pairs/groups at up to eight tokens, using
  the same per-row arithmetic as separate Qwen kernels. Linear attention
  input projections, sparse attention/indexer projections and CPU-layer
  shared-expert gate/up can use this path.
- HC residual copy folded into its combine kernel. It reads the preceding
  residual and writes the next residual with the same multiply/add expression.

These are automatic, without new public tuning flags. Diagnostic environment
switches `DS4_QWEN4_MOE_SERIAL`, `DS4_QWEN4_PAGEABLE_STAGING`,
`DS4_QWEN4_CPU_STATIC`, `DS4_QWEN4_CPU_SEPARATE_GATE_UP`,
`DS4_QWEN4_NO_MULTI_PROJECTION` and `DS4_QWEN4_HC_COPY_REFERENCE` expose
the corresponding reference operations for comparison.

Sequential IQ3 comparisons on the same hardware and command as the previous
section, with the preceding float-SIMD binary archived as the baseline:

| Short prompt | Previous generation | New generation | Previous prefill | New prefill |
| --- | ---: | ---: | ---: | ---: |
| Counting, warm repeat | 5.56 t/s | 6.59 t/s | 33.31 t/s | 38.13 t/s |
| Explanation, capped at 32 tokens | 4.57 t/s | 5.97 t/s | 25.30 t/s | 35.38 t/s |

All paired greedy answers matched. Counting is the cleaner warm comparison;
the first previous-binary run and some other prompts have more major page
faults. Individual diagnostic comparisons are short and cannot establish a
precise isolated gain for each change. Total sampled chat VRAM stayed at
about 13.14 GiB with three trunk expert layers and MTP on GPU. The requested
ten layers still means a maximum fitted to available memory.

Actual-model expert parity now includes trunk layers 0/46 and MTP layer 48
at batches 1, 2, 3 and 9. Routed MTP mid/down outputs are checked separately
against the CPU reference. This exposed a Q4_K scale-bit shift mistake in
the first cooperative implementation; it was corrected before acceptance,
without relaxing tolerance. Failed attempt logs are preserved.
Full-vocabulary target logits keep top-1 agreement on all seven archived
reference rows, with maximum absolute difference 0.00037861. These checks
support numerical correctness on the tested inputs; they do not constitute
a new model-quality benchmark or proof of identical tokens on every prompt.

The ROCm hybrid variant of `tests/test_qwen4_ngram_state` checks snapshot
restore, ordinary/MTP batch cycles, forced draft rejection, n-gram disk
failure/recovery, context limits and arena growth. It requests one trunk
GPU expert layer to leave room for simultaneous sessions. CPU and CUDA host
compilation also pass; Metal/CUDA device execution is unavailable here.

Artifacts, commands, source/binary hashes, residency, sampled memory and
page-fault counts are in `gguf/benchmarks/hybrid-quality-optimized/`.
`stage1/`, `stage2/` and `failed-mtp-check/` preserve intermediate results;
the root report identifies the accepted implementation.

HIP graph capture is not enabled by this pass: cached temporary allocations
can move, and speculative snapshot settings change the captured pointer set.
Correct capture requires explicit ownership and invalidation of those
resources. The current release retains the verified ordering rather than
capturing a graph with unproven pointer lifetime. CPU Q8 activation dots
likewise require a separate accuracy evaluation before adoption.

## Cooperative attention, GDN and routing

The next ROCm pass replaces the one-thread-per-head attention decoder with
a 32-lane head. Adjacent lanes read adjacent KV columns and retain at most
eight output accumulators each. It keeps the online softmax, causal filtering
and sparse token ordering. GDN similarly assigns a wave to each value row,
loads the existing state layout contiguously and keeps the row in registers
through the scan. Both speculative state snapshots retain their existing
layout. Float reduction trees change; quantization formats do not.

The router stores probabilities in shared memory and selects top-k with a
wave reduction, retaining smallest-expert-ID tie ordering. The full softmax
denominator and the selected-weight denominator keep their serial ordering.
The shared-expert gate dot is cooperative. CPU IQ3 grouping now also covers
one-token decode, using the same float gate/up kernel as verifier batches.

ROCm HC mixing activates the low-rank vector once, projects it using the
dense kernels and mixes the resulting channels. This avoids recomputing SiLU
for each output dimension. For small F16 batches, rows up to width 512 use
four waves per block and share weights across tokens; the model's stored HC
rank is 320. Wider rows retain the existing matmul. The verifier's inputs
remain FP32, and Metal/CUDA HC dispatch is unchanged.

Sequential paired warm runs use the same IQ3 file, 16 threads, context 4096,
chunk 128, 32-token cap, temperature zero, MTP and a 14 GiB device budget.
The actual placement in both binaries is four trunk expert layers plus GPU
MTP, with fewer desktop allocations than in the earlier three-layer runs.

| Prompt | Before generation | Current generation | Before prefill | Current prefill |
| --- | ---: | ---: | ---: | ---: |
| Counting | 6.50 t/s | 14.37 t/s | 36.22 t/s | 45.72 t/s |
| Date only | 5.91 t/s | 12.44 t/s | 34.22 t/s | 42.42 t/s |
| Sky explanation, output capped | 5.86 t/s | 12.96 t/s | 35.09 t/s | 43.72 t/s |

All paired runs have zero major page faults and identical generated text.
Forcing two draft tokens (three verifier rows) reaches 15.73 t/s on counting
and 13.38 t/s on the explanation with identical text; the adaptive default is
retained. These are short samples, and this depth is still smaller than the
three drafts in the user's llama command. Sampled paired-chat GPU usage stays
below 12.52 GiB including other GPU users, within the requested budget.

A separate 1243-token synthetic context uses chunk 512, three fixed GPU
expert layers, temperature zero, an eight-token output cap and no MTP in
both binaries. After an archived cold warmup, prefill improves from 34.47
to 82.48 t/s and ordinary generation from 1.78 to 10.03 t/s. Text is
identical. The before run has three major faults; the current run has 915,
so the speedup is not explained by the current run having a warmer cache.
This is still a repetitive synthetic prompt, distinct from the user's
llama UI request, and the short ordinary decode is not comparable to the
MTP chat numbers above.

Seven full-vocabulary rows retain the previous top-1 token, with maximum
logit difference 0.000590801. Synthetic tests cover real head widths, dense
and sparse attention, stable router ties, odd F16 dimensions and recurrent
scans/snapshots. The session test passes rejected drafts, disk read failure
and recovery, context boundaries and shared-arena growth. The CPU SIMD test
and CPU/CUDA host syntax checks pass. These establish numerical/state
regressions on ROCm, not a general task accuracy benchmark or device testing
on Metal/CUDA.

Commands, outputs, hashes, sampled memory and the archived before binary
are in `gguf/benchmarks/hybrid-llama-port/`. `stage1/` preserves intermediate
GPU results. A tested AVX512 CPU prototype was removed because it did not
improve end-to-end performance over AVX2. CPU Q8 activation products, a
four-row MTP verifier and HIP graph capture remain separate work; this pass
does not claim the llama server's 33 t/s with our current custom weights.

### Experimental CPU IQ3 × Q8_K assessment

The hybrid ROCm expert path has a temporary diagnostic switch,
`DS4_QWEN4_CPU_Q8=1`. Float activations remain the production default: the
Q8 candidate has measurable numerical and quality drift and is not accepted
as an optimization that preserves quality.

The candidate quantizes each token's gate/up input once and each selected
expert's SwiGLU output once. AVX2 integer products share the row decode across
up to four routed slots. The down projection zero-pads 640 logical inputs to
768 stored inputs. CPU graph buffers are reused and freed with the graph;
weights, routing, GPU experts, MTP weights and SSD n-gram tables are unchanged.
The signed-byte activation range is restricted to [-127,127], avoiding sign
reversal overflow. Router weights are applied by the existing GPU reducer.

The final paired warm tests (same binary, 16 threads, 14 GB budget, four GPU
trunk expert layers, GPU MTP, greedy, 32-token cap) measured:

| Prompt | Float generation | Q8 generation |
| --- | ---: | ---: |
| Count 1–10 | 14.55 t/s | 16.42 t/s |
| Berlusconi date | 12.98 t/s | 14.55 t/s |
| Blue sky explanation | 13.69 t/s | 15.93 t/s |

All six chat runs had zero major faults. Peak total VRAM, including desktop,
was 13.01 GiB. These short runs are a pilot, not a throughput guarantee. The
cached-row microbenchmark improved approximately 1.8× relative to the float
AVX2 product. Q8's speedup on the whole graph was only 12–16%.

On a fixed text corpus formed from the existing ARC sample, teacher-forced
scoring of 256 tokens after a 32-token prefix changed perplexity from
5.944938 to 5.999246 (+0.9135%). This measures a paired NLL regression, not
ARC question accuracy. The count and date responses matched; the sky
explanation changed. Seven full-vocabulary rows retained their top token,
but differed by up to 1.4488 in logits. The independent dot-product oracle
uses dequantized Q8 inputs: kernel correctness is assessed separately from
the approximation to original float activations. Existing float tolerances
are not relaxed to accept Q8.

Run the sequential diagnostic with:

```sh
python gguf-tools/quality-testing/run_hybrid_cpu_q8.py
```

It saves source/binary/model provenance, commands, corpus, logits, responses,
NLL, timing, page faults, RSS, total VRAM and MTP/state regression output under
`gguf/benchmarks/hybrid-cpu-q8/`. It requires the local IQ3 model, the existing
ARC sample and the built dot/state test executables. Only one model process
runs at a time. The CPU-only reference and other GPU backends retain their
existing arithmetic.

### Extended activation-quantization perplexity protocol

`gguf-tools/quality-testing/run_hybrid_perplexity.py` compares fixed raw-text
windows from WikiText-2 test, Tiny Shakespeare, Italian Pinocchio and the
existing ARC sample. The manifest retains source URLs, source hashes, exact
token sequences, offsets and context lengths. All model processes run in
sequence, with MTP disabled, a 14 GB budget and four GPU expert layers.

Before scoring, it compares repeated teacher-forced probabilities and an
8-token target microbatch against single-token teacher forcing on every
source. Float passed microbatch calibration (maximum log-probability drift
below 0.0004 on those controls). Q8 failed (up to 1.3524), despite exact
repeatability within each method. The grouped integer dot kernel separately
passes an independent double-precision oracle and yields bit-identical
results to single-input dots for distinct quantized inputs. This rules out
input-slot mixing in that kernel; the cause of the whole-model batch
sensitivity still needs isolation. Therefore the extended Q8 assessment uses
single-token targets for both modes. Large all-row prefill batches also
failed float calibration and are excluded from this study.

The first-token diagnostic can export compact next-token probabilities with
`DS4_QWEN4_FT_SCORES=1`. Its default binary logit export is unchanged.
`DS4_QWEN4_FT_TEACHER_BATCH=1..8` controls only the teacher-forcing diagnostic,
never generation. It defaults to 1. The comparison uses the same 128-token
prefill chunks, then scores targets sequentially. The CSV retains absolute
sequence position, ground-truth token ID, normalized log probability and top
predicted token. The harness verifies token alignment before computing NLL.

Perplexities are computed as `exp(total NLL / total scored tokens)`, rather
than averaging per-window perplexity. Differences are paired on the exact
same tokens. Confidence intervals resample whole windows within source and
context strata; they do not treat correlated tokens as independent. Book
windows share a source/author, and context strata use different articles;
these intervals describe the sampled windows, not all text or a causal
context-length effect. This is not the canonical full WikiText score and
cannot certify unchanged chat accuracy.

After downloading the sources listed in the manifest into the output's
`data/` directory:

```sh
python gguf-tools/quality-testing/run_hybrid_perplexity.py --stage prepare
python gguf-tools/quality-testing/run_hybrid_perplexity.py --stage calibrate
python gguf-tools/quality-testing/run_hybrid_perplexity.py --stage score
python gguf-tools/quality-testing/run_hybrid_perplexity.py --stage analyze
```

The default output is `gguf/benchmarks/hybrid-cpu-q8-perplexity/`. Each mode
also repeats an entire short-context and long-context window after other
windows, verifying graph reset and probability reproducibility. Original
256-token pilot artifacts are retained in `hybrid-cpu-q8/`.

The completed study scored **14,336 tokens in 28 windows** per mode, plus
separate repeated-window controls. Token-weighted perplexity was **3.750271
float versus 3.760689 Q8**, a **+0.278%** change. The paired, stratified
window-bootstrap 95% interval was **−0.504% to +1.275%**. It includes zero
and the original +0.9135% pilot observation; it establishes neither a clear
overall regression nor equivalence. These intervals exclude execution
variability, which the failed Q8 repeat controls exposed.

| Source | Scored tokens | Float PPL | Q8 PPL | Relative change |
| --- | ---: | ---: | ---: | ---: |
| WikiText-2 sample | 6,144 | 4.091777 | 4.073292 | −0.452% |
| Tiny Shakespeare | 2,048 | 1.313715 | 1.314721 | +0.077% |
| Pinocchio, Italian | 2,048 | 4.687572 | 4.665537 | −0.470% |
| ARC text sample | 4,096 | 4.973093 | 5.065823 | +1.865% |

Thirteen windows had higher NLL under Q8 and fifteen had lower NLL. One ARC
window had a +12.91% perplexity regression. The top predicted token changed
on 4.499% of all scored positions; perplexity similarity does not imply
identical outputs. The sources are archived from [WikiText document-level
conversion](https://huggingface.co/datasets/EleutherAI/wikitext_document_level),
[Karpathy's Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt),
[Project Gutenberg's Italian Pinocchio](https://www.gutenberg.org/ebooks/52484),
and the existing local ARC-Challenge sample. Source hashes and token offsets
are in the manifest. The previous pilot automatically placed five GPU
expert layers with MTP disabled; this study fixes four in both modes.

Float repeated both entire control windows with **zero** probability drift.
Q8 repeated the short window exactly, but the long window differed by up to
**0.645602** in log probability: nine top predictions changed and the repeated
window's perplexity increased **0.2384%**. Further focused runs explicitly
disabled prefix caching and repeated a 1,536-token prefix with 64 targets.
Float remained bit-identical both consecutively and after a different text.
Q8's consecutive runs differed by up to **0.376697** in log probability;
a subsequent run after a different text matched the first. This reproduces
Q8 sensitivity without prefix reuse. Its cause remains unisolated; the data
must not be reported as a passed quality validation. Float stays default.
The independent grouped integer dot oracle passes, also under production
`fast-math` flags; this is not evidence that the whole Q8 graph is stable.

All raw CSVs, failed calibration outputs, repeat controls, focused-reset
runs, hardware metadata, logs and summaries remain under
`gguf/benchmarks/hybrid-cpu-q8-perplexity/`. The compact exporter rejects
nonfinite logits and verifies target-position alignment through the harness.
Unit tests cover shifted targets, token-weighted NLL aggregation, paired
bootstrap arithmetic, and retention of failed repeat controls.
