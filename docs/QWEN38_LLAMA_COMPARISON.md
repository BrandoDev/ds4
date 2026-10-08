# Qwen hybrid performance: local llama.cpp comparison

Inspected local fork: `/home/moltencream/llama.cpp-flash-next-rdna4`,
commit `1f35edfeefa9166cabcaa60f0f512aa3c1c14cdc`.
The live local server was measured on 2026-10-04. Its native timings reproduce
33.99 generation tokens/s on the Berlusconi question. Requests, responses,
process arguments, selected environment variables and tensor inventories are
archived in `gguf/benchmarks/llama-reproduced-baseline/`. These are short local
measurements, not a statistically stable throughput or quality benchmark.

The subsequent ds4 implementation pass is documented in
`QWEN38_ROCM_RDNA4.md` under "Cooperative attention, GDN and routing".
Its paired warm counting result is 6.50 to 14.37 generation tokens/s using
the original custom IQ3 weights and a 14 GiB budget. Other paired replies
reach 12.44–12.96 t/s. Results and regression checks are archived separately
in `gguf/benchmarks/hybrid-llama-port/`; the earlier ds4 table below is retained
as the pre-port comparison with the live llama server.

## Measured live reference

The running server has the supplied CPU_OD=8 placement, MTP with up to three
drafts, 16 CPU threads, Q8 KV, chunk 512 and batch 1536. Its context is 16384,
whereas the supplied command specifies 32768. Requests use temperature zero,
disable prompt-cache reuse, and run sequentially without restarting the server.

| Request | Prompt tokens processed | Generated tokens | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| Berlusconi, thinking enabled | 58 | 71 | 58.89 t/s | 33.99 t/s |
| Berlusconi, thinking disabled | 18 | 58 | 47.58 t/s | 31.16 t/s |
| Counting, limit 128 | 31 | 128 | 61.70 t/s | 46.31 t/s |
| Berlusconi with repeated context | 1273 | 93 | 264.80 t/s | 30.79 t/s |

The long context is synthetic and repetitive, unlike the user's 1432-token
request. It confirms efficient larger prefill batches but does not reproduce
the screenshot's exact 153.19 t/s workload. All four responses report zero
cached prompt tokens. Thinking-enabled generation includes reasoning tokens.
Counting accepts 95 of 95 drafted tokens; the first question accepts 48 of 72.

Three additional requests use the same user prompts, temperature zero,
thinking disabled and a 32-token output cap as the latest ds4 measurements:

| Prompt | llama.cpp generation | Latest ds4 generation |
| --- | ---: | ---: |
| Counting | 43.33 t/s | 6.59 t/s |
| Date only | 42.31 t/s | 5.94 t/s |
| Sky explanation | 25.24 t/s | 5.97 t/s |

Templates, weights and speculative depth still differ. Output lengths differ
when models stop early; the date-only llama response has just nine generated
tokens. These comparisons establish the practical gap, not its causal split.
ds4 figures come from `hybrid-quality-optimized/summary.json` and were measured
earlier, with no concurrent llama process.

The live server's sampled device usage after requests is 15.81 GiB out of
15.92 GiB, including other GPU users. This is a snapshot, not a sampled peak.
The latest ds4 paired chats peaked at 13.14 GiB. Keeping substantial VRAM
headroom is therefore a stricter constraint than the supplied configuration.
The server process occupies about 39 GiB RSS; no second model was launched.

## CPU experts

In `ggml/src/ggml-cpu/ggml-cpu.c:349`, IQ3_XXS dispatches to
`ggml_vec_dot_iq3_xxs_q8_K`; the activation conversion in `MUL_MAT_ID`
prepares Q8_K once before the matrix rows consume it. The x86 implementation
in `arch/x86/quants.c:3260` uses byte sign operations, integer multiply/add
and floating scales per block. ds4's optimized hybrid kernel still consumes
F32 activations and expands eight weight values to float per inner step.
These compute different numerical approximations, not just different SIMD
widths. Porting the Q8 path requires an activation-quantization accuracy
evaluation; exact greedy parity cannot be assumed.

The fork additionally contains `iqp.cpp`: decode panels of eight weight rows,
consume activation rows in groups of four, and use VNNI when compiled with
the relevant ISA. It supports IQ3_XXS as well as several other IQ formats.
The actual `MUL_MAT_ID` panel dispatch requires at least eight token/slot
occurrences for that individual expert (`ggml-cpu.c:1672`), not merely eight
tokens in the whole batch. Thus it is mainly a prefill opportunity and does
not explain the two-row MTP verifier speed by itself. IQ4_NL is absent from
the panel type list. ds4 already groups up to four activations per expert,
but does not panel multiple output rows or use integer activation products.

llama.cpp's ordinary CPU matmul distributes small chunks dynamically using
atomic counters. ds4 divides flattened rows statically. Unequal group sizes
can imbalance those slices. Its pthread workers sleep on condition variables
and are awakened twice per CPU expert layer. The current hybrid grouped path
now dynamically distributes 64-row slices. `OMP_WAIT_POLICY=active` from
the old command affects an OpenMP runtime; it does not tune ds4's pthread
pool. Scheduling, affinity and short active waits are experiments, not an
established source of a particular speedup.

## GPU experts and dense work

The current `rocm/ds4_rocm_qwen4_moe.cuh` distributes routed dot products
across 32 lanes, including the MTP Q4_K/MXFP4 layer. CPU IQ3 gate/up share
activation loads, and groups of up to four tokens share weight decoding.
Activations remain float. More residency does not replace those kernels with
llama.cpp's quantized-activation MMVQ/MMQ implementation.

The local llama build enables HIP graphs, HIP MMQ MFMA, native CPU ISA and
OpenMP in `build-hip/CMakeCache.txt`. Those build settings do not prove that
every node in the earlier run used graphs or matrix instructions. The
`GGML_JOHNV8_MMQ_ID_JMAX=32` patch in `mmq.cuh:1482` caps the tile-selection
column count for expert matmuls, avoiding wide mostly empty tiles. It mainly
targets prefill. A decode kernel should distribute a row across a wave and
reuse weights across verifier tokens rather than copy that tile limit alone.

`hc-block.cu` fuses norm/activation quantization, HC down/activation and HC
up/gating/mixing for batches up to four. The fork's own comment reports a
small MTP gain; this is not our hardware measurement. ds4 has some HC fusion,
but HC mix still calls separate normalization, projection and gate/mix
operations. Current ds4 fuses small Q8 multi-projection launches and folds
the HC residual copy into its combine kernel. HC normalization and projection
still require separate operations, and larger batches retain fallback kernels.

## Transfers and graph scheduling

ds4 now enqueues mixed activations and expert IDs together with two async
copies into pinned staging, waits once, computes CPU experts, then enqueues
the upload of all expert partial outputs before the GPU reducer.
For T=2, top-10 and width 2560 the partial upload is 200 KiB per CPU layer,
about 8.79 MiB across 45 CPU layers per verification. This is activation
traffic, not transfer of the expert weights. There are three synchronous
copy operations per CPU layer; their latency and remaining host dependency
still merit measurement. They are no longer three synchronous API calls.

llama.cpp's backend scheduler uses async transfers and events between splits.
Its CUDA/HIP backend can capture supported GPU graphs. ds4's ROCm
begin-commands hook is a no-op and kernels launch individually. Capture
should start with stable GPU segments, especially MTP and dense work;
CPU expert computation and host routing dependencies prevent treating the
whole hybrid forward as one ordinary device-only graph.

Pinned reusable staging and combined readback ordering are implemented,
preserving the existing GPU shared-expert overlap. Removing further waits
without dependencies would introduce races. Measure
CPU projections, GPU routed/shared work, transfers, output head and MTP
separately before attributing all aggregate MoE time to the CPU.

## Model and MTP differences

The archived previous model inventory is in
`gguf/benchmarks/hybrid-iq3-validation/previous-llama-model-inventory.json`.
The previous UD file has mixed routed formats and IQ4_NL n-grams; our IQ3
file uses IQ3_XXS primary experts and BF16 n-grams. The old command explicitly
places eight trunk expert layers on GPU. The current budget fits three plus
GPU MTP. The old command uses Q8 KV while ds4 uses F16 KV. These differences
prevent interpreting the reported 30 tokens/s as a controlled runtime-only
comparison. Quantizing the SSD table is a separate model-conversion project;
it is not a justified first target from the current warm short-prompt profile.

Fresh tensor-header inventories confirm that the UD main model uses IQ2_S
and IQ3_S routed weights plus IQ4_NL down projections; its name does not
mean uniform IQ3_XXS weights. It also uses Q6_K for many dense projections
and the output head. Our dense tensors include Q8, F16 and F32 instead.
Switching to the UD file or quantizing float activations changes numerical
results and needs a separate quality evaluation; it is not a byte-preserving
optimization of the current custom IQ3 model.

llama.cpp's `--spec-draft-n-max 3` means up to three draft tokens. ds4's
diagnostic depth 3 means three verifier rows: one committed input and two
drafts. These numbers have different meanings. ds4 defaults to adaptive
depth 2/3 and splits some depth-3 operations into 2+1 to preserve arithmetic.

Sequential current-binary experiments, same IQ3/14 GiB budget/16 threads,
greedy sampling and 32-token cap:

| Prompt | Depth 2 generation | Depth 3 generation |
| --- | ---: | ---: |
| Counting | 6.33 t/s | 6.18 t/s |
| Explanation | 5.28 t/s | 5.03 t/s |

Answers were identical in each pair; sampled total VRAM stayed below
12.85 GiB. These are single short comparisons, insufficient to select a
new default, and show no benefit from forcing more drafts here. Commands,
outputs and reproduction script are in `gguf/benchmarks/llama-comparison/`.
The existing acceptance percentage counts first-draft acceptance per cycle,
not acceptance of every drafted token at deeper settings.

## Implementation priorities

1. Repeat granular profiling on the current cooperative/pinned/fused code;
   the older synchronized trunk profile below predates these changes. Separate
   GPU execution, CPU projections, transfers, output head and MTP predictor.
2. Prototype IQ3 x Q8_K on CPU, with reusable activation quantization and
   zero padding from logical width 640 to physical width 768. Compare against
   a float reference and evaluate logit drift and model quality, not just
   argmax on a tiny sample. Decide using end-to-end speed and accuracy.
3. Measure remaining pool wakeup overhead and host/device synchronization;
   keep the explicit buffer reuse dependencies of pinned staging.
4. Extend HC/dense fusion and capture supported GPU segments only after
   graph-owned stable allocation and pointer invalidation are enforced. Add CPU
   panel matmul and occupancy-aware GPU tiling for larger prefills.

The current synchronized trunk profile has mean MoE 155.69 ms and other
groups 134.63 ms per two-row verification, excluding output head and MTP
predictor. Even eliminating the entire measured MoE would give only about
2.16x for those trunk groups. Reaching 30 tokens/s would require improvements
across several stages and a controlled comparison with the old model.
