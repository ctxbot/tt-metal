# GLM routed-expert cache hydration experiment (8-P150 Loudbox)

## Objective

Determine whether storing each GLM MoE layer's local routed-expert projections as a
small number of packed cache entries reduces warm-cache model construction time
without changing routed-expert output or the layout required by the Blackhole
kernel.

The production symptom is the GLM-5.2 8x4 / 78-layer chunked-prefill test: model
construction takes about 15 minutes of the roughly 24-minute pytest runtime.  The
current 8x4 arrangement loads 8 local experts per device, with gate, up, and down
projections as separate cached tensors.  Across 75 MoE layers that is 1,800 tensor
loads.

## Scope and assumptions

* Hardware: one Loudbox with eight Blackhole P150s, opened as a 2x4 mesh.
* Model: GLM-5.2 FP8 source weights (`zai-org/GLM-5.2-FP8`) and the corresponding
  2x4 TTNN cache.  Use `GLM52_HF_MODEL` and `TT_GLM52_PREFILL_TTNN_CACHE` to point
  at writable experiment directories; do not modify shared CI caches.
* Primary experiment: one sparse/MoE layer on a warm cache.  The per-device expert
  count is 32 on 2x4, making load-call reduction at least as visible as on the
  production 8x4 topology.  In the absence of the staged GLM model/cache, run a
  synthetic 2x4 probe with the same GLM projection shapes to establish the
  per-entry hydration cost; do not treat it as a model-construction result.
* Non-goal: replacing the full 8x4 chunked-prefill performance gate.  Galaxy is
  still required to validate the production mesh, fabric, and end-to-end timing.

## Baseline

1. Use a clean process and the existing 2x4 cache to construct the selected GLM
   MoE block three times.  Synchronize after construction and record:
   * wall-clock construction time;
   * number of routed-expert cache entries opened/loaded;
   * allocated DRAM bytes; and
   * one routed-expert PCC result.
2. Keep the inputs, mesh shape, dtype, selected layer, and cache filesystem fixed.
   Report median and range, rather than a single run.
3. Retain the existing Loudbox GLM MoE PCC proxy as an operation regression check:

   ```sh
   scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe.py::test_ds_moe \
     -k 'mesh-2x4 and pcc-device-glm-256 and not pad50 and not fabric2d-' -xvs
   ```

   This uses the stated 2x4 mesh and checks the routed-expert operation with
   generated test weights.  It does not read the real GLM cache, so a dedicated
   cache-hit test is still required.

### Loudbox evidence (2026-08-15)

The 2x4 `pcc-device-glm-256` proxy passed on the available 8-P150 Loudbox.
Its initial synthetic-cache run took **364.97 s** and wrote **101 tensorbins**:
**96** routed-expert tensors (`32 local experts * 3 projections`) plus five
gate/shared tensors.  The cache occupied **5.10 GiB**.  A subsequent completed-
cache run also passed in **171.54 s**.

This establishes the entry-count baseline and validates cache-only TTNN
construction at the target geometry.  It is not a pure hydration benchmark:
both PCC runs construct host reference weights, while the first additionally
converts and writes the tensorbins.  A packed cache will retain approximately
the same payload bytes (about 5.1 GiB for this proxy), but use three routed-
expert tensorbins of roughly 1.69 GiB each rather than 96 ~54 MiB files.

## Implementation stages

### 1. Make the experiment observable

Add a focused 2x4 Blackhole test/benchmark that constructs `TtRoutedExpert` from
the cache only and records structured `routed_expert_hydration_ms`, peak host RSS,
and DRAM allocation metrics.  It should accept a layer index and run in a fresh
process so allocations from a prior case cannot mask loads.  Time both an
OS-page-cache-cold run and three page-cache-warm fresh processes; report the latter
median and range separately.

The test must assert the cache is complete before timing and must fail clearly when
the 2x4 cache or raw weights are unavailable.  Instrument the actual `as_tensor`
flatbuffer-load calls (not `FastCacheChecker`, which only counts existence checks)
to record physical cache loads, bytes, and timing.  Add host-only tests for the
format metadata and source-to-packed ordering, plus a device test that verifies
the base address and expert offsets used by the kernel.

### 2. Introduce an opt-in packed cache format

Use the existing GLM `.tensorbin` cache convention, not the
`FusionGroupSpec` / `TensorCache` machinery from `deepseek_v3_b1`.  That machinery
builds L1-backed `OverlappedTensor` byte views, whereas this path requires
ordinary DRAM `ttnn.Tensor` inputs to the routed-expert kernel.  Create a versioned
packed cache suffix/schema; never reinterpret an existing per-expert cache entry.

Initially pack the three projection families independently:

* all local gate projections;
* all local up projections; and
* all local down projections.

This gives three physical loads per MoE layer instead of `3 * experts_per_chip`.
Do not fuse gate and up in the first experiment: their logical fusion is a separate
kernel-layout decision.

Each packed tensor must be local-expert-major with a fixed, tile-aligned per-expert
stride.  GLM's expert shapes are uniform, so shape, offset, and size are derived
from the versioned schema; persist and validate the schema version, mesh shape,
expert count, logical shape, tile/layout, dtype, and stride.  The cache writer must
build this format from real FP8 source weights, preserving the current
`ExpertMapping.gather_weights_for_mesh_distribution` ordering.  The reader must
detect a missing or incompatible complete set before *any* packed allocation/load
begins.

### 3. Add packed-base kernel support before replacing the fast path

The present `unified_routed_expert_moe` API takes `std::vector<ttnn::Tensor>` and
indexes one logical tensor per expert.  There is no established ordinary DRAM
tensor-view path here: the existing overlap abstraction is an `OverlappedTensor`
plus explicit byte offsets for specialised kernels.  Therefore make the packed
base plus stride/offset interface the planned implementation, not a fallback.

Add a second, opt-in composite/kernel path accepting three packed DRAM tensors and
the fixed expert stride (or validated offsets).  In the per-expert loop, derive
the per-expert DRAM base address for each projection and pass it to the reader/
matmul setup.  Preserve the current list-based API and implementation unchanged
until the packed path passes output, PCC, and allocation checks.

Do not ship a packed cache that loads a packed tensor and then materializes 32
independent device tensors; that would retain the allocation cost being targeted.

### 4. Validate on Loudbox

For a sparse GLM layer on 2x4:

1. Build the new packed cache once into an isolated cache directory.
2. In fresh processes, compare old and packed warm-cache hydration over at least
   three repetitions.
3. Compare logical packed slices with legacy tensorbins before the kernel test,
   then compare routed-expert outputs against the existing reference/PCC path.
4. Run the existing 2x4 `pcc-device-glm-256` proxy.
5. Check that packed and baseline paths use equivalent DRAM capacity, and that the
   packed path does not introduce a forward-pass regression beyond measurement
   noise.

Success criteria for proceeding to Galaxy:

* identical logical weights after validating each packed expert slice against
  its legacy tensorbin;
* existing PCC threshold passes;
* exactly three physical packed cache loads per MoE layer for the initial format
  (with the count captured at the flatbuffer loader, rather than inferred from
  cache files); and
* a repeatable, material reduction in cache-hit hydration time.

If the time reduction is insignificant, capture filesystem, flatbuffer-deserialization,
and device-transfer breakdowns before investing in a kernel interface change.

## Galaxy confirmation

After Loudbox passes, generate the corresponding 8x4 packed cache in an isolated
location and run:

1. a one-layer GLM-5.2 8x4 smoke/PCC case; then
2. the existing L78 warm-cache chunked-prefill test from the failing job.

Compare `tt_transformer_creation`, forward time, total test time, PCC, and peak
DRAM with the job baseline.  The expected direct benefit is lower transformer
creation time; total bytes hydrated to DRAM may remain similar, so the outcome must
be decided by measurement rather than cache-entry count alone.

## Implementation result (2026-08-15)

Implemented the opt-in `packed_v1` format and validated it on the available 2x4
Loudbox fixture.  The cache now stores three expert-major tensorbins plus a JSON
schema sidecar, validates format/mesh/expert count/dtype/layout/logical shapes and
tile strides before loading, and falls back to the legacy per-expert format for a
missing or incompatible packed set.  Actual `ttnn.as_tensor` cache loads emit
structured `routed_expert_hydration` metrics.

The fused routed-expert composite accepts the three packed DRAM tensors without
creating per-expert tensor views.  Its reader and split-up writer apply an
expert-specific *tile page* offset.  This is required because TTNN DRAM pages are
bank-interleaved; a raw byte-address offset selected the wrong page sequence.

The focused 2x4 test passed host schema/order checks and device output PCC = 1.0.
Across the three warm-cache runs performed while validating the implementation,
the 8-local-expert synthetic fixture loaded the packed path in **5.7, 7.2, and
8.8 ms** (median **7.2 ms**) versus legacy warm loads of **14.9, 17.5, and
20.0 ms** (median **17.5 ms**).  This is a **2.4x** median reduction while
hydrating equivalent weight payloads.  The final run logged exactly three packed
physical cache loads (one each for gate, up, and down).

The `GLM52_HF_MODEL` and `TT_GLM52_PREFILL_TTNN_CACHE` locations were not staged
on this Loudbox, so the real GLM/Galaxy confirmation remains a follow-up.  The
implementation uses isolated cache directories and is intentionally opt-in via
`TT_ROUTED_EXPERT_CACHE_FORMAT=packed_v1` for that validation.

## Rollout and rollback

Keep the current per-expert cache reader as a fallback.  Update
`TtRoutedExpert.check_cache_complete` and the transformer-level cache-completeness
path to recognise an all-or-nothing packed set, then enable packed loading only
when the set validates against cache format version, mesh shape, expert count,
shape, stride, dtype, and layout.  A missing or invalid packed cache must fall
back to the proven per-expert reader rather than silently rebuilding or
overwriting a shared cache during a test run.
