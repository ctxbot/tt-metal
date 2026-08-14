# Blackhole Concurrent Sub-Device Profiler Protocol

## Status and scope

This is the Milestone 0 decision record for
`REALTIME_PROFILER_CONCURRENT_SUBDEVICE_REDESIGN_PLAN.md`.

The protocol applies only to the Blackhole, single-command-queue,
worker-dispatch path. It does not define model integration, operation core
allocation, Wormhole behavior, Quasar behavior, multi-CQ behavior, trace replay,
or remote-chip transport.

## Current-path evidence

The current branch contains the concurrent-profiler changes from
`20e839e1ff9032573690c031751ad58b5e967fed`. Inspection identifies two waits
that make that implementation unsuitable for asynchronous profiling:

1. `finish_realtime_profiled_program()` in
   `cq_dispatch_subordinate.cpp` waits until TRISC0's sampled stream count is
   exactly equal to the command target. This is profiler-only backpressure on
   dispatch and exact equality can be missed if the sampled counter advances
   past the target.
2. `drain_realtime_profiler_records()` waits for the reserved profiler BRISC to
   acknowledge every record. It is called on local-queue pressure, profiler
   flush, and termination. This couples dispatch progress to BRISC, NCRISC, D2H
   socket, and host progress.

There are two more downstream waits. They do not currently execute directly on
the application dispatch RISC, but they must not be allowed to propagate
backpressure into it:

- `realtime_profiler_read_and_enqueue()` waits while the reserved-core L1 ring
  is full.
- `push_entries_to_host()` waits in the D2H socket reserve path when host FIFO
  capacity is unavailable.

The authoritative Blackhole completion signal is the dispatch stream's
`STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX` value. The dispatch_s TRISC0
kernel already samples this register on the same physical dispatch core as the
dispatch_s NCRISC and captures the device wall clock on a change. The current
implementation writes only the latest timestamp and count per stream, which can
be overwritten before dispatch_s consumes them.

The current go-signal command does **not** carry enough information to calculate
the completion target for a partitioned sub-device. Its compile-time
`num_worker_cores_to_mcast` value is the whole physical compute grid, while only
the selected sub-device's workers increment that sub-device stream.

The host already calculates the exact per-launch completion contribution as
`num_workers` in `FDMeshCommandQueue::enqueue_mesh_workload()`. Milestone 1 will
carry that value as an eight-bit `profiler_num_workers` field in
`CQDispatchGoSignalMcastCmd`. Blackhole's maximum Tensix plus virtual ETH worker
count fits in eight bits and the host must assert that before encoding it.

The command remains 16 bytes by narrowing `wait_stream` from 16 bits to 8 bits;
the dispatch stream index is bounded well below 256 by the hardware stream
count. This avoids doubling every go-signal command to 32 bytes. The field is
patched together with `profiler_program_id` for ordinary and traced command
sequences, but the new concurrent profiler does not claim trace-replay support.

The target for a profiled launch is therefore:

```text
target = wait_count
       + profiler_num_workers
```

using the stream counter's existing wrap semantics.

The device stream counter is a `MEM_WORD_ADDR_WIDTH == 17` field. Descriptor and
watermark targets are masked to this 17-bit domain, and TRISC0 uses the existing
half-range modular greater-than-or-equal comparison. A natural transition from
`0x1ffff` to `0` is forward progress; it does not reset profiler state or discard
any descriptor or watermark. Each per-launch advance is at most 255, far below
the half-range ambiguity boundary.

The virtualized-ETH workaround currently injects its synthetic completion delta
into `first_stream_used`. Concurrent profiling is enabled for such a launch only
when `wait_stream == first_stream_used`; otherwise dispatch_s counts the launch
as unsupported profiler loss and does not publish a descriptor. This guard
prevents a profiler descriptor from waiting on a target that the selected stream
cannot reach. It does not change the dispatch workaround itself.

TRISC0 and dispatch_s NCRISC share the dispatch core's L1. This permits a local,
bounded producer/consumer protocol without adding NOC traffic to the program
start or completion critical paths. The reserved profiler BRISC remains the
only producer of its existing BRISC-to-NCRISC L1 ring.

## Baseline evidence

Hardware used for Milestone 0 is the local four-chip Blackhole P150b QuietBox
(`bh-qb-12-special-pjosipovic-for-reservation-62599`), firmware bundle
19.10.0, KMD 2.10.0, IOMMU enabled, and 800 MHz AI clock.

After rebuilding `unit_tests_dispatch` to eliminate a stale test/library ABI,
the current branch passes all five existing Blackhole sanity tests:

```text
RealtimeProfilerSanity.FiveProgramsBackToBack
RealtimeProfilerSanity.CloseDrainsRegisteredCallback
RealtimeProfilerSanity.ThrowingCallbackIsIsolated
RealtimeProfilerSanity.LastProgramRecordDeliveredOnFinish
RealtimeProfilerSanity.TraceReplayResolvesKernelSources
```

A one-second `RealtimeProfilerStress.PeakLoadPreservesRecords` attempt did not
reach its replay summary after more than eight minutes. A debugger snapshot
showed the host blocked in `FDMeshCommandQueue::finish_nolock()` waiting for the
device completion event. The run was stopped locally. It is retained as failure
evidence for the current blocking path, not treated as a throughput baseline.

Current static footprint from the generated Blackhole kernels is:

| Component | Current maximum measured image payload |
| --- | ---: |
| dispatch_s NCRISC | 7,848 B text, 80 B data, 452 B BSS |
| dispatch_s TRISC0 monitor | 1,232 B text |
| reserved profiler BRISC | 2,704 B text, 24 B data |
| reserved profiler NCRISC | 3,336 B text, 96 B data, 40 B BSS |
| dispatch_s profiler message | 4,416 B L1 |
| reserved profiler-core ring and socket config | 262,336 B L1 |

The values above are regression anchors, not architecture limits. Milestone 1
must report the same measurements after implementation.

## Selected protocol

### Ownership

| State | Producer | Consumer |
| --- | --- | --- |
| Per-stream start descriptor ring | dispatch_s NCRISC | dispatch_s TRISC0 |
| Completed interval ring | dispatch_s TRISC0 | dispatch_s NCRISC |
| A/B transport mailbox | dispatch_s NCRISC | reserved profiler BRISC |
| Reserved-core L1 ring | reserved profiler BRISC | reserved profiler NCRISC |
| D2H pages | reserved profiler NCRISC | host receiver |

Every mutable queue has exactly one producer and one consumer. No atomic
read-modify-write operation or lock is required.

### Per-stream start descriptors

Each monitored stream receives a power-of-two ring of four descriptors. A
descriptor contains:

- runtime ID;
- start timestamp high and low words;
- stream completion target;
- descriptor kind and schema version.

dispatch_s captures the start tick and publishes the descriptor immediately
before sending the go signal. Publishing consists of ordinary L1 stores, a
Blackhole `fence w,w`, and a final producer-index store.

If the descriptor ring is full, dispatch_s increments
`start_descriptor_drop_count` and launches the program without profiling it.
It never waits for TRISC0.

Each descriptor also carries the stream reset generation current at publication.

### Completion correlation

TRISC0 continuously samples all configured stream counters. For each stream it
examines the descriptor at the consumer index and applies the dispatch stream's
wrap-aware greater-than-or-equal comparison to the descriptor target.

When one descriptor target is satisfied, TRISC0 captures the end tick and emits
one interval. If more than one descriptor is already satisfied in the same
sample, only the newest satisfied descriptor can receive an accurate end tick;
older satisfied descriptors are discarded and counted in
`completion_observer_drop_count`. They must not be emitted with an invented or
duplicated completion timestamp.

TRISC0 advances the descriptor consumer index whether the interval is emitted
or counted as lost. It never waits for downstream capacity.

### Completed interval ring

TRISC0 publishes completed intervals into the existing 128-entry dispatch-core
record ring. Each entry remains eight words so the existing A/B and D2H record
format can be retained:

```text
word 0  start timestamp high
word 1  start timestamp low
word 2  runtime ID
word 3  schema/type/stream metadata
word 4  end timestamp high
word 5  end timestamp low
word 6  successful interval sequence
word 7  cumulative source-drop snapshot
```

The successful interval sequence advances only when a complete record is
accepted into this ring. If the ring is full, TRISC0 increments
`completed_record_drop_count`, consumes the descriptor, and continues.

TRISC0 stores the record words, executes `fence w,w`, and publishes the producer
index last. dispatch_s uses the two-step Blackhole cache invalidation described
below before reading the producer index and record payload.

### Bounded dispatch forwarding

dispatch_s replaces `drain_realtime_profiler_records()` with
`service_realtime_profiler_once()`:

1. If the A/B mailbox is not idle, return immediately.
2. If the completed interval ring is empty, try to forward one ready watermark;
   otherwise return.
3. Copy one completed record into the next A/B mailbox.
4. Signal the reserved profiler BRISC with the existing inline NOC dword write.
5. Advance the completed-ring consumer index and return.

There is no acknowledgement loop. At most one record is forwarded per call.
The service function is invoked at bounded progress points already visited by
dispatch_s:

- once per command-loop iteration;
- while waiting for command-buffer input;
- while waiting for dispatch_d permission or worker completion;
- after processing a profiler flush request.

This keeps the common action to a few local loads and branches. A NOC signal is
issued only when a record is ready and the existing A/B mailbox is idle.

Blackhole has no distinct uncached L1 alias: `uncached_l1_ptr()` is an identity
operation on this architecture. Both cross-RISC consumers therefore use an
explicit cache-invalidation sequence. Before reading a producer index, the
consumer calls `invalidate_l1_cache()`. After observing a non-empty queue and
before reading the reusable payload slot, it calls `invalidate_l1_cache()`
again. A fresh producer index must never be paired with a payload line retained
from an earlier use of that ring slot.

This rule applies to TRISC0 consuming NCRISC-published start descriptors and to
dispatch_s NCRISC consuming TRISC0-published completed records. It is required
regardless of the consumer's initial cache state; the implementation must not
depend on TRISC0 inheriting a disabled or empty data cache.

### Reserved profiler core

The BRISC retains the existing A/B NOC-read protocol and remains the sole
producer of the reserved-core L1 ring. Its full-ring behavior changes:

- interval record: increment `transport_drop_count`, acknowledge the A/B
  mailbox, and continue;
- watermark record: retain or replace a single local pending-watermark slot,
  acknowledge the A/B mailbox, and enqueue the newest pending watermark after
  ring capacity returns;
- clock-sync marker: retain the existing explicit sync behavior; sync does not
  execute on the application dispatch path.

The BRISC must not wait for reserved-ring capacity while handling an interval
from dispatch_s. The NCRISC and D2H socket can stall independently without
stalling program dispatch; pressure becomes counted interval loss.

## Watermark protocol

### Device request

One `finish_nolock()` call allocates a monotonically increasing 32-bit batch
watermark ID and registers the participating sub-device stream mask with the
profiler manager. Every `CQ_DISPATCH_CMD_RT_PROFILER_FLUSH` emitted by that
Finish carries the same batch ID. A normal Finish enqueues these requests but
does not wait for profiler delivery.

After its existing worker wait completes, dispatch_s publishes a per-stream
watermark request containing:

- watermark ID;
- stream completion target from the flush command;
- a request generation.

This is a dedicated per-stream slot, not part of the start descriptor ring, so
a full descriptor ring cannot lose the request.

### Device completion

TRISC0 marks a per-stream watermark ready only after:

- the stream counter reached the requested target; and
- every start descriptor on that stream whose target is at or before the
  requested target has been emitted or counted as dropped.

The ready state snapshots:

- successful interval sequence;
- start-descriptor drops;
- completion-observer drops;
- completed-record drops;
- completed-record producer index that must be forwarded first.

dispatch_s forwards a watermark only after its completed-record consumer index
has reached the snapshotted producer index. Thus the watermark cannot overtake a
successfully accepted interval on the dispatch core.

The watermark uses the existing eight-word transport payload with a reserved
record type. The reserved profiler BRISC adds its cumulative transport-drop
snapshot before enqueueing the control page in the local ring.

### Host completion

The host receiver consumes watermark pages internally and does not publish them
as program callbacks. Per device and batch ID, it stores the set of participating
streams observed plus their cumulative counter snapshots, then notifies
collection waiters.

The host collection result contains:

- requested watermark;
- observed participating-stream mask per active device;
- records received since the caller's baseline snapshot;
- source and transport drop deltas;
- host callback-ring drop information remains callback-specific and is not
  conflated with device loss;
- timeout and protocol-error state.

A batch is complete only when every registered participating stream is observed
for every active device. A later batch does not satisfy an earlier batch, because
streams can become ready out of order. Ring emptiness is never used as proof of
completion.

## Lifecycle

Host initialization zeros all queue indices, drop counters, sequences, stream
reset generations, watermark generations, and A/B state before launching the
participating kernels. TRISC0 waits for the existing profiler enable word before
touching the protocol. dispatch_s treats a zero profiler-core NOC coordinate as
disabled and does not publish descriptors.

On the supported Blackhole worker-dispatch route, dispatch_d owns every explicit
`CLEAR_STREAM`; `process_dispatch_s_wait_cmd()` does not execute. dispatch_d and
dispatch_s are co-located on the same worker tile and already receive the same
`REALTIME_PROFILER_MSG_ADDR`. Immediately after the required worker wait and
before clearing a stream counter, dispatch_d:

1. clears the hardware stream counter through the existing stream update;
2. executes a RISC I/O-to-memory fence so the clear is ordered before shared-L1
   publication;
3. increments `stream_reset_generation[stream - first_stream_used]` in the
   shared profiler L1 block;
4. executes `fence w,w` and performs no profiler wait.

This covers sub-device-manager loads, event/reset paths, and host 32-bit
completion-count wrap because all of them ultimately execute dispatch_d's
`process_wait(... CLEAR_STREAM ...)` path. Natural 17-bit counter rollover does
not execute that path and therefore does not change the generation.

dispatch_s and TRISC0 each maintain a local adopted generation per stream.
Before publishing a descriptor, dispatch_s reads the shared generation. On a
change it counts unread old-generation descriptors and an old watermark request
as reset loss, resets its producer state, clears watermark generations, and
adopts the new generation before publishing new work. TRISC0 checks the same
generation before its counter and descriptor scan. On a change it consumes and
counts old-generation descriptors, clears old ready-watermark state, adopts the
new generation, and samples the newly reset stream before consuming new
descriptors. Every descriptor carries and must match the adopted generation.

TRISC0 calls `invalidate_l1_cache()` before reading the shared generation and
producer index, and again before reading descriptor payload words. dispatch_s
performs the same two-step invalidation when it consumes TRISC0 records. Host
initialization starts both sides at generation zero.

Termination occurs only after application work is quiesced. It is the one
explicit exception to steady-state nonblocking forwarding: dispatch_s executes
a fixed-budget terminal handoff loop capped by the dispatch completed-ring
capacity plus the number of participating stream watermarks. Each iteration may
forward at most one item and may wait only for the existing A/B acknowledgement;
the loop also has a device-cycle deadline. It never waits for the host or D2H
socket directly. If the item/count or cycle budget expires, remaining entries
are counted as terminal loss and termination proceeds. The reserved profiler
BRISC drains accepted A/B items into its ring before setting its terminate flag;
the NCRISC drains that ring before exit. Milestone 2 moves the final host
collection wait before CQ teardown so a healthy close observes the final batch;
the bounded device drain remains the failure-safe path.

A timeout or terminal loss is reported as an incomplete/lossy collection; it
must not trigger a D2H tensor fallback or host-duration substitution.

Sequence and unsigned producer/consumer differences use natural 32-bit
wraparound. Ring capacity is far below 2^31, so the standard unsigned-distance
full/empty tests remain unambiguous.

## Ordering proof obligations

Milestone 1 must preserve these edges:

1. dispatch_s descriptor words happen before descriptor producer-index publish;
2. TRISC0 invalidates its Blackhole L1 cache before reading the producer index
   and invalidates again before reading descriptor words;
3. TRISC0 interval words happen before completed-ring producer-index publish;
4. dispatch_s invalidates its Blackhole L1 cache before reading the
   completed-ring producer index and invalidates again before reading payload
   words;
5. A/B words are visible before the inline NOC state notification;
6. BRISC completes the NOC read before acknowledging A/B idle;
7. BRISC ring-slot data is visible before its producer-index publish;
8. NCRISC completes D2H writes before advancing the local consumer index;
9. a device watermark is enqueued after every accepted interval through its
   snapshotted producer index.

The Blackhole implementation will use explicit RISC `fence w,w` instructions at
local-L1 publication points and existing NOC read/write barriers at NOC
handoffs. Any weaker sequence requires device evidence and a new review.

## Resource and performance gates

Milestone 1 limits are:

- dispatch_s profiler-message L1: at most 8 KiB total;
- reserved profiler-core L1: no increase over 262,336 B;
- dispatch_s NCRISC text: at most +1,536 B;
- dispatch_s TRISC0 text: at most +2,048 B;
- profiler BRISC text: at most +1,536 B;
- profiler NCRISC text: no intentional increase in Milestone 1;
- no added NOC transaction when profiling is disabled;
- no loop that waits for profiler consumer progress on dispatch_s or TRISC0
  during steady-state execution; the fixed item/cycle-budget terminal handoff
  after application quiescence is the only exception;
- disabled dispatch-throughput regression at most 0.5% outside paired-run
  noise;
- enabled blank-program dispatch-throughput regression at most 2.0% outside
  paired-run noise.

The throughput thresholds are qualification gates. They do not permit a device
timing error, silent loss, or application-dispatch wait in exchange for speed.

## Rejected alternatives

### Keep the current dispatch_s drain and call it only at Finish

Rejected because ordinary Finish would remain proportional to record count and
dependent on BRISC/NCRISC/host progress. It also does not remove the exact TRISC
completion-count wait.

### Have the reserved profiler BRISC continuously poll dispatch_s L1

Rejected because it adds continuous NOC reads while the profiler is always on.
Bounded opportunistic forwarding reuses dispatch_s progress points and sends NOC
traffic only for actual records.

### Have dispatch_s capture completion after `wait_for_workers()`

Rejected because no subsequent command may arrive near the real completion
time. A flush could therefore produce an end timestamp delayed by arbitrary host
idle time. TRISC0 is the existing device observer at the completion source.

### Let TRISC0 overwrite one timestamp slot per stream

Rejected because dispatch_s can start a later program after the previous worker
completion but before TRISC0's slot is consumed. Per-stream descriptor rings and
a completed-record ring make this race explicit and bounded.

### Emit one interval for every descriptor already satisfied in a TRISC0 poll

Rejected because one sampled wall-clock value cannot prove distinct completion
times for multiple programs. Ambiguous intervals are counted as lost.

## Milestone 1 implementation surface

Expected files are limited to:

- `tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h`;
- `tt_metal/impl/dispatch/kernel_config/dispatch_s.cpp`;
- `tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp`;
- `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` for the nonblocking explicit
  `CLEAR_STREAM` generation publication;
- `tt_metal/impl/dispatch/kernels/cq_realtime_profiler_dispatch_subordinate.hpp`;
- `tt_metal/impl/dispatch/kernels/cq_realtime_profiler.cpp`;
- `tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp`;
- `tt_metal/impl/dispatch/kernels/cq_commands.hpp`;
- `tt_metal/impl/dispatch/device_command.hpp`;
- `tt_metal/impl/dispatch/device_command.cpp`;
- `tt_metal/impl/program/dispatch.hpp`;
- `tt_metal/impl/program/dispatch.cpp`;
- `tt_metal/distributed/fd_mesh_command_queue.cpp` for exact worker-count and
  shared batch-watermark plumbing;
- focused profiler tests and this documentation.

Host collection and public result changes belong to Milestone 2. If Milestone 1
requires edits outside this list, the reason must be documented before the diff
is reviewed.
