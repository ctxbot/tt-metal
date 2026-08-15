// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CHARACTERIZATION probe: can a Quasar DM kernel deliver 32-bit values into a REMOTE core's L1
// with noc_inline_dw_write? Context: the NON-posted variant WEDGES on this RTL (watcher-verified
// 2026-08-14: hart parked at NWIW inside the call) and is deliberately NOT exercised here. This
// probe characterizes the posted variant -- the gap-8 candidate, since posted writes involve no
// response path at all:
//   V1 (word 0): one posted inline write of a recognizable value.
//   V2 (word 1): 64 ascending posted inline writes -- the LAST value must win (same-source
//                same-address ordering, the property gap-8's running-value protocol needs).
//   marker (word 4): a PROVEN non-posted 4B noc_async_write on the same src->dst route, then a
//                write barrier. Same-route ordering means the marker's ack implies every earlier
//                posted write already landed -- so the host can distinguish "posted writes were
//                silently dropped" from "still in flight".
// Word 2, 3, 5..7 are canaries: they must still hold the host's prefill (no byte-enable spill).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t base = get_arg(args::base_addr);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);

    // V1: single posted inline write.
    const uint64_t w0 = get_noc_addr(remote_noc_x, remote_noc_y, base);
    noc_inline_dw_write<InlineWriteDst::DEFAULT, /*posted=*/true>(w0, 0xC0DE0001u);

    // V2: ascending posted writes to one word; the last (64) must win.
    const uint64_t w1 = get_noc_addr(remote_noc_x, remote_noc_y, base + 4);
    for (uint32_t v = 1; v <= 64; v++) {
        noc_inline_dw_write<InlineWriteDst::DEFAULT, /*posted=*/true>(w1, v);
    }

    // Marker: stage 0x4ACCED11 in OUR local scratch (same offset on this node), then a proven
    // non-posted async write of it into the remote word 4, then barrier it.
    volatile tt_l1_ptr uint32_t* stage =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(base) + MEM_L1_UNCACHED_BASE);
    stage[0] = 0x4ACCED11u;
    const uint64_t w4 = get_noc_addr(remote_noc_x, remote_noc_y, base + 16);
    noc_async_write(base + MEM_L1_UNCACHED_BASE, w4, sizeof(uint32_t));
    noc_async_write_barrier();
}
