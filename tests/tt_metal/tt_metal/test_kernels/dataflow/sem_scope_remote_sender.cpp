// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Remote-up SENDER (off the semaphore's node): every thread bumps sem::counter on the
// semaphore's node, increment_times times, through the class's remote up(). Kept in its own
// file (not an #ifdef role) so the census access scan sees exactly one role: the scan proves
// this kernel's ONLY semaphore op is the remote up(), which -- for a sole 1-instance sender --
// lets the host bake REMOTE_POSTED (private count + plain value-writes) instead of EXTERNAL.
// The scope is host-picked (invisible table), so the same source also runs under EXTERNAL
// (e.g. the multi-thread contrast test, whose instance count keeps it atomic).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);

    Semaphore counter(sem::counter);  // mechanism comes from the host's scope table
    Noc noc;
    for (uint32_t i = 0; i < increment_times; i++) {
        counter.up(noc, remote_noc_x, remote_noc_y, 1);
    }
    // Drain this hart's in-flight NoC ops before exit: atomics for an EXTERNAL run, plain
    // writes for a REMOTE_POSTED run (remote up() never fences internally).
    noc.async_atomic_barrier();
    noc.async_write_barrier();
}
