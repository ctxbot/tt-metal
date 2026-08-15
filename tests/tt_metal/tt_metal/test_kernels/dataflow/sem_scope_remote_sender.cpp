// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Remote-up SENDER (off the semaphore's node): every thread bumps sem::counter on the
// semaphore's node, increment_times times, through the class's remote up(). Kept in its own
// file (not an #ifdef role) so the usage check sees exactly one role: the compiler proves
// this kernel's ONLY semaphore op is the remote up(), which -- for a sole 1-instance sender --
// lets the host bake REMOTE_POSTED (private count + plain value-writes) instead of EXTERNAL.
// The scope is host-picked (invisible table), so the same source also runs under EXTERNAL
// (e.g. the multi-thread contrast test, whose instance count keeps it atomic).
// Reports its OWN baked scope so the tests pin the sender-side bake, not just the count.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);

    Semaphore counter(sem::counter);  // mechanism comes from the host's scope table
    Noc noc;
    for (uint32_t i = 0; i < increment_times; i++) {
        counter.up(noc, remote_noc_x, remote_noc_y, 1);
    }
    // Drain in-flight NoC atomics before exit (the EXTERNAL remote up() does not fence).
    // Deliberately NO write barrier in USER code: a REMOTE_POSTED up() flushes its staged
    // write to departure itself, and the generated kernel_main exit stub (genfiles.cpp)
    // drains the final ack -- that split being sufficient is part of what this kernel proves.
    noc.async_atomic_barrier();

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = static_cast<uint32_t>(sem_scope_of(sem::counter));
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Make the report visible to the host readback of TL1.
    flush_l2_cache_line(report_addr);
#endif
}
