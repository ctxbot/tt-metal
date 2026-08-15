// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RE-CONSTRUCTION control for the posted fast path: same sole off-node remote-up shape as
// sem_scope_remote_sender.cpp, but the Semaphore is constructed INSIDE the loop -- a fresh
// object every iteration. The posted running count is kernel-image state
// (tt_sem_posted_count_, wrapper-zeroed once per launch), NOT object state, so the census
// bakes REMOTE_POSTED for this kernel too and every increment must still land exactly.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t remote_noc_x = get_arg(args::remote_noc_x);
    const uint32_t remote_noc_y = get_arg(args::remote_noc_y);

    Noc noc;
    for (uint32_t i = 0; i < increment_times; i++) {
        Semaphore counter(sem::counter);  // fresh object every iteration: the count must survive
        counter.up(noc, remote_noc_x, remote_noc_y, 1);
    }
    noc.async_atomic_barrier();

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = static_cast<uint32_t>(sem_scope_of(sem::counter));
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Make the report visible to the host readback of TL1.
    flush_l2_cache_line(report_addr);
#endif
}
