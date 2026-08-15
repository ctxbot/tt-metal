// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NEGATIVE control for the gap-8 construction-once gate: same sole off-node remote-up shape as
// sem_scope_remote_sender.cpp, but the Semaphore is constructed INSIDE the loop. The access
// scan still classifies REMOTE_UP_ONLY (one construction site, remote ups only), yet the census
// must NOT bake REMOTE_POSTED: the posted running count lives in the object, so a per-iteration
// re-construction would restart it at zero and strand the receiver. The construction-once gate
// (ProvesSingleTopLevelConstruction) rejects this shape and the semaphore stays EXTERNAL --
// under which per-iteration construction is harmless (the object holds no protocol state).

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
        Semaphore counter(sem::counter);  // re-constructed every iteration: POSTED must not bake
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
