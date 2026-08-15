// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Remote-up RECEIVER (on the semaphore's node): waits for the exact expected total, then
// reports the scope-table entry and value(). Read-only by construction -- the usage check
// proves it, which is what lets a sole remote sender take the posted-write fast path.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t expected = get_arg(args::expected);

    Semaphore counter(sem::counter);  // mechanism comes from the host's scope table
    counter.wait_min(expected);

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = static_cast<uint32_t>(sem_scope_of(sem::counter));
    report[1] = counter.value();
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Make the report visible to the host readback of TL1.
    flush_l2_cache_line(report_addr);
    flush_l2_cache_line(report_addr + sizeof(uint32_t));
#endif
}
