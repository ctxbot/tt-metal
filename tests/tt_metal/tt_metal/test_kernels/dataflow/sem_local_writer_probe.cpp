// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Provably-local sole writer: increments sem::counter with the plain-local up() only, then
// reports {baked scope, value}. Unlike the census probe, it never takes the semaphore's raw
// ring address, so the usage probe classifies it LOCAL_WRITER -- the class the gap-3
// refinement requires before letting a read-only remote observer ride along on the plain word.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t report_addr = get_arg(args::report_addr);
    const uint32_t increment_times = get_arg(args::increment_times);

    Semaphore counter(sem::counter);  // mechanism comes from the host's scope table
    for (uint32_t i = 0; i < increment_times; i++) {
        counter.up(1);
    }

    volatile tt_l1_ptr uint32_t* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(report_addr);
    report[0] = static_cast<uint32_t>(sem_scope_of(sem::counter));
    report[1] = counter.value();
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
    // Make the report visible to the host readback of TL1.
    flush_l2_cache_line(report_addr);
    flush_l2_cache_line(report_addr + sizeof(uint32_t));
#endif
}
