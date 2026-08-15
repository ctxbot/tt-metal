// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pure read-only observer: binds sem::counter and only reads it once. The usage check
// proves this binding read-only, so its presence never forces an atomic mechanism -- used by
// the gap-3 census pin (sole on-node writer + off-node observer must stay LOCAL_NONATOMIC).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Semaphore counter(sem::counter);  // mechanism comes from the host's scope table
    // A single read (of this core's own cell -- meaningless data, but a legal read shape).
    volatile uint32_t sink = counter.value();
    (void)sink;
}
