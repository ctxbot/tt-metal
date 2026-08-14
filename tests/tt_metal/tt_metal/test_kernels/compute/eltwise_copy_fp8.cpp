// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// L1MetaData id-free datacopy verification: the two ops (unpack+math copy, and pack) take an L1MetaData
// (data format + tile geometry as NTTPs, absolute L1 address as the only runtime state) -- NO CB id
// on the op surface. The register format is derived on-device from the L1 format. Legacy inits are
// kept so this run isolates the id-free OP path + address seam + infer fn. Identity bf16->bf16 output
// must be bit-identical to the legacy path.
#include "api/compute/experimental/tile_move_copy_spec.h"
#include "api/compute/experimental/pack_spec.h"

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    // Input/output L1 data formats come from the host as compile-time args (id-free: no CB lookup).
    // 32x32 tile = 2x2 faces of 16x16 for every format (face geometry is format-independent). Format +
    // geometry are compile-time (fold/DCE); the L1 address is resolved per-iteration from the CB via
    // the format-agnostic address seam.
    constexpr auto in_fmt = static_cast<DataFormat>(get_compile_time_arg_val(1));
    constexpr auto out_fmt = static_cast<DataFormat>(get_compile_time_arg_val(2));
    constexpr auto shape = TensorShape{16, 16, 2, 2};
    using InInfo = experimental::L1MetaData<in_fmt, shape>;
    using OutInfo = experimental::L1MetaData<out_fmt, shape>;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        cb0.wait_front(1);
        cb16.reserve_back(1);
        experimental::copy_tile(InInfo(experimental::cb_read_address(tt::CBIndex::c_0)), 0);

        tile_regs_commit();
        tile_regs_wait();

        experimental::pack_tile(OutInfo(experimental::cb_write_address(tt::CBIndex::c_16)), 0);
        cb0.pop_front(1);
        cb16.push_back(1);

        tile_regs_release();
    }
}
