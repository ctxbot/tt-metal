// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/l1_spec.h"

#ifdef TRISC_PACK
#include "experimental/llk_pack_tile_spec_api.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free pack init. The op is NOT keyed on a CB id: the output buffer L1 format + tile
 * geometry are NTTPs deduced from the L1MetaData argument. The Dest register format is derived INSIDE the
 * LLK wrapper from the L1 format -- the compute API never sees a register format. Legacy
 * ckernel::pack_init is untouched.
 *
 * | Template | Format | Output buffer L1 data format (deduced from L1MetaData) | DataFormat  |  | True |
 * | Template | Shape  | Output tile geometry (deduced from L1MetaData)         | TensorShape |  | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_init(L1MetaData<Format, Shape> /*out*/) {
    PACK((llk_pack_init<L1MetaData<Format, Shape>::descriptor, DST_ACCUM_MODE>()));
}

// clang-format off
/**
 * Experimental id-free pack. Copies one tile from DST to the absolute L1 address in the output L1MetaData.
 * Formats/geometry were programmed at pack_init; the pack op needs only the runtime write address
 * (out.l1_address) -- absolute (out-of-order) addressing, no running fifo pointer. No id, no formats.
 *
 * | Template | Format    | Output buffer L1 data format (deduced from L1MetaData) | DataFormat  |         | True |
 * | Template | Shape     | Output tile geometry (deduced from L1MetaData)         | TensorShape |         | True |
 * | Function | out       | The output L1 region (format+shape+write address)  | L1MetaData      |         | True |
 * | Function | ifrom_dst | Tile index in the DST register                     | uint32_t    | 0 to 15 | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_tile(L1MetaData<Format, Shape> out, std::uint32_t ifrom_dst) {
    // out_of_order_output=true: pack to the absolute address in the L1MetaData (no fifo_wr_tile_ptr bump).
    PACK((llk_pack<
          L1MetaData<Format, Shape>::descriptor,
          DST_ACCUM_MODE,
          /*out_of_order_output=*/true,
          PackMode::Default>(ifrom_dst, out.l1_address)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
