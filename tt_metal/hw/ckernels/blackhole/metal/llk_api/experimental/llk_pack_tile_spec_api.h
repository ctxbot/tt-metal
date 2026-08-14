// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_pack_tile_api.h"  // legacy CB-id API + unified llk_pack_init_impl / llk_pack_impl
#include "data_format_derive.h"
#include "api/compute/experimental/l1_spec.h"

/*************************************************************************
 * LLK PACK -- L1MetaData (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_pack_init / llk_pack), distinguished by taking an
 * L1Descriptor as the first NTTP. DESC carries the output buffer L1 format + geometry; the Dest
 * register format is derived HERE from DESC.format + the fp32-dest-acc flag and never exposed above the
 * LLK. Both overloads forward to the same unified cores as the CB-id API. The runtime write address
 * comes from the caller via l1_spec.h::cb_write_address (absolute/out-of-order) -- no fifo bookkeeping.
 *************************************************************************/

template <
    ckernel::experimental::L1Descriptor DESC,
    bool is_fp32_dest_acc_en = false,
    PackMode pack_mode = PackMode::Default,
    bool zero_output = false,
    bool skip_addrmod_config = false,
    bool skip_packer_strides = false>
inline void llk_pack_init(const std::uint32_t num_tiles = 1) {
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(
        ckernel::infer_pack_src_format(static_cast<DataFormat>(DESC.format), is_fp32_dest_acc_en));
    // is_input_8bit_format only affects the tilize workaround; irrelevant for PackMode::Default datacopy.
    constexpr bool is_input_8bit_format = false;
    llk_pack_init_impl<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        RegFmt,
        DESC.shape.face_r_dim,
        DESC.shape.total_col_dim(),
        DESC.shape.total_num_faces(),
        num_tiles,
        is_input_8bit_format);
}

// DESC is required only to disambiguate this overload from the CB-id llk_pack (same runtime arg count);
// the pack op itself needs only the runtime write address (format/geometry were set at llk_pack_init).
template <
    ckernel::experimental::L1Descriptor DESC,
    bool is_fp32_dest_acc_en = false,
    bool out_of_order_output = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t base_ptr) {
    llk_pack_impl<is_fp32_dest_acc_en, pack_mode>(tile_index, base_ptr);
}
