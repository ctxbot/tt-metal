// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/l1_spec.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_unary_datacopy_spec_api.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_A_spec_api.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free datacopy init. The op is NOT keyed on a CB id: it takes an L1MetaData whose data
 * format + tile geometry are NTTPs (deduced from the argument type). The register format is derived
 * INSIDE the LLK wrapper from the L1 format -- the compute API never sees a register format. Legacy
 * ckernel::copy_tile_init is untouched.
 *
 * | Template | Format | Buffer L1 data format (deduced from the L1MetaData argument) | DataFormat  |  | True |
 * | Template | Shape  | Tile geometry (deduced from the L1MetaData argument)         | TensorShape |  | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_tile_init(L1MetaData<Format, Shape> /*src*/) {
    // The descriptor (buffer L1 format + geometry the LLK APIs accept) is passed directly as an NTTP;
    // register format is derived inside the LLK overload.
    UNPACK((llk_unpack_A_init<
            L1MetaData<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          L1MetaData<Format, Shape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE>()));
}

// clang-format off
/**
 * Experimental id-free datacopy. Copies one tile from an L1 region described by L1MetaData into DST.
 * Compile-time "what" = the L1MetaData NTTPs (Format + Shape, fold/DCE); runtime "where" = the absolute
 * L1 address in the L1MetaData (src.l1_address). No CB id, no source type, no register format on the API.
 *
 * | Template | Format         | Buffer L1 data format (deduced from L1MetaData)   | DataFormat  |         | True |
 * | Template | Shape          | Tile geometry (deduced from L1MetaData)           | TensorShape |         | True |
 * | Function | src            | The source L1 region (format+shape+address)   | L1MetaData      |         | True |
 * | Function | dst_tile_index | Tile index in the DST register                | uint32_t    | 0 to 15 | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_tile(L1MetaData<Format, Shape> src, std::uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<
            L1MetaData<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(src.l1_address)));
    MATH((llk_math_eltwise_unary_datacopy<
          L1MetaData<Format, Shape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          UnpackToDestEn>(dst_tile_index)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
