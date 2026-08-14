// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"                        // ckernel::TensorShape + geometry helpers
#include "api/compute/common_globals.h"          // DataFormat enum, DST_ACCUM_MODE, ALWI
#include "internal/circular_buffer_interface.h"  // get_local_cb_interface (universal, all TRISCs)

// =====================================================================================================
// L1MetaData: the id-free L1 metadata object the compute datacopy ops consume (Phase 1: datacopy).
//
// One type carries both halves of "an L1 tile", split by compile-time vs runtime:
//   * Format + Shape are NON-TYPE TEMPLATE PARAMETERS (-ftt-nttp) -- the compile-time "what". Format is
//     the buffer's L1 data format; Shape (a ckernel::TensorShape) is the tile geometry. Being NTTPs is
//     the whole point: the per-format switches / register writes / asserts fold and DCE away.
//   * l1_address is the ONLY runtime member -- the "where". A runtime value cannot be an NTTP, so the
//     split lives INSIDE the type (NTTP vs member) rather than across two separate objects.
//
// L1MetaData carries NO register-side format (that is op state, derived from Format inside the LLK),
// NO CB id, and NO knowledge of the source (CB / DataflowBuffer / Scratchpad / LocalTensorAccessor).
// The op template deduces Format/Shape from the argument type.
// =====================================================================================================

namespace ckernel {
namespace experimental {

// Internal wrapper ABI only: the plain (format, geometry) descriptor the LLK APIs accept as an NTTP.
// Not a public type -- kernels use L1MetaData; the op builds L1MetaData::descriptor from its NTTPs.
struct L1Descriptor {
    std::uint8_t format;  // buffer L1 format (what the unpacker reads / the packer writes)
    TensorShape shape;    // tile geometry; derive num_faces / tile dims via TensorShape helpers
};

// The public, id-free metadata object. Construction is what varies across phases; the op signature
// copy_tile(L1MetaData<F,S>, dst) never changes:
//   * Phase 0 (manual):     L1MetaData<fmt, shape>(raw_l1_addr)
//   * Phase 1 (translators): to_l1metadata(dfb, idx) etc. return the same L1MetaData<fmt,shape> type.
template <DataFormat Format, TensorShape Shape>
struct L1MetaData {
    std::uint32_t l1_address;  // runtime "where"; Format/Shape are the compile-time "what"
    constexpr explicit L1MetaData(std::uint32_t addr) : l1_address(addr) {}

    // The descriptor the LLK APIs accept (buffer L1 format + geometry).
    static constexpr L1Descriptor descriptor = L1Descriptor{static_cast<std::uint8_t>(Format), Shape};
};

// -----------------------------------------------------------------------------------------------------
// Format-agnostic ADDRESS seam (runtime "where"). Resolves an absolute L1 tile address from a CB, with
// NO data format / geometry and NO side effects: get_operand_id / get_output_id are identity on
// Blackhole (interface index == cb_id), and these are pure reads of the local CB interface (valid on
// every TRISC), so a kernel can build the address on any thread and hand it to an id-free op.
//
// Absolute (out-of-order) addressing: the op packs/unpacks at exactly this address rather than
// advancing a running fifo pointer. The kernel still owns CB flow control (cb_wait_front /
// cb_reserve_back / cb_push_back); this only reads the current base pointer. In Phase 1 these become
// the CB specializations of the to_l1info(source) translators.
// -----------------------------------------------------------------------------------------------------
ALWI std::uint32_t cb_read_address(std::uint32_t cb_id, std::uint32_t tile_index = 0) {
    const auto& cb = get_local_cb_interface(cb_id);
    return cb.fifo_rd_ptr - 1 + cb.fifo_page_size * tile_index;
}

ALWI std::uint32_t cb_write_address(std::uint32_t cb_id, std::uint32_t out_tile_index = 0) {
    const auto& cb = get_local_cb_interface(cb_id);
    return cb.fifo_wr_ptr - 1 + cb.fifo_page_size * out_tile_index;
}

}  // namespace experimental
}  // namespace ckernel
