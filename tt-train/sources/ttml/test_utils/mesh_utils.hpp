// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttml::test_utils {

// True when this host has enough chips for `shape`. A 1x2 mesh, for instance, is all of an
// N300 but also a slice of a T3K or a Blackhole tray, so tests that need one gate on the
// devices being there rather than on a specific board type.
//
// Deliberately counts chips instead of comparing against SystemMesh::local_shape(): the
// system mesh reflects the mesh graph descriptor currently installed on the control plane,
// so once any test opens a 1x2 mesh it reports 1x2 for the rest of the process and a later
// test wanting a bigger mesh would skip itself on a host that can host it. A host that has
// the chips but can't form the mesh should fail loudly in the open, not skip.
inline bool system_supports_mesh(const tt::tt_metal::distributed::MeshShape& shape) {
    return tt::tt_metal::GetNumAvailableDevices() >= shape.mesh_size();
}

}  // namespace ttml::test_utils

// GTEST_SKIP() expands into a return statement, so it has to sit in the caller's body: from a
// helper function it would return out of the helper and let SetUp() go on to open the device.
//
// `shape` is bound once because the macro reads it three times, so a caller passing a temporary
// or a call expression doesn't get it evaluated repeatedly. The const reference also keeps a
// temporary alive for the body.
#define SKIP_UNLESS_MESH_SUPPORTED(shape)                                      \
    do {                                                                       \
        const auto& s = (shape);                                               \
        if (!ttml::test_utils::system_supports_mesh(s)) {                      \
            GTEST_SKIP() << fmt::format(                                       \
                "Skipping test: a {} mesh needs {} devices, this host has {}", \
                s,                                                             \
                s.mesh_size(),                                                 \
                tt::tt_metal::GetNumAvailableDevices());                       \
        }                                                                      \
    } while (0)
