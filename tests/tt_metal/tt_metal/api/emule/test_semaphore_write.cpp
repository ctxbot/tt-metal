// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// To run (from the tt-metal repo root, after an emule build):
//   build_emule/test/tt_metal/unit_tests_api --gtest_filter="UnitMeshFixture.Semaphore_*"

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include "device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal {

TEST_F(UnitMeshFixture, Semaphore_Direct_Write_SanityCheck) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // EMULE_SEM_BASE is a JIT-time define injected by the runner: the
    // firmware-style L1 offset of the reserved Semaphore region. Any scalar
    // pointer access into that range must trip the ASAN guard inside
    // __emule_local_l1_to_ptr and abort the kernel thread.
    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t sem_addr = EMULE_SEM_BASE;
            volatile uint32_t* illegal_ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(sem_addr);
            *illegal_ptr = 0xABCD;
        }
    )";

    CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    EXPECT_DEATH(
        LaunchProgram(this->device(), std::move(program)),
        ".*Illegal Semaphore Access: Offset 0x.*is inside the reserved Semaphore region.*");
}

// Positive control: a scalar access to an ordinary allocated L1 buffer — well
// OUTSIDE the reserved semaphore region — must NOT abort. Guards the semaphore
// check from flagging normal L1 addressing (the region test must be a precise
// [start, end) containment, not an over-broad lower-bound).
TEST_F(UnitMeshFixture, Semaphore_OutsideRegion_NoViolation) {
    ::setenv("TT_METAL_EMULE_ASAN", "1", 1);

    CoreCoord logical_core = {0, 0};
    Program program = CreateProgram();

    // A normal L1 buffer is allocated well away from the reserved semaphore
    // region (which lives in the low system area near EMULE_SEM_BASE).
    auto buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = 1024},
        {.page_size = 1024, .buffer_type = BufferType::L1},
        &this->device());
    uint32_t addr = static_cast<uint32_t>(buf->address());

    std::string kernel_src = R"(
        #include "api/dataflow/dataflow_api.h"
        void kernel_main() {
            uint32_t a = get_arg_val<uint32_t>(0);
            volatile uint32_t* ptr = (volatile uint32_t*)__emule_local_l1_to_ptr(a);
            *ptr = 0x1234;
        }
    )";
    auto kernel = CreateKernelFromString(
        program,
        kernel_src,
        logical_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, logical_core, {addr});

    // Must NOT abort.
    LaunchProgram(this->device(), std::move(program));
    SUCCEED();

    ::unsetenv("TT_METAL_EMULE_ASAN");
}

}  // namespace tt::tt_metal
