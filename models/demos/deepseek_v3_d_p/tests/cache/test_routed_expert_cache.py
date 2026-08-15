# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_routed_expert")
PACKED_CACHE_DIR = Path("/tmp/DS_PREFILL_routed_expert_packed")


@pytest.fixture(autouse=True)
def cleanup_cache():
    for cache_dir in [CACHE_DIR, PACKED_CACHE_DIR]:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    yield
    report_and_clear()


def test_packed_cache_schema_and_expert_ordering(tmp_path):
    """Packed metadata rejects incompatibility and preserves mesh expert order."""
    mesh_device = SimpleNamespace(shape=(2, 2))
    experts_per_chip, emb_dim, hidden_dim = 2, 32, 32
    torch_weights = [
        {
            "gate_proj": torch.full((hidden_dim, emb_dim), global_expert, dtype=torch.float32),
            "up_proj": torch.zeros(hidden_dim, emb_dim),
            "down_proj": torch.zeros(emb_dim, hidden_dim),
        }
        for global_expert in range(8)
    ]
    packed_gate = TtRoutedExpert._pack_expert_projection_family(
        torch_weights, experts_per_chip, *mesh_device.shape, "gate"
    )
    for row in range(mesh_device.shape[0]):
        for col in range(mesh_device.shape[1]):
            for local_expert in range(experts_per_chip):
                global_expert = ExpertMapping.get_global_expert_idx(
                    group=col,
                    chip=row,
                    local_expert=local_expert,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=mesh_device.shape[0],
                    num_dispatch_groups=mesh_device.shape[1],
                )
                start = local_expert * emb_dim
                assert torch.equal(
                    packed_gate[row, col, start : start + emb_dim],
                    torch.full((emb_dim, hidden_dim), global_expert, dtype=torch.float32),
                )

    TtRoutedExpert._write_packed_cache_schema(
        tmp_path, "routed_expert", mesh_device, experts_per_chip, emb_dim, hidden_dim, ttnn.bfloat4_b
    )
    metadata_path = tmp_path / "routed_expert.packed_v1.json"
    assert json.loads(metadata_path.read_text())["format"] == TtRoutedExpert.PACKED_CACHE_FORMAT
    assert TtRoutedExpert.validate_packed_cache_schema(
        tmp_path, "routed_expert", mesh_device, experts_per_chip, emb_dim, hidden_dim, ttnn.bfloat4_b
    )
    metadata_path.write_text("{}")
    assert not TtRoutedExpert.validate_packed_cache_schema(
        tmp_path, "routed_expert", mesh_device, experts_per_chip, emb_dim, hidden_dim, ttnn.bfloat4_b
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_routed_expert_weights_cold_warm_cache(mesh_device, device_params, monkeypatch):
    """Legacy and packed cache hydration produce identical routed-expert output."""
    torch.manual_seed(42)

    # Use realistic parameters
    seq_len_per_chip = 320
    emb_dim = 1024
    hidden_dim = 512
    num_routed_experts = 64
    num_experts_per_tok = 2
    # ceil(N/2) of the most conservative integer N such that dgs*seq*N >= theoretical
    # worst-case dispatch buffer. Real traffic never approaches the worst case.
    dispatch_buffer_capacity_factor = 2

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    # Compute constants (same as PCC test)
    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    total_experts = num_devices * experts_per_chip

    # Create random weights for all experts (HF format)
    torch_weights = []
    for _ in range(total_experts):
        torch_weights.append(
            {
                "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
            }
        )

    # Create input with proper sharding (follows PCC test pattern).
    # Flat 4D layout: (num_dispatch_groups, dispatch_group_size,
    # max_dispatch_buffer_token_size, emb_dim) — each chip's experts
    # are concatenated along the token dim, matching the real dispatch kernel layout.
    dispatched_buffer_torch = torch.randn(
        num_dispatch_groups,
        dispatch_group_size,
        max_dispatch_buffer_token_size,
        emb_dim,
        dtype=torch.float32,
    )

    # Shard across devices and reshape per-device to 2D (what extract/insert require).
    per_device_shape = (max_dispatch_buffer_token_size, emb_dim)
    mesh_mapper = get_ep_mesh_mapper(mesh_device)
    dispatched_buffer_tt = ttnn.from_torch(
        dispatched_buffer_torch,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    dispatched_buffer_tt = ttnn.reshape(dispatched_buffer_tt, per_device_shape)

    # Build (group, chip, local_expert) -> global expert id table, sharded across
    # the EP mesh so each device holds (1, 1, experts_per_chip). Then squeeze to a
    # 1D (experts_per_chip,) vector (required by extract/insert validators). Shared
    # across all TtRoutedExpert instances built below.
    global_expert_idx_tt = ttnn.from_torch(
        ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        ),
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)
    global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)

    # Build synthetic token counts / region offsets from random routing indices.
    # For a cache test we only need all three expert instances to see the same
    # counts/offsets — the specific values don't matter for the PCC comparison.
    _, _, routing_indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
        skip_x_initialization=True,
    )
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    _, expert_token_counts_torch, expert_region_offsets_torch, _ = get_gate_outputs(
        routing_indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    expert_token_counts_tt = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_token_counts_torch)
    expert_region_offsets_tt = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_region_offsets_torch)

    # Helper to convert output back to torch (follows PCC test pattern)
    mesh_composer = get_ep_mesh_composer(mesh_device)

    def to_torch_expert(tt_tensor):
        """Convert expert output back to torch with proper mesh composer."""
        # Output shape per device: (max_dispatch_buffer_token_size, emb_dim) — 2D.
        # Unsqueeze to 4D and compose back across the EP mesh.
        tt_expanded = ttnn.unsqueeze(ttnn.unsqueeze(tt_tensor, dim=0), dim=0)
        return ttnn.to_torch(tt_expanded, mesh_composer=mesh_composer)

    # Use consistent dtype across all paths
    weights_dtype = ttnn.bfloat4_b

    # === Path 1: From Weights ===
    logger.info(f"Test params: experts_per_chip={experts_per_chip}, max_tokens={max_dispatched_tokens_per_expert}")
    logger.info(f"Dimensions: emb_dim={emb_dim}, hidden_dim={hidden_dim}")
    logger.info(f"dispatched_buffer_tt.shape={dispatched_buffer_tt.shape}")

    expert_from_weights = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=torch_weights,
        weights_dtype=weights_dtype,
        weight_cache_path=None,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    # The unified op writes its FFN results back into the dispatched buffer in
    # place (no separate output allocation), so each path must run on its own
    # pristine copy of the input — otherwise path 2/3 would extract from path 1's
    # already-overwritten rows. Clone per path; the determinism we are checking is
    # over the weights->cold->warm cache, not the input buffer.
    output1_tt = expert_from_weights(ttnn.clone(dispatched_buffer_tt), expert_token_counts_tt, expert_region_offsets_tt)
    output1 = to_torch_expert(output1_tt)

    # === Path 2: Cold Cache ===
    init_checker(CACHE_DIR)
    assert not TtRoutedExpert.check_cache_complete(
        CACHE_DIR, "routed_expert", experts_per_chip
    ), "Cache should be empty before build"

    logger.info(f"Building cache to {CACHE_DIR}")
    profiler.clear()
    profiler.start("build_cache")
    TtRoutedExpert.build_ttnn_cache(
        torch_weights,
        experts_per_chip,
        mesh_device,
        weights_dtype,
        CACHE_DIR,
        "routed_expert",
    )
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert TtRoutedExpert.check_cache_complete(
        CACHE_DIR, "routed_expert", experts_per_chip
    ), "Cache should be complete after build"

    profiler.start("cold_load")
    expert_cold = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="routed_expert",
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    profiler.end("cold_load")
    output2_tt = expert_cold(ttnn.clone(dispatched_buffer_tt), expert_token_counts_tt, expert_region_offsets_tt)
    output2 = to_torch_expert(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    expert_warm = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="routed_expert",
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    profiler.end("warm_load")
    output3_tt = expert_warm(ttnn.clone(dispatched_buffer_tt), expert_token_counts_tt, expert_region_offsets_tt)
    output3 = to_torch_expert(output3_tt)

    # === Path 4: Packed Cache ===
    # The packed format is deliberately opt-in and isolated from the legacy
    # directory. It must load just three physical routed-expert tensorbins and
    # pass their base tensors to the packed-base C++ path without materialising
    # one TTNN tensor per local expert.
    monkeypatch.setenv("TT_ROUTED_EXPERT_CACHE_FORMAT", TtRoutedExpert.PACKED_CACHE_FORMAT)
    init_checker(PACKED_CACHE_DIR)
    TtRoutedExpert.build_ttnn_cache(
        torch_weights,
        experts_per_chip,
        mesh_device,
        weights_dtype,
        PACKED_CACHE_DIR,
        "routed_expert",
    )
    init_checker(PACKED_CACHE_DIR)
    assert TtRoutedExpert.check_packed_cache_complete(PACKED_CACHE_DIR, "routed_expert")
    packed_files = list(PACKED_CACHE_DIR.glob("routed_expert.packed_v1_*.tensorbin"))
    assert len(packed_files) == 3

    profiler.start("packed_load")
    expert_packed = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=PACKED_CACHE_DIR,
        cache_name_prefix="routed_expert",
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    profiler.end("packed_load")
    assert expert_packed.using_packed_cache
    assert expert_packed.gate_projs == []
    assert expert_packed.packed_gate_proj is not None
    output4_tt = expert_packed(ttnn.clone(dispatched_buffer_tt), expert_token_counts_tt, expert_region_offsets_tt)
    output4 = to_torch_expert(output4_tt)

    # === Validation ===
    # Debug: check output stats
    logger.info(
        f"Output1 (from weights): shape={output1.shape}, min={output1.min():.4f}, max={output1.max():.4f}, mean={output1.mean():.4f}"
    )
    logger.info(
        f"Output2 (cold cache): shape={output2.shape}, min={output2.min():.4f}, max={output2.max():.4f}, mean={output2.mean():.4f}"
    )
    logger.info(
        f"Output3 (warm cache): shape={output3.shape}, min={output3.min():.4f}, max={output3.max():.4f}, mean={output3.mean():.4f}"
    )

    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)
    passed_packed, pcc_packed = comp_pcc(output1, output4)

    logger.info(f"Routed Expert Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")
    logger.info(f"  packed_load: {profiler.get('packed_load')*1000:.1f} ms")
    logger.info(f"  Weights vs Packed Cache PCC: {pcc_packed}")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
    assert passed_packed, f"Packed cache mismatch: PCC={pcc_packed}"
