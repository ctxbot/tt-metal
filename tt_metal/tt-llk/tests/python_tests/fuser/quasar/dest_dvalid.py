# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

from fuser.sfpu_node import SfpuNode

if TYPE_CHECKING:
    from fuser.compute_pipeline import ComputePipeline
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation

UNPACK = "UNPACK"
FPU = "FPU"
SFPU = "SFPU"
PACK = "PACK"

CHAIN_ORDER = (UNPACK, FPU, SFPU, PACK)


def chain(pipeline: "ComputePipeline") -> List[str]:
    clients = {PACK}

    for node in pipeline.math_nodes:
        if isinstance(node, SfpuNode):
            clients.add(SFPU)
        elif node.unpack_to_dest.value:
            clients.add(UNPACK)
        else:
            clients.add(FPU)

    return [client for client in CHAIN_ORDER if client in clients]


def enable(config: "GlobalConfig", operation: "L1Operation", client: str) -> str:
    if not config.quasar_use_dvalid:
        return ""

    members = chain(operation.math)

    code = ""
    for other in CHAIN_ORDER:
        call = "include" if other in members else "exclude"
        code += f"_llk_dest_dvalid_{call}_<dest_dvalid_client::{other}>();\n"

    action = "enable" if client in members else "disable"
    return code + f"_llk_dest_dvalid_{action}_<dest_dvalid_client::{client}>();\n"


def signal(config: "GlobalConfig", operation: "L1Operation", client: str) -> str:
    if not config.quasar_use_dvalid or config.skip_sync:
        return ""
    if client not in chain(operation.math):
        return ""

    params = f"dest_dvalid_client::{client}, {operation.dest_sync.cpp_enum_value}"
    if client == PACK:
        params += f", {config.dest_acc.cpp_enum_value}"

    return f"_llk_dest_dvalid_signal_<{params}>();\n"


def disable(config: "GlobalConfig", operation: "L1Operation", client: str) -> str:
    if not config.quasar_use_dvalid or not operation.is_last_stage:
        return ""
    if client not in chain(operation.math):
        return ""

    return (
        f"_llk_dest_dvalid_disable_<dest_dvalid_client::{client}>();\n"
        f"_llk_dest_dvalid_exclude_<dest_dvalid_client::{client}>();\n"
    )
