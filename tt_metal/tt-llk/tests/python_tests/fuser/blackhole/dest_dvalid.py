# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation

UNPACK = "UNPACK"
FPU = "FPU"
SFPU = "SFPU"
PACK = "PACK"


def enable(config: "GlobalConfig", operation: "L1Operation", client: str) -> str:
    return ""


def signal(config: "GlobalConfig", operation: "L1Operation", client: str) -> str:
    return ""


def disable(config: "GlobalConfig", operation: "L1Operation", client: str) -> str:
    return ""
