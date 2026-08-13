# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Capability checks for tests that need a multi-device mesh.

The rule the mesh fixtures follow: a host that is too small for the requested mesh
skips, but a host that should be able to open that mesh and then fails must fail the
test. Collapsing both into a skip is how a fabric bring-up failure on a Blackhole
quietbox turned every mesh test into a skip while the job still reported green.
"""

import math
from typing import Optional, Sequence

import pytest

import ttnn


def num_available_devices() -> Optional[int]:
    """Chips visible to this host, or ``None`` if the cluster can't be queried."""
    try:
        return int(ttnn.get_num_devices())
    except Exception:  # noqa: BLE001
        return None


def system_supports_mesh(shape: Sequence[int]) -> bool:
    """Whether this host has enough chips for ``shape``.

    Counts chips rather than comparing against the system mesh shape, which reflects
    the mesh graph descriptor currently installed on the control plane: once any test
    opens a 1x2 mesh the system mesh reads 1x2 for the rest of the process, and a
    later test wanting a bigger mesh would skip itself on a host that can host it.
    A host with the chips that still can't form the mesh should fail in the open.

    An unrecognised host counts as supported, for the same reason: that's the open
    path's story to tell, not a reason to skip silently.
    """
    available = num_available_devices()
    return available is None or available >= math.prod(shape)


def skip_if_system_too_small(shape: Sequence[int], what: str) -> None:
    """Skip when the host is too small for ``shape``, otherwise return normally.

    Callers are expected to let any later failure from the open itself propagate.
    """
    if system_supports_mesh(shape):
        return
    pytest.skip(
        f"{what} needs a {tuple(shape)} mesh ({math.prod(shape)} devices); this host has {num_available_devices()}"
    )
