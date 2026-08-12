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
from typing import Optional, Sequence, Tuple

import pytest

import ttnn


def system_mesh_shape() -> Optional[Tuple[int, ...]]:
    """Shape of the devices visible to this host, or ``None`` if it can't be queried."""
    try:
        # Only reachable through the binding module; ttnn.distributed doesn't re-export it.
        descriptor = ttnn._ttnn.multi_device.SystemMeshDescriptor()  # type: ignore[attr-defined]
        return tuple(descriptor.local_shape())
    except Exception:  # noqa: BLE001
        return None


def system_supports_mesh(shape: Sequence[int]) -> bool:
    """Whether this host's devices can host ``shape``.

    Falls back to a plain device count when the system mesh isn't queryable, and to
    ``True`` when neither is available: an unrecognised host is something the open
    path should report, not a reason to skip silently.
    """
    system = system_mesh_shape()
    if system is not None:
        if len(system) < len(shape):
            return False
        return all(have >= want for have, want in zip(system, shape))
    try:
        return int(ttnn.get_num_devices()) >= math.prod(shape)
    except Exception:  # noqa: BLE001
        return True


def skip_if_system_too_small(shape: Sequence[int], what: str) -> None:
    """Skip when the host is too small for ``shape``, otherwise return normally.

    Callers are expected to let any later failure from the open itself propagate.
    """
    if system_supports_mesh(shape):
        return
    system = system_mesh_shape()
    have = f"{system}" if system is not None else "unavailable"
    pytest.skip(f"{what} needs a {tuple(shape)} mesh; devices visible to this host: {have}")
