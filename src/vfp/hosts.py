"""Host placement and contact accounting.

Cummins model a host patch as a square block of emitters. The patch geometry is
what the whole attack-abatement question turns on: their point is that contact
rate depends on the spatial arrangement of hosts and not only on their number,
which is a valid criticism of compartmental models.

Capacity is implemented here and defaults to unlimited, reproducing Cummins. They
invoked host saturation -- "mosquitoes only require a fixed amount of blood and
will not attack additional individuals" -- as one of two mechanisms explaining
attack abatement, but no such mechanism exists in their agent model (audit 3.8),
so the effect they report comes entirely from plume geometry. Having capacity as
a switch is what lets the two contributions be separated rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HostSet:
    positions_m: np.ndarray  # (M, 2)
    patch_id: np.ndarray  # (M,) 0 or 1, for the two-patch attack-abatement geometry
    capacity: int  # 0 => unlimited

    @property
    def n(self) -> int:
        return int(self.positions_m.shape[0])

    def patch_sizes(self) -> tuple[int, ...]:
        return tuple(int(np.sum(self.patch_id == p)) for p in np.unique(self.patch_id))


def _square_block(centre: tuple[float, float], count: int, spacing: float) -> np.ndarray:
    """`count` hosts on the tightest centred square lattice at `spacing`."""
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    ix, iy = np.meshgrid(np.arange(cols), np.arange(rows))
    offsets = np.column_stack([ix.ravel(), iy.ravel()])[:count].astype(float)
    offsets -= offsets.mean(axis=0)
    return np.asarray(centre, dtype=float) + offsets * spacing


def build(cfg) -> HostSet:
    h = cfg.hosts
    if h.layout == "single":
        positions = np.asarray(h.patch_center_m, dtype=float).reshape(1, 2)
        patch_id = np.zeros(1, dtype=np.int8)
    elif h.layout == "grid_patch":
        positions = _square_block(h.patch_center_m, h.n_hosts, h.spacing_m)
        patch_id = np.zeros(h.n_hosts, dtype=np.int8)
    elif h.layout == "two_patches":
        first = _square_block(h.patch_center_m, h.n_hosts, h.spacing_m)
        second = _square_block(h.patch2_center_m, h.n_hosts2, h.spacing_m)
        positions = np.vstack([first, second])
        patch_id = np.concatenate(
            [np.zeros(h.n_hosts, dtype=np.int8), np.ones(h.n_hosts2, dtype=np.int8)]
        )
    else:
        raise ValueError(f"hosts.layout {h.layout!r} is not implemented")

    margin = cfg.contact.radius_m
    if (
        positions[:, 0].min() < margin
        or positions[:, 0].max() > cfg.domain.lx_m - margin
        or positions[:, 1].min() < margin
        or positions[:, 1].max() > cfg.domain.ly_m - margin
    ):
        raise ValueError(
            f"hosts.layout {h.layout!r} places a host within one contact radius "
            f"({margin} m) of the domain edge, so its capture disc is clipped"
        )
    return HostSet(positions, patch_id, h.capacity)
