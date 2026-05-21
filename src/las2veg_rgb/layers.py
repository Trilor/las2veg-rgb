"""Layer definitions for vertical point cloud bins.

All Z bounds are half-open [z_min, z_max). Heights come from PDAL HAG
(Height Above Ground), not absolute Z.

z3 upper bound is 100m to cover the global range of forest canopies
(largest known tree is 115m). Values above are clipped, which also
filters HAG anomalies from PDAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CANOPY_UPPER_BOUND_M, VOXEL_SIZE_M


@dataclass(frozen=True)
class LayerSpec:
    name: str
    z_min: float
    z_max: float

    @property
    def z_subvoxels(self) -> int:
        return int(round((self.z_max - self.z_min) / VOXEL_SIZE_M))


@dataclass
class LayerCounts:
    """Per-cell point counts for one layer (intermediate Phase 1 output)."""

    spec: LayerSpec
    count: np.ndarray  # shape (H, W) int32 — number of points in the layer per 1m cell


@dataclass
class RatioMetrics:
    """The 5 ratio-based indicators stored in Phase 1 GeoTIFF / Phase 2 RGBA."""

    density_z1: np.ndarray  # z1 / (z0 + z1), shape (H, W) float32, range [0, 1]
    density_z2: np.ndarray  # z2 / (z0 + z1 + z2)
    density_z3: np.ndarray  # z3 / total
    occupancy_z1: np.ndarray
    occupancy_z2: np.ndarray


LAYERS: tuple[LayerSpec, ...] = (
    LayerSpec("z0", 0.00, 0.25),
    LayerSpec("z1", 0.25, 1.00),
    LayerSpec("z2", 1.00, 2.00),
    LayerSpec("z3", 2.00, CANOPY_UPPER_BOUND_M),
)
