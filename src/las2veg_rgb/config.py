"""Global constants for las2veg-rgb."""

from __future__ import annotations

GRID_SIZE_M: float = 1.0
VOXEL_SIZE_M: float = 0.25
SUBVOXELS_PER_M: int = int(round(GRID_SIZE_M / VOXEL_SIZE_M))

CANOPY_UPPER_BOUND_M: float = 100.0

NOISE_CLASSIFICATION: int = 7
