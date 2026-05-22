"""Global constants for las2veg-rgb."""

from __future__ import annotations

# デフォルトのメッシュサイズ (m)。CLI --mesh-size で上書き可能。
GRID_SIZE_M: float = 5.0

# サブボクセル一辺 (m)。常に 0.25 で固定。
VOXEL_SIZE_M: float = 0.25

# 1m あたりのサブボクセル数 (常に 4)。
SUBVOXELS_PER_METER: int = int(round(1.0 / VOXEL_SIZE_M))

# 後方互換: 古いコードが SUBVOXELS_PER_M を参照する場合用。
# (1m メッシュ時代のグローバル定数。新しいコードは subvoxels_per_grid(mesh) を使う)
SUBVOXELS_PER_M: int = int(round(GRID_SIZE_M / VOXEL_SIZE_M))

CANOPY_UPPER_BOUND_M: float = 100.0

NOISE_CLASSIFICATION: int = 7


def subvoxels_per_grid(mesh_size_m: float) -> int:
    """Return number of 25cm subvoxels along one side of a mesh cell."""
    return int(round(mesh_size_m / VOXEL_SIZE_M))
