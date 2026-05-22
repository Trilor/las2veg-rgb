"""Phase 1 verification script for las2veg-rgb (ratio-based version).

Pipeline:
    LAS -> PDAL (drop noise, SMRF ground classify, HAG) -> cached LAZ
        -> laspy read
        -> per-layer point counts (z0/z1/z2/z3)
        -> 6 ratio indicators:
            density_z1        = z1 / (z0 + z1)
            density_z2        = z2 / (z0 + z1 + z2)
            density_z3        = z3 / total
            occupancy_z1      = subvoxel-occupied / 48
            occupancy_z2      = subvoxel-occupied / 64
            canopy_height_p95 = percentile(HAG, 95) within z3 / 100m
        -> PNG previews + 6-band float32 GeoTIFF + metadata JSON

All ratios are laser-density independent and live in [0.0, 1.0].
canopy_height_p95 follows the NASA GEDI / ALS convention (RH95) for
canopy top height, normalized by the 100m global upper bound.

Run:
    python scripts/phase1_preview.py \
        --input 08ME3204/08ME3204.las \
        --output data/output \
        --crs EPSG:6676 \
        [--skip-pdal]
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path

import click
import laspy
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy.stats import binned_statistic_2d

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from las2veg_rgb.config import (  # noqa: E402
    CANOPY_UPPER_BOUND_M,
    GRID_SIZE_M as DEFAULT_GRID_SIZE_M,
    NOISE_CLASSIFICATION,
    VOXEL_SIZE_M,
    subvoxels_per_grid,
)
from las2veg_rgb.layers import LAYERS, LayerSpec  # noqa: E402
from las2veg_rgb.runs import create_run_dir, update_latest_link  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")


def run_pdal_pipeline(input_path: Path, cached_laz: Path, override_crs: str | None) -> int:
    """Run PDAL: drop noise, SMRF ground classify, HAG. Cache to LAZ.

    Returns the number of points written.
    """
    import pdal

    stages: list[dict] = []
    reader: dict = {"type": "readers.las", "filename": str(input_path)}
    if override_crs:
        reader["override_srs"] = override_crs
    stages.append(reader)

    stages.append(
        {
            "type": "filters.range",
            "limits": f"Classification![{NOISE_CLASSIFICATION}:{NOISE_CLASSIFICATION}]",
        }
    )
    stages.append({"type": "filters.smrf"})
    stages.append({"type": "filters.hag_nn"})
    stages.append(
        {
            "type": "writers.las",
            "filename": str(cached_laz),
            "compression": "true",
            "extra_dims": "HeightAboveGround=float32",
        }
    )

    pipeline_json = json.dumps({"pipeline": stages})
    t0 = time.time()
    pipeline = pdal.Pipeline(pipeline_json)
    count = pipeline.execute()
    log.info(
        "  [PDAL] %s: %d points in %.1fs",
        input_path.name, count, time.time() - t0,
    )
    return count


def _pdal_worker(args: tuple[Path, Path, str | None]) -> tuple[str, int | None, str | None]:
    """multiprocessing worker. Returns (input_name, point_count, error_str)."""
    input_path, cached_laz, override_crs = args
    try:
        count = run_pdal_pipeline(input_path, cached_laz, override_crs)
        return (input_path.name, count, None)
    except Exception as e:  # noqa: BLE001
        return (input_path.name, None, str(e))


def ensure_hag_cache(
    input_paths: list[Path],
    override_crs: str | None,
    parallel: int,
) -> list[Path]:
    """For each LAS, ensure a *_hag.laz cache exists next to it. Runs PDAL in parallel.

    Returns the list of cached LAZ paths in the same order as input_paths.
    """
    cached_paths: list[Path] = [p.with_name(p.stem + "_hag.laz") for p in input_paths]

    pending: list[tuple[Path, Path, str | None]] = []
    for las_path, cache_path in zip(input_paths, cached_paths, strict=True):
        if cache_path.exists():
            log.info("  [skip PDAL] cached: %s", cache_path.name)
        else:
            pending.append((las_path, cache_path, override_crs))

    if not pending:
        return cached_paths

    log.info(
        "Running PDAL for %d file(s) with %d parallel worker(s)...",
        len(pending), parallel,
    )
    t0 = time.time()
    if parallel > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=parallel) as pool:
            for name, count, err in pool.imap_unordered(_pdal_worker, pending):
                if err:
                    log.error("  [PDAL FAIL] %s: %s", name, err)
                else:
                    log.info("  [PDAL OK] %s: %d points", name, count)
    else:
        for args in pending:
            name, count, err = _pdal_worker(args)
            if err:
                log.error("  [PDAL FAIL] %s: %s", name, err)

    log.info("PDAL stage total: %.1fs", time.time() - t0)
    return cached_paths


def compute_global_bbox(cached_laz_paths: list[Path]) -> tuple[float, float, float, float]:
    """Compute the global xy bbox across all LAZ files (header-only, fast)."""
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for laz_path in cached_laz_paths:
        with laspy.open(str(laz_path)) as f:
            h = f.header
            x_min = min(x_min, h.mins[0])
            y_min = min(y_min, h.mins[1])
            x_max = max(x_max, h.maxs[0])
            y_max = max(y_max, h.maxs[1])
    return x_min, y_min, x_max, y_max


def resolve_input_paths(input_arg: Path) -> list[Path]:
    """Expand --input into a sorted list of LAS files.

    Accepts:
      - a single .las/.laz file
      - a directory (all *.las / *.laz inside, non-recursive, excluding *_hag.laz)
    """
    if input_arg.is_file():
        return [input_arg]
    if input_arg.is_dir():
        las_files = sorted(
            [p for p in input_arg.iterdir() if p.suffix.lower() in (".las", ".laz")
             and not p.name.endswith("_hag.laz")]
        )
        if not las_files:
            raise click.ClickException(f"No LAS/LAZ files in {input_arg}.")
        return las_files
    raise click.ClickException(f"Input does not exist: {input_arg}")


def detect_crs(las_path: Path, override_crs: str | None) -> CRS:
    """Read CRS from LAS header; error out if missing and no override given."""
    with laspy.open(str(las_path)) as f:
        las_crs = f.header.parse_crs()

    if override_crs:
        crs = CRS.from_user_input(override_crs)
        log.info("Using user-specified CRS: %s", crs)
        return crs

    if las_crs is None:
        raise click.ClickException(
            f"LAS at {las_path} has no embedded CRS. "
            f"Re-run with --crs EPSG:xxxx (e.g. EPSG:6676 for JGD2011 zone VIII)."
        )

    crs = CRS.from_wkt(las_crs.to_wkt())
    log.info("Detected CRS from LAS header: %s", crs)
    return crs


def snap_grid_to_crs_origin(
    x_min: float, x_max: float, y_min: float, y_max: float, mesh_size_m: float
) -> tuple[int, int, int, int]:
    """Snap bbox to a grid anchored at the CRS absolute origin (0,0).

    Returns (gx_min, gy_min, gx_max, gy_max) all aligned to multiples of mesh_size_m.
    """
    gx_min = math.floor(x_min / mesh_size_m) * mesh_size_m
    gy_min = math.floor(y_min / mesh_size_m) * mesh_size_m
    gx_max = math.ceil(x_max / mesh_size_m) * mesh_size_m
    gy_max = math.ceil(y_max / mesh_size_m) * mesh_size_m
    return int(gx_min), int(gy_min), int(gx_max), int(gy_max)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise numerator/denominator; returns 0 where denominator is 0."""
    out = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=out, where=(denominator > 0))
    return out.astype(np.float32)


def _build_layer_edges() -> np.ndarray:
    """Edges for np.digitize: returns LAYERS' z_min boundaries (skipping the first).

    Points with hag < LAYERS[0].z_min get layer_id 0 too (dropped later).
    Points with hag >= LAYERS[-1].z_max also get the last id (and are then masked out).
    """
    # Order matters: z0..z3 sequential
    return np.array([s.z_min for s in LAYERS[1:]] + [LAYERS[-1].z_max], dtype=np.float64)


def _streaming_accumulator(
    cached_laz_paths: list[Path],
    gx_min: int,
    gy_min: int,
    nx: int,
    ny: int,
    mesh_size_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Stream all LAZ files through one pass, accumulating:
      - counts[layer_name][ny, nx] uint32: per-layer point counts (4 layers)
      - subvoxel_set[layer_name] bool (only for z1, z2): occupied subvoxel flags
        Each is a 3D bool array sized (ny*subs, nx*subs, nz_sub).
      - z3_x, z3_y, z3_hag: concatenated arrays of z3 points (for P95 later)

    Memory: peak per file is ~LAS-size in RAM; global accumulators are
      counts: 4 * nx * ny * 8B = small (< 100MB even for 10km tile)
      subvoxel_set: bool arrays sized (ny*subs) * (nx*subs) * nz_sub
        for 5m mesh + 0.25m subvoxel on 2km*2km: 8000*8000*4 = 256MB (z1: *3, z2: *4)
    """
    subs = subvoxels_per_grid(mesh_size_m)

    counts: dict[str, np.ndarray] = {s.name: np.zeros((ny, nx), dtype=np.uint32) for s in LAYERS}

    # 占有率はサブボクセル単位の bool 配列で集積（複数 LAZ 間で OR 合成可能）
    subvoxel_sets: dict[str, np.ndarray] = {}
    for s in LAYERS:
        if s.name in ("z1", "z2"):
            subvoxel_sets[s.name] = np.zeros(
                (ny * subs, nx * subs, s.z_subvoxels), dtype=bool
            )

    # canopy P95 用の z3 点座標は concat する (P95 はストリーミング不可)
    z3_x_chunks: list[np.ndarray] = []
    z3_y_chunks: list[np.ndarray] = []
    z3_h_chunks: list[np.ndarray] = []

    # 各 LAZ ごとの bin edges (全範囲共通)
    x_edges = np.linspace(gx_min, gx_min + nx * mesh_size_m, nx + 1, dtype=np.float64)
    y_edges = np.linspace(gy_min, gy_min + ny * mesh_size_m, ny + 1, dtype=np.float64)
    layer_z_edges = _build_layer_edges()

    sub_x_edges = np.linspace(gx_min, gx_min + nx * mesh_size_m, nx * subs + 1, dtype=np.float64)
    sub_y_edges = np.linspace(gy_min, gy_min + ny * mesh_size_m, ny * subs + 1, dtype=np.float64)

    total_points = 0
    t_start = time.time()
    for idx, laz_path in enumerate(cached_laz_paths, start=1):
        t0 = time.time()
        las = laspy.read(str(laz_path))
        x_arr = np.asarray(las.x, dtype=np.float64)
        y_arr = np.asarray(las.y, dtype=np.float64)
        try:
            h_arr = np.asarray(las["HeightAboveGround"], dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(
                f"HeightAboveGround missing in {laz_path}. Delete it and re-run."
            ) from e
        del las
        n_pts = x_arr.size
        total_points += n_pts

        # 1 回だけ層 ID を計算: layer_id 0,1,2,3 == z0,z1,z2,z3
        # hag < 0.25 -> 0 (z0), 0.25-1.0 -> 1, 1.0-2.0 -> 2, 2.0-100.0 -> 3, else -> 4
        layer_id = np.digitize(h_arr, layer_z_edges)  # 0..len(LAYERS)
        valid = layer_id < len(LAYERS)  # 100m 超の点を除外

        # ── 層ごとの 1m count 集計 ──
        for k, spec in enumerate(LAYERS):
            layer_mask = valid & (layer_id == k)
            if not np.any(layer_mask):
                continue
            cnt, _, _ = np.histogram2d(
                x_arr[layer_mask], y_arr[layer_mask], bins=[x_edges, y_edges],
            )
            # cnt の形状は (nx, ny) なので転置 + 上下反転で (ny, nx) に
            counts[spec.name] += np.flipud(cnt.T.astype(np.uint32))

        # ── z1, z2 の占有サブボクセル更新 ──
        for k, spec in enumerate(LAYERS):
            if spec.name not in subvoxel_sets:
                continue
            layer_mask = valid & (layer_id == k)
            if not np.any(layer_mask):
                continue
            xl = x_arr[layer_mask]
            yl = y_arr[layer_mask]
            hl = h_arr[layer_mask]
            sub_z_edges = np.linspace(
                spec.z_min, spec.z_max, spec.z_subvoxels + 1, dtype=np.float64,
            )
            # サブボクセルインデックスを算出 (digitize で edges 内インデックスを 1-based に取り、 -1 で 0-based)
            ix = np.digitize(xl, sub_x_edges) - 1
            iy = np.digitize(yl, sub_y_edges) - 1
            iz = np.digitize(hl, sub_z_edges) - 1
            in_range = (
                (ix >= 0) & (ix < nx * subs)
                & (iy >= 0) & (iy < ny * subs)
                & (iz >= 0) & (iz < spec.z_subvoxels)
            )
            ix = ix[in_range]
            iy = iy[in_range]
            iz = iz[in_range]
            # bool 配列にフラグ立て (重複代入は OR と等価)
            subvoxel_sets[spec.name][iy, ix, iz] = True

        # ── z3 点の concat ──
        z3_mask = valid & (layer_id == 3)
        if np.any(z3_mask):
            z3_x_chunks.append(x_arr[z3_mask])
            z3_y_chunks.append(y_arr[z3_mask])
            z3_h_chunks.append(h_arr[z3_mask])

        log.info(
            "  [%2d/%d] %s: %d pts in %.1fs (total %d pts, %.1fs)",
            idx, len(cached_laz_paths), laz_path.name, n_pts,
            time.time() - t0, total_points, time.time() - t_start,
        )

    z3_x = np.concatenate(z3_x_chunks) if z3_x_chunks else np.zeros(0, dtype=np.float64)
    z3_y = np.concatenate(z3_y_chunks) if z3_y_chunks else np.zeros(0, dtype=np.float64)
    z3_h = np.concatenate(z3_h_chunks) if z3_h_chunks else np.zeros(0, dtype=np.float64)

    log.info("Streaming aggregation total: %d points in %.1fs",
             total_points, time.time() - t_start)
    return counts, subvoxel_sets, z3_x, z3_y, z3_h


def occupancy_from_subvoxel_set(
    subvoxel_set: np.ndarray, subs: int, nz_sub: int,
) -> np.ndarray:
    """Reduce a (ny*subs, nx*subs, nz_sub) bool array to (ny, nx) float occupancy ratio.

    Uses ny/nx inferred from the input shape; total subvoxels per cell = subs*subs*nz_sub.
    Output is flipud-reversed so that row 0 = north (= same orientation as density).
    """
    ny_sub, nx_sub, _ = subvoxel_set.shape
    ny = ny_sub // subs
    nx = nx_sub // subs
    # Reshape: (ny, subs, nx, subs, nz_sub) and sum over subvoxel axes
    occupied_count = subvoxel_set.reshape(ny, subs, nx, subs, nz_sub).sum(axis=(1, 3, 4))
    total = float(subs * subs * nz_sub)
    occupancy = (occupied_count / total).astype(np.float32)
    # density と同じ向き (row 0 = 北側) に揃えるため上下反転
    return np.flipud(occupancy)


def canopy_p95_from_z3(
    z3_x: np.ndarray, z3_y: np.ndarray, z3_h: np.ndarray,
    gx_min: int, gy_min: int, nx: int, ny: int, mesh_size_m: float,
) -> np.ndarray:
    """Compute (ny, nx) float32 P95 canopy height normalized by CANOPY_UPPER_BOUND_M."""
    if z3_x.size == 0:
        return np.zeros((ny, nx), dtype=np.float32)
    x_edges = np.linspace(gx_min, gx_min + nx * mesh_size_m, nx + 1, dtype=np.float64)
    y_edges = np.linspace(gy_min, gy_min + ny * mesh_size_m, ny + 1, dtype=np.float64)
    stat, _, _, _ = binned_statistic_2d(
        z3_x, z3_y, z3_h,
        statistic=lambda v: np.percentile(v, 95),
        bins=[x_edges, y_edges],
    )
    stat = np.nan_to_num(stat, nan=0.0)
    normalized = (stat / CANOPY_UPPER_BOUND_M).astype(np.float32)
    return np.flipud(normalized.T)


def write_geotiff_indicators(
    out_path: Path,
    indicators: dict[str, np.ndarray],
    gx_min: int,
    gy_max: int,
    nx: int,
    ny: int,
    mesh_size_m: float,
    crs: CRS,
) -> None:
    """Write a multi-band float32 GeoTIFF with all ratio indicators.

    Each pixel represents one mesh cell (mesh_size_m on each side).
    """
    band_order = [
        "density_z1",
        "density_z2",
        "density_z3",
        "occupancy_z1",
        "occupancy_z2",
        "canopy_height_p95",
    ]
    transform = from_origin(gx_min, gy_max, mesh_size_m, mesh_size_m)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=len(band_order),
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
        nodata=-1,
    ) as dst:
        for i, name in enumerate(band_order, start=1):
            dst.write(indicators[name].astype(np.float32), i)
            dst.set_band_description(i, name)


def save_png(out_path: Path, arr: np.ndarray, title: str, cmap: str) -> None:
    """Render a [0, 1] float array as a quick-look PNG."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(arr, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def array_stats(arr: np.ndarray) -> dict:
    """Summary statistics ignoring zeros (most cells are zero in skewed layers)."""
    nonzero = arr[arr > 0]
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "nonzero_count": int(nonzero.size),
        "nonzero_mean": float(nonzero.mean()) if nonzero.size > 0 else 0.0,
        "nonzero_p50": float(np.percentile(nonzero, 50)) if nonzero.size > 0 else 0.0,
        "nonzero_p95": float(np.percentile(nonzero, 95)) if nonzero.size > 0 else 0.0,
    }


@click.command()
@click.option(
    "--input",
    "input_arg",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input LAS/LAZ file, or directory containing multiple .las/.laz files.",
)
@click.option(
    "--output",
    "output_root",
    default="data/output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output root. A timestamped run_YYYYMMDD_HHMMSS subdirectory is created inside.",
)
@click.option(
    "--crs",
    "override_crs",
    default=None,
    help="Override CRS (e.g. EPSG:6676). Required if LAS has no embedded CRS.",
)
@click.option(
    "--skip-pdal",
    is_flag=True,
    help="Skip PDAL even when *_hag.laz cache is missing (will likely fail).",
)
@click.option(
    "--mesh-size",
    "mesh_size_m",
    default=DEFAULT_GRID_SIZE_M,
    type=float,
    help=f"Mesh cell size in meters (default: {DEFAULT_GRID_SIZE_M}). "
         "Subvoxel size stays at 0.25m regardless.",
)
@click.option(
    "--parallel",
    default=4,
    type=int,
    help="Number of parallel workers for PDAL stage (default: 4).",
)
def main(
    input_arg: Path,
    output_root: Path,
    override_crs: str | None,
    skip_pdal: bool,
    mesh_size_m: float,
    parallel: int,
) -> None:
    """Generate Phase 1 ratio-based preview rasters from one or many LAS files."""
    # メッシュサイズを run ディレクトリ名のサフィックスに (例: 'run_20260522_080000_1m')
    mesh_suffix = f"{mesh_size_m:g}m"
    output_dir = create_run_dir(output_root, suffix=mesh_suffix)
    update_latest_link(output_root, output_dir)
    log.info("Output directory: %s", output_dir)
    log.info("  (also reachable via: %s)", (output_root / "latest").resolve())
    input_paths = resolve_input_paths(input_arg)
    log.info("Found %d input LAS file(s):", len(input_paths))
    for p in input_paths:
        log.info("  - %s", p.name)

    # PDAL stage: ensure cached *_hag.laz next to each input
    if skip_pdal:
        cached_laz_paths = [p.with_name(p.stem + "_hag.laz") for p in input_paths]
        missing = [c for c in cached_laz_paths if not c.exists()]
        if missing:
            raise click.ClickException(
                f"--skip-pdal but missing caches: {[m.name for m in missing]}"
            )
    else:
        cached_laz_paths = ensure_hag_cache(input_paths, override_crs, parallel)

    crs = detect_crs(input_paths[0], override_crs)

    subs = subvoxels_per_grid(mesh_size_m)
    log.info(
        "Mesh: %.2fm x %.2fm; subvoxel: %.2fm; subvoxels per cell side: %d",
        mesh_size_m, mesh_size_m, VOXEL_SIZE_M, subs,
    )

    log.info("Computing global bbox from %d LAZ headers...", len(cached_laz_paths))
    x_min_g, y_min_g, x_max_g, y_max_g = compute_global_bbox(cached_laz_paths)
    gx_min, gy_min, gx_max, gy_max = snap_grid_to_crs_origin(
        x_min_g, x_max_g, y_min_g, y_max_g, mesh_size_m,
    )
    nx = int(round((gx_max - gx_min) / mesh_size_m))
    ny = int(round((gy_max - gy_min) / mesh_size_m))
    log.info(
        "Grid: origin=(%d, %d), cells=%d x %d, ground=%.0f x %.0f m",
        gx_min, gy_min, nx, ny, nx * mesh_size_m, ny * mesh_size_m,
    )

    # サブボクセル bool 配列のサイズを事前に予算チェック (1.5GB を超えたら警告)
    subvoxel_bytes = sum(
        (ny * subs) * (nx * subs) * s.z_subvoxels
        for s in LAYERS if s.name in ("z1", "z2")
    )
    log.info(
        "Subvoxel buffer total size: %.2f GB (z1+z2 bool arrays)",
        subvoxel_bytes / 1e9,
    )
    if subvoxel_bytes > 4e9:
        log.warning(
            "Subvoxel buffer exceeds 4GB. Consider increasing --mesh-size."
        )

    log.info("Streaming aggregation across %d LAZ files...", len(cached_laz_paths))
    counts, subvoxel_sets, z3_x, z3_y, z3_h = _streaming_accumulator(
        cached_laz_paths, gx_min, gy_min, nx, ny, mesh_size_m,
    )

    log.info("Computing ratio indicators...")
    z0, z1, z2, z3 = counts["z0"], counts["z1"], counts["z2"], counts["z3"]
    total = z0 + z1 + z2 + z3
    indicators: dict[str, np.ndarray] = {
        "density_z1": safe_ratio(z1, z0 + z1),
        "density_z2": safe_ratio(z2, z0 + z1 + z2),
        "density_z3": safe_ratio(z3, total),
    }

    log.info("Computing occupancy for z1, z2...")
    indicators["occupancy_z1"] = occupancy_from_subvoxel_set(
        subvoxel_sets["z1"], subs, LAYERS[1].z_subvoxels,
    )
    indicators["occupancy_z2"] = occupancy_from_subvoxel_set(
        subvoxel_sets["z2"], subs, LAYERS[2].z_subvoxels,
    )
    # 解放: もう使わない
    del subvoxel_sets

    log.info("Computing canopy_height_p95 from %d z3 points...", z3_x.size)
    indicators["canopy_height_p95"] = canopy_p95_from_z3(
        z3_x, z3_y, z3_h, gx_min, gy_min, nx, ny, mesh_size_m,
    )
    del z3_x, z3_y, z3_h

    log.info("Writing PNG previews...")
    cmap_for: dict[str, str] = {
        "density_z1": "viridis",
        "density_z2": "viridis",
        "density_z3": "viridis",
        "occupancy_z1": "magma",
        "occupancy_z2": "magma",
        "canopy_height_p95": "cividis",
    }
    for name, arr in indicators.items():
        save_png(
            output_dir / f"preview_{name}.png",
            arr,
            title=f"{name}  (range 0.0 - 1.0)  [mesh={mesh_size_m}m]",
            cmap=cmap_for[name],
        )

    log.info("Writing 6-band float32 GeoTIFF...")
    write_geotiff_indicators(
        output_dir / "preview_indicators.tif",
        indicators,
        gx_min=gx_min,
        gy_max=gy_max,
        nx=nx,
        ny=ny,
        mesh_size_m=mesh_size_m,
        crs=crs,
    )

    meta = {
        "input": str(input_arg),
        "input_files": [p.name for p in input_paths],
        "crs": crs.to_string(),
        "mesh_size_m": mesh_size_m,
        "voxel_size_m": VOXEL_SIZE_M,
        "subvoxels_per_grid_side": subs,
        "grid_origin_xy": [gx_min, gy_min],
        "grid_size_cells": [nx, ny],
        "ground_extent_m": [nx * mesh_size_m, ny * mesh_size_m],
        "layer_point_counts": {name: int(arr.sum()) for name, arr in counts.items()},
        "indicators": {name: array_stats(arr) for name, arr in indicators.items()},
    }
    (output_dir / "preview_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote outputs to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
