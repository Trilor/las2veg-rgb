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
    NOISE_CLASSIFICATION,
    SUBVOXELS_PER_M,
    VOXEL_SIZE_M,
)
from las2veg_rgb.layers import LAYERS, LayerSpec  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")


def run_pdal_pipeline(input_path: Path, cached_laz: Path, override_crs: str | None) -> None:
    """Run PDAL: drop noise, SMRF ground classify, HAG. Cache to LAZ."""
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
    log.info("Running PDAL pipeline (SMRF + HAG)...")
    t0 = time.time()
    pipeline = pdal.Pipeline(pipeline_json)
    count = pipeline.execute()
    log.info("PDAL processed %d points in %.1fs", count, time.time() - t0)


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
    x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[int, int, int, int]:
    """Snap bbox to a 1m grid anchored at the CRS absolute origin (0,0)."""
    gx_min = math.floor(x_min)
    gy_min = math.floor(y_min)
    gx_max = math.ceil(x_max)
    gy_max = math.ceil(y_max)
    return gx_min, gy_min, gx_max, gy_max


def count_points_per_cell(
    x: np.ndarray,
    y: np.ndarray,
    hag: np.ndarray,
    spec: LayerSpec,
    gx_min: int,
    gy_min: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Return (H, W) int32 count of points belonging to a layer per 1m cell."""
    mask = (hag >= spec.z_min) & (hag < spec.z_max)
    log.info("  %s [%.2f, %.2f): %d points", spec.name, spec.z_min, spec.z_max, int(mask.sum()))

    x_edges = np.arange(gx_min, gx_min + width + 1, dtype=np.float64)
    y_edges = np.arange(gy_min, gy_min + height + 1, dtype=np.float64)

    count, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
    return np.flipud(count.T.astype(np.int32))


def compute_occupancy(
    x: np.ndarray,
    y: np.ndarray,
    hag: np.ndarray,
    spec: LayerSpec,
    gx_min: int,
    gy_min: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Return (H, W) float32 subvoxel-occupancy in [0, 1] for the given layer."""
    mask = (hag >= spec.z_min) & (hag < spec.z_max)
    if not mask.any():
        return np.zeros((height, width), dtype=np.float32)

    nx_sub = width * SUBVOXELS_PER_M
    ny_sub = height * SUBVOXELS_PER_M
    nz_sub = spec.z_subvoxels

    sx_edges = np.linspace(gx_min, gx_min + width, nx_sub + 1, dtype=np.float64)
    sy_edges = np.linspace(gy_min, gy_min + height, ny_sub + 1, dtype=np.float64)
    sz_edges = np.linspace(spec.z_min, spec.z_max, nz_sub + 1, dtype=np.float64)

    voxel_counts, _ = np.histogramdd(
        np.stack([x[mask], y[mask], hag[mask]], axis=1),
        bins=[sx_edges, sy_edges, sz_edges],
    )
    occupied = (voxel_counts > 0).astype(np.float32)
    occupied_per_cell = occupied.reshape(
        width, SUBVOXELS_PER_M, height, SUBVOXELS_PER_M, nz_sub
    ).sum(axis=(1, 3, 4))
    total = float(SUBVOXELS_PER_M * SUBVOXELS_PER_M * nz_sub)
    occupancy = (occupied_per_cell / total).astype(np.float32)
    return np.flipud(occupancy.T)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise numerator/denominator; returns 0 where denominator is 0."""
    out = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=out, where=(denominator > 0))
    return out.astype(np.float32)


def compute_canopy_height_p95(
    x: np.ndarray,
    y: np.ndarray,
    hag: np.ndarray,
    gx_min: int,
    gy_min: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Compute per-cell P95 canopy height (z3 points only), normalized by 100m.

    Returns (H, W) float32 in [0, 1]. Cells without z3 points get 0.
    Uses scipy.stats.binned_statistic_2d for vectorized percentile over 1m bins.
    """
    z3_mask = (hag >= 2.0) & (hag < CANOPY_UPPER_BOUND_M)
    log.info("  canopy P95 source: %d z3 points", int(z3_mask.sum()))

    if not z3_mask.any():
        return np.zeros((height, width), dtype=np.float32)

    x_edges = np.arange(gx_min, gx_min + width + 1, dtype=np.float64)
    y_edges = np.arange(gy_min, gy_min + height + 1, dtype=np.float64)

    stat, _, _, _ = binned_statistic_2d(
        x[z3_mask],
        y[z3_mask],
        hag[z3_mask],
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
    width: int,
    height: int,
    crs: CRS,
) -> None:
    """Write a multi-band float32 GeoTIFF with all ratio indicators."""
    band_order = [
        "density_z1",
        "density_z2",
        "density_z3",
        "occupancy_z1",
        "occupancy_z2",
        "canopy_height_p95",
    ]
    transform = from_origin(gx_min, gy_max, 1.0, 1.0)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
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
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input LAS or LAZ file.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for previews and GeoTIFFs.",
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
    help="Skip PDAL stage and reuse the cached *_hag.laz next to the input.",
)
def main(input_path: Path, output_dir: Path, override_crs: str | None, skip_pdal: bool) -> None:
    """Generate Phase 1 ratio-based preview rasters from a LAS file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_laz = input_path.with_name(input_path.stem + "_hag.laz")

    if not skip_pdal or not cached_laz.exists():
        run_pdal_pipeline(input_path, cached_laz, override_crs)
    else:
        log.info("Reusing cached HAG LAZ: %s", cached_laz)

    crs = detect_crs(input_path, override_crs)

    log.info("Reading cached LAZ: %s", cached_laz)
    las = laspy.read(str(cached_laz))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    try:
        hag = np.asarray(las["HeightAboveGround"], dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(
            "HeightAboveGround dimension not found in cached LAZ. "
            "Re-run without --skip-pdal."
        ) from e
    log.info("Loaded %d points", x.size)

    gx_min, gy_min, gx_max, gy_max = snap_grid_to_crs_origin(
        float(x.min()), float(x.max()), float(y.min()), float(y.max())
    )
    width = gx_max - gx_min
    height = gy_max - gy_min
    log.info("Grid: origin=(%d, %d), size=%d x %d (1m cells)", gx_min, gy_min, width, height)

    counts: dict[str, np.ndarray] = {}
    log.info("Counting points per cell per layer...")
    for spec in LAYERS:
        counts[spec.name] = count_points_per_cell(
            x, y, hag, spec, gx_min, gy_min, width, height
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
    indicators["occupancy_z1"] = compute_occupancy(
        x, y, hag, LAYERS[1], gx_min, gy_min, width, height
    )
    indicators["occupancy_z2"] = compute_occupancy(
        x, y, hag, LAYERS[2], gx_min, gy_min, width, height
    )

    log.info("Computing canopy_height_p95...")
    indicators["canopy_height_p95"] = compute_canopy_height_p95(
        x, y, hag, gx_min, gy_min, width, height
    )

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
            title=f"{name}  (range 0.0 - 1.0)",
            cmap=cmap_for[name],
        )

    log.info("Writing 6-band float32 GeoTIFF...")
    write_geotiff_indicators(
        output_dir / "preview_indicators.tif",
        indicators,
        gx_min=gx_min,
        gy_max=gy_max,
        width=width,
        height=height,
        crs=crs,
    )

    meta = {
        "input": str(input_path),
        "crs": crs.to_string(),
        "grid_origin_xy": [gx_min, gy_min],
        "grid_size_cells": [width, height],
        "voxel_size_m": VOXEL_SIZE_M,
        "subvoxels_per_m": SUBVOXELS_PER_M,
        "layer_point_counts": {name: int(arr.sum()) for name, arr in counts.items()},
        "indicators": {name: array_stats(arr) for name, arr in indicators.items()},
    }
    (output_dir / "preview_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote outputs to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
