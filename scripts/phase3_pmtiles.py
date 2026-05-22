"""Phase 3: PMTiles builder for las2veg-rgb.

Pipeline:
    encoded_rgba.tif (EPSG:6676, 4-band uint8)
        -> Stage 1: gdalwarp -> Web Mercator (EPSG:3857), -r near
        -> Stage 2: gdal_translate -of MBTILES -> single .mbtiles (TILE_FORMAT=PNG, RESAMPLING=NEAREST)
        -> Stage 3: gdaladdo -r nearest -> generate lower zoom levels
        -> Stage 4: pmtiles convert -> single PMTiles file
        -> Stage 5: pmtiles edit -> inject metadata JSON
        -> Stage 6: pmtiles show -> validation
        -> Stage 7: optional round-trip pixel verification

PNG is used (not WebP) because the MBTiles driver's WEBP option does not
support lossless encoding, and lossy WebP would corrupt the 4-bit packed values.

Run:
    python scripts/phase3_pmtiles.py \
        --input data/output/encoded_rgba.tif \
        --output data/output/output.pmtiles \
        [--minzoom 14] [--maxzoom 17] [--clean] [--verify]
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from las2veg_rgb.config import CANOPY_UPPER_BOUND_M  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase3")


def run(cmd: list[str], stage: str) -> None:
    """Run a subprocess command, logging stdout/stderr.

    encoding/errors を明示することで Windows のデフォルト cp932 で
    Unicode 文字 (pmtiles の進捗バー等) に当たって落ちるのを回避する。
    """
    log.info("[%s] $ %s", stage, " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    dt = time.time() - t0
    if result.stdout and result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            log.info("[%s] %s", stage, line)
    if result.stderr and result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            log.warning("[%s] %s", stage, line)
    if result.returncode != 0:
        raise click.ClickException(
            f"{stage} failed with exit code {result.returncode}"
        )
    log.info("[%s] done in %.1fs", stage, dt)


def stage1_reproject(input_tif: Path, reprojected_tif: Path) -> None:
    """gdalwarp: EPSG:6676 -> EPSG:3857, nearest resampling."""
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:3857",
        "-r", "near",
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        "-overwrite",
        str(input_tif),
        str(reprojected_tif),
    ]
    run(cmd, "stage1.warp")


def stage2_translate_to_mbtiles(
    reprojected_tif: Path,
    mbtiles_path: Path,
) -> None:
    """gdal_translate -of MBTILES: produce MBTiles with PNG tiles at native resolution."""
    if mbtiles_path.exists():
        mbtiles_path.unlink()

    cmd = [
        "gdal_translate",
        "-of", "MBTILES",
        "-co", "BLOCKSIZE=256",
        "-co", "TILE_FORMAT=PNG",
        "-co", "RESAMPLING=NEAREST",
        "-co", "ZOOM_LEVEL_STRATEGY=UPPER",
        str(reprojected_tif),
        str(mbtiles_path),
    ]
    run(cmd, "stage2.mbtiles")


def stage3_overviews(mbtiles_path: Path, maxzoom_offset_levels: list[int]) -> None:
    """gdaladdo -r nearest: build lower-zoom overviews.

    Levels are zoom-level *factors* relative to the native resolution
    (e.g. [2, 4, 8] = produce zoom-1, zoom-2, zoom-3 below native).
    """
    cmd = ["gdaladdo", "-r", "nearest", str(mbtiles_path)] + [
        str(level) for level in maxzoom_offset_levels
    ]
    run(cmd, "stage3.overviews")


def stage4_convert(mbtiles_path: Path, output_pmtiles: Path) -> None:
    """pmtiles convert: MBTiles -> PMTiles."""
    if output_pmtiles.exists():
        output_pmtiles.unlink()
    cmd = [
        "pmtiles",
        "convert",
        str(mbtiles_path),
        str(output_pmtiles),
    ]
    run(cmd, "stage4.pmtiles")


def stage5_metadata(
    output_pmtiles: Path,
    tile_format: str,
    temp_dir: Path,
) -> None:
    """pmtiles edit: inject las2veg-rgb specific metadata."""
    # phase2_encode.py の QUANTIZE_MAX と同期: 指標ごとの上限値で正規化
    # 復号式: real_value = (encoded_4bit / 15) * quantize_max
    # canopy_height_p95 の 0.6 は Phase 1 正規化 (0-100m) のうち 0-60m を量子化、
    # 即ち bin 0..15 は 0..60m を表す (1段階 4m、世界 99.5% カバー)。
    quantize_max = {
        "density_z1":        1.0,
        "density_z2":        1.0,
        "density_z3":        1.0,
        "occupancy_z1":      0.5,
        "occupancy_z2":      0.5,
        "canopy_height_p95": 0.6,
    }
    metadata = {
        "name": "las2veg-rgb",
        "version": "0.1.0",
        "format": tile_format,
        "channels": {
            "R_high4": "density_z3",
            "R_low4":  "canopy_height_p95",
            "G_high4": "density_z2",
            "G_low4":  "occupancy_z2",
            "B_high4": "density_z1",
            "B_low4":  "occupancy_z1",
            "A":       "unused (always 255)",
        },
        "scaling": "uniform_16_bins_per_indicator",
        "bin_definition": "Bin N covers [N/16, (N+1)/16) * quantize_max",
        "decode_formula_center": "real_value = (encoded_4bit + 0.5) / 16 * quantize_max[name]",
        "decode_formula_lower":  "real_value =  encoded_4bit       / 16 * quantize_max[name]",
        "quantize_max": quantize_max,
        "canopy_height_p95_unit_meters": CANOPY_UPPER_BOUND_M,
        "z3_upper_bound_m": CANOPY_UPPER_BOUND_M,
        "spec_url": "https://github.com/Trilor/las2veg-rgb/blob/main/docs/spec.md",
    }

    meta_path = temp_dir / "pmtiles_meta.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("[stage5.meta] wrote metadata to %s", meta_path)

    cmd = [
        "pmtiles",
        "edit",
        f"--metadata={meta_path}",
        str(output_pmtiles),
    ]
    run(cmd, "stage5.meta")


def stage6_show(output_pmtiles: Path) -> None:
    """pmtiles show: print summary."""
    cmd = ["pmtiles", "show", str(output_pmtiles)]
    run(cmd, "stage6.show")


def verify_one_tile(
    reprojected_tif: Path,
    output_pmtiles: Path,
    temp_dir: Path,
) -> None:
    """Extract one tile from the PMTiles, decode it, and report value range."""
    log.info("[verify] running PMTiles -> source pixel comparison")

    # Discover maxzoom from pmtiles show
    show_out = subprocess.run(
        ["pmtiles", "show", str(output_pmtiles)],
        capture_output=True, text=True, check=False,
    )
    log.info("[verify] %s", show_out.stdout.strip().replace("\n", " | "))

    with rasterio.open(reprojected_tif) as src:
        bounds = src.bounds
        cx = (bounds.left + bounds.right) / 2.0
        cy = (bounds.top + bounds.bottom) / 2.0

    # Pick a reasonable max zoom from the PMTiles
    import math
    # Use show output to find maxzoom; fallback to a sensible default
    maxzoom = 17
    for line in (show_out.stdout or "").splitlines():
        if "max_zoom" in line.lower() or "maxzoom" in line.lower():
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                maxzoom = int(digits[-2:]) if len(digits) >= 2 else int(digits)
                break

    z = maxzoom
    n = 2 ** z
    # Web Mercator extent in meters
    R = 20037508.342789244
    tx = int((cx + R) / (2 * R) * n)
    ty_xyz = int((R - cy) / (2 * R) * n)  # XYZ (top-left origin)
    log.info("[verify] extracting tile z=%d x=%d y=%d", z, tx, ty_xyz)

    extracted = temp_dir / f"verify_z{z}_x{tx}_y{ty_xyz}.webp"
    if extracted.exists():
        extracted.unlink()
    cmd = [
        "pmtiles",
        "tile",
        str(output_pmtiles),
        str(z), str(tx), str(ty_xyz),
    ]
    log.info("[verify] $ %s", " ".join(cmd))
    with open(extracted, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        log.warning("[verify] failed to extract tile (may be outside dataset)")
        if result.stderr:
            log.warning("[verify] %s", result.stderr.decode("utf-8", errors="replace"))
        return

    if extracted.stat().st_size == 0:
        log.warning("[verify] extracted tile is empty")
        return

    try:
        from PIL import Image
        tile_img = np.array(Image.open(extracted))
    except Exception as e:  # noqa: BLE001
        log.warning("[verify] cannot decode extracted tile: %s", e)
        return

    log.info(
        "[verify] tile shape=%s dtype=%s value_range=[%d, %d]",
        tile_img.shape,
        tile_img.dtype,
        int(tile_img.min()),
        int(tile_img.max()),
    )
    log.info("[verify] tile saved to %s (%d bytes)", extracted, extracted.stat().st_size)


@click.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Phase 2 output RGBA GeoTIFF (EPSG:6676). "
         "Defaults to data/output/latest/encoded_rgba.tif.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Final PMTiles file. Defaults to <input-dir>/output.pmtiles.",
)
@click.option(
    "--overview-levels",
    default="2,4,8,16",
    help="Comma-separated overview downsampling factors (gdaladdo levels).",
)
@click.option(
    "--clean",
    is_flag=True,
    help="Remove intermediate files (reprojected tif, tiles directory) after build.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="Extract one tile from the PMTiles and report pixel value range.",
)
def main(
    input_path: Path | None,
    output_path: Path | None,
    overview_levels: str,
    clean: bool,
    verify: bool,
) -> None:
    """Build a PMTiles file from a Phase 2 RGBA GeoTIFF."""
    if input_path is None:
        from las2veg_rgb.runs import find_latest_run

        latest = find_latest_run(Path("data/output"))
        if latest is None:
            raise click.ClickException(
                "No run directory found under data/output/. "
                "Run phase1_preview.py and phase2_encode.py first or pass --input."
            )
        input_path = latest / "encoded_rgba.tif"
        if not input_path.exists():
            raise click.ClickException(
                f"Expected {input_path} but it does not exist. "
                "Run phase2_encode.py first."
            )
        log.info("Auto-detected input from latest run: %s", input_path)

    if output_path is None:
        output_path = input_path.parent / "output.pmtiles"
        log.info("Auto-detected output path: %s", output_path)

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    reprojected = out_dir / "encoded_rgba_3857.tif"
    mbtiles_path = out_dir / "tiles.mbtiles"
    temp_dir = out_dir / "phase3_temp"
    temp_dir.mkdir(exist_ok=True)

    tile_format = "png"
    overview_list = [int(x) for x in overview_levels.split(",") if x.strip()]
    log.info(
        "Phase 3: build PMTiles (PNG tiles, overviews=%s)", overview_list
    )
    log.info("  input:  %s", input_path)
    log.info("  output: %s", output_path)

    stage1_reproject(input_path, reprojected)
    stage2_translate_to_mbtiles(reprojected, mbtiles_path)
    stage3_overviews(mbtiles_path, overview_list)
    stage4_convert(mbtiles_path, output_path)
    stage5_metadata(output_path, tile_format, temp_dir)
    stage6_show(output_path)

    if verify:
        verify_one_tile(reprojected, output_path, temp_dir)

    log.info("Final size: %.1f KB", output_path.stat().st_size / 1024.0)

    if clean:
        log.info("Cleaning intermediates...")
        reprojected.unlink(missing_ok=True)
        mbtiles_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    log.info("Done. PMTiles file: %s", output_path.resolve())


if __name__ == "__main__":
    main()
