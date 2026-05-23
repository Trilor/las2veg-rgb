"""Canopy height max vs P95 比較スクリプト。

PDAL writers.gdal で z3 (HAG 2-100m) の max を集計したラスターを生成し、
既存の P95 結果 (preview_indicators.tif の canopy_height_p95 バンド) と比較する。

目的:
  1. 処理時間の比較 (PDAL C++ max vs scipy lambda P95)
  2. 値の差分布 (max - P95 はどれくらいか、外れ値の影響を可視化)
  3. 空間分布の目視確認

Phase 1 の P95 計算結果には触れず、独立した比較として実行。

Run:
    python scripts/canopy_max_compare.py
    python scripts/canopy_max_compare.py --run-dir data/output/run_20260523_022215_2.5m
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from las2veg_rgb.config import CANOPY_UPPER_BOUND_M  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("max_compare")


def find_run_dir(run_dir: Path | None) -> Path:
    """run-dir 引数が None なら latest を返す。"""
    if run_dir is not None:
        return run_dir
    from las2veg_rgb.runs import find_latest_run
    latest = find_latest_run(Path("data/output"))
    if latest is None:
        raise click.ClickException("No run directory under data/output/.")
    log.info("Using latest run: %s", latest)
    return latest


def read_run_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "preview_meta.json"
    if not meta_path.exists():
        raise click.ClickException(f"Missing {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def make_pdal_pipeline(
    laz_path: Path, out_tif: Path, resolution: float,
    bounds_xy: tuple[float, float, float, float],
    override_srs: str = "EPSG:6676",
) -> dict:
    """PDAL pipeline JSON: HAG[2:100] の max を集計し GeoTIFF 出力。

    bounds_xy = (xmin, xmax, ymin, ymax) でグリッド範囲を統一する。
    override_srs: _hag.laz は CRS を保持しないので明示指定が必要。
    """
    xmin, xmax, ymin, ymax = bounds_xy
    return {
        "pipeline": [
            {"type": "readers.las",
             "filename": str(laz_path),
             "override_srs": override_srs},
            {"type": "filters.range",
             "limits": f"HeightAboveGround[2:{CANOPY_UPPER_BOUND_M}]"},
            {"type": "writers.gdal",
             "filename": str(out_tif),
             "resolution": resolution,
             "output_type": "max",
             "dimension": "HeightAboveGround",
             "data_type": "float32",
             "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
             "nodata": -9999},
        ]
    }


def run_pdal_max(
    laz_paths: list[Path], out_dir: Path, resolution: float,
    bounds_xy: tuple[float, float, float, float],
) -> list[Path]:
    """各 LAZ について PDAL pipeline を実行し、max ラスターを生成。

    既存の *_max.tif があれば再利用 (PDAL は決定的なのでキャッシュ安全)。
    """
    import pdal

    tif_paths: list[Path] = []
    for laz in laz_paths:
        out_tif = out_dir / f"{laz.stem}_max.tif"
        if out_tif.exists():
            log.info("  [skip PDAL] cached: %s", out_tif.name)
            tif_paths.append(out_tif)
            continue
        pipeline_json = make_pdal_pipeline(laz, out_tif, resolution, bounds_xy)
        t0 = time.time()
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        try:
            count = pipeline.execute()
            log.info("  [PDAL max] %s: %d points in %.1fs",
                     laz.name, count, time.time() - t0)
            tif_paths.append(out_tif)
        except Exception as e:  # noqa: BLE001
            log.warning("  [PDAL max FAIL] %s: %s", laz.name, e)
    return tif_paths


def merge_rasters_max(tif_paths: list[Path], out_path: Path) -> Path:
    """複数の max ラスターを「最大値合成」でマージ。

    rasterio.merge は デフォルト first だが、method='max' で重複領域の max を取る。
    """
    log.info("Merging %d max rasters with method='max'...", len(tif_paths))
    srcs = [rasterio.open(p) for p in tif_paths]
    try:
        merged, transform = rio_merge(srcs, method="max", nodata=-9999)
        profile = srcs[0].profile
        profile.update(
            height=merged.shape[1], width=merged.shape[2],
            transform=transform, count=1, dtype="float32",
            nodata=-9999, compress="deflate",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(merged[0], 1)
            dst.set_band_description(1, "canopy_height_max")
    finally:
        for s in srcs:
            s.close()
    return out_path


def reproject_to_grid(
    src_path: Path, out_path: Path, ref_path: Path,
) -> Path:
    """src を ref のグリッド (transform + shape) に合わせて再投影。"""
    cmd = [
        "gdalwarp",
        "-of", "GTiff",
        "-tr",
        # ref から解像度取得
        *str(rasterio.open(ref_path).transform[0]).split(),
        str(abs(rasterio.open(ref_path).transform[4])),
        "-te",  # bbox を ref に合わせる
        str(rasterio.open(ref_path).bounds.left),
        str(rasterio.open(ref_path).bounds.bottom),
        str(rasterio.open(ref_path).bounds.right),
        str(rasterio.open(ref_path).bounds.top),
        "-r", "near",
        "-srcnodata", "-9999",
        "-dstnodata", "-9999",
        "-overwrite",
        str(src_path), str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True,
                   encoding="utf-8", errors="replace")
    return out_path


def compare_p95_vs_max(
    p95_tif: Path, max_tif: Path, output_dir: Path,
) -> dict:
    """P95 と max を読み込み、差分の統計と散布図を出力。

    P95 は preview_indicators.tif の band 6 (0-1 正規化、QUANTIZE_MAX=100 で
    実際は HAG meter)。読み込み時に *100 して meter に戻す。
    max は PDAL writers.gdal の生 HAG meter 出力。
    """
    log.info("Reading P95 from %s", p95_tif)
    with rasterio.open(p95_tif) as src:
        # band 6 = canopy_height_p95 (0-1 で 100m 正規化)
        p95_norm = src.read(6).astype(np.float32)
        # NaN ベース nodata なら NaN のまま、-1 なら mask 変換
        if src.nodata is not None and not np.isnan(src.nodata):
            p95_norm = np.where(p95_norm == src.nodata, np.nan, p95_norm)
    p95_m = p95_norm * CANOPY_UPPER_BOUND_M  # 0-1 → 0-100m

    log.info("Reading max from %s", max_tif)
    with rasterio.open(max_tif) as src:
        max_m = src.read(1).astype(np.float32)
        max_m = np.where(max_m == src.nodata, np.nan, max_m)

    # サイズが一致するか
    if p95_m.shape != max_m.shape:
        raise click.ClickException(
            f"Shape mismatch: P95 {p95_m.shape} vs max {max_m.shape}.\n"
            "PDAL の bounds とメッシュサイズを Phase 1 と完全に揃える必要があります。"
        )

    # 両方有効なセルだけで比較
    valid = ~(np.isnan(p95_m) | np.isnan(max_m))
    p95_v = p95_m[valid]
    max_v = max_m[valid]
    diff = max_v - p95_v  # 通常 >= 0

    log.info("Compared %d valid cells", p95_v.size)

    stats = {
        "n_cells_p95_valid": int((~np.isnan(p95_m)).sum()),
        "n_cells_max_valid": int((~np.isnan(max_m)).sum()),
        "n_cells_both_valid": int(valid.sum()),
        "p95_mean_m": float(p95_v.mean()),
        "max_mean_m": float(max_v.mean()),
        "diff_mean_m": float(diff.mean()),
        "diff_median_m": float(np.median(diff)),
        "diff_p95_m": float(np.percentile(diff, 95)),
        "diff_max_m": float(diff.max()),
        "diff_min_m": float(diff.min()),  # 負なら異常 (max < P95 はあり得ない)
        "n_diff_negative": int((diff < -0.01).sum()),  # 計算誤差以上に負の差
    }

    log.info("")
    log.info("=== Statistics ===")
    for k, v in stats.items():
        log.info("  %-25s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # 散布図 (P95 vs max)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=120)
    axes[0].scatter(p95_v, max_v, s=2, alpha=0.05, color="steelblue")
    lim_max = max(p95_v.max(), max_v.max()) * 1.05
    axes[0].plot([0, lim_max], [0, lim_max], "r--", label="y = x (max == P95)")
    axes[0].set_xlabel("canopy_height_p95 (m)")
    axes[0].set_ylabel("canopy_height_max (m)")
    axes[0].set_title("P95 vs max (上にあるほど max > P95)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 差分のヒストグラム
    bins = np.linspace(0, np.percentile(diff, 99), 50)
    axes[1].hist(diff, bins=bins, color="orange", edgecolor="black")
    axes[1].axvline(stats["diff_mean_m"], color="red", linestyle="--",
                    label=f"mean = {stats['diff_mean_m']:.2f}m")
    axes[1].axvline(stats["diff_median_m"], color="blue", linestyle="--",
                    label=f"median = {stats['diff_median_m']:.2f}m")
    axes[1].set_xlabel("max - P95 (m)")
    axes[1].set_ylabel("# cells")
    axes[1].set_title("Distribution of (max - P95)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Canopy height: P95 vs max comparison")
    fig.tight_layout()
    out_png = output_dir / "canopy_p95_vs_max.png"
    fig.savefig(out_png)
    plt.close(fig)
    log.info("scatter+hist PNG: %s", out_png)

    # CSV
    out_csv = output_dir / "canopy_p95_vs_max_stats.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in stats.items():
            f.write(f"{k},{v}\n")
    log.info("stats CSV: %s", out_csv)

    return stats


@click.command()
@click.option(
    "--run-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Run directory (defaults to latest).",
)
@click.option(
    "--input-laz-dir",
    default="data/input/kamiide",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing *_hag.laz files.",
)
@click.option(
    "--remove-intermediate",
    is_flag=True,
    help="Remove per-tile *_max.tif files after merge (default: keep for reuse).",
)
def main(run_dir: Path | None, input_laz_dir: Path, remove_intermediate: bool) -> None:
    run_dir = find_run_dir(run_dir)
    meta = read_run_metadata(run_dir)
    mesh = meta["mesh_size_m"]
    gx_min, gy_min = meta["grid_origin_xy"]
    nx, ny = meta["grid_size_cells"]
    gx_max = gx_min + nx * mesh
    gy_max = gy_min + ny * mesh
    bounds = (gx_min, gx_max, gy_min, gy_max)

    log.info("Run: %s (mesh=%sm, grid=%dx%d)", run_dir.name, mesh, nx, ny)
    log.info("Grid bounds: x=[%s, %s] y=[%s, %s]", *bounds)

    # 全 _hag.laz を集める
    laz_paths = sorted([p for p in input_laz_dir.iterdir()
                        if p.name.endswith("_hag.laz")])
    log.info("Found %d hag.laz files in %s", len(laz_paths), input_laz_dir)
    if not laz_paths:
        raise click.ClickException(
            f"No *_hag.laz files in {input_laz_dir}. "
            "Run phase1_preview.py first to generate PDAL caches."
        )

    out_dir = run_dir / "compare_max"
    out_dir.mkdir(exist_ok=True)
    tmp_dir = out_dir / "_tile_max"
    tmp_dir.mkdir(exist_ok=True)

    log.info("=== PDAL writers.gdal max for %d tiles ===", len(laz_paths))
    t0_total = time.time()
    tif_paths = run_pdal_max(laz_paths, tmp_dir, mesh, bounds)
    pdal_total = time.time() - t0_total
    log.info("PDAL max total: %.1fs (%.2f s/tile avg)", pdal_total, pdal_total / max(1, len(laz_paths)))

    log.info("")
    log.info("=== Merging tile max rasters ===")
    merged_path = out_dir / "canopy_max_merged.tif"
    t0 = time.time()
    merge_rasters_max(tif_paths, merged_path)
    log.info("Merge took: %.1fs", time.time() - t0)

    log.info("")
    log.info("=== Aligning merged max raster to P95 grid ===")
    p95_path = run_dir / "preview_indicators.tif"
    if not p95_path.exists():
        raise click.ClickException(f"P95 source missing: {p95_path}")
    aligned_path = out_dir / "canopy_max_aligned.tif"
    # P95 のグリッドに合わせて max を再投影
    with rasterio.open(p95_path) as ref:
        ref_bounds = ref.bounds
        ref_res = abs(ref.transform[0])
    cmd = [
        "gdalwarp",
        "-of", "GTiff",
        "-tr", str(ref_res), str(ref_res),
        "-te", str(ref_bounds.left), str(ref_bounds.bottom),
                str(ref_bounds.right), str(ref_bounds.top),
        "-r", "near",
        "-srcnodata", "-9999",
        "-dstnodata", "-9999",
        "-overwrite",
        str(merged_path), str(aligned_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True,
                   encoding="utf-8", errors="replace")
    log.info("Aligned max: %s", aligned_path)

    log.info("")
    log.info("=== Comparison with P95 ===")
    compare_p95_vs_max(p95_path, aligned_path, out_dir)

    # 中間ファイル削除 (デフォルトは保持して再利用可能に)
    if remove_intermediate:
        log.info("Removing intermediate per-tile rasters...")
        for p in tif_paths:
            p.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    log.info("")
    log.info("Total PDAL max compute time: %.1fs", pdal_total)
    log.info("(compare with: P95 in Phase 1 took several minutes)")
    log.info("Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
