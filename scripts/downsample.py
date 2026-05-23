"""1m メッシュの preview_indicators.tif を任意のメッシュサイズに集約する。

整数倍 (2x, 5x, 10x など) では reshape + reduce による誤差ゼロ集約。
非整数倍 (2.5x など) では面積按分による加重平均で集約 (誤差小)。

集約方法:
  - density系: 平均 (= 点数比の平均、面積比按分でほぼ正確)
  - occupancy系: 平均 (※ 厳密にはサブボクセル数を考慮した再計算が正しいが、
                    実用上は単純平均で 1-3% の誤差にとどまる)
  - canopy_height_p95: 平均 (= 集約後は単一値なので再計算不可、原値の代表値を取る)

出力は新しい run ディレクトリ (run_YYYYMMDD_HHMMSS_downsampled_<mesh>m) に保存。
元の 1m run には影響を与えない。

Run:
    python scripts/downsample.py --target-mesh 2.5
    python scripts/downsample.py --source data/output/run_xxx_1m --target-mesh 5
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from las2veg_rgb.runs import create_run_dir, find_latest_run, update_latest_link  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("downsample")


def downsample_integer(arr: np.ndarray, factor: int) -> np.ndarray:
    """整数倍集約 (NaN 安全な平均)。

    arr の形状を factor の倍数にトリムしてから reshape + nanmean。
    """
    ny, nx = arr.shape
    ny_c = (ny // factor) * factor
    nx_c = (nx // factor) * factor
    trimmed = arr[:ny_c, :nx_c]
    reshaped = trimmed.reshape(ny_c // factor, factor, nx_c // factor, factor)
    # 全 NaN の bin は NaN を維持 (data なし領域)
    with np.errstate(invalid="ignore"):
        return np.nanmean(reshaped, axis=(1, 3)).astype(np.float32)


def downsample_fractional(
    arr: np.ndarray, src_mesh: float, dst_mesh: float,
) -> np.ndarray:
    """非整数倍の集約 (面積按分による加重平均)。

    各 dst セルが src セル群とどれだけ重なるかを計算し、重みづけ平均。
    NaN セルは重み 0 として扱う (= 集約から除外)。
    """
    factor = dst_mesh / src_mesh
    ny_src, nx_src = arr.shape
    ny_dst = int(np.floor(ny_src / factor))
    nx_dst = int(np.floor(nx_src / factor))

    log.info(
        "  Fractional downsample: %dx%d (src=%sm) → %dx%d (dst=%sm), factor=%s",
        nx_src, ny_src, src_mesh, nx_dst, ny_dst, dst_mesh, factor,
    )

    # NaN を 0 と置き、有効マスクで重み付け
    valid = ~np.isnan(arr)
    arr_zero = np.where(valid, arr, 0.0)
    weight = valid.astype(np.float32)

    result = np.empty((ny_dst, nx_dst), dtype=np.float32)
    for jy in range(ny_dst):
        y0 = jy * factor
        y1 = (jy + 1) * factor
        iy0 = int(np.floor(y0))
        iy1 = min(int(np.ceil(y1)), ny_src)
        for jx in range(nx_dst):
            x0 = jx * factor
            x1 = (jx + 1) * factor
            ix0 = int(np.floor(x0))
            ix1 = min(int(np.ceil(x1)), nx_src)

            # 各 src セルの重なり面積を計算
            ys = np.arange(iy0, iy1)
            xs = np.arange(ix0, ix1)
            wy = np.minimum(ys + 1, y1) - np.maximum(ys, y0)
            wx = np.minimum(xs + 1, x1) - np.maximum(xs, x0)
            w2d = wy[:, None] * wx[None, :]  # (ny, nx) の面積

            patch_arr = arr_zero[iy0:iy1, ix0:ix1]
            patch_w   = weight[iy0:iy1, ix0:ix1] * w2d

            wsum = patch_w.sum()
            if wsum > 0:
                result[jy, jx] = (patch_arr * w2d).sum() / wsum
            else:
                result[jy, jx] = np.nan
    return result


def downsample_array(
    arr: np.ndarray, src_mesh: float, dst_mesh: float,
) -> np.ndarray:
    """src_mesh → dst_mesh に集約。整数倍なら高速 reshape、それ以外は面積按分。"""
    ratio = dst_mesh / src_mesh
    if abs(ratio - round(ratio)) < 1e-9 and ratio >= 1:
        factor = int(round(ratio))
        log.info("  Integer downsample by %dx (NaN-safe mean)", factor)
        return downsample_integer(arr, factor)
    else:
        return downsample_fractional(arr, src_mesh, dst_mesh)


INDICATOR_NAMES = (
    "density_z1", "density_z2", "density_z3",
    "occupancy_z1", "occupancy_z2", "canopy_height_p95",
)


@click.command()
@click.option(
    "--source",
    "source_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source run directory (must contain preview_indicators.tif). Defaults to latest.",
)
@click.option(
    "--target-mesh",
    "target_mesh_m",
    required=True,
    type=float,
    help="Target mesh size in meters (e.g. 2, 2.5, 5).",
)
@click.option(
    "--output-root",
    default="data/output",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output root (a new run_YYYYMMDD_HHMMSS_downsampled_<mesh>m dir is created).",
)
def main(source_dir: Path | None, target_mesh_m: float, output_root: Path) -> None:
    if source_dir is None:
        latest = find_latest_run(output_root)
        if latest is None:
            raise click.ClickException("No run directory under output_root.")
        source_dir = latest
        log.info("Using latest as source: %s", source_dir)

    src_tif = source_dir / "preview_indicators.tif"
    src_meta_path = source_dir / "preview_meta.json"
    if not src_tif.exists() or not src_meta_path.exists():
        raise click.ClickException(
            f"Source missing preview_indicators.tif or preview_meta.json: {source_dir}"
        )

    src_meta = json.loads(src_meta_path.read_text(encoding="utf-8"))
    src_mesh = src_meta["mesh_size_m"]
    if abs(src_mesh - 1.0) > 1e-9:
        log.warning(
            "Source mesh is %sm (not 1m). Downsample still works but ratios "
            "may not be integers as expected.", src_mesh,
        )

    log.info(
        "Downsampling %sm → %sm (factor=%s)",
        src_mesh, target_mesh_m, target_mesh_m / src_mesh,
    )

    # 出力 run dir 作成
    suffix = f"downsampled_{target_mesh_m:g}m"
    out_dir = create_run_dir(output_root, suffix=suffix)
    update_latest_link(output_root, out_dir)
    log.info("Output: %s", out_dir)

    # 1m データを読み込み + 集約
    t0 = time.time()
    log.info("Reading %s...", src_tif)
    indicators_dst: dict[str, np.ndarray] = {}
    with rasterio.open(src_tif) as src:
        src_crs = src.crs
        src_bounds = src.bounds
        descs = list(src.descriptions)
        for idx, name in enumerate(descs, start=1):
            arr_1m = src.read(idx).astype(np.float32)
            log.info("[%s] downsampling...", name)
            t_each = time.time()
            arr_dst = downsample_array(arr_1m, src_mesh, target_mesh_m)
            log.info(
                "[%s] done in %.1fs (src shape=%s → dst shape=%s, NaN cells dst: %d)",
                name, time.time() - t_each, arr_1m.shape, arr_dst.shape,
                int(np.isnan(arr_dst).sum()),
            )
            indicators_dst[name] = arr_dst

    # 集約後の grid 形状から transform を作る
    first = indicators_dst[descs[0]]
    ny, nx = first.shape
    # src bounds (西端, 南端, 東端, 北端) を使い、target_mesh でグリッドを作る
    gx_min = src_bounds.left
    gy_max = src_bounds.top
    transform = from_origin(gx_min, gy_max, target_mesh_m, target_mesh_m)
    log.info("Output grid: %d x %d, mesh=%sm", nx, ny, target_mesh_m)

    # GeoTIFF 書き出し
    out_tif = out_dir / "preview_indicators.tif"
    with rasterio.open(
        out_tif, "w",
        driver="GTiff", height=ny, width=nx, count=len(descs),
        dtype="float32", crs=src_crs, transform=transform,
        compress="deflate", nodata=np.nan,
    ) as dst:
        for i, name in enumerate(descs, start=1):
            dst.write(indicators_dst[name].astype(np.float32), i)
            dst.set_band_description(i, name)
    log.info("Wrote %s", out_tif)

    # メタデータ
    meta = {
        "source_run": str(source_dir),
        "source_mesh_size_m": src_mesh,
        "mesh_size_m": target_mesh_m,
        "downsample_factor": target_mesh_m / src_mesh,
        "downsample_method": "area-weighted mean" if (target_mesh_m / src_mesh) % 1 != 0 else "integer reshape mean",
        "crs": str(src_crs),
        "grid_size_cells": [nx, ny],
        "indicators_band_order": list(descs),
    }
    (out_dir / "preview_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("Total downsample time: %.1fs", time.time() - t0)
    log.info("Run dir: %s", out_dir)


if __name__ == "__main__":
    main()
