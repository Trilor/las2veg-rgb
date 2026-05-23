"""2 つのオープン判定指標を 16 段階で並べて比較する画像を生成。

候補 A: density_z3 = z3 / total              (現状の指標、2m 以上の遮蔽率)
候補 B: density_z23 = (z2 + z3) / total       (1m 以上の遮蔽率)

両方を 1m メッシュで集計し、Phase 2 と同じ 16-bin 量子化で色分け PNG 出力。

入力: 既存の *_hag.laz (PDAL キャッシュ)
出力:
  <run_dir>/compare_openness/
    ├── density_z3.png        (16色)
    ├── density_z23.png       (16色)
    ├── diff_z23_minus_z3.png  (差の可視化)
    └── stats.csv

Phase 1 とは独立した検証スクリプト。preview_indicators.tif などには影響しない。

Run:
    python scripts/openness_two_metrics.py
    python scripts/openness_two_metrics.py --mesh-size 1
    python scripts/openness_two_metrics.py --mesh-size 2.5
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import time
from pathlib import Path

import click
import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from las2veg_rgb.config import CANOPY_UPPER_BOUND_M  # noqa: E402
from las2veg_rgb.runs import find_latest_run  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("openness2")


# 16色離散カラーマップ (tab20 から 16 色)
def make_16color_cmap() -> ListedColormap:
    base = plt.get_cmap("tab20")
    colors = [base(i / 19) for i in range(16)]
    cmap = ListedColormap(colors, name="quantized16")
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def quantize_to_bins(values: np.ndarray, max_value: float) -> np.ndarray:
    """0..max_value の値を 0..15 のビン番号に量子化 (NaN は NaN のまま)。"""
    nan_mask = np.isnan(values)
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, max_value)
    scaled = clipped / max_value * 16.0
    bins = np.minimum(np.floor(scaled), 15.0).astype(np.float32)
    bins[nan_mask] = np.nan
    return bins


def save_quantized_png(
    out_path: Path,
    bin_values: np.ndarray,
    title: str,
    cmap: ListedColormap,
    dpi: int = 300,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    bounds = np.arange(0, 17, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    masked = np.ma.masked_invalid(bin_values)
    im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(17))
    cbar.set_ticklabels([str(i) for i in range(17)])
    cbar.set_label("bin (each color N covers [N, N+1))")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_diff_png(
    out_path: Path, diff: np.ndarray, title: str, dpi: int = 300,
) -> None:
    """差を viridis-ish なグラデーションで表示。NaN は透明。"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad((0, 0, 0, 0))
    masked = np.ma.masked_invalid(diff)
    vmax = float(np.nanmax(np.abs(masked)))
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("diff = (z2+z3)/total − z3/total")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@click.command()
@click.option(
    "--input-dir",
    default="data/input/kamiide",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing *_hag.laz files.",
)
@click.option(
    "--mesh-size",
    "mesh_size_m",
    default=2.5,
    type=float,
    help="Mesh size in meters (default: 2.5).",
)
@click.option(
    "--output",
    "output_root",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output dir (defaults to latest run / compare_openness).",
)
@click.option(
    "--dpi",
    default=300,
    type=int,
    help="PNG DPI (default: 300).",
)
def main(input_dir: Path, mesh_size_m: float, output_root: Path | None, dpi: int) -> None:
    """全 LAZ をストリーミング処理して z3/total と (z2+z3)/total を計算し、両方を色分け PNG 出力。"""
    if output_root is None:
        latest = find_latest_run(Path("data/output"))
        if latest is None:
            raise click.ClickException("No run directory under data/output/.")
        output_root = latest / "compare_openness"
    output_root.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", output_root)

    hag_paths = sorted(input_dir.glob("*_hag.laz"))
    log.info("Found %d *_hag.laz files", len(hag_paths))

    # 全 LAZ から bbox を取得
    log.info("Computing global bbox...")
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for p in hag_paths:
        with laspy.open(str(p)) as f:
            h = f.header
            x_min = min(x_min, h.mins[0])
            y_min = min(y_min, h.mins[1])
            x_max = max(x_max, h.maxs[0])
            y_max = max(y_max, h.maxs[1])

    # CRS 絶対原点 (0,0) スナップ
    gx_min = math.floor(x_min / mesh_size_m) * mesh_size_m
    gy_min = math.floor(y_min / mesh_size_m) * mesh_size_m
    gx_max = math.ceil(x_max / mesh_size_m) * mesh_size_m
    gy_max = math.ceil(y_max / mesh_size_m) * mesh_size_m
    nx = int(round((gx_max - gx_min) / mesh_size_m))
    ny = int(round((gy_max - gy_min) / mesh_size_m))
    log.info("Grid: origin=(%d, %d), cells=%d x %d, mesh=%sm",
             int(gx_min), int(gy_min), nx, ny, mesh_size_m)

    # 各層の点数集計用 (z0..z3 を別々に集計)
    counts_z0 = np.zeros((ny, nx), dtype=np.uint32)
    counts_z1 = np.zeros((ny, nx), dtype=np.uint32)
    counts_z2 = np.zeros((ny, nx), dtype=np.uint32)
    counts_z3 = np.zeros((ny, nx), dtype=np.uint32)

    layer_edges = np.array([0.25, 1.0, 2.0, CANOPY_UPPER_BOUND_M], dtype=np.float64)
    x_edges = np.linspace(gx_min, gx_min + nx * mesh_size_m, nx + 1, dtype=np.float64)
    y_edges = np.linspace(gy_min, gy_min + ny * mesh_size_m, ny + 1, dtype=np.float64)

    log.info("Streaming aggregation...")
    t0 = time.time()
    for idx, p in enumerate(hag_paths, 1):
        las = laspy.read(str(p))
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        h = np.asarray(las["HeightAboveGround"], dtype=np.float64)
        del las

        layer_id = np.digitize(h, layer_edges)  # 0..4
        valid = layer_id < 4  # z0..z3 のみ

        for k, target in enumerate([counts_z0, counts_z1, counts_z2, counts_z3]):
            m = valid & (layer_id == k)
            if not m.any():
                continue
            cnt, _, _ = np.histogram2d(x[m], y[m], bins=[x_edges, y_edges])
            target += np.flipud(cnt.T.astype(np.uint32))
        log.info("  [%2d/%d] %s done", idx, len(hag_paths), p.name)
    log.info("Aggregation: %.1fs", time.time() - t0)

    total = counts_z0 + counts_z1 + counts_z2 + counts_z3
    data_mask = total > 0

    log.info("Computing density_z3 and density_z23...")
    density_z3 = np.where(data_mask, counts_z3.astype(np.float32) / total, np.nan)
    density_z23 = np.where(
        data_mask,
        (counts_z2 + counts_z3).astype(np.float32) / total,
        np.nan,
    )

    # 16 bin に量子化
    bins_z3 = quantize_to_bins(density_z3, 1.0)
    bins_z23 = quantize_to_bins(density_z23, 1.0)
    cmap = make_16color_cmap()

    # 画像出力
    save_quantized_png(
        output_root / "density_z3.png", bins_z3,
        f"density_z3 = z3 / total  (mesh={mesh_size_m}m, 16 bins)",
        cmap, dpi=dpi,
    )
    save_quantized_png(
        output_root / "density_z23.png", bins_z23,
        f"density_z23 = (z2 + z3) / total  (mesh={mesh_size_m}m, 16 bins)",
        cmap, dpi=dpi,
    )

    # 差の画像
    diff = density_z23 - density_z3  # = z2 / total (常に >= 0)
    save_diff_png(
        output_root / "diff_z23_minus_z3.png",
        diff,
        f"(z2+z3)/total − z3/total = z2/total  (mesh={mesh_size_m}m)",
        dpi=dpi,
    )

    # 統計 CSV
    valid_d3 = density_z3[~np.isnan(density_z3)]
    valid_d23 = density_z23[~np.isnan(density_z23)]
    valid_diff = diff[~np.isnan(diff)]

    log.info("")
    log.info("=== Statistics ===")
    stats: list[dict] = []
    for name, arr in [("density_z3", valid_d3),
                       ("density_z23", valid_d23),
                       ("diff (= z2/total)", valid_diff)]:
        s = {
            "metric": name,
            "n_cells": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "n_zero": int((arr == 0).sum()),
            "n_zero_pct": float((arr == 0).sum() / arr.size * 100),
        }
        stats.append(s)
        log.info("  %-20s n=%d min=%.4f max=%.4f mean=%.4f median=%.4f n_zero=%d (%.2f%%)",
                 s["metric"], s["n_cells"], s["min"], s["max"], s["mean"], s["median"],
                 s["n_zero"], s["n_zero_pct"])

    # 「ゼロの一致率」 = どれだけ判定が一致するか
    zero_d3 = density_z3 == 0
    zero_d23 = density_z23 == 0
    both_zero = (zero_d3 & zero_d23 & data_mask).sum()
    only_d3_zero = (zero_d3 & ~zero_d23 & data_mask).sum()  # ← d3=0 だが z2 はある
    only_d23_zero = (~zero_d3 & zero_d23 & data_mask).sum()  # ← 論理的に発生しない
    neither_zero = (~zero_d3 & ~zero_d23 & data_mask).sum()

    log.info("")
    log.info("=== Zero classification agreement (cells where each metric == 0) ===")
    log.info("  Both == 0           : %d", int(both_zero))
    log.info("  Only z3 == 0        : %d (z3 ゼロだが z2 はある = 樹冠なしだが顔の高さに何かある)", int(only_d3_zero))
    log.info("  Only z23 == 0       : %d (論理的に発生しない)", int(only_d23_zero))
    log.info("  Neither == 0        : %d", int(neither_zero))

    # CSV 保存
    csv_path = output_root / "stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        for row in stats:
            writer.writerow(row)
    log.info("")
    log.info("PNGs and stats saved to: %s", output_root)


if __name__ == "__main__":
    main()
