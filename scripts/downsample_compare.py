"""Phase 1 直接実行 と 1m から集約 の preview_indicators.tif を比較する。

例えば 2.5m メッシュを 2 通りで作って比較:
  - Phase 1 を mesh-size=2.5 で実行 (= 点群から直接 2.5m メッシュ集計)
  - Phase 1 を mesh-size=1   で実行し、その結果を downsample.py で 2.5m に集約

両者の各指標の差を統計化、空間分布として可視化、CSV/PNG 出力。

Run:
    python scripts/downsample_compare.py \\
        --direct  data/output/run_xxx_2.5m \\
        --downsampled data/output/run_yyy_downsampled_2.5m
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ds_compare")

INDICATOR_NAMES = (
    "density_z1", "density_z2", "density_z3",
    "occupancy_z1", "occupancy_z2", "canopy_height_p95",
)


@click.command()
@click.option(
    "--direct",
    "direct_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Phase 1 で直接生成した run dir.",
)
@click.option(
    "--downsampled",
    "ds_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="1m から集約して作った run dir.",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output dir (defaults to <downsampled>/compare_with_direct/).",
)
def main(direct_dir: Path, ds_dir: Path, output_dir: Path | None) -> None:
    if output_dir is None:
        output_dir = ds_dir / "compare_with_direct"
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_tif = direct_dir / "preview_indicators.tif"
    ds_tif = ds_dir / "preview_indicators.tif"

    with rasterio.open(direct_tif) as src_d, rasterio.open(ds_tif) as src_s:
        direct = {n: src_d.read(i + 1).astype(np.float64)
                  for i, n in enumerate(INDICATOR_NAMES)}
        ds = {n: src_s.read(i + 1).astype(np.float64)
              for i, n in enumerate(INDICATOR_NAMES)}

    # サイズ一致確認
    shape_d = direct["density_z1"].shape
    shape_s = ds["density_z1"].shape
    if shape_d != shape_s:
        # 1 行/列ずれる場合があるので最小サイズに合わせて切る
        ny = min(shape_d[0], shape_s[0])
        nx = min(shape_d[1], shape_s[1])
        log.warning(
            "Shape mismatch direct=%s vs downsampled=%s → clipping to (%d, %d)",
            shape_d, shape_s, ny, nx,
        )
        direct = {n: a[:ny, :nx] for n, a in direct.items()}
        ds = {n: a[:ny, :nx] for n, a in ds.items()}

    log.info("Comparing direct=%s vs downsampled=%s", direct_dir.name, ds_dir.name)

    stats_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=120)
    axes_flat = axes.flatten()

    for idx, name in enumerate(INDICATOR_NAMES):
        d = direct[name]
        s = ds[name]
        # 両方有効なセルのみで比較
        valid = ~(np.isnan(d) | np.isnan(s))
        if not valid.any():
            continue
        d_v = d[valid]
        s_v = s[valid]
        diff = d_v - s_v  # direct - downsampled

        max_abs_err = float(np.abs(diff).max())
        mean_abs_err = float(np.abs(diff).mean())
        p95_abs_err = float(np.percentile(np.abs(diff), 95))
        bias = float(diff.mean())
        # 相関 (1 に近いほど一致)
        if len(d_v) > 1:
            r = float(np.corrcoef(d_v, s_v)[0, 1])
        else:
            r = float("nan")

        log.info(
            "[%-20s] valid=%d  bias=%+.5f  mean|err|=%.5f  p95|err|=%.5f  max|err|=%.5f  r=%.5f",
            name, int(valid.sum()), bias, mean_abs_err, p95_abs_err, max_abs_err, r,
        )

        stats_rows.append({
            "indicator": name,
            "valid_cells": int(valid.sum()),
            "bias_direct_minus_downsampled": bias,
            "mean_abs_error": mean_abs_err,
            "p95_abs_error": p95_abs_err,
            "max_abs_error": max_abs_err,
            "pearson_correlation": r,
        })

        # 散布図
        ax = axes_flat[idx]
        ax.scatter(d_v, s_v, s=1, alpha=0.03, color="steelblue")
        # 対角線
        lim = max(d_v.max() if d_v.size else 1.0, s_v.max() if s_v.size else 1.0) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
        ax.set_xlabel(f"{name} (direct)")
        ax.set_ylabel(f"{name} (downsampled)")
        ax.set_title(f"{name}  r={r:.4f}, bias={bias:+.4f}")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f"Comparison: direct vs downsampled\n{direct_dir.name}  vs  {ds_dir.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = output_dir / "direct_vs_downsampled_scatter.png"
    fig.savefig(png_path)
    plt.close(fig)

    # CSV
    csv_path = output_dir / "direct_vs_downsampled_stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    log.info("")
    log.info("scatter PNG: %s", png_path)
    log.info("stats CSV:   %s", csv_path)


if __name__ == "__main__":
    main()
