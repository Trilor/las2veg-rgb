"""指標分布分析ツール。

preview_indicators.tif の各バンド (6 指標) について、0.0-1.0 を 0.1 刻みの
10 ビンに分けたヒストグラムを CSV と PNG で出力する。

ISOM 閾値設計 / odrop 側の閾値スライダー初期値決めに使用する。

Run:
    python scripts/analyze_distribution.py \
        --input data/output/preview_indicators.tif \
        --output data/output/distribution
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
log = logging.getLogger("analyze")

INDICATOR_NAMES: tuple[str, ...] = (
    "density_z1",
    "density_z2",
    "density_z3",
    "occupancy_z1",
    "occupancy_z2",
    "canopy_height_p95",
)


def histogram_per_indicator(arr: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute counts for each bin, and separately a zero-cell tally.

    Returns (counts, zero_count, nonzero_count).
    counts is the histogram across nonzero cells only (so the zero spike
    doesn't drown out the rest of the distribution).
    NaN セル (= データなし) は集計から除外。
    """
    valid = arr[~np.isnan(arr)]
    zero_count = int((valid == 0).sum())
    nonzero = valid[valid > 0]
    counts, _ = np.histogram(nonzero, bins=bins)
    return counts, zero_count, nonzero.size


@click.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Phase 1 multi-band indicators GeoTIFF. "
         "Defaults to data/output/latest/preview_indicators.tif.",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for CSV and PNG. Defaults to <input-dir>/distribution/.",
)
def main(input_path: Path | None, output_dir: Path | None) -> None:
    if input_path is None:
        from las2veg_rgb.runs import find_latest_run

        latest = find_latest_run(Path("data/output"))
        if latest is None:
            raise click.ClickException(
                "No run directory found under data/output/. "
                "Run phase1_preview.py first or pass --input."
            )
        input_path = latest / "preview_indicators.tif"
        if not input_path.exists():
            raise click.ClickException(
                f"Expected {input_path} but it does not exist."
            )
        log.info("Auto-detected input from latest run: %s", input_path)

    if output_dir is None:
        output_dir = input_path.parent / "analyze"
        log.info("Auto-detected output: %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading %s", input_path)
    with rasterio.open(input_path) as src:
        if src.count != len(INDICATOR_NAMES):
            raise click.ClickException(
                f"Expected {len(INDICATOR_NAMES)} bands, got {src.count}."
            )
        bands = {
            name: src.read(idx + 1).astype(np.float32)
            for idx, name in enumerate(INDICATOR_NAMES)
        }
        descriptions = list(src.descriptions)
        log.info("Band descriptions: %s", descriptions)

    bins = np.arange(0.0, 1.05, 0.1)
    bin_labels = [
        f"[{bins[i]:.1f}, {bins[i+1]:.1f}{')' if i + 1 < len(bins) - 1 else ']'}"
        for i in range(len(bins) - 1)
    ]

    csv_path = output_dir / "distribution.csv"
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["indicator", "zero_cells", "nonzero_cells", "total_cells"]
            + [f"bin_{label}" for label in bin_labels]
            + [f"bin_pct_{label}" for label in bin_labels]
        )

        log.info("=" * 90)
        log.info(
            "Distribution (counts of nonzero cells per 0.1-bin; "
            "percentage is of nonzero cells)"
        )
        log.info("=" * 90)

        for idx, name in enumerate(INDICATOR_NAMES):
            arr = bands[name]
            # NaN セル (データなし) は除外して total_cells を計算
            valid = arr[~np.isnan(arr)]
            total_cells = valid.size
            counts, zero_count, nonzero_count = histogram_per_indicator(arr, bins)
            pct = (counts / nonzero_count * 100.0) if nonzero_count > 0 else np.zeros_like(counts, dtype=float)

            # ロガー出力
            log.info("")
            log.info(
                "[%s]  total=%d  zero=%d (%.1f%%)  nonzero=%d (%.1f%%)",
                name,
                total_cells,
                zero_count,
                zero_count / total_cells * 100.0,
                nonzero_count,
                nonzero_count / total_cells * 100.0,
            )
            for label, c, p in zip(bin_labels, counts, pct, strict=True):
                bar = "#" * int(p)
                log.info("  %-14s  %8d  (%5.1f%%) %s", label, c, p, bar)

            # CSV
            writer.writerow(
                [name, zero_count, nonzero_count, total_cells]
                + counts.tolist()
                + [f"{p:.2f}" for p in pct]
            )

            # ヒストグラム描画
            ax = axes_flat[idx]
            ax.bar(bin_labels, counts, color="steelblue", edgecolor="black")
            ax.set_title(
                f"{name}\nnonzero={nonzero_count}/{total_cells} "
                f"({nonzero_count / total_cells * 100:.1f}%)",
                fontsize=10,
            )
            ax.set_xlabel("Value bin (0.0-1.0)")
            ax.set_ylabel("Count (nonzero cells)")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            for label_, p in zip(bin_labels, pct, strict=True):
                ax.text(
                    label_,
                    counts[bin_labels.index(label_)],
                    f"{p:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.suptitle("las2veg-rgb indicator distributions (0.1 bins, excluding zeros)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = output_dir / "distribution.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    log.info("")
    log.info("CSV: %s", csv_path)
    log.info("PNG: %s", png_path)


if __name__ == "__main__":
    main()
