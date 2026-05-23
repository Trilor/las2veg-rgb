"""density と occupancy の相関分析。

z1, z2 それぞれについて density と occupancy の関係を見て、
冗長性 (片方を捨てて良いか) を判定する材料を出す。

出力:
  - correlation.csv:  ピアソン相関係数、スピアマン順位相関、サンプル数
  - correlation.png:  z1, z2 それぞれの散布図 + 回帰線

Run:
    python scripts/correlation_analysis.py
    python scripts/correlation_analysis.py --input <path-to-preview_indicators.tif>
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
log = logging.getLogger("corr")

# preview_indicators.tif のバンド順 (Phase 1 と一致)
BAND_INDEX = {
    "density_z1":        1,
    "density_z2":        2,
    "density_z3":        3,
    "occupancy_z1":      4,
    "occupancy_z2":      5,
    "canopy_height_p95": 6,
}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without scipy.stats (rank then Pearson)."""
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra.astype(np.float64), rb.astype(np.float64))


def regression_line(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) for b = slope*a + intercept."""
    if a.size < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(a, b, 1)
    return float(slope), float(intercept)


@click.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Phase 1 multi-band indicators GeoTIFF (defaults to latest run).",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory (defaults to <input-dir>/correlation/).",
)
def main(input_path: Path | None, output_dir: Path | None) -> None:
    if input_path is None:
        from las2veg_rgb.runs import find_latest_run

        latest = find_latest_run(Path("data/output"))
        if latest is None:
            raise click.ClickException("No run directory under data/output/.")
        input_path = latest / "preview_indicators.tif"
        log.info("Auto-detected input: %s", input_path)

    if output_dir is None:
        output_dir = input_path.parent / "analyze"
        log.info("Auto-detected output: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading %s", input_path)
    with rasterio.open(input_path) as src:
        d1 = src.read(BAND_INDEX["density_z1"]).astype(np.float64).ravel()
        d2 = src.read(BAND_INDEX["density_z2"]).astype(np.float64).ravel()
        o1 = src.read(BAND_INDEX["occupancy_z1"]).astype(np.float64).ravel()
        o2 = src.read(BAND_INDEX["occupancy_z2"]).astype(np.float64).ravel()

    # NaN セル (データなし) を除外して同じインデックスで揃える
    valid = ~(np.isnan(d1) | np.isnan(d2) | np.isnan(o1) | np.isnan(o2))
    d1, d2, o1, o2 = d1[valid], d2[valid], o1[valid], o2[valid]

    csv_path = output_dir / "correlation.csv"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer", "n_total", "n_nonzero",
             "pearson_all", "spearman_all",
             "pearson_nonzero", "spearman_nonzero",
             "regression_slope", "regression_intercept"]
        )

        for label, (d, o, ax) in {
            "z1": (d1, o1, axes[0]),
            "z2": (d2, o2, axes[1]),
        }.items():
            # 全セル (両方とも 0 を含む)
            r_all = pearson(d, o)
            rho_all = spearman(d, o)

            # 非ゼロのみ (両方とも > 0)
            both_nonzero = (d > 0) & (o > 0)
            d_nz = d[both_nonzero]
            o_nz = o[both_nonzero]
            r_nz = pearson(d_nz, o_nz)
            rho_nz = spearman(d_nz, o_nz)
            slope, intercept = regression_line(d_nz, o_nz)

            log.info("")
            log.info("[%s]", label)
            log.info("  total cells:      %d", d.size)
            log.info("  both-nonzero:     %d  (%.1f%%)", d_nz.size, d_nz.size / d.size * 100)
            log.info("  Pearson  (all):   %.4f", r_all)
            log.info("  Spearman (all):   %.4f", rho_all)
            log.info("  Pearson  (>0):    %.4f", r_nz)
            log.info("  Spearman (>0):    %.4f", rho_nz)
            log.info("  Regression: occupancy = %.4f * density + %.4f", slope, intercept)

            writer.writerow([
                label, d.size, d_nz.size,
                f"{r_all:.4f}", f"{rho_all:.4f}",
                f"{r_nz:.4f}", f"{rho_nz:.4f}",
                f"{slope:.4f}", f"{intercept:.4f}",
            ])

            # 散布図
            ax.scatter(d, o, s=3, alpha=0.3, color="steelblue", label="cells (all)")
            if d_nz.size > 1:
                x_line = np.linspace(d_nz.min(), d_nz.max(), 100)
                ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5, label="regression (>0)")
            ax.set_xlim(0, max(0.1, d.max() * 1.05))
            ax.set_ylim(0, max(0.1, o.max() * 1.05))
            ax.set_xlabel(f"density_{label}")
            ax.set_ylabel(f"occupancy_{label}")
            ax.set_title(
                f"{label}: Pearson={r_nz:.3f}, Spearman={rho_nz:.3f} (nonzero)"
            )
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

    fig.suptitle("density vs occupancy correlation", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = output_dir / "correlation.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    log.info("")
    log.info("CSV: %s", csv_path)
    log.info("PNG: %s", png_path)


if __name__ == "__main__":
    main()
