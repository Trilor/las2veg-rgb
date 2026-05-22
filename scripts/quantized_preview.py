"""16段階量子化プレビュー画像生成。

preview_indicators.tif の各バンドを Phase 2 と同じ QUANTIZE_MAX で 0-15 の
ビン番号に量子化し、16 色離散カラーマップで色分け PNG を出力する。

各ビン番号 0..15 は全指標で同じ色を使うため、画像間で同じ色 = 同じビン番号
として比較できる (実値は QUANTIZE_MAX で異なる)。

Run:
    python scripts/quantized_preview.py
    python scripts/quantized_preview.py --input <path-to-preview_indicators.tif>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("quantized")

# Phase 2 の QUANTIZE_MAX と同期 (occupancy は 0-0.5、canopy_height_p95 は 0-0.6)
QUANTIZE_MAX: dict[str, float] = {
    "density_z1":        1.0,
    "density_z2":        1.0,
    "density_z3":        1.0,
    "occupancy_z1":      0.5,
    "occupancy_z2":      0.5,
    "canopy_height_p95": 0.6,
}

# preview_indicators.tif のバンド順 (Phase 1 と一致)
INDICATOR_NAMES: tuple[str, ...] = (
    "density_z1",
    "density_z2",
    "density_z3",
    "occupancy_z1",
    "occupancy_z2",
    "canopy_height_p95",
)


def make_16color_cmap() -> ListedColormap:
    """16 色の離散カラーマップを作る (全指標共通)。

    Matplotlib の tab20 は 20 色なので 16 色だけ取り出す。
    隣接ビンが明確に違う色になるよう、tab20 の前 16 を採用 (色相が交互配置)。
    """
    base = plt.get_cmap("tab20")
    colors = [base(i / 19) for i in range(16)]  # 0..15 で 16 色を取り出す
    return ListedColormap(colors, name="quantized16")


def quantize_to_bins(values: np.ndarray, max_value: float) -> np.ndarray:
    """0..max_value の値を 0..15 のビン番号に量子化 (Phase 2 と同じ式)。

    Bin N は [N/16, (N+1)/16) * max_value の範囲を表す均等幅区間。
    """
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, max_value)
    scaled = clipped / max_value * 16.0
    return np.minimum(np.floor(scaled), 15.0).astype(np.uint8)


def save_quantized_png(
    out_path: Path,
    bin_values: np.ndarray,
    title: str,
    cmap: ListedColormap,
) -> None:
    """量子化済み (0-15) 配列を 16 色離散カラーマップで PNG 出力。"""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    bounds = np.arange(-0.5, 16.5, 1.0)  # bin 中央 = 0,1,...,15
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(bin_values, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(16))
    cbar.set_label("bin (0-15)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_legend(out_path: Path, cmap: ListedColormap) -> None:
    """16 色 + 凡例 (bin 番号 + 各 QUANTIZE_MAX に対する実値) を 1 枚の PNG に。"""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    # 16 個のカラースウォッチを横一列に
    swatch = np.arange(16).reshape(1, 16)
    bounds = np.arange(-0.5, 16.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(swatch, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(np.arange(16))
    ax.set_xticklabels([str(i) for i in range(16)])
    ax.set_xlabel("bin number")
    ax.set_title("16-step discrete colormap (shared across all indicators)")

    # 各 QUANTIZE_MAX に対応する bin の値域 (下限, 上限) を表に追記
    # bin N = [N/16, (N+1)/16) * qmax (bin 15 のみ閉区間)
    rows = sorted(set(QUANTIZE_MAX.values()))
    table_text = "Each bin N covers [N/16, (N+1)/16) * QUANTIZE_MAX:\n"
    for qmax in rows:
        labels = [f"{i/16.0*qmax:.3f}-{(i+1)/16.0*qmax:.3f}" for i in range(16)]
        table_text += f"  QUANTIZE_MAX={qmax}:  " + "  ".join(labels) + "\n"
    fig.text(0.05, 0.05, table_text, fontsize=8, family="monospace")
    fig.subplots_adjust(bottom=0.45)
    fig.savefig(out_path)
    plt.close(fig)


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
    help="Output directory (defaults to <input-dir>/quantized/).",
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
        output_dir = input_path.parent / "quantized"
        log.info("Auto-detected output: %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = make_16color_cmap()

    log.info("Reading %s", input_path)
    with rasterio.open(input_path) as src:
        if src.count != len(INDICATOR_NAMES):
            raise click.ClickException(
                f"Expected {len(INDICATOR_NAMES)} bands, got {src.count}."
            )
        for idx, name in enumerate(INDICATOR_NAMES, start=1):
            arr = src.read(idx).astype(np.float32)
            qmax = QUANTIZE_MAX[name]
            bin_values = quantize_to_bins(arr, qmax)

            out_path = output_dir / f"{name}.png"
            title = f"{name}  (QUANTIZE_MAX={qmax}, bins 0-15)"
            save_quantized_png(out_path, bin_values, title, cmap)
            log.info(
                "  [%s] saved %s  (used bins: %s)",
                name, out_path.name,
                sorted(np.unique(bin_values).tolist()),
            )

    legend_path = output_dir / "legend.png"
    save_legend(legend_path, cmap)
    log.info("  legend: %s", legend_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
