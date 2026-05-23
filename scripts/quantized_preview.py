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
    set_bad で NaN セルを透明 (RGBA = (0,0,0,0)) として描画する。
    """
    base = plt.get_cmap("tab20")
    colors = [base(i / 19) for i in range(16)]  # 0..15 で 16 色を取り出す
    cmap = ListedColormap(colors, name="quantized16")
    cmap.set_bad((0, 0, 0, 0))  # NaN/masked → 透明
    return cmap


def quantize_to_bins(values: np.ndarray, max_value: float) -> np.ndarray:
    """0..max_value の値を 0..15 のビン番号に量子化 (Phase 2 と同じ式)。

    Bin N は [N/16, (N+1)/16) * max_value の範囲を表す均等幅区間。
    NaN 入力は出力でも NaN (float) として残す → masked array で透明描画。
    """
    nan_mask = np.isnan(values)
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, max_value)
    scaled = clipped / max_value * 16.0
    bins = np.minimum(np.floor(scaled), 15.0)
    # NaN 入力箇所は NaN にして masked array で扱えるようにする
    bins_f = bins.astype(np.float32)
    bins_f[nan_mask] = np.nan
    return bins_f


def save_quantized_png(
    out_path: Path,
    bin_values: np.ndarray,
    title: str,
    cmap: ListedColormap,
    dpi: int = 300,
) -> None:
    """量子化済み (0-15) 配列を 16 色離散カラーマップで PNG 出力。

    BoundaryNorm の境界は量子化方式に合わせて [0, 1, 2, ..., 16] とする:
      整数値 N (0 <= N < 16) を [N, N+1) の区間として色 N に対応させる。
      (= 連続値 [N/16, (N+1)/16) * qmax と一貫した解釈)
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    bounds = np.arange(0, 17, 1.0)  # [0, 1, 2, ..., 16]
    norm = BoundaryNorm(bounds, cmap.N)
    # NaN を masked array で扱い、set_bad で透明描画させる
    masked = np.ma.masked_invalid(bin_values)
    im = ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X (grid cells)")
    ax.set_ylabel("Y (grid cells)")
    # カラーバーのティックは境界 (0, 1, ..., 16) に配置
    # 各色 N は [N, N+1) の区間を表す
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(17))
    cbar.set_ticklabels([str(i) for i in range(17)])
    cbar.set_label("bin boundary (each color N covers [N, N+1))")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_legend(out_path: Path, cmap: ListedColormap, dpi: int = 300) -> None:
    """16 色 + 凡例 (bin 境界 + 各 QUANTIZE_MAX に対する実値) を 1 枚の PNG に。

    境界 0, 1, 2, ..., 16 を x 軸の左端から右端に配置し、
    各色 N は [N, N+1) の区間 (= bin N) を占める。
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    # 各色を [N, N+1) の区間で表示するため、imshow の extent で X 軸を [0, 16] に設定
    swatch = np.arange(16).reshape(1, 16)
    bounds = np.arange(0, 17, 1.0)  # [0, 1, 2, ..., 16]
    norm = BoundaryNorm(bounds, cmap.N)
    ax.imshow(swatch, cmap=cmap, norm=norm, aspect="auto", extent=[0, 16, 0, 1])
    ax.set_yticks([])
    # ティックは境界 (0, 1, ..., 16) に配置
    ax.set_xticks(np.arange(17))
    ax.set_xticklabels([str(i) for i in range(17)])
    ax.set_xlabel("bin boundary (each color N covers [N, N+1))")
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
@click.option(
    "--dpi",
    default=300,
    type=int,
    help="Output image DPI (default: 300). Higher = larger file, sharper grid cells.",
)
def main(input_path: Path | None, output_dir: Path | None, dpi: int) -> None:
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
            save_quantized_png(out_path, bin_values, title, cmap, dpi=dpi)
            # NaN を除いた有効 bin の一覧
            valid_bins = bin_values[~np.isnan(bin_values)].astype(np.uint8)
            unique_bins = sorted(np.unique(valid_bins).tolist())
            log.info(
                "  [%s] saved %s  (used bins: %s, nodata cells: %d)",
                name, out_path.name, unique_bins,
                int(np.isnan(bin_values).sum()),
            )

    legend_path = output_dir / "legend.png"
    save_legend(legend_path, cmap, dpi=dpi)
    log.info("  legend: %s", legend_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
