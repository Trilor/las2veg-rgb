"""水系判別の検証スクリプト (Intensity + dropout ベース、Phase 1 とは独立)。

手法: レーザー反射強度 (Intensity) の低さと点群ドロップアウトを利用する。
近赤外線レーザーは水面で吸収・鏡面反射されるため、水域では:
  - 点群が極端に少ない (dropout)
  - 取得できても Intensity が著しく低い

1 m メッシュで以下 3 指標を集計し、二値水マスクと統計を出力する。
本スクリプトは Phase 1 出力に手を加えない (独立検証用)。

指標:
  - point_density    : セル内総点数 (pts/m^2)
  - intensity_p10    : セル内 Intensity の 10 パーセンタイル (低い順)
  - intensity_mean   : セル内 Intensity の平均

z0 層 (HAG < 0.25 m) の点だけを使う。樹冠の反射は水検出に邪魔なので除外。

出力:
  data/output/water_check_<timestamp>/
    raw_point_density.png
    raw_intensity_p10.png
    raw_intensity_mean.png
    rgb_composite.png           (R=低密度, G=低 P10, B=低 Mean)
    water_mask_binary.png       (二値: 水=青、陸=透明)
    histograms.png              (3 指標の分布)
    stats.csv

Run:
    python scripts/water_intensity_dropout.py
    python scripts/water_intensity_dropout.py --mesh-size 1 --input-dir data/input/kamiide
"""

from __future__ import annotations

import csv
import datetime as dt
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
from matplotlib.colors import ListedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("water")

# z0 上限 (これより低い点を地面付近として扱う)
GROUND_HAG_MAX = 0.25


def streaming_aggregate(
    hag_paths: list[Path],
    gx_min: float,
    gy_min: float,
    nx: int,
    ny: int,
    mesh_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全 LAZ をストリーミング読込で 1m メッシュに集計。

    各セルについて (Σ点数, ΣIntensity, ΣIntensity^2) と、低 Intensity 用に
    intensity 配列を保持して P10 を後で計算する。点数の合計は安く、P10 は
    全点を保持するとメモリが大きくなる。本実装はセルごとの低位 50 点だけ
    バッファして P10 を近似する (典型的に十分)。

    Returns:
      counts (ny, nx) uint32      地面付近の点数
      sum_inten (ny, nx) float64  Intensity の合計
      p10 (ny, nx) float32        Intensity の P10 (NaN = データなし)
    """
    counts = np.zeros((ny, nx), dtype=np.uint32)
    sum_inten = np.zeros((ny, nx), dtype=np.float64)
    # 低位 K 点バッファ: 形状 (ny, nx, K) の uint16、未使用は 65535
    K = 50
    low_buf = np.full((ny, nx, K), 65535, dtype=np.uint16)
    # バッファの現在の最大値 (これより大きい新規 Intensity はスキップ)
    low_max = np.full((ny, nx), 65535, dtype=np.uint16)

    x_edges = np.linspace(gx_min, gx_min + nx * mesh_size_m, nx + 1, dtype=np.float64)
    y_edges = np.linspace(gy_min, gy_min + ny * mesh_size_m, ny + 1, dtype=np.float64)

    t0 = time.time()
    for idx, p in enumerate(hag_paths, 1):
        t_start = time.time()
        las = laspy.read(str(p))
        x = np.asarray(las.x, dtype=np.float64)
        y = np.asarray(las.y, dtype=np.float64)
        hag = np.asarray(las["HeightAboveGround"], dtype=np.float64)
        inten = np.asarray(las.intensity, dtype=np.uint16)
        del las

        # z0 層 (地面付近) のみ
        m = hag < GROUND_HAG_MAX
        if not m.any():
            log.info("  [%2d/%d] %s skipped (no ground points)", idx, len(hag_paths), p.name)
            continue
        xg = x[m]
        yg = y[m]
        ing = inten[m]

        # セルインデックス
        ix = np.clip(np.searchsorted(x_edges, xg, side="right") - 1, 0, nx - 1)
        iy_top = ny - 1 - np.clip(np.searchsorted(y_edges, yg, side="right") - 1, 0, ny - 1)
        # (注: y_edges は南→北。画像は北を row=0 にしたいので ny-1 から反転)

        # 点数の加算
        np.add.at(counts, (iy_top, ix), 1)
        # Intensity の合計
        np.add.at(sum_inten, (iy_top, ix), ing)

        # 低位 K 点バッファ更新 (Python ループは遅いので、セルでなく点単位の
        # ナイーブ実装: 一旦全点ループで低位ヒープ更新。25 タイル合計でも
        # 数百万点なのでこれで動く)。
        # 高速化のため、low_max を超える点は最初に除外する。
        keep = ing < low_max[iy_top, ix]
        if keep.any():
            ix_k = ix[keep]
            iy_k = iy_top[keep]
            in_k = ing[keep]
            # 各点について、そのセルのバッファ最大要素を新値で置き換え
            for j in range(in_k.size):
                cell_buf = low_buf[iy_k[j], ix_k[j]]
                # 最大値の位置を探す (K=50 で argmax は数十 ns)
                pos = int(np.argmax(cell_buf))
                if in_k[j] < cell_buf[pos]:
                    cell_buf[pos] = in_k[j]
                    # low_max 更新
                    low_max[iy_k[j], ix_k[j]] = cell_buf.max()
        log.info(
            "  [%2d/%d] %s: %d ground pts in %.1fs",
            idx, len(hag_paths), p.name, int(m.sum()), time.time() - t_start,
        )
    log.info("Aggregation total: %.1fs", time.time() - t0)

    # P10 を low_buf から算出
    log.info("Computing P10 from low-K buffer...")
    valid_buf = low_buf < 65535
    n_valid = valid_buf.sum(axis=2)
    p10 = np.full((ny, nx), np.nan, dtype=np.float32)
    # P10 は各セルの低位 K 点中の 10 パーセンタイル = K=50, P10 = 5 番目
    # 有効点数が少ないセルは min を採用
    has_data = n_valid > 0
    for jy in range(ny):
        for jx in range(nx):
            if not has_data[jy, jx]:
                continue
            v = low_buf[jy, jx][low_buf[jy, jx] < 65535]
            if v.size == 0:
                continue
            if v.size >= 10:
                p10[jy, jx] = float(np.percentile(v, 10))
            else:
                p10[jy, jx] = float(v.min())

    return counts, sum_inten, p10


def save_grayscale_png(
    out_path: Path, arr: np.ndarray, title: str, cmap: str, vmin=None, vmax=None,
    dpi: int = 150,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad((0, 0, 0, 0))
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X (1m cells)")
    ax.set_ylabel("Y (1m cells)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_rgb_composite(
    out_path: Path,
    low_density_score: np.ndarray,
    low_p10_score: np.ndarray,
    low_mean_score: np.ndarray,
    title: str,
    dpi: int = 150,
) -> None:
    """3 つの [0,1] スコアを RGB に詰めた合成画像。

    R = 低密度, G = 低 P10, B = 低 Mean。水域では 3 つすべて高くなる (= 白っぽく)。
    """
    ny, nx = low_density_score.shape
    rgb = np.zeros((ny, nx, 3), dtype=np.float32)
    rgb[..., 0] = low_density_score
    rgb[..., 1] = low_p10_score
    rgb[..., 2] = low_mean_score
    rgb = np.clip(rgb, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("R=low density, G=low P10 intensity, B=low mean intensity")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_water_mask(
    out_path: Path, water_mask: np.ndarray, title: str, dpi: int = 150,
) -> None:
    """二値水マスクを青で表示 (水=不透明青、陸=透明)。"""
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    cmap = ListedColormap([(0, 0, 0, 0), (0.1, 0.4, 0.9, 1.0)])
    ax.imshow(water_mask.astype(np.uint8), cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("X (1m cells)")
    ax.set_ylabel("Y (1m cells)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_three_class_map(
    out_path: Path,
    interior_mask: np.ndarray,
    data_mask: np.ndarray,
    water_low_intensity: np.ndarray,
    title: str,
    dpi: int = 150,
) -> None:
    """3 クラス分類画像。

    クラス:
      0 = 外部 (interior_mask=False): 透明
      1 = 陸地 (data セルで low intensity ではない): 茶色
      2 = dropout (interior 内の NoData): 薄灰
      3 = low intensity 水セル: 水色

    お互いハッキリ区別できる配色にする。
    """
    cls = np.zeros(interior_mask.shape, dtype=np.uint8)
    cls[interior_mask & data_mask & ~water_low_intensity] = 1   # 陸地
    cls[interior_mask & ~data_mask] = 2                          # dropout
    cls[water_low_intensity] = 3                                 # low intensity = 水

    # 0=透明, 1=茶, 2=薄灰, 3=水色
    cmap = ListedColormap([
        (0, 0, 0, 0),               # 0: 透明
        (0.55, 0.45, 0.35, 1.0),    # 1: 茶 (陸地)
        (0.85, 0.85, 0.85, 1.0),    # 2: 薄灰 (dropout)
        (0.10, 0.55, 0.90, 1.0),    # 3: 水色 (low intensity 水セル)
    ])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.imshow(cls, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
    ax.set_title(title)
    ax.set_xlabel("X (1m cells)")
    ax.set_ylabel("Y (1m cells)")

    # 凡例
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=(0.10, 0.55, 0.90), label="water (low intensity)"),
        Patch(facecolor=(0.55, 0.45, 0.35), label="land (data cell)"),
        Patch(facecolor=(0.85, 0.85, 0.85), label="dropout (no data)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_histograms(
    out_path: Path,
    density: np.ndarray, p10: np.ndarray, mean: np.ndarray,
    thresholds: dict, dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)
    for ax, arr, name, key in [
        (axes[0], density, "point_density (pts/m^2)", "density"),
        (axes[1], p10, "intensity_p10", "p10"),
        (axes[2], mean, "intensity_mean", "mean"),
    ]:
        v = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr[arr > 0]
        if v.size == 0:
            ax.set_title(f"{name}: no data")
            continue
        ax.hist(v, bins=100, color="steelblue", edgecolor="none")
        if key in thresholds:
            ax.axvline(thresholds[key], color="red", linestyle="--",
                       label=f"threshold = {thresholds[key]:.2f}")
            ax.legend()
        ax.set_title(name)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@click.command()
@click.option(
    "--input-dir",
    default="data/input/kamiide",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="*_hag.laz を含むディレクトリ",
)
@click.option(
    "--mesh-size",
    "mesh_size_m",
    default=1.0,
    type=float,
    help="メッシュサイズ (m). デフォルト 1m (細い水系判別用).",
)
@click.option(
    "--density-threshold",
    default=2.0,
    type=float,
    help="低密度判定の閾値 (pts/m^2). これ未満を 'dropout' とみなす. "
         "Phase 1 の典型 ground 密度は中央値 5 pts/m^2 程度なので、それより十分低い値.",
)
@click.option(
    "--p10-threshold",
    default=200.0,
    type=float,
    help="低 Intensity 判定の閾値. data セルの P10 = ~48 なので、"
         "それより少し高い 200 を水候補のラインに.",
)
@click.option(
    "--mean-threshold",
    default=500.0,
    type=float,
    help="低 Intensity (Mean) 判定の閾値. data セルの P10 = ~352, P50 = ~2176 なので、"
         "500 で下位 ~15% をカット.",
)
@click.option(
    "--trim-nodata-border/--no-trim-nodata-border",
    default=True,
    help="グリッド周縁の NoData 領域 (タイル境界外) を可視化からトリムする.",
)
@click.option(
    "--cache",
    "cache_path",
    default=None,
    type=click.Path(path_type=Path),
    help="集計結果 (counts, sum_inten, p10) を保存/再利用する .npz パス. "
         "指定があり既存なら集計をスキップし可視化のみ高速に再実行できる.",
)
@click.option(
    "--output-root",
    default="data/output",
    type=click.Path(file_okay=False, path_type=Path),
    help="出力 root.",
)
def main(
    input_dir: Path, mesh_size_m: float,
    density_threshold: float, p10_threshold: float, mean_threshold: float,
    trim_nodata_border: bool,
    cache_path: Path | None,
    output_root: Path,
) -> None:
    """1m メッシュ Intensity / dropout ベースで水系を判別する検証スクリプト。"""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"water_check_{stamp}_{mesh_size_m:g}m"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", out_dir)

    hag_paths = sorted(input_dir.glob("*_hag.laz"))
    if not hag_paths:
        raise click.ClickException(f"No *_hag.laz in {input_dir}.")
    log.info("Found %d *_hag.laz files", len(hag_paths))

    # bbox
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
    gx_min = math.floor(x_min / mesh_size_m) * mesh_size_m
    gy_min = math.floor(y_min / mesh_size_m) * mesh_size_m
    gx_max = math.ceil(x_max / mesh_size_m) * mesh_size_m
    gy_max = math.ceil(y_max / mesh_size_m) * mesh_size_m
    nx = int(round((gx_max - gx_min) / mesh_size_m))
    ny = int(round((gy_max - gy_min) / mesh_size_m))
    log.info("Grid: origin=(%.0f, %.0f), cells=%d x %d, mesh=%sm",
             gx_min, gy_min, nx, ny, mesh_size_m)

    # 集計 (キャッシュがあればスキップ)
    if cache_path is not None and cache_path.exists():
        log.info("Loading cached aggregates from %s", cache_path)
        z = np.load(cache_path)
        counts = z["counts"]
        sum_inten = z["sum_inten"]
        p10 = z["p10"]
        if counts.shape != (ny, nx):
            raise click.ClickException(
                f"Cache shape mismatch: cache={counts.shape}, expected=({ny},{nx}). "
                f"Delete cache or use different mesh-size."
            )
    else:
        counts, sum_inten, p10 = streaming_aggregate(
            hag_paths, gx_min, gy_min, nx, ny, mesh_size_m,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, counts=counts, sum_inten=sum_inten, p10=p10)
            log.info("Saved cache: %s", cache_path)
    cell_area = mesh_size_m * mesh_size_m
    point_density = counts.astype(np.float32) / cell_area  # pts/m^2
    intensity_mean = np.where(counts > 0, sum_inten / np.maximum(counts, 1), np.nan).astype(np.float32)

    # NoData マスク
    data_mask = counts > 0
    n_nodata = int((~data_mask).sum())
    log.info("Data cells: %d (%.1f%%), NoData cells: %d (%.1f%%)",
             int(data_mask.sum()), data_mask.sum() / data_mask.size * 100,
             n_nodata, n_nodata / data_mask.size * 100)

    # 周縁 NoData (= タイル取得範囲外) を内部 NoData (= ground 点ゼロ = 水/建物影など)
    # と区別する。データセル全体の凸包近似として、各行・各列で最初/最後の data セル
    # 位置からトリム範囲を決める。
    interior_mask = np.zeros_like(data_mask)
    if trim_nodata_border:
        rows_has_data = data_mask.any(axis=1)
        cols_has_data = data_mask.any(axis=0)
        if rows_has_data.any() and cols_has_data.any():
            y0 = int(np.argmax(rows_has_data))
            y1 = ny - int(np.argmax(rows_has_data[::-1]))
            x0 = int(np.argmax(cols_has_data))
            x1 = nx - int(np.argmax(cols_has_data[::-1]))
            interior_mask[y0:y1, x0:x1] = True
            log.info("Interior box: y=[%d, %d), x=[%d, %d) = %d cells",
                     y0, y1, x0, x1, (y1 - y0) * (x1 - x0))
        # interior_mask 内の NoData = 内部 dropout (水候補)
        internal_dropout = interior_mask & ~data_mask
        log.info("Internal dropout cells (interior NoData): %d (%.2f%% of interior)",
                 int(internal_dropout.sum()),
                 internal_dropout.sum() / max(interior_mask.sum(), 1) * 100)
    else:
        interior_mask[:] = True
        internal_dropout = ~data_mask

    # NaN マスクで非データセルを覆う
    pd_nan = point_density.copy()
    pd_nan[~data_mask] = np.nan
    p10_nan = p10.copy()  # 既に NaN 化済み

    # 各指標の生 PNG (色分布が見やすいよう自動範囲)
    pd_vmin = 0.0
    pd_vmax = float(np.nanpercentile(pd_nan, 99))
    save_grayscale_png(
        out_dir / "raw_point_density.png", pd_nan,
        f"point_density (pts/m^2), vmax=p99={pd_vmax:.1f}",
        cmap="viridis", vmin=pd_vmin, vmax=pd_vmax,
    )
    inten_max = float(np.nanpercentile(intensity_mean, 99))
    save_grayscale_png(
        out_dir / "raw_intensity_p10.png", p10_nan,
        f"intensity_p10 (raw uint16), vmax=p99={inten_max:.0f}",
        cmap="magma", vmin=0, vmax=inten_max,
    )
    save_grayscale_png(
        out_dir / "raw_intensity_mean.png", intensity_mean,
        f"intensity_mean (raw uint16), vmax=p99={inten_max:.0f}",
        cmap="magma", vmin=0, vmax=inten_max,
    )

    # RGB 合成 (各指標を [0,1] の「水らしさスコア」に変換)
    # 内部 dropout は「水らしさ最大 (= 1)」として扱う (Intensity 情報がそもそも無い)
    # data セルでは閾値ベースのスコア化
    # 低密度スコア = 1 - clip(density / density_threshold, 0, 1)
    score_density = np.where(
        data_mask,
        1.0 - np.clip(point_density / max(density_threshold, 1e-9), 0, 1),
        1.0,  # 内部 dropout は最大
    )
    score_density[~interior_mask] = 0  # 外側 NoData は無視
    # 低 P10 スコア
    score_p10 = np.where(
        data_mask,
        1.0 - np.clip(np.nan_to_num(p10, nan=p10_threshold) / max(p10_threshold, 1e-9), 0, 1),
        1.0,
    )
    score_p10[~interior_mask] = 0
    # 低 Mean スコア
    score_mean = np.where(
        data_mask,
        1.0 - np.clip(intensity_mean / max(mean_threshold, 1e-9), 0, 1),
        1.0,
    )
    score_mean[~interior_mask] = 0

    save_rgb_composite(
        out_dir / "rgb_composite.png",
        score_density.astype(np.float32),
        score_p10.astype(np.float32),
        score_mean.astype(np.float32),
        "RGB composite (R=low density, G=low P10, B=low Mean), 1m mesh",
    )

    # 二値水マスク
    # ルール1: 内部 dropout (= interior_mask 内かつ data_mask 外) は水候補
    # ルール2: data セルでは「density 低 AND P10 低 AND Mean 低」を水候補とする
    water_from_intensity = (
        data_mask
        & (point_density < density_threshold)
        & (~np.isnan(p10)) & (p10 < p10_threshold)
        & (intensity_mean < mean_threshold)
    )
    water_mask = (interior_mask & ~data_mask) | water_from_intensity
    n_water_dropout = int((interior_mask & ~data_mask).sum())
    n_water_intensity = int(water_from_intensity.sum())
    log.info("Water (from internal dropout): %d cells", n_water_dropout)
    log.info("Water (from low intensity): %d cells", n_water_intensity)
    n_water = int(water_mask.sum())
    log.info("Water mask: %d cells (%.2f%% of data cells)",
             n_water, n_water / max(data_mask.sum(), 1) * 100)

    save_water_mask(
        out_dir / "water_mask_binary.png", water_mask,
        f"water mask (dropout={n_water_dropout}, intensity={n_water_intensity}, "
        f"total={n_water}). density<{density_threshold}, P10<{p10_threshold}, "
        f"mean<{mean_threshold:.0f}",
    )

    # 3 クラス分類画像 (low intensity 水セルだけ水色)
    save_three_class_map(
        out_dir / "three_class_map.png",
        interior_mask=interior_mask,
        data_mask=data_mask,
        water_low_intensity=water_from_intensity,
        title=f"3 classes: water (low intensity) = {n_water_intensity} cells "
              f"({n_water_intensity / max(interior_mask.sum(), 1) * 100:.2f}% of interior). "
              f"density<{density_threshold}, P10<{p10_threshold}, mean<{mean_threshold:.0f}",
    )

    # ヒストグラム
    save_histograms(
        out_dir / "histograms.png",
        point_density[data_mask],
        p10[data_mask & ~np.isnan(p10)],
        intensity_mean[data_mask],
        thresholds={"density": density_threshold, "p10": p10_threshold, "mean": mean_threshold},
    )

    # 統計 CSV
    csv_path = out_dir / "stats.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "n_cells", "min", "p1", "p10", "p50", "p90", "p99", "max", "mean"])
        for name, arr in [
            ("point_density", point_density[data_mask]),
            ("intensity_p10", p10[data_mask & ~np.isnan(p10)]),
            ("intensity_mean", intensity_mean[data_mask]),
        ]:
            if arr.size == 0:
                writer.writerow([name, 0] + [""] * 8)
                continue
            stats = [
                arr.size,
                float(arr.min()),
                float(np.percentile(arr, 1)),
                float(np.percentile(arr, 10)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 90)),
                float(np.percentile(arr, 99)),
                float(arr.max()),
                float(arr.mean()),
            ]
            writer.writerow([name] + stats)
        writer.writerow([])
        writer.writerow(["threshold_density (pts/m^2)", density_threshold])
        writer.writerow(["threshold_p10", p10_threshold])
        writer.writerow(["threshold_mean", mean_threshold])
        writer.writerow(["n_data_cells", int(data_mask.sum())])
        writer.writerow(["n_interior_cells", int(interior_mask.sum())])
        writer.writerow(["n_water_from_dropout", n_water_dropout])
        writer.writerow(["n_water_from_low_intensity", n_water_intensity])
        writer.writerow(["n_water_total", n_water])
        writer.writerow(["water_pct_of_interior", n_water / max(interior_mask.sum(), 1) * 100])
    log.info("CSV: %s", csv_path)
    log.info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()
