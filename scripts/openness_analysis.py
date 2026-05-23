"""オープンエリア判定のための density_z3 vs canopy_height_p95 分析。

両指標は「樹冠の有無」を別の角度から測る:
  density_z3:        z3 層 (2-100m) に当たったパルスの全体に対する比率
  canopy_height_p95: z3 点群の高さ P95 (100m 正規化)

オープンエリア (= 樹冠なし) を判定する用途で、どちらを使うべきかを
実データから検証する。

出力 (input ディレクトリ /analyze/ に):
  - openness_scatter.png     2D 散布図 (色 = 出現頻度)
  - openness_joint_hist.csv  10x10 ジョイントヒストグラム
  - openness_summary.txt     エッジケース集計と推奨

Run:
    python scripts/openness_analysis.py
    python scripts/openness_analysis.py --input <path-to-preview_indicators.tif>
"""

from __future__ import annotations

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
log = logging.getLogger("openness")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra.astype(np.float64), rb.astype(np.float64))


@click.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Phase 1 multi-band indicators GeoTIFF.",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory (defaults to <input-dir>/analyze/).",
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
        descs = list(src.descriptions)
        bands = {name: src.read(idx + 1).astype(np.float64)
                 for idx, name in enumerate(descs)}

    d3 = bands["density_z3"].ravel()
    ch = bands["canopy_height_p95"].ravel()
    d1 = bands["density_z1"].ravel()
    d2 = bands["density_z2"].ravel()
    o1 = bands["occupancy_z1"].ravel()
    o2 = bands["occupancy_z2"].ravel()

    # NaN セル (データなし) を除外
    valid = ~(np.isnan(d3) | np.isnan(ch))
    d3, ch = d3[valid], ch[valid]
    d1, d2 = d1[valid], d2[valid]
    o1, o2 = o1[valid], o2[valid]
    total_cells = d3.size

    # ── 1. 全体相関 ──
    r_pearson_all  = pearson(d3, ch)
    r_spearman_all = spearman(d3, ch)
    # canopy_height は z3 がないと意味がないので、d3 > 0 のセルだけでも見る
    mask_d3pos = d3 > 0
    r_pearson_d3pos  = pearson(d3[mask_d3pos], ch[mask_d3pos])
    r_spearman_d3pos = spearman(d3[mask_d3pos], ch[mask_d3pos])

    # ── 2. オープン判定の合致性 ──
    # 「樹冠なし」を density_z3 で判定: d3 < threshold_d3
    # 「樹冠なし」を canopy_height_p95 で判定: ch < threshold_ch
    # 候補閾値ごとに「両者の一致率」と「片方だけ open 判定するセル数」を集計

    log.info("\n=== density_z3 vs canopy_height_p95 (オープン判定の食い違い) ===")
    log.info(
        f"  total cells: {total_cells}\n"
        f"  Pearson  (all):       {r_pearson_all:.4f}\n"
        f"  Spearman (all):       {r_spearman_all:.4f}\n"
        f"  Pearson  (d3>0):      {r_pearson_d3pos:.4f}\n"
        f"  Spearman (d3>0):      {r_spearman_d3pos:.4f}\n"
    )

    # ── 3. 閾値比較表 ──
    summary_lines: list[str] = []
    summary_lines.append("=== Openness threshold agreement table ===\n")
    summary_lines.append(
        f"Total cells: {total_cells}\n"
        f"\n"
        f"Correlation density_z3 vs canopy_height_p95:\n"
        f"  Pearson  (all):  {r_pearson_all:.4f}\n"
        f"  Spearman (all):  {r_spearman_all:.4f}\n"
        f"  Pearson  (d3>0): {r_pearson_d3pos:.4f}\n"
        f"  Spearman (d3>0): {r_spearman_d3pos:.4f}\n"
        f"\n"
    )

    # 「オープン」を表現する候補閾値
    d3_thresholds = [0.0, 0.05, 0.10, 0.20, 0.30]
    ch_thresholds = [0.0, 0.02, 0.05, 0.10, 0.15]  # 0-0.15 = 0-15m
    summary_lines.append(
        "Threshold table: open := (d3 < d3_th) and (ch < ch_th).\n"
        "Columns show how many cells each definition catches.\n\n"
    )
    summary_lines.append(
        f"  {'d3_th':>8s} {'ch_th':>8s} {'open by d3':>12s} {'open by ch':>12s} "
        f"{'both':>8s} {'only-d3':>8s} {'only-ch':>8s} {'agreement':>10s}\n"
    )
    for d3_th in d3_thresholds:
        for ch_th in ch_thresholds:
            open_d3 = d3 < d3_th if d3_th > 0 else d3 == 0
            open_ch = ch < ch_th if ch_th > 0 else ch == 0
            both    = open_d3 & open_ch
            only_d3 = open_d3 & ~open_ch
            only_ch = ~open_d3 & open_ch
            # agreement: 一致率 (どちらも open / どちらも closed)
            agree = (open_d3 == open_ch).sum() / total_cells * 100
            summary_lines.append(
                f"  {d3_th:8.2f} {ch_th:8.2f} {int(open_d3.sum()):12d} "
                f"{int(open_ch.sum()):12d} {int(both.sum()):8d} "
                f"{int(only_d3.sum()):8d} {int(only_ch.sum()):8d} "
                f"{agree:10.1f}%\n"
            )

    # ── 4. エッジケース分析: 片方だけ「オープン」と判定するセルの特徴 ──
    summary_lines.append("\n=== Edge cases: cells where the two metrics disagree ===\n\n")

    # ケース A: density_z3 高い (= 樹冠あり) なのに canopy_height_p95 低い (= 樹高低い)
    #          → 若い人工林・低木林
    case_A_mask = (d3 >= 0.5) & (ch < 0.10)  # キャノピー > 50%、樹高 < 10m
    n_case_A = int(case_A_mask.sum())
    summary_lines.append(
        f"Case A: 若い人工林・低木林 (d3 >= 0.5 AND ch < 0.10 = 樹高 < 10m)\n"
        f"  cells: {n_case_A} ({n_case_A/total_cells*100:.2f}%)\n"
    )
    if n_case_A > 0:
        summary_lines.append(
            f"  this group avg: d1={d1[case_A_mask].mean():.3f} "
            f"d2={d2[case_A_mask].mean():.3f} "
            f"d3={d3[case_A_mask].mean():.3f} "
            f"ch={ch[case_A_mask].mean():.3f} ({ch[case_A_mask].mean()*100:.1f}m)\n"
        )

    # ケース B: density_z3 低い (= 樹冠スカスカ) なのに canopy_height_p95 高い (= 高木孤立)
    #          → 孤立樹 / まばらな森
    case_B_mask = (d3 < 0.2) & (ch >= 0.10)
    n_case_B = int(case_B_mask.sum())
    summary_lines.append(
        f"\nCase B: 孤立樹/まばら林 (d3 < 0.2 AND ch >= 0.10 = 樹高 >= 10m)\n"
        f"  cells: {n_case_B} ({n_case_B/total_cells*100:.2f}%)\n"
    )
    if n_case_B > 0:
        summary_lines.append(
            f"  this group avg: d1={d1[case_B_mask].mean():.3f} "
            f"d2={d2[case_B_mask].mean():.3f} "
            f"d3={d3[case_B_mask].mean():.3f} "
            f"ch={ch[case_B_mask].mean():.3f} ({ch[case_B_mask].mean()*100:.1f}m)\n"
        )

    # ケース C: 両方とも「オープン」(完全に空き地)
    case_C_mask = (d3 == 0) & (ch == 0)
    n_case_C = int(case_C_mask.sum())
    summary_lines.append(
        f"\nCase C: 完全オープン (d3 == 0 AND ch == 0)\n"
        f"  cells: {n_case_C} ({n_case_C/total_cells*100:.2f}%)\n"
    )

    # ── 5. 散布図とジョイントヒストグラム ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 散布図 (透明度で密度表現)
    axes[0].scatter(d3, ch, s=2, alpha=0.05, color="steelblue")
    axes[0].set_xlabel("density_z3 (canopy closure)")
    axes[0].set_ylabel("canopy_height_p95 (normalized by 100m)")
    axes[0].set_title(f"Scatter (Pearson={r_pearson_all:.3f}, Spearman={r_spearman_all:.3f})")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, max(0.3, ch.max() * 1.05))
    axes[0].grid(alpha=0.3)

    # ジョイントヒストグラム (10x10)
    H, xedges, yedges = np.histogram2d(
        d3, ch,
        bins=[np.linspace(0, 1, 11), np.linspace(0, 0.3, 11)],
    )
    # 対数スケールで濃淡 (各ビンの cell 数が大きく違うため)
    H_log = np.log10(H + 1)
    im = axes[1].imshow(
        H_log.T, origin="lower", aspect="auto",
        extent=[0, 1, 0, 0.3], cmap="viridis",
    )
    axes[1].set_xlabel("density_z3")
    axes[1].set_ylabel("canopy_height_p95 (0-0.3 = 0-30m)")
    axes[1].set_title("Joint histogram (log10 scale)")
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label("log10(cells + 1)")

    fig.suptitle("Openness analysis: density_z3 vs canopy_height_p95", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    scatter_path = output_dir / "openness_scatter.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)

    # CSV (ジョイントヒストグラム)
    csv_path = output_dir / "openness_joint_hist.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Rows = density_z3 bin (low->high), Cols = canopy_height_p95 bin (low->high)\n")
        f.write("# density_z3 bins: " + ", ".join(f"{x:.2f}" for x in xedges) + "\n")
        f.write("# canopy_height_p95 bins: " + ", ".join(f"{x:.3f}" for x in yedges) + "\n")
        f.write("d3_low,d3_high," + ",".join(
            f"ch_{i}" for i in range(len(yedges)-1)
        ) + "\n")
        for i in range(len(xedges) - 1):
            row = [f"{xedges[i]:.3f}", f"{xedges[i+1]:.3f}"]
            row += [str(int(v)) for v in H[i]]
            f.write(",".join(row) + "\n")

    # ── 6. 物理的解釈と推奨 ──
    summary_lines.append("\n=== 推奨 ===\n\n")
    summary_lines.append(
        f"density_z3 と canopy_height_p95 の Spearman 相関は {r_spearman_all:.3f}\n"
        f"(d3>0 セルだけだと {r_spearman_d3pos:.3f})\n"
    )
    if r_spearman_d3pos > 0.7:
        summary_lines.append(
            "→ 両指標は強く相関しているため、どちらでもオープン判定は可能。\n"
            "  density_z3 の方が「真のゼロ」が明確 (= ch は地表点でも 0 になりうる)\n"
        )
    elif r_spearman_d3pos > 0.3:
        summary_lines.append(
            "→ 中程度の相関。両指標は独立した情報を含む。\n"
            "  Case A (若い人工林) や Case B (孤立樹) のあるテレインでは\n"
            "  両方使うか、組み合わせ判定が望ましい。\n"
        )
    else:
        summary_lines.append(
            "→ 弱い相関。両指標は別の側面を見ている。\n"
            "  オープン判定は density_z3 が直接的、canopy_height_p95 は補助情報。\n"
        )

    summary_lines.append(
        f"\n物理的には:\n"
        f"  density_z3 < 0.05  → 樹冠ほぼゼロ (= オープン土地)\n"
        f"  canopy_height_p95 < 0.05 (=5m) → 樹冠が低い (オープン or 低木地)\n"
        f"\n"
        f"判定の食い違い (上表参照):\n"
        f"  - Case A {n_case_A} cells: 「キャノピー密 + 樹高低」(= 若い植林地)\n"
        f"      density_z3 はオープンと判定しない\n"
        f"      canopy_height_p95 < 0.10 だとオープンと誤判定する可能性\n"
        f"  - Case B {n_case_B} cells: 「キャノピー疎 + 高木」(= 孤立樹)\n"
        f"      density_z3 はオープンと判定 (= 401)\n"
        f"      canopy_height_p95 はオープンと判定しない\n"
        f"\n"
        f"結論:\n"
        f"  「キャノピーで覆われているか」を見るのが目的なら density_z3 を使う。\n"
        f"  canopy_height_p95 は補助 (ISOM 402/404 の「散在木」判定に有用)。\n"
    )

    summary_path = output_dir / "openness_summary.txt"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    # ログにも要約を出力
    log.info("\n".join(summary_lines))

    log.info("")
    log.info("scatter PNG:    %s", scatter_path)
    log.info("joint hist CSV: %s", csv_path)
    log.info("summary TXT:    %s", summary_path)


if __name__ == "__main__":
    main()
