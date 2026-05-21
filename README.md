# las2veg-rgb

LAS/LAZ 点群を高さ別の density と空間分散 (occupancy) で 1m グリッド集計し、最終的に MapLibre GL JS で復号可能な RGBA PMTiles を生成する CLI ツール。

詳細仕様は [docs/spec.md](docs/spec.md) を参照。

## 必要環境

- Windows / macOS / Linux
- miniforge (conda) — PDAL を C++ バインディング込みで入れるため必須
- Python 3.11

## セットアップ (PowerShell)

```powershell
# miniforge 未インストールなら:
winget install --id CondaForge.Miniforge3

# プロジェクト環境を作成
conda create -n las2veg -c conda-forge python=3.11 pdal python-pdal gdal -y
conda activate las2veg

# 純Python依存を uv pip で
pip install uv
uv pip install -e .
```

## Phase 1: 検証用プレビュー出力

```powershell
python scripts/phase1_preview.py `
  --input 08ME3204/08ME3204.las `
  --output data/output
```

CRS が LAS ヘッダに無い場合はエラーになる。その場合は `--crs` で明示:

```powershell
python scripts/phase1_preview.py `
  --input 08ME3204/08ME3204.las `
  --output data/output `
  --crs EPSG:6677
```

PDAL の処理結果は `08ME3204/08ME3204_hag.laz` にキャッシュされる。集計ロジックだけ繰り返し試したい場合は `--skip-pdal` でキャッシュを再利用できる:

```powershell
python scripts/phase1_preview.py `
  --input 08ME3204/08ME3204.las `
  --output data/output `
  --skip-pdal
```

### 出力

`data/output/` に以下が生成される:

- `preview_z{0,1,2,3}_density.png` — density のクイックビュー (log1p スケール表示)
- `preview_z{1,2}_occupancy.png` — occupancy のクイックビュー
- `preview_z{0,1,2,3}.tif` — GeoTIFF (QGIS で開ける)
- `preview_meta.json` — bbox / CRS / 層ごとの統計

## ロードマップ

- **Phase 1 (完了)**: 集計ロジックの視覚的検証
- **Phase 2**: 対数スケーリング + 4bit パッキングで RGBA GeoTIFF 出力
- **Phase 3**: PMTiles ビルド + MapLibre GL JS 復号サンプル
