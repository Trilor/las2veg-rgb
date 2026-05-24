# las2veg-rgb 仕様書

LAS/LAZ 点群を、植生・テレイン解析向けマルチバンド RGBA PMTiles に変換する CLI ツール。

## 設計哲学

**DEM-PNG 流の自己完結型タイル**: Mapbox Terrain-RGB や地理院標高タイルと同じ思想で、PMTiles 単体でデコード可能なフォーマットにする。サイドカー JSON ファイル (bins.json 等) は不要。デコード式は本書に明記し、MapLibre スタイル定義に直接埋め込む。

全ての保存値は **比率 [0.0, 1.0]** に正規化されているため、固定の線形/対数式で復元可能。レーザー密度に非依存で、テレイン間で値を直接比較できる。

## 1. パイプライン構成

| Step | 入力 | 出力 | 役割 |
|---|---|---|---|
| 1 | LAS/LAZ | float32 マルチバンド GeoTIFF | 5指標の比率を 1m グリッドで集計 |
| 2 | float32 GeoTIFF | RGBA(8bit) GeoTIFF | 16段階に量子化して 4bit パッキング |
| 3 | RGBA GeoTIFF | PMTiles | タイル化、必要なら Web メルカトル再投影 |

各ステップは独立した CLI として呼び出せる。中間成果物は常にファイルとして保存し、再実行を可能に保つ。

## 2. 層 (Layer) 定義

全て半開区間 `[z_min, z_max)`、高さは PDAL の HAG (Height Above Ground) ベース。

| ID | 範囲 (m) | 用途 | density | occupancy |
|---|---|---|---|---|
| z0 | 0.00 - 0.25 | Ground | ―（常に1.0で意味なし） | ― |
| z1 | 0.25 - 1.00 | Low Bush | あり | あり |
| z2 | 1.00 - 2.00 | High Bush | あり | あり |
| z3 | 2.00 - canopy_height_p95 | Canopy | あり | あり（セル単位で上端可変） |

z3 の occupancy は当初「サブボクセル数が多すぎて計算しない」としていたが、セルごとに上端を canopy_height_p95 まで切り詰めれば現実的なサイズで計算できる。サブボクセル分母もセル単位で可変 (`4*4*ceil((p95-2.0)/0.25)`)。canopy_height_p95 < 2.0m のセル (= キャノピー無し) は occupancy_z3 = 0.0 とする。

z3 の上限を 100m にする理由:
- 世界中の森林（最高樹高はカリフォルニアのハイペリオン 115m）を網羅
- PDAL HAG の異常値（200m+）を自動除外
- 全タイルで統一基準にしてタイル間整合を保つ

## 3. 7 指標の定義（比率ベース）

全て [0.0, 1.0] の比率値。レーザー密度に非依存。

| ID | 名前 | 計算式 | 物理的意味 |
|---|---|---|---|
| 1 | `density_z1` | `z1 / (z0 + z1)` | z1 まで届いたパルスのうち z1 で止まった割合 (Low Bush 遮蔽率) |
| 2 | `density_z2` | `z2 / (z0 + z1 + z2)` | z2 まで届いたパルスのうち z2 で止まった割合 (顔の高さ遮蔽率) |
| 3 | `density_z3` | `z3 / (z0 + z1 + z2 + z3)` | 全パルスのうち z3 で止まった割合 (キャノピー遮蔽率) |
| 4 | `occupancy_z1` | (1点以上ある 25cm³ サブボクセル数) / 48 | Low Bush の空間広がり |
| 5 | `occupancy_z2` | (1点以上ある 25cm³ サブボクセル数) / 64 | High Bush の空間広がり |
| 6 | `occupancy_z3` | (1点以上ある 25cm³ サブボクセル数) / `16 * ceil((p95-2.0)/0.25)` | Canopy 内部の空間充填率 (セル単位で上端=p95 まで切る; p95 < 2.0m は 0.0) |
| 7 | `canopy_height_p95` | z3 内点群の高さ 95 パーセンタイル / 100m | 樹冠高指標 (NASA GEDI 互換、林学・生態学・防災で標準) |

**式の物理的解釈**: `density_zN = zN / (zN まで届いたパルス数)`。レーザーは上から下に降ってくるので、ある層に届いたパルス数 = その層より下に到達した点の総数 (= zN 自身 + それ以下の層の合計)。

**z0 自身の density は計算しない理由**: `z0 / z0 = 1.0` で常に同じ値になり情報量ゼロ。ただし z0 の点数は z1, z2 の分母として、また z3 の `total` として使われる。

## 4. ボクセル定義

- 25cm × 25cm × 25cm の立方体に一律分割
- 1m グリッド 1 セルあたり: 4 × 4 × Nz サブボクセル
  - z1: 4 × 4 × 3 = 48
  - z2: 4 × 4 × 4 = 64
  - z3: 4 × 4 × `ceil((p95-2.0)/0.25)` (セル単位で可変、p95 < 2.0m は 0)
- メッシュサイズが N×N m の場合は xy 方向に `(4N) × (4N)` サブボクセル
- occupancy = (1点以上を含むボクセル数) / (総ボクセル数)

## 5. 空間整合性

### グリッドスナップ
1m グリッドは入力データの bbox ではなく、**CRS の絶対原点 (0, 0) を基準にスナップ**する。`math.floor` / `math.ceil` で平面直角座標系の整数メートルに揃える。これにより複数 LAS ファイル間の継ぎ目が完全に一致する。

### CRS 管理
- LAS ヘッダの CRS を解析。
- 欠落している場合は**自動推測せずエラー終了**し、`--crs EPSG:xxxx` での明示指定を促す。
- 出力 GeoTIFF には入力 CRS をそのまま付与（QGIS 等で正しい位置に重なる）。
- Web メルカトル化は Step 3 で実施する。

## 6. 点群フィルタ

- `Classification == 7` (Noise) のみ除外。
- 他のクラスは未分類含めすべて使用。

## 7. 高さ正規化

- PDAL の `filters.smrf` で地表面分類
- PDAL の `filters.hag_nn` で各点に Height Above Ground を付与
- 山岳地形でも層境界が崩れない

## 8. RGBA パッキング設計 (Step 2)

### 配置 (Plan D: 高さ順 + 同一層を同一チャンネルに集約)

| Channel | 上位 4bit (16段階) | 下位 4bit (16段階) |
|---|---|---|
| **R** | density_z3 (Canopy) | canopy_height_p95 |
| **G** | density_z2 (High Bush) | occupancy_z2 |
| **B** | density_z1 (Low Bush) | occupancy_z1 |
| **A** | 未使用 (常に 255) | ― |

### 設計の根拠
- **高さ - 色対応**: R = 上空 (Canopy)、B = 地表近く (Low Bush)。RGB 合成画像が「赤=森、青=ヤブ」と直感的に読める。
- **同一層を同一チャンネルに**: density と occupancy が同じバンド内にあり、MapLibre 式で 1 バンドのビット操作だけで層単位の判定ができる。
- **A チャンネルは使わない**: ブラウザの画像処理がアルファを乗算してデータを壊す可能性があるため、データ用には使わない。

### パッキング
```
packed = (high_4bit << 4) | (low_4bit & 0x0F)
```

### ブラウザ側復元式 (MapLibre GL JS)

```javascript
// 上位 4bit (0-15) を取り出す
const high4 = ["floor", ["/", ["band", N], 16]];

// 下位 4bit (0-15) を取り出す
const low4  = ["%", ["band", N], 16];
```

### 各指標の復元 (デフォルト = 線形スケーリング想定。Phase 2 で対数等に変更の可能性)

```javascript
// バンド番号 (1-indexed):  R=1, G=2, B=3
const density_z3        = ["/", ["floor", ["/", ["band", 1], 16]], 15];
const canopy_height_p95 = ["/", ["%",     ["band", 1], 16],      15];
// canopy_height_p95 を実際のメートル単位に戻す場合:
// const canopy_h_m  = ["*", canopy_height_p95, 100];

const density_z2        = ["/", ["floor", ["/", ["band", 2], 16]], 15];
const occupancy_z2      = ["/", ["%",     ["band", 2], 16],      15];

const density_z1        = ["/", ["floor", ["/", ["band", 3], 16]], 15];
const occupancy_z1      = ["/", ["%",     ["band", 3], 16],      15];
```

### スケーリング (確定)

全 5 指標を**線形 16 段階**で量子化する。

```
エンコード: encoded = round(value * 15)     ※ value ∈ [0.0, 1.0]
デコード:   decoded = encoded / 15
```

対応表 (match 式) は不要。MapLibre のスタイル定義では純粋な算術式で値が復元できる。

#### 採用理由
- スタイル定義が極めて単純（除算1つで完結）
- バグが入り込む余地が少ない
- 全タイルで同じ式が使えるためフォーマット互換性が高い
- ISOM2017-2 の主要閾値 (0.1, 0.3, 0.5, 0.7, 0.85) はすべて異なるビンに割り当てられる

#### 観測済みデータ分布 (08ME3204、富士山周辺樹林帯)
| 指標 | 全体平均 | 非ゼロ平均 | 非ゼロ中央値 | 非ゼロ p95 |
|---|---|---|---|---|
| density_z1 | 0.087 | 0.339 | 0.25 | 1.0 |
| density_z2 | 0.154 | 0.421 | 0.375 | 1.0 |
| density_z3 | 0.792 | 0.825 | 0.875 | 1.0 |
| occupancy_z1 | 0.012 | 0.047 | 0.042 | 0.125 |
| occupancy_z2 | 0.022 | 0.060 | 0.047 | 0.172 |

低い値の解像度損失は許容範囲（ISOM 判定は「ほぼゼロ」と「微量」を区別しないため）。

## 9. ISOM2017-2 記号への対応 (下流 TeleDrop の責務)

5指標の組み合わせで ISOM2017-2 の植生記号 401-410 を全て判定可能。閾値は TeleDrop 側のスタイル定義でスライダー調整できるようにする。詳細は下流アプリケーション側の文書を参照。

## 10. PMTiles 配信形式 (Phase 3)

### タイル形式
- **PNG** (常に可逆圧縮)
- 256×256 ピクセル
- 4バンド RGBA (実態は 3バンド RGB + Alpha=255 固定)

WebP Lossless 採用も検討したが、GDAL の MBTiles ドライバが WebP Lossless オプションを提供していないため PNG を採用。サイズ差は小さく、可逆性が確実に得られる利点が勝る。

### Web メルカトル再投影
- 入力 CRS (例: EPSG:6676) → EPSG:3857
- リサンプリング: **nearest** 必須（bilinear/cubic は 4bit パッキングを破壊する）

### ズームレベル
- 1m データの解像度に最も適合: z17 (約 1m/pixel @ 北緯35°)
- 低ズームは gdaladdo の overview 機能で nearest ダウンサンプリング

### PMTiles メタデータ (注入される JSON)
```json
{
  "name": "las2veg-rgb",
  "version": "0.1.0",
  "format": "png",
  "channels": {
    "R_high4": "density_z3",
    "R_low4":  "canopy_height_p95",
    "G_high4": "density_z2",
    "G_low4":  "occupancy_z2",
    "B_high4": "density_z1",
    "B_low4":  "occupancy_z1",
    "A":       "unused (always 255)"
  },
  "scaling": "linear_16_steps",
  "decode_formula": "value = encoded_4bit / 15",
  "canopy_height_p95_unit_meters": 100,
  "z3_upper_bound_m": 100,
  "spec_url": "https://github.com/Trilor/las2veg-rgb/blob/main/docs/spec.md"
}
```

### ブラウザ側で必要なもの
- MapLibre GL JS v3 以降
- pmtiles プラグイン (`https://unpkg.com/pmtiles@3/dist/pmtiles.js`)
- 上記メタデータ (`pmtiles meta output.pmtiles` で取得可能)
- 本仕様書の復元式

## 11. Phase 1 出力 (現在の実装)

| ファイル | 内容 |
|---|---|
| `preview_density_z1.png`, `preview_density_z2.png`, `preview_density_z3.png` | density 指標のクイックビュー (viridis colormap, 0.0-1.0) |
| `preview_occupancy_z1.png`, `preview_occupancy_z2.png`, `preview_occupancy_z3.png` | occupancy 指標のクイックビュー (magma colormap, 0.0-1.0) |
| `preview_canopy_height_p95.png` | canopy_height_p95 のクイックビュー (cividis, 0.0-1.0 = 0-100m) |
| `preview_indicators.tif` | 7バンド float32 GeoTIFF (CRS 付き)。バンド順: density_z1, density_z2, density_z3, occupancy_z1, occupancy_z2, occupancy_z3, canopy_height_p95 |
| `preview_meta.json` | bbox / CRS / 各層の点数 / 各指標の統計 (min/max/mean/nonzero percentiles) |
| `<入力名>_hag.laz` | PDAL 処理済みの中間 LAZ (入力と同じディレクトリにキャッシュ) |
