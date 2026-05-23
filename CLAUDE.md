# CLAUDE.md — las2veg-rgb 作業ルール

このファイルは Claude / AI エージェントが本リポジトリで作業する際の運用ルールと、
過去に詰まった環境固有の落とし穴を集約する。技術仕様は [docs/spec.md](docs/spec.md) を参照。

## 0. 大原則

- **このファイルを毎回読む**: 作業開始時に必ず一度読む。
- **重要情報が出たら即追記**: 環境固有の落とし穴、再発防止すべき事故、判断根拠など
  「次回も役立つ」と判断したら即この CLAUDE.md か対応する md に追記する。
  迷ったら書く。書く判断はエージェント自身が下す (ユーザに毎回確認しない)。
- **毎回コミットする**: 何らかのファイル変更を伴う作業が一区切りついたら、
  ユーザの明示指示が無くても日本語コミットメッセージで commit する。
  ただし `data/` 配下や生成物 (PNG, GeoTIFF 等) は基本コミットしない (.gitignore 参照)。
- 言語: ユーザ応答・コミットメッセージ・コメントすべて日本語。
- LaTeX のコンパイルはしない。

## 1. 実行環境 (最重要)

### Python: conda env `las2veg` を使う

```
C:\Users\kurag\AppData\Local\anaconda3\envs\las2veg\python.exe
```

ユーザは PowerShell で `conda activate las2veg` してから作業している。
グローバル Python / QGIS Python には `laspy` / `pdal` が入っていないので必ず las2veg env を使うこと。

conda env 一覧は `C:\Users\kurag\.conda\environments.txt` に記録されている。
`conda` コマンドが PATH 上に無いため、env の python.exe を絶対パスで直接呼ぶのが確実。

### Bash 経由で実行するときの落とし穴

**matplotlib が描画前にハードクラッシュ (exit 127) する。**

Git Bash から las2veg env の Python を起動して `plt.subplots()` を呼ぶと、デフォルト
GUI backend (Tk/Qt) の DLL 解決に失敗してプロセスが落ちる。エラーメッセージも出ず
シェルが exit 127 を返すだけなので、原因の特定が難しい。

**対策**: 描画を行うスクリプトには必ず以下を入れる。

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
```

または PowerShell から実行する。PowerShell では再現しない。

### 推奨: PowerShell で実行する

```powershell
conda activate las2veg
cd C:\Users\kurag\Documents\GitHub\las2veg-rgb
python scripts/phase1_preview.py --input data/input/kamiide --crs EPSG:6676 --mesh-size 1
```

エージェントが Bash 経由で動かす場合は env の python.exe を絶対パスで叩く:

```bash
PY="/c/Users/kurag/AppData/Local/anaconda3/envs/las2veg/python.exe"
"$PY" scripts/phase1_preview.py --input data/input/kamiide --crs EPSG:6676
```

## 2. データレイアウト

- 入力 LAS/LAZ: `data/input/<area_name>/` (例: `data/input/kamiide/`)
- PDAL HAG キャッシュ: 入力と同じディレクトリに `<stem>_hag.laz` として並ぶ。
  存在すれば PDAL ステージをスキップ。削除すれば再生成。
- 出力: `data/output/run_YYYYMMDD_HHMMSS_<suffix>/`
  - `latest` ジャンクションが最新 run を指す。
  - 過去の run は削除しない (比較に使う)。
- `data/` 配下は `.gitignore` で除外。コミットしない。

## 3. 既知の事実 (再調査しない)

- **kamiide データには有効な Classification が無い** (Class 1=91.6%, Class 2=8.4% のみ)。
  建物・植生分類は SMRF + HAG_NN で自前で行う必要がある。
- **マルチリターンによる建物判定は針葉樹林 (杉・檜) で 22% 偽陽性**。実装したが棄却済み。
  再挑戦しないこと。判定したいなら別の特徴量 (例: 平面性、点群密度パターン) を検討。
- **canopy_height_p95 は scipy で計算 (PDAL writers.gdal max ではない)**。検証済み
  (avg max-P95=+2.2m、最大外れ値 +20m)。P95 で十分な外れ値耐性がある。
- **density は集約 (1m → 2.5m など) で精度劣化が大きい** (r=0.88-0.95)。
  occupancy と density_z3 は集約に強い (r>0.99)。
  density_z1/z2 を正確に扱いたい場合は Phase 1 を目的メッシュサイズで直接実行する。
- **PMTiles の WebP Lossless は MBTiles driver 非対応**。PNG タイルで運用する。
- **Windows cp932 で subprocess の出力をデコードする時は `encoding="utf-8", errors="replace"`** を付ける
  (Phase 3 の pmtiles 呼び出しで対応済み)。

## 4. コーディング規約

- スクリプトは `scripts/` 直下に置く。共通ロジックは `src/las2veg_rgb/` に。
- 全スクリプトは `click` でオプションを定義し、`logging` で出力する。
- 描画スクリプトは前述の `matplotlib.use("Agg")` を必ず先頭に。
- 新しい指標を加える / 既存指標の式を変える場合は `docs/spec.md` も同時に更新する。
- 16-bin 量子化は floor+clip (Method B) で統一。バイアスを変えないこと。
- NaN を NoData として伝搬する。0 と NaN を混同しない。

## 5. コミット運用

- ファイル変更を伴う作業が一段落したら、ユーザの明示指示が無くても commit する。
  「コミットしますか?」と毎回確認しない。
- 一区切りの目安: スクリプト追加、バグ修正、リファクタ、ドキュメント更新の各単位。
- コミットメッセージは日本語、本文に「何を / なぜ / 検証結果」を簡潔に。
  既存履歴 (`git log`) のスタイルに合わせる。
- 共著フッタは付ける:
  ```
  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  ```
- **push は「プッシュして」と明示指示があるまで絶対にしない**。`git push` を勝手に実行しない。
- `data/` 配下や生成物は staging しない (`.gitignore` 任せ)。
  `git add -A` は使わずファイルを明示する。

## 6. このファイルの更新ルール

以下に該当する事実が発生したら **その場で追記** する (別途確認不要):

- 環境固有の落とし穴 (再現条件 + 回避策)
- 「やったけど棄却した」アプローチ (理由付き、再挑戦防止)
- 検証済みの数値根拠 (再検証コストが高いもの)
- ユーザから受けた運用上のフィードバック

技術仕様 (指標の定義、パイプラインの構造など) は `docs/spec.md` に書く。
このファイルは作業エージェントの行動ルールと環境メモに絞る。
