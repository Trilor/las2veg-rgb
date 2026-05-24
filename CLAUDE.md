# CLAUDE.md — las2veg-rgb 作業ルール

このファイルは Claude / AI エージェントが本リポジトリで作業する際の運用ルールと、
過去に詰まった環境固有の落とし穴を集約する。

参照ドキュメント:
- 技術仕様 (パイプライン構造・指標式): [docs/spec.md](docs/spec.md)
- **ISOM2017-2 シンボル定義 (401–419、判定の正典)**: [docs/isom2017-2_vegetation.md](docs/isom2017-2_vegetation.md)
  植生分類ロジックを書く時は必ずこちらを参照する。原文 PDF (日本オリエンテーリング協会版) を
  2026-05-24 に取得し定義を構造化保存済み。

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

### Python を呼ぶときの最重要事項: PATH に env の `Library/bin` を必ず前置すること

**原因 (2026-05-24 確定)**: las2veg env の python.exe を絶対パスで叩くだけでは不十分。
PATH に `<env_root>\Library\bin` が無いと、Windows が `freetype.dll`, `libpng.dll`,
`zlib.dll` などを **別アプリ (Inkscape, gfortran, QGIS, mingw 等)** から拾ってロードし、
ABI 不一致で matplotlib の C 拡張 (`canvas.draw()` / `savefig()` の中身) が SEGV する。

クラッシュは `plt.subplots()` ではなく **`canvas.draw()` または `savefig()` で発生**する。
そのため「PNG 出力の直前まで動いて死ぬ」「以前は exit 127、今は exit 0 だが run dir が空」
のような症状になる。tee の buffering とは無関係。

**過去の誤認 (反省記録)**:
- 「`matplotlib.use("Agg")` を入れれば直る」→ 不十分。PATH が正しくないと Agg でも落ちる。
- 「PowerShell では再現しない」→ 誤り。前回動いたのは `conda activate las2veg` 相当の
  PATH 前置を手動でしていたから。素の PowerShell でも env の `Library/bin` が PATH に
  無ければ落ちる。

### 正しい実行方法

**PowerShell から (ユーザの通常運用)**:
```powershell
conda activate las2veg
cd C:\Users\kurag\Documents\GitHub\las2veg-rgb
python scripts/phase1_preview.py --input data/input/kamiide --crs EPSG:6676 --mesh-size 1
```
`conda activate` が PATH に env の `Library/bin` を前置するので問題なく動く。

**エージェントが Bash 経由で動かす場合 (PATH 前置必須)**:
```bash
ENV="/c/Users/kurag/AppData/Local/anaconda3/envs/las2veg"
PATH="$ENV/Library/bin:$ENV:$PATH" "$ENV/python.exe" scripts/phase1_preview.py ...
```

**エージェントが PowerShell ツール経由で動かす場合**:
```powershell
$ENV_ROOT = "C:\Users\kurag\AppData\Local\anaconda3\envs\las2veg"
$env:Path = "$ENV_ROOT;$ENV_ROOT\Library\bin;$ENV_ROOT\Scripts;" + $env:Path
& "$ENV_ROOT\python.exe" scripts/phase1_preview.py ...
```

### 描画スクリプト側の保険

念のため Agg backend 強制も入れておく (GUI backend を回避するため):

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
```

これは GUI backend (Tk/Qt) の初期化失敗を避けるためで、PATH 問題の根治策ではない。
両方やる。

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
- **Intensity dropout 法による水系判別は kamiide では効果薄 (2026-05-24 検証)**。
  `scripts/water_intensity_dropout.py` として独立スクリプトを保持しているが、
  kamiide 25 タイルに**大きな水系が無く**有効性を判定できなかった。手法自体は標準的
  (LiDAR の近赤外が水で吸収/鏡面反射 → 低 Intensity & dropout) で、湖沼や大河川を
  含むエリアでは有効と期待される。再検証時の前提:
  - 既存の `*_hag.laz` キャッシュ (Intensity 引き継ぎ済み) を使う
  - 集計結果は `.npz` キャッシュで保存できるので、閾値調整は高速 (--cache オプション)
  - z0 層 (HAG < 0.25m) の点のみで Intensity を集計
  - 細い水系 (幅 1-3m) では SMRF が水面を地面と認識して点が残るケースあり、不向き
  - 閾値の現在値: density<2 pts/m², P10<200, mean<500
  別エリアで湖沼が支配的なテレインを処理する際は本スクリプトを再利用すること。
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

## 6. odrop との連携

las2veg-rgb は PMTiles を吐くだけ、odrop はそれを `veg2isom://` プロトコルでデコードして表示する。
両者は **PMTiles ファイル経由でのみ繋がる**。コード上の依存は無い。

### 開発フローで連携する場合

1. las2veg-rgb で Phase 1→2→3 を実行し `data/output/run_xxx/vegetation.pmtiles` を生成
2. 生成物を `../odrop/public/veg/<name>.pmtiles` にコピー (odrop 開発時のみ)
3. odrop dev server を起動して、対象 PMTiles の URL を指定してブラウザ確認

ブラウザでの表示確認は人間しかできない。エージェントは PMTiles 生成 + コピー + dev server
起動までで止まり、視覚的検証はユーザに依頼する。

### エンコード/デコード式の変更時

Phase 2 の bit packing と odrop 側 `src/core/protocols/veg2isom.ts` のデコード式は
**ペアで動く**。片方だけ変えると壊れる。変更時は両 repo を同時に編集する必要があり、
VSCode のワークスペースに両方が見える状態 (multi-root か `GitHub/` ルート開き) で作業する。

odrop 側の関連事実: [odrop/CLAUDE.md](../odrop/CLAUDE.md) を参照。

## 7. このファイルの更新ルール

以下に該当する事実が発生したら **その場で追記** する (別途確認不要):

- 環境固有の落とし穴 (再現条件 + 回避策)
- 「やったけど棄却した」アプローチ (理由付き、再挑戦防止)
- 検証済みの数値根拠 (再検証コストが高いもの)
- ユーザから受けた運用上のフィードバック

技術仕様 (指標の定義、パイプラインの構造など) は `docs/spec.md` に書く。
このファイルは作業エージェントの行動ルールと環境メモに絞る。
