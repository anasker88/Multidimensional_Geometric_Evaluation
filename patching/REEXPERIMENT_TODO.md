# 再実験・追加検証 TODO — 査読監査からの派生

最終更新: 2026-07-09
出典: 進捗アーティファクト(次元幾何学推論の活性化パッチング)への外部査読監査。
本ファイルは **追加実験または追加集計が必要**な項目のみを列挙する(文面修正で済む項目はアーティファクト側で対応済み)。
各項目に: 動機 / 現状 / 実施内容(具体コマンド) / 期待成果物 / 優先度 を記す。

凡例(優先度): **P1**=結論の妥当性に直結(要・追加実験)。**P2**=頑健性・統計。**P3**=機構的フォローアップ・スコープ拡張。
凡例(状態): 🟢=実装済み(要 GPU 実行) / ✅=実装済み+実行検証済み(GPU 不要) / ⬜=未実装。

## 実装サマリ(2026-07-09 時点)

| 項目 | 状態 | 実装 |
|---|---|---|
| P1-1 レター counterbalance | 🟢 | `patch_pairs.py --counterbalance`(rotation で clean 答えを非A化) |
| P1-2 mean/resample ablation | 🟢 | `patch_ablate.py --ablation {zero,mean,resample}` |
| P1-3 中間位置・全ヘッド窓 | 🟢 | 既存 `patch_heads.py --layers all` / `patch_run.py --positions edit,last`(全位置版は要拡張、下記) |
| P1-4 解けない事例トレース | ⬜ | 未実装(設計のみ) |
| P2-1 top-k 同時 denoise patch | 🟢 | 新規 `patch_joint.py` |
| P2-2 random 多 seed | 🟢 | `patch_ablate.py --rand-seeds`(+ per-pair `rand{k}_std` 出力) |
| P2-3 split-half 信頼性(多分割+SB) | ✅ | 新規 `patch_reliability.py`(全5モデルで実行済 → `results/patching/reliability/`) |
| P2-4 (type×family) 層別 ablation | ⬜ | 未実装(集計のみ・要データ拡張) |
| P3-1〜3 | ⬜ | 未実装(設計のみ) |

**P2-3 の実行結果(検証済)**: 200 ランダム分割の split-half 信頼性(Spearman-Brown 補正天井)と次元間 cosine
を全5モデルで算出。cosine は SB 天井を **わずかに下回る**(Qwen3-8B 系統残差 ~0.015、gemma-2-9b ~0.046)=
「ほぼ不変+小さな系統残差」を定量確認。**gemma-2-27b は 4D 信頼性が低く**(200分割 raw 0.47 / SB 0.64。
アーティファクト旧値の単一偶奇分割 0.36 は不安定だった)**信号不足で判定不能** — C5(単一分割の不安定性)を実証。

---

## P1-1. レター counterbalance(A/B 非均衡の解消)

- **動機**: 全ペアで clean(正答)=トークン `A`・corrupted=`B` に固定されている
  (`patch_pairs.py`: `rotation=0` 固定 + 「clean = 辞書順で小さい answer」規約)。
  正答とレターが完全相関するため、同定した mover が運ぶのが *幾何関係* か *正答レター* かを分離できない。
  ※ 差分ベースの patching ゆえ「常時 A を書く定数ヘッド」は recovery≈0 で検出されないので、交絡は
  「検出ミス」ではなく **解釈レベル**。だが意味論的主張には分離が必須。
  ※ noise/denoise の混合では解決しない(両方向とも clean=A 構成を共有)。**レター側の counterbalance が必要**。
- **現状**: 未実施。`clean_answer` は全整列ペアで `A`(全6モデルで確認済み)。
- **実施内容**:
  1. `patch_pairs.py` を拡張し、**ペアの半数で正答を選択肢 B に置く**(rotation で選択肢順を入れ替え、
     `clean_answer` を A/B で均衡させる)。`answer <= b.answer` の canonical 規約を「rotation 込みでレター均衡」に変更。
  2. 生成し直したペアで `patch_run.py` / `patch_components.py` / `patch_heads.py` / `patch_ablate.py` を再実行。
  3. **判定**: 真の関係 mover は correct=A / correct=B の両サブセットで正の recovery を示す。
     常時-A mover は correct=B で符号が反転する。両サブセットで一致すれば「関係を運ぶ」と結論できる。
- **期待成果物**: `results/patching/pairs/*_balanced.json`、correct=A/B 別の mover recovery 比較表。
- **優先度**: **P1**(意味論的解釈の土台)。

## P1-2. mean / resample ablation(zero-ablation 単独依拠の解消)

- **動機**: §04 の必要性・過補償(drop_frac<0)・Hydra(gemma-2-9b L28 H13)の結論はすべて
  **zero-ablation 単独**に依拠。ゼロ注入は分布外介入で、`drop_frac<0` は機構的過補償ではなく
  OOD artifact でも生じうる(Heimersheim & Nanda が最も注意喚起する点)。「能動的競合」解釈は現状 **仮説**。
- **現状**: `patch_ablate.py` は zero-ablation のみ(`_make_ablate_hook` が該当ヘッドを 0 埋め)。mean/resample 未実装。
- **実施内容**:
  1. `patch_ablate.py` に `--ablation {zero,mean,resample}` を追加。
     - `mean`: 該当ヘッド `hook_z` を **baseline-ok ペア平均**で置換。
     - `resample`: 別ペア(または corrupted 側)からの値で置換。
  2. 全標準GQA5モデルで再実行し、特に **gemma-2-9b の drop_frac<0(pooled −0.11・4D −0.27)** と
     **Phase 6 の L28 H13 過補償(−0.38→+0.04)** が mean/resample でも再現するか確認。
  3. 再現すれば「能動的競合/相互調整」を主張可能。消えれば zero-ablation artifact として撤回。
- **期待成果物**: `results/patching/ablate_mean/`, `ablate_resample/` と zero との比較表。
- **優先度**: **P1**(RQ の答え「依存度が次元で動く」の頑健性)。

## P1-3. 中間位置・中間層のスイープ(探索範囲の過小性)

- **動機**: 位置スイープは **編集スパンと最終トークンの2位置のみ**。read(≈L15)→decide(≈L24)間の
  **中間層・中間位置(実際の幾何計算が起きるはずの場所)が未探索**。ヘッド解析も attn·last ピーク±2層窓に限定。
  「位置・ヘッドが次元不変」は入出力口の主張に留まり、次元依存が最も出そうな中間計算に及んでいない。
  (注: 退化するのは *全位置同時* の一括パッチであって、位置ごとの個別スイープは有効。)
- **現状**: `patch_run.py` は `--positions edit,last`。中間位置・全ヘッド窓の掃引は未実施。
- **実施内容**:
  1. `patch_run.py` を **全トークン位置 × 全層**の個別スイープに拡張(edit と last の間の位置を含む)。
     計算量が大きい場合は代表 Qwen3-8B + 次元別で先行。
  2. `patch_heads.py` の窓を attn·last ピーク±2から **全層×全ヘッド**へ拡張(少なくとも代表モデル)。
  3. 中間で次元依存(2D/3D/4D で異なる位置/ヘッドが立つ)が出るかを確認。出れば主結論の限定、出なければ強化。
- **期待成果物**: (層×全位置)ヒートマップ、全層ヘッド recovery マップ(次元別)。
- **優先度**: **P1**(「回路は次元不変」の一般化範囲を確定)。

## P1-4. 解けない事例のトレース(選択効果 / P1 の未完)

- **動機**: 全分析は **両条件正解ペア**限定。よって言えるのは「*解けた* 事例では回路が次元不変」まで。
  正解率低下を駆動する *解けない* 事例は観測対象外で、「正解率低下は再配置/伝達能力低下ではない」は
  **消去法として未完**。「baseline solvability が原因」は同語反復に近く仮説として空。
- **現状**: 未実施。baseline フィルタで incorrect ペアを除外している。
- **実施内容**:
  1. **解けない高次元ペア**(clean で誤答、または両条件で不正解)を抽出し、正答モデルの回路と比較。
  2. 解けないケースで計算がどこで壊れるか(mover 回路の *外* か *内* か)を活性で局在化。
  3. 併せて Figure 1(次元別正解率)を条件付き提示 → 無条件の因果主張へ格上げできるか検討。
- **期待成果物**: 正答 vs 誤答ペアの回路差分、失敗の局在。
- **優先度**: **P1**(RQ 本体 P1「なぜ落ちるか」への実質的回答)。

---

## P2-1. top-k 同時パッチの累積 recovery(十分性の定量)

- **動機**: 現在の「集中度」は各ヘッドの *単一* recovery に基づく(加法的でない)。
  「top-k を同時に patch したときの累積回復」= 十分性の正しい指標は **未測定**(アーティファクトにも明記)。
- **現状**: `patch_heads.py` は単一ヘッド patch のみ。`patch_ablate.py` は cumulative *ablation* はあるが
  denoise 側の cumulative *patch-in* は無い。
- **実施内容**: `patch_heads.py`(または新スクリプト)に **top-k 同時 denoise patch** を追加し、
  k=1,2,3,5 の累積 recovery を測定。単一和との乖離を報告。
- **期待成果物**: 累積 recovery 曲線(k 対 recovery)、単一和との比較。
- **優先度**: **P2**。

## P2-2. random 対照を多 seed 分布に

- **動機**: random-N 対照が **単一 seed**(`patch_ablate.py --seed 0`)。分布として示すべき。
- **実施内容**: `--n-rand` を維持しつつ **複数 seed(例 20)** で回し、random drop_frac の分布(平均±CI)を提示。
- **期待成果物**: random 対照の分布(現状の点推定を区間へ)。
- **優先度**: **P2**。

## P2-3. split-half 信頼性を多分割平均に

- **動機**: cosine の信頼性(§04)が **1回の偶奇分割**による点推定。反復平均でないため不安定。
- **実施内容**: `patch_heads.py` per-pair から **多数(例 100)のランダム二分割**で split-half cosine を計算し、
  平均と CI を報告(Spearman-Brown 補正込みの信頼性天井も併記)。
- **期待成果物**: 信頼性の区間、cosine との差の有意性判定。
- **優先度**: **P2**。

## P2-4. 依存度の次元トレンドを (type×family) セル固定で層別化

- **動機**: drop_frac の次元トレンドは **family 構成比と部分交絡**(次元ごとに type×family 分布が違う)。
  「高次元で依存度が再編成」か「構成比の変化」かを分離するには、同一 (type×family) セル内で次元比較すべき。
- **現状**: `by_dim` 集計のみ。セル固定の層別は小 n でノイズ大。
- **実施内容**: `by_dim_type` × family でセル別 drop_frac を集計。n 不足セルは **追加ペア生成**(`patch_pairs.py --per-family-cap` 引き上げ)。
- **期待成果物**: (type×family) 固定の次元別 drop_frac(交絡を除いた純次元効果)。
- **優先度**: **P2**(必要性の次元依存が交絡かの決着)。

---

## P3-1. attn vs MLP の書き込み/抑制の logit attribution

- **動機**: 「attn が運び MLP が書く」というスローガンは **未検証**(§03 で撤回済み)。
  データは attn·last 支配だが、成分が正答を *書く* のか競合を *抑制* するのかは未帰属。
- **実施内容**: 各成分・各ヘッドの **direct logit attribution**(DLA: `logit(A)`/`logit(B)` への直接寄与)を測定。
  gemma-2-27b の「後期は競合抑制の尾部」(§02 脚注)も DLA で裏取り。
- **期待成果物**: 成分/ヘッド別の write(A↑) vs suppress(B↓)分解。
- **優先度**: **P3**。

## P3-2. Phase 6 backup(L28 H13)の機構同定

- **動機**: gemma-2-9b の backup L28 H13 が **mover コピー**か **copy-suppression** かは未同定(既存ロードマップ Phase 6+)。
- **実施内容**: L28 H13 の **attention パターン / OV 方向**を解析(copy vs copy-suppression の切り分け)。
  同型の backup が他セル(IC 以外・他モデル)にも現れるか横断確認。P1-2(mean/resample)と併せて実施。
- **期待成果物**: L28 H13 の OV/attention 特性、backup の一般性。
- **優先度**: **P3**。

## P3-3. phi-4 / Qwen3-8B の採用数一致(335/146)の再検証

- **動機**: phi-4 が整列 335・baseline-ok 146 で Qwen3-8B と完全一致。異なるトークナイザで三つ組が
  一致するのはコピペ/パイプラインバグの疑いがあり **要検証**(結果ファイル上は同値)。
- **実施内容**: 両モデルのトークン整列・baseline フィルタを再導出し、独立に 335/146 を再現できるか確認。
- **期待成果物**: 一致が真の偶然か否かの確定。
- **優先度**: **P3**(整合性チェック)。

---

## 付記: 追加実験を要さない項目(集計・文面のみ)

- 成分分解(§03)の family 別内訳: 保存集計が family をプールしているだけで、整列ペアの family タグを
  index-join すれば **再集計のみ**で出せる(§02・§04 は実施済み)。再実験不要。
- 「不変」の許容幅(±5% 相対深さ)・CI 記法・cosine の系統残差の記述などは文面対応済み。
