# Activation Patching ロードマップ — 次元別・層ごとの幾何推論回路調査

最終更新: 2026-06-29

## 0. 目的と科学的問い

「Multidimensional Geometric Evaluation」の中核は **2D→3D→4D で精度が劣化する**こと
(summary: 2D≈88–93% → 4D≈56–77%)。本実験はその背後の計算機構を
activation patching で因果的に調べる。

主問い: **同じ関係推論が、次元(2D/3D/4D)ごとに、どの層・どのトークン位置で
行われているか。次元が上がると回路はどう変質するか。**

## 1. 設計の確定事項

### 主軸: MC・次元内・線参照スワップ
clean と corrupted は **同一次元内**で、幾何参照(線/面/超平面のラベル)を
1スパンだけ入れ替え、**答えが反転(A↔B)**するペアにする。

```
clean:     In rectangle ABEF, AB=5. ... relationship between line AB and line EF?  → AB‖EF = Parallel (A)
corrupted: In rectangle ABEF, AB=5. ... relationship between line AB and line BE?  → AB⊥BE = Perpendicular (B)
```
- 編集は末尾ラベル `EF`→`BE` の単一スパンのみ(それ以外は完全一致)。
- 2D/3D/4D の各列で同型のペアを作れる → 次元間で回路を比較できる。
- データ上、clean/corrupted は `questions_augmented.csv` の type=A 行 / type=B 行に
  対応(例: 行2 と 行26)。ペアは**メタデータではなく文字列の最小差分で自動検出**する。

### なぜ「図形名スワップ」をMCで使わないか
MC関係問題の答えは図形トポロジーに依存し、図形名(rectangle)だけ替えると
頂点ラベル構造が壊れて問題が成立しない。図形名のみで答えが変わるのは
**数値タスク**(例: vertices in a square=4 → cube=8)で、これは次元をまたぐため
副軸として別扱い(将来 Phase 5 で operand/図形同定回路の調査に使う)。

### 確定パラメータ
| 項目 | 値 | 理由 |
|---|---|---|
| 対象モデル | Qwen3.5-9B(評価セット中で最小の密モデル) | MoE/gpt-oss は patching に不利。次元劣化が明確(2D84/4D56)。 |
| バックエンド | TransformerLens / TransformerBridge(`recon_eval.py` と同経路) | フックで resid/attn/mlp に介入。**SAEは不使用**。 |
| プロンプト形式 | `simple_prompt`(assistant prefill 直後に回答レター) | 回答が次トークン → 1 forward で logit 取得、生成不要。 |
| rotation | **0 に固定** | ペア内の差分を線スワップ1点に限定(rotationは交絡)。A=Parallel,B=Perp が不変。 |
| メトリクス | `logit(clean_ans) − logit(corr_ans)` を主、確率・正解率を従 | 連続・線形で段階的効果が見える(Zhang & Nanda)。 |
| patch 方向 | denoising 主(corrupted に clean を注入)/ noising 従 | denoising の方が局在しやすい(Heimersheim & Nanda)。 |
| デコード | greedy, **repetition_penalty=1.0** | memory: rep=1.1 が MC primacy aversion を誘発。 |

### 不変条件(patching 成立の前提)
1. **トークン整列**: clean/corrupted は同じトークン長で、編集スパン以外の全位置が一致。
   → スワップ部のトークン長が一致するペアのみ採用(Phase 0 でフィルタ)。
2. **ベースライン選別**: clean→正解 かつ corrupted→正解 の**両方を当てている**
   ペアのみ採用。両条件が分離していないと正規化メトリクスの分母≈0で破綻。

### メトリクス正規化
```
effect = (logitdiff_patched − logitdiff_corrupted) / (logitdiff_clean − logitdiff_corrupted)
```
denoising では 0(corrupted のまま)→ 1(clean を完全回復)。

## 2. フェーズ計画

### Phase 0 — ペア構築とトークン整列  ← 完了
`validation/patch_pairs.py`
- `questions_augmented.csv` から次元内・同type・答え反転・単一スパン編集のペアを自動抽出。
- 対象モデルのトークナイザで、編集スパンのトークン長一致と整列を検証・フィルタ。
- **2D/3D の偶然一致不足対策**: `--balance N` で各次元をベンチ同一テンプレの**合成ペア**
  (rectangle / rectangular solid / tesseract、対辺=平行A↔隣辺=垂直B、トークン整列保証)で N までトップアップ。
- 出力 JSON: clean/corrupted プロンプト(chat 整形済)、正解レター、編集トークン位置 [start,end)、次元、type、出所(`questions_augmented` / `synthetic_{d}d`)。

**option-set type の網羅**: 抽出ペアは type1/2 が中心、type3 は実データに単一ラベル編集の
最小ペアが存在しない(各構成が単一の答えに固定)。`--balance-types 1,2,3` で各 type を合成補完:
- **type1**(平行/垂直): rectangle/solid/tesseract、対辺=平行A↔隣辺=垂直B。
- **type2**(交差/非交差): 同一幾何で option set だけ差し替え、隣辺=交差A↔対辺=非交差B。
- **type3**(共線/共面, Yes/No): 構成点(2D=midpoint, 3D/4D=innercenter)が問う平面上にあれば
  Yes(A)、外れれば No(B)。編集は構成対象ラベル1スパン(AC→BC / ACD→BCD / ACDE→BCDE)。

**実行結果**(Qwen3.5-9B トークナイザ, simple_prompt, `--balance 40 --balance-types 1,2,3`):
- (次元 × type)= 各40整列(4D type1のみ実データ43)→ **計 363 整列ペア**、全 A→B。
- 編集スパンは 1〜2 トークン。出力: `output/patch_pairs/qwen35_9b_aligned.json`。
- コマンド: `python validation/patch_pairs.py --balance 40 --balance-types 1,2,3 --aligned-only --out output/patch_pairs/qwen35_9b_aligned.json`
- type3 注記: collinear/coplanar は line-relationship とは別タスク。次元横断比較は type 内で行う。

### Phase 1 — 残差ストリーム (層 × 位置) スイープ  ← 実装済
`validation/patch_run.py`
- clean を `run_with_cache`、corrupted のキャッシュを取得。各層 L の `blocks.L.hook_resid_post` を
  **特定位置だけ** patch → 最終位置の logit-diff を記録。denoise/noise 両方向。
- **重要な修正**: 当初の「全位置 patch」は **degenerate**(resid_post 全位置は完全な状態なので
  下流は donor で完全に決まり、正規化効果は全層で 1.0 になり局在不能)。スモークテストで確認。
  → 位置を限定: `--positions edit,last`(`edit`=編集ラベルのトークンスパン、`last`=回答位置)。
  これは causal tracing の (層 × 位置) 分解そのもの。`all` は degenerate 確認用に残す。
- ベースライン選別(clean→A・corrupted→B 両正解、`logit(A)-logit(B)` の符号)をここで実施。
- 全ペア平均で **層 × 効果** 曲線を (次元 × type) ごとに描画・JSON 出力。
- 実行: `CUDA_VISIBLE_DEVICES=0 .venv-patch/bin/python validation/patch_run.py
  --pairs output/patch_pairs/qwen35_9b_aligned.json --positions edit,last --out output/patch_run/qwen35_9b`
- 集計は (次元/type/次元×type/real-vs-synthetic) で分解、type別プロットを出力。
- **スモーク検証(9ペア)で causal-tracing シグネチャ確認**(denoise, pooled):
  edit位置 = 早期1.0 → L8-14で受け渡し → 後期≈0、last位置 = 早期≈0 → L16-28で上昇、
  all位置 = 全層1.0(退化=非情報、サニティ用)。位置分解が正しく機能。
- 速度: 約30s/ペア(positions=edit,last × directions=denoise,noise × 32層)。363ペアで ~3h。

### Phase 2 — 位置・成分の局在化(確認)
- **(層 × 位置) ヒートマップ**(causal tracing): resid_post を
  (a) 編集スパンのみ / (b) 回答位置のみ / (c) 選択肢トークン / (d) 全位置 で別々に patch。
  「読み取り(編集位置・早期)」と「計算・決定(回答位置・後期)」を分離。
- 受け渡し層で `attn_out` を patch → query=回答位置・key/value=編集位置の mover head を同定。
  続いて `mlp_out` を patch して計算が MLP に乗るか確認。

### type 横断分析の妥当性(解釈ガードレール)
- **比較は妥当・推奨、合算は不可**。ペア内正規化+全ペア A→B 方向統一で type 間でも曲線は
  比較可能。type1(方向)/type2(接続)/type3(アフィン従属)は別タスクなので「次元局在が
  タスク共有か固有か」を**並べて比較**する価値がある。一方 3 type を**プールした単一平均**は
  異機構を混ぜるため主結果にしない(`all` は粗サマリ)。
- **第一単位 = type 固定 × 次元比較**(`by_dim_type`)。**第二 = type 間比較**(`by_type`)。
- 交絡注意: 編集位置が type で違う(type3 は文中)→ type 間比較は `last` 位置を主に。
  後段の A/B 読み出しは 3 type 共通ゆえ後段の一致は自明、差は**中盤層**で見る。
  baseline-ok の n を群ごとに併記。

### Phase 3 — 次元間比較と頑健性
- 2D/3D/4D の (層×位置) マップを重ね、次元上昇で回路がどう後方/分散化するかを比較。
- テンプレ/パラフレーズ横断で同層が出るか(=回路主張の根拠)。
- backup/Hydra 効果を考慮し sufficiency 表現に統一。distribution-bound を注記。

### Phase 5(任意・副軸) — 数値タスクの図形同定/operand 回路
- 図形名スワップ(square↔cube 等, span patching)で図形クラス/次元表現を局在化。
- パラメータスワップ(side length 1↔2)で operand 処理回路(Stolfo 型)。

## 3. 既存コードの活用
- モデルロード: `recon_eval.py` の TransformerBridge 初期化を雛形に、SAE抜きの薄いローダを新規追加。
- プロンプト: `prompting.make_prompt_mc` / `apply_chat_template`(rotation=0 で呼ぶ)。
- 回答抽出・採点: `validation/ablation_eval._extract_answer`。
- 可視化: `validation/visualize.py` に層曲線/ヒートマップを追記。

## 4. リスクと分岐 — モデルロード(Phase 1 着手前の最重要決定)
**実測**: インストール済 `transformer-lens==2.17.0` の `OFFICIAL_MODEL_NAMES` は
Qwen3 系(0.6B/1.7B/4B/8B/14B)を持つが **Qwen3.5 は未収録**(`Qwen/Qwen3.5-9B` なし)。
評価セットの密モデル(Qwen3.5-9B/27B, Qwen3-32B, gemma-4)はいずれも 2.17.0 では非ネイティブ。
TransformerLens のリリースノートでは新しめのバージョンで Qwen3.5 対応が追加されている。

選択肢:
- **(A) transformer-lens を Qwen3.5 対応版へ更新**(評価セットと完全一致)。更新後 `Qwen/Qwen3.5-9B`
  が `OFFICIAL_MODEL_NAMES` に入ることを確認。SAE未使用なので sae-lens 連動は問題になりにくいが、
  依存衝突回避のため **patching 専用 venv** を別に切るのが安全。
- **(B) Qwen3-8B を使用**(2.17.0 でネイティブ・密・~8B で patching 最適)。評価セット外なので
  `evaluate.py` で同条件ベースラインを取り直す(同 Qwen3 系で Qwen3.5-9B に近い)。即着手可。
- **(C) TransformerBridge**(`recon_eval.py` の `boot_transformers` 経路)で Qwen3.5-9B を HF 直読み。
  非収録でもフック可能な場合がある。`model.hook_dict` に `blocks.{L}.hook_resid_post` が出るか要検証。

推奨: **(A) を第一**(評価整合)、ダメなら **(B) を即時フォールバック**。
- 9B のメモリ → bf16・単一GPUで全層 resid キャッシュが乗るか確認(必要なら層を分割キャッシュ)。

### 解決済(2026-06-29 スパイク結果)
- **環境**: 評価env非破壊のため `.venv-patch` を新設(base `.venv` の site-packages を
  `.pth` でリンクし torch/transformers を再利用)、`transformer-lens==3.4.0` を `--no-deps` で重ねた。
  起動: `.venv-patch/bin/python ...`。
- **ロード経路確定**: `from transformer_lens.model_bridge import TransformerBridge;
  TransformerBridge.boot_transformers("Qwen/Qwen3.5-9B", device="cuda")` で成功。
  32層 / d_model=4096 / d_vocab=248320。`blocks.{L}.hook_resid_post` 発火確認済。
  回答トークン: ` A`=357, ` B`=417(単一トークン)。
- **注意1**: ロードに ~480s。Phase 1 は1回ロードで全ペア処理する設計。
- **注意2(Phase 2 に影響)**: 本モデルは(一部層が)**linear attention**(`blocks.{L}.linear_attn.*`,
  `hook_recurrence_out` 等)。標準 `hook_attn_out` は未解決。**Phase 1 の resid_post は無影響**だが、
  Phase 2 のヘッド/attention 解析は linear attention 用に再設計が必要。
- 実行GPU: 4×80GB 空き。bf16 9B は単一GPUで余裕。

## 参考(設計根拠)
- Heimersheim & Nanda, How to use and interpret activation patching (2024) — denoising 推奨, 解釈の落とし穴
- Zhang & Nanda, Best Practices of Activation Patching (2024) — logit diff 推奨
- Meng et al., Causal Tracing (2022) — (層×位置) マップ
- Stolfo et al. (EMNLP'23) — attn が operand→answer 移送, 後段 MLP が計算
- Nikankin et al. (2024) — 後段 MLP の sparse heuristics
