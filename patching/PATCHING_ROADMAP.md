# Activation Patching ロードマップ — 次元別・構成別の幾何推論回路調査

最終更新: 2026-07-02
状態: **Phase 0–5 完了**(6モデル・3ファミリー、データセット多様化済み)。
残タスク: 全モデル full 再eval / gemma-2-9b backup 解明(任意) / 数値タスク operand 回路(任意・副軸)。

## 0. 目的と科学的問い

「Multidimensional Geometric Evaluation」の中核は **2D→3D→4D で精度が劣化する**こと
(summary: 2D≈88–93% → 4D≈56–77%)。本実験はその背後の計算機構を activation patching で因果的に調べる。

主問い:
1. **次元**: 同じ関係推論が 2D/3D/4D でどの層・どの位置で行われ、次元が上がると回路はどう変質するか。
2. **構成**: その回路は幾何構成(箱/円/角柱/…)に依存するか、それとも構成非依存か(Phase 5)。
3. **普遍性**: 知見はモデル/ファミリーを超えて普遍か、特定モデル固有か(3ファミリー・6モデルで検証)。

## 1. 設計の確定事項

### 主軸: MC・次元内・線参照スワップ
clean と corrupted は **同一次元内**で、幾何参照(線/面/超平面のラベル)を1スパンだけ入れ替え、
**答えが反転(A↔B)**するペアにする。

```
clean:     In rectangle ABEF, AB=5. ... relationship between line AB and line EF?  → AB‖EF = Parallel (A)
corrupted: In rectangle ABEF, AB=5. ... relationship between line AB and line BE?  → AB⊥BE = Perpendicular (B)
```
- 編集は末尾ラベル `EF`→`BE` の単一スパンのみ(それ以外は完全一致)。
- 2D/3D/4D の各列で同型のペアを作れる → 次元間で回路を比較できる。
- ペアは**メタデータではなく文字列の最小差分で自動検出**(`patch_pairs.build_pairs`)。

### なぜ「図形名スワップ」をMCで使わないか
MC関係問題の答えは図形トポロジーに依存し、図形名(rectangle)だけ替えると頂点ラベル構造が壊れて
問題が成立しない。図形名のみで答えが変わるのは**数値タスク**(square=4→cube=8)で、次元をまたぐため
副軸として別扱い(Phase 7)。

### 構成の多様化と A/B 統一(commit f16a8e7) ← Phase 5 の基盤
当初 type1/2(PPC/IC)は**箱系の単一構成**(rectangle→cuboid→tesseract)に偏っていた。構成一般性を
検証するため、以下を追加し**全 minimal-pair を A/B に統一**、各ペアに構成 **family タグ**を付与:
- **type1/2 に追加**: 円/球/超球(接線＋平行直径)、正三角柱/四面体プリズム、平行垂直の推移
  (CD∥AB・EF∥CD・GH⊥CD → AB&EF=∥ / AB&GH=⊥、EF↔GH の1スパン反転)。
- **type3(CC)**: `_LABEL_RE {2,5}→{1,5}` 修正で頂点1文字 swap のデータセット抽出を解禁
  (従来は合成 midpoint/innercenter のみに退化していた)。接線/交差/中点/垂線/等脚 等が mine される。
- **重要な仕様**: patching は実質 **A/B 対比のみ**を使う(合成 builder は clean=A/corr=B 固定、
  C=「どちらでもない」/D=「不能」は保持が弱く baseline filter で落ちる)。よって追加構成は全て
  「平行 vs 垂直 / 交差 vs 非交差」の A/B を1図形から出すよう設計。
- **均等化**: `family` タグ + `patch_run --per-family-cap N`(各 (次元,family) を N 件に間引き) +
  集計軸 `by_family`/`by_dim_family` で、box 優位のプールでも構成別に circuit を比較可能。

### 確定パラメータ
| 項目 | 値 | 理由 |
|---|---|---|
| 対象モデル | **6モデル・3ファミリー**(下記) | 普遍性の切り分け。標準GQA=per-head/ablation 可能。 |
| バックエンド | TransformerLens / TransformerBridge(`boot_transformers`) | resid/attn/mlp/hook_z にフック。SAE 不使用。 |
| プロンプト形式 | `simple_prompt`(assistant prefill 直後に回答レター) | 回答が次トークン → 1 forward、生成不要。 |
| rotation | **0 固定** | ペア内差分を線スワップ1点に限定(rotation は交絡)。 |
| メトリクス | `logit(clean_ans) − logit(corr_ans)` 主 | 連続・線形で段階効果(Zhang & Nanda)。 |
| patch 方向 | denoising 主 / noising 従 | denoising が局在しやすい(Heimersheim & Nanda)。 |
| デコード | greedy, **repetition_penalty=1.0** | rep=1.1 は MC primacy aversion を誘発(memory)。 |

**対象6モデル**: Qwen3.5-9B(Qwen・hybrid) / Qwen3-8B(Qwen・GQA) / Qwen3-14B(Qwen・GQA) /
gemma-2-9b-it(Google・GQA) / **gemma-2-27b-it(Google・GQA)** / phi-4(Microsoft・GQA)。
選定経緯: `patching/MODEL_SELECTION.md`。

### 不変条件(patching 成立の前提)
1. **トークン整列**: clean/corrupted は同じトークン長、編集スパン以外全一致(モデル毎に整列判定 →
   整列ペア数はトークナイザ依存で 335〜425 と変動=設計通り)。
2. **ベースライン選別**: clean→正解 かつ corrupted→正解 の両方を当てるペアのみ採用
   (`ld_clean>margin かつ ld_corr<−margin`)。分母≈0 の破綻を回避。

### メトリクス正規化
```
effect = (logitdiff_patched − logitdiff_corrupted) / (logitdiff_clean − logitdiff_corrupted)
```
denoising では 0(corrupted のまま)→ 1(clean を完全回復)。

## 2. フェーズ計画(実施状況)

### Phase 0 — ペア構築・多様化・トークン整列  ← 完了
`patching/patch_pairs.py`
- `questions_augmented.csv`(多様化済 804行相当)から次元内・同type・答え反転・単一スパン編集のペアを自動抽出。
- 各構成に **family タグ**を付与(`_classify_family`: box / prism / circle·sphere / transitivity /
  simplex·tetra / intersect / midpoint / isosceles·equilateral / altitude / …)。
- **各モデルのトークナイザで再 mine**(`--dims 2,3,4 --prompt-type simple_prompt`、合成 top-up=なし)。
  整列ペア: Qwen3.5-9B 342 / Qwen3-8B 335 / Qwen3-14B 335 / gemma-2-9b 389 / gemma-2-27b 425 / phi-4 335。
  **全ペア A/B**。出力: `results/patching/pairs/{model}_aligned.json`。

### Phase 1 — 残差ストリーム (層 × 位置) スイープ  ← 完了(6モデル)
`patching/patch_run.py`
- 各層 `blocks.L.hook_resid_post` を **特定位置だけ** patch(`--positions edit,last`)→ 最終位置の
  logit-diff 回復を記録。denoise/noise 両方向。`all` は degenerate(全層 1.0)確認用。
- 集計軸: `by_dim` / `by_type` / `by_dim_type` / `by_kind`(real/synthetic) / **`by_family`** / **`by_dim_family`**。
- **普遍知見**: モデルが解けるペアでは **編集は早期に読まれ(edit ハンドオフ L8–L19)、答えは後期に
  最終位置で組み立てられる(decision rise)**。次元不変・6モデル普遍(深いモデルほど絶対層が後ろへ)。
- 実行(例): `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m patching.patch_run
  --pairs results/patching/pairs/qwen3_8b_aligned.json --positions edit,last --out results/patching/run/qwen3_8b`
  (14B/gemma/phi4 は `--dtype bfloat16`)。速度 ~15–30s/ペア。

#### type 横断分析のガードレール
- **比較は妥当・推奨、合算は不可**。type1(方向)/type2(接続)/type3(アフィン従属)は別タスク。
  第一単位 = `by_dim_type`、第二 = `by_type`。3type プールの単一平均(`all`)は主結果にしない。
- 交絡注意: 編集位置が type で違う(type3 は文中)→ type 間比較は `last` 位置を主に、差は中盤層で見る。
  baseline-ok の n を群ごとに併記。

### Phase 2 — 成分の局在化(attn vs mlp)  ← 完了(6モデル)
`patching/patch_components.py`(`linear_attn.hook_out` / `attn.hook_out` を自動解決)
- **普遍**: MLP は後期に書く(mlp·last 0.20–0.36) / 標準GQA は最終位置に late attention mover
  (attn·last: 8B 0.59・14B 0.51・gemma-2-9b 0.48・phi-4 0.28、hybrid 9B のみ弱い 0.16)。
- **Qwen 特有**: 「attn が edit を L0 で読む」(gemma-2/phi-4 は MLP が読む)。
- 旧「Phase 3(次元間比較)」は Phase 1/2 のクロスファミリ検証に吸収済。

### Phase 3 — per-head mover 解析(hook_z)  ← 完了(標準GQA 5モデル)
`patching/patch_heads.py`(標準 attention のみ = hook_z 必須。hybrid 9B は対象外)
- attn·last ピーク層周辺で `blocks.L.attn.hook_z` をヘッド単位に patch(最終位置・denoise)。
- **普遍**: 後期 mover は少数の専門ヘッド(top2 で正 recovery の 27–43%、top5 で 44–76%)。
  トップ: Qwen3-8B L24 H29/H31、Qwen3-14B L28 H21/L29 H20、gemma-2-9b L26 H12/L28 H8、
  phi-4 L22 H28/L23 H1、gemma-2-27b L23 H2/H3。IOI の name-mover 的スパース性が普遍。

### Phase 4 — ablation による因果検証(necessity)  ← 完了(標準GQA 5モデル)
`patching/patch_ablate.py`(Phase 3 の順位から top-k を clean 実行で zero-ablate)
- `drop_frac = (ld_clean − ld_ablated)/(ld_clean − ld_corr)`(1.0=corruption 完全再現、負=除去で確信↑=backup)。
- **多様化データでの結果(top5 mover / random5)**:

  | モデル | top5 | random5 | 判定 |
  |---|---|---|---|
  | Qwen3-8B | +0.10 | +0.00 | 必要 |
  | Qwen3-14B | +0.14 | −0.00 | 必要 |
  | phi-4 | +0.15 | +0.01 | 必要 |
  | **gemma-2-27b** | **+0.19** | +0.00 | **必要** |
  | gemma-2-9b | **−0.11** | −0.00 | **backup 冗長** |

- **知見**: sufficiency(Phase 3 の recovery)はスパース&普遍。necessity は **gemma-2-9b のみ backup 冗長**で、
  **gemma-2-27b は必要**=backup は gemma ファミリー全体ではなく **9B 固有(スケール/冗長性依存)**。
  「復元すれば少数ヘッドで足りる」≠「除去すると壊れる」。

### Phase 5 — 構成一般性(by_family)  ← 完了 ★
Phase 0 の多様化(A/B・family タグ)を活かし、**同じ mover 回路が構成に依らず成立するか**を検証。
- **指標**: mover 立ち上がり層 = denoise·last の half-recovery(mean≥0.5 の最初の層)を **family 別**に比較。
- **結果**: どのモデルでも **全構成ファミリでほぼ同一の深さ**で答えが組み立てられる:

  | モデル | pooled(深さ%) | family 間の幅 | baseline |
  |---|---|---|---|
  | Qwen3.5-9B | 59% | 59–59%(9 family 完全一致) | 218/342 |
  | Qwen3-8B | 67% | 64–67% | 146/335 |
  | Qwen3-14B | 65% | 65–72% | 186/335 |
  | gemma-2-9b | 62% | 57–62% | 161/389 |
  | gemma-2-27b | 63% | 57–72% | 141/425 |
  | phi-4 | 50% | 48–52% | 146/335 |

- **含意**: family 間の幅は各モデル ±0–3層(≈0–7% 深さ)。**box が数で支配的でも、構成別に見れば全て同一局在**
  → mover 回路は**構成非依存**であり「box アーティファクト」ではない。多様化の目的を達成。
- 再現(read-only 集計例): `by_family`/`by_dim_family` を各 `patch_results.json` から抽出。

### Phase 6(任意) — gemma-2-9b の backup 解明
- backup 冗長は **gemma-2-9b 固有**(Phase 4)。反復/二重ヘッド ablation で backup ヘッドを特定し、
  mover の attend 先を読んで「単一経路」でなく「冗長回路(Hydra)」として記述する。

### Phase 7(任意・副軸) — 数値タスクの図形同定/operand 回路
- 図形名スワップ(square↔cube 等, span patching)で図形クラス/次元表現を局在化。
- パラメータスワップ(side length 1↔2)で operand 処理回路(Stolfo 型)。

## 3. 知見サマリ(普遍 vs モデル特有)
| 知見 | 区分 | 詳細 |
|---|---|---|
| 次元不変回路(早期read→後期decide) | **普遍** | 2D/3D/4D 同一・6モデル(絶対層は深さでシフト) |
| 4D 難化は回路移動でなく正解率低下 | **普遍** | 次元が上がるほど baseline 通過ペアが減る(回路の位置は不変) |
| MLP 後期書き込み / 標準GQA late attn mover | **普遍** | mlp·last 0.20–0.36、attn·last は hybrid 除き強い |
| 少数の専門 mover ヘッド(sufficient) | **普遍** | top2 27–43%(標準GQA 5モデル) |
| **mover 回路の構成非依存(Phase 5)** | **普遍** | box/円/角柱/推移/simplex が同一深さ(6モデル) |
| attn が edit を L0 で読む | Qwen 特有(要再確認) | gemma-2/phi-4 は MLP が読む。components 多様化再実行で確認中 |
| mover の necessity(除去で崩れる) | 規模/冗長性依存 | Qwen/phi-4/gemma-2-**27b** 必要、gemma-2-**9b** は backup |

> **削除した旧知見**: 「4D-CC 確信崩壊(Yes→No 反転)は Qwen 特有」は、多様化 CC で精査した結果 **構成特有**(simplex では小型 Qwen、box では 14B/phi-4、intersect では 14B/gemma-2-9b/phi-4 が崩壊、円/接線は頑健)であり、クリーンな「Qwen 特有」ではないと判明。重要度が低いためレポートから除外。

## 4. 残タスク
- **全モデル full 再eval**(`evaluation/evaluate.py`, `data/questions_augmented.csv`) — 多様化データでの
  サマリ整合 + 新構成の解答可能性。patching の baseline filter とは独立(必須ではないが推奨)。
  gemma-2-27b は A6000 単体不可(A100 80GB / device_map)、vLLM TP=2 は NCCL 不安定 → HF フォールバック。
- Phase 6(gemma-2-9b backup)/ Phase 7(数値 operand) は任意。

## 5. 既存コードの活用
- モデルロード: `boot_transformers`(SAE 抜きの薄いローダ)。
- プロンプト: `prompting.make_prompt_mc` / `apply_chat_template`(rotation=0)。
- 可視化: `patch_run`/`patch_heads`/`patch_ablate` が各自プロット出力。

## 6. 環境・モデルロード(解決済)
- **環境**: 単一 `.venv`(`transformer-lens` 3.3.0 が vllm/sae-lens と共存)。起動 `.venv/bin/python`。
- **ロード**: `TransformerBridge.boot_transformers("<model>", device="cuda", dtype=…)`。ロード ~8分。
- **dtype**: 14B/gemma-2/phi-4 は 48GB A6000 に fp32 不可 → `--dtype bfloat16`(指標は dtype 頑健)。
  27B は 48GB 単体不可(A100 80GB / device_map / shard)。
- **実行**: 4× RTX A6000(48GB)。長時間ジョブは **`setsid` でデタッチ**(ssh セッション終了に巻き込まれない)、
  **同時2本**に抑えて host RAM/swap 逼迫を回避(SLURM 割り当て外で直接実行のため)。
- **アーキ制約**: hybrid 9B は per-head hook_z なし(Phase 3/4 対象外)。gemma-3(vision tower)/gemma-4
  (TL 未対応)は bridge 不可。gpt-oss-20B は 16tok で empty ≈100%(推論漏れ)。

## 参考(設計根拠)
- Heimersheim & Nanda, How to use and interpret activation patching (2024) — denoising 推奨, 落とし穴
- Zhang & Nanda, Best Practices of Activation Patching (2024) — logit diff 推奨
- Meng et al., Causal Tracing (2022) — (層×位置) マップ
- Wang et al. (ICLR'23) — IOI, name-mover ヘッドと最小ペア
- Conmy et al. (NeurIPS'23) — ACDC, 成分/ヘッド単位の自動回路同定
- Stolfo et al. (EMNLP'23) — attn が operand→answer 移送, 後段 MLP が計算
