# Patching — モデル選定の経緯と結果

活性化パッチングの知見が **モデル/ファミリーを超えて普遍か** を検証するために、どのモデルを
選び・除外したかの記録。最終更新 2026-07-01。

## 選定基準
1. **標準 Attention**（将来の per-head / `hook_z` 解析が可能。hybrid/linear は per-head 不可）
2. **十分な幾何 baseline**（clean→正解 かつ corrupted→正解 のペアが残る）
3. **TransformerLens bridge 対応**（`TransformerBridge.boot_transformers` でロード可能）
4. **48GB GPU に bf16 で載る**（単一GPUシャードで4並列）
5. **次トークンで回答**（`simple_prompt`/16tok。推論モデルは思考漏れで baseline が崩れる）
6. **ファミリー多様性**（Qwen 以外の系統を増やす）

## 検証済みモデル（パッチング実施）
| モデル | ファミリー | attn種別 | dtype | 状態 |
|---|---|---|---|---|
| Qwen3.5-9B | Qwen | hybrid/linear | fp32 | ✅ Phase 0–2 |
| Qwen3-8B | Qwen | 標準GQA | fp32 | ✅ Phase 0–2 |
| Qwen3-14B | Qwen | 標準GQA | bf16¹ | ✅ Phase 0–2 |
| gemma-2-9b-it | Google | 標準GQA | bf16 | ✅ Phase 0–2 |
| phi-4 (14B) | Microsoft | 標準GQA | bf16 | ✅ Phase 0–2 |

¹ 14B は fp32 で 48GB に載らず OOM → `--dtype bfloat16` を追加（指標は dtype 頑健と確認）。

## 除外モデルと理由
| モデル | ファミリー | 除外理由 |
|---|---|---|
| Llama-3.1-8B-Instruct | Meta | 幾何 eval が低すぎ（51/43/39）→ baseline ペアが痩せる |
| Nemotron-Nano-8B | NVIDIA(Llama基盤) | eval 49/46/35 と弱い（empty 0% で機能はする）。phi-4+gemma 優先で見送り |
| **gpt-oss-20B** | OpenAI | **eval empty ~95–100%**：16tok で思考トークンを出し次トークンで答えない → baseline 0 → パッチング不可（gpt-oss-120b を分析対象外にした理由と同じ） |
| **gemma-3-12b-it** | Google | eval は良好（67/61/48）だが **bridge 不可**：マルチモーダル（SigLIP vision tower）で gemma adapter が `'SiglipVisionModel' object has no attribute 'vision_model'` で失敗。text-only 抽出（`hf_model=language_model`）も不発 |
| **gemma-4-12B-it** | Google | eval 高性能（88/75/65）だが **bridge 未対応**：`Gemma4UnifiedForConditionalGeneration` が TransformerLens の SUPPORTED_ARCHITECTURES に無い |

### Meta「7–30B の新世代 dense」が存在しない件
- Llama-4（Scout/Maverick）は **大規模 MoE のみ**（109B/400B total）→ 48GB 不可・非 dense。
- Llama-3.x dense は **8B → 70B** で中間が無い（3.2 dense は 1B/3B、3.3 は 70B のみ）。
- 結果、3.1-8B より新しく 7–30B で 48GB に載る Meta dense は**無い**。Nemotron-Nano-8B（fine-tune）が代替候補だったが弱く見送り。

## 検証の流れ（時系列）
1. **初期**: Qwen3.5-9B（hybrid）で Phase 0–2 を確立。
2. **標準Attn候補 eval**: Qwen3-8B/14B を最有力に選定（Qwen2.5-7B / gemma-2-9b / Llama-3.1-8B も評価）。
3. **Qwen 標準GQA をパッチング**: Qwen3-8B(fp32)・Qwen3-14B(bf16)。findings の **Qwen 内** 普遍性を確認。
4. **非Qwen 拡張**: gemma-2-9b-it（Google, bridge実証）をパッチング。
5. **eval-first ゲート**: phi-4 / gemma-3-12b / Nemotron / gpt-oss-20b を先に評価し適性判定
   → phi-4 採用、gpt-oss 除外（思考漏れ）、Nemotron 見送り、gemma-3 は eval 良好も bridge 不可。
6. **gemma-4 検討**: DL 済だが bridge 未対応で除外。
7. **最終セット**: Qwen×3 + gemma-2 + phi-4 = **3ファミリー・5モデル**。

## クロスファミリー知見（普遍 vs 特有）
denoise・ピーク `L=値`、F3 は true-Yes CC の clean logit 差(A−B)。

| | attn·edit | attn·last | mlp·edit | mlp·last | F3 4D(ld/argmax) | 4D-CC baseline |
|---|---|---|---|---|---|---|
| Qwen3.5-9B (hybrid) | **L0=0.81** | L13=0.16 | L0=0.25 | L20=0.20 | −0.81 / B✗ | 0 |
| Qwen3-8B (GQA) | **L0=0.79** | L24=0.59 | L0=0.96 | L24=0.20 | −2.56 / B✗ | 0 |
| Qwen3-14B (GQA) | **L0=0.94** | L29=0.51 | L0=0.95 | L35=0.22 | −2.33 / B✗ | 0 |
| gemma-2-9b (GQA) | L12=0.13 | L28=0.48 | L14=0.19 | L38=0.28 | +3.70 / **A✓** | 0 |
| phi-4 (GQA) | L12=0.03 | L23=0.28 | **L0=0.68** | L24=0.36 | +0.58 / **A✓** | 20 |

**普遍（全5モデル・3ファミリー）**
- **resid 回路の局在（Phase 1）**: edit を早期で読み・last で後期に決定。各モデル内で**次元不変**、モデル間は深さでスケール。
- **MLP は後期で書く（mlp·last が後段でピーク）**: 全モデルで成立（≈0.20–0.36）。
- **標準GQA は最終位置に late attention mover（attn·last）を持つ**: Qwen-GQA/gemma-2/phi-4 で 0.28–0.59、hybrid 9B のみ弱い（0.16）。

**ファミリー特有（要注意）**
- **「Attention が edit を L0 で読む」は Qwen 特有**: gemma-2/phi-4 では attn·edit が弱く（0.03–0.13）遅い。代わりに **MLP が edit を読む**（phi-4 mlp·edit L0=0.68、gemma-2 L14=0.19）。＝「誰が edit を読むか」はファミリー依存。
- **F3「4D-CC が確信を持って No に崩壊」は Qwen 特有**: gemma-2(+3.70)・phi-4(+0.58) は 4D でも clean は **Yes 正答のまま**崩壊しない。
- **4D-CC baseline 崖**: Qwen×3 と gemma-2 は 0 だが phi-4 は 20 残る（崖は強いが絶対ではない）。

## bridge 非対応モデルの扱い
gemma-3（マルチモーダル）・gemma-4（未対応アーキ）は **ツール（TransformerLens 3.3.0）の制約**であり科学的な不適ではない。
TL 側の対応が進めば再検討可。代替に **gemma-2-27b-it**（text-only・gemma-2系で bridge 実証済・9B より高性能）が候補。
