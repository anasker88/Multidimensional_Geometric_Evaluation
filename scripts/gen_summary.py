#!/usr/bin/env python
"""現在の環境(numeric_mc + confidence/empty 反映)で評価した全モデルの summary.md を生成する。"""
import csv, os, glob, re
from collections import defaultdict, Counter
csv.field_size_limit(10**9)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("EVAL_BASE", "results/eval/final_greedy_20260626")
OUT = os.path.join(ROOT, BASE, "summary.md")

# (display, dir = results/<BASE>/<model_safe>, prompt, TP, maxtok)
MODELS = [
 ("Qwen3.5-2B",                 f"{BASE}/Qwen_Qwen3.5-2B",                 "simple_prompt", 1, 16),
 ("Qwen3.5-4B",                 f"{BASE}/Qwen_Qwen3.5-4B",                 "simple_prompt", 1, 16),
 ("Qwen3.5-9B",                 f"{BASE}/Qwen_Qwen3.5-9B",                 "simple_prompt", 1, 16),
 ("Qwen3.5-27B",                f"{BASE}/Qwen_Qwen3.5-27B",                "simple_prompt", 2, 16),
 ("Qwen3.5-35B-A3B",            f"{BASE}/Qwen_Qwen3.5-35B-A3B",            "simple_prompt", 2, 16),
 ("Qwen3.5-122B-A10B",          f"{BASE}/Qwen_Qwen3.5-122B-A10B",          "simple_prompt", 4, 16),
 ("Qwen3-30B-A3B",              f"{BASE}/Qwen_Qwen3-30B-A3B",              "simple_prompt", 2, 16),
 ("Qwen3-32B",                  f"{BASE}/Qwen_Qwen3-32B",                  "simple_prompt", 2, 16),
 ("Qwen3-Next-80B-A3B-Instruct",f"{BASE}/Qwen_Qwen3-Next-80B-A3B-Instruct","simple_prompt", 4, 16),
 ("gemma-4-E4B-it",             f"{BASE}/google_gemma-4-E4B-it",           "simple_prompt", 1, 16),
 ("gemma-4-12B-it",             f"{BASE}/google_gemma-4-12B-it",           "simple_prompt", 1, 16),
 ("gemma-4-26B-A4B-it",         f"{BASE}/google_gemma-4-26B-A4B-it",       "simple_prompt", 2, 16),
 ("gemma-4-31B-it",             f"{BASE}/google_gemma-4-31B-it",           "simple_prompt", 2, 16),
 ("Olmo-3-7B-Instruct",         f"{BASE}/allenai_Olmo-3-7B-Instruct",      "simple_prompt", 1, 16),
 ("Olmo-3-7B-RL-Zero-Math",     f"{BASE}/allenai_Olmo-3-7B-RL-Zero-Math",  "simple_prompt", 1, 16),
]
TYPES = ["1","2","3","numeric","numeric_mc"]
TYPELABEL = {"1":"PPC","2":"IC","3":"CC","numeric":"Numeric","numeric_mc":"Numeric-MC"}

# Per-type accumulator layout (index): 0 count, 1 correct, 2 empty,
# 3 conf_sum, 4 conf_n, 5 conf_ok_sum, 6 conf_ok_n, 7 conf_no_sum, 8 conf_no_n
def _new():
    return [0, 0, 0, 0.0, 0, 0.0, 0, 0.0, 0]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, d, "dim_*_per_question.csv"))):
        dim = os.path.basename(f).split("_")[1]
        by = defaultdict(_new)
        for x in csv.DictReader(open(f)):
            t = x["type"]; a = by[t]
            a[0] += 1
            ok = x.get("is_correct") == "1"
            if ok: a[1] += 1
            if not (x.get("predicted_raw") or "").strip(): a[2] += 1
            c = (x.get("confidence") or "").strip()
            if c:
                cv = float(c)
                a[3] += cv; a[4] += 1
                if ok: a[5] += cv; a[6] += 1
                else:  a[7] += cv; a[8] += 1
        out[dim] = by
    return out


def overall(by):
    agg = _new()
    for v in by.values():
        for i in range(9):
            agg[i] += v[i]
    return agg


def acc(a):    return 100 * a[1] / a[0] if a[0] else None
def emp(a):    return 100 * a[2] / a[0] if a[0] else None
def conf(a):   return a[3] / a[4] if a[4] else None
def conf_ok(a):return a[5] / a[6] if a[6] else None
def conf_no(a):return a[7] / a[8] if a[8] else None
def _f(v, p=1): return f"{v:.{p}f}" if v is not None else "—"


data = {name: load(d) for name, d, _, _, _ in MODELS}

# 考察は手書き。下のマーカー間を編集すれば、本スクリプト再生成でも保持される。
FINDINGS_PLACEHOLDER = "_(考察未記入。FINDINGS の START/END コメント間に手書きで追記してください。)_"
def extract_findings(path):
    if os.path.exists(path):
        m = re.search(r"<!-- FINDINGS:START -->(.*?)<!-- FINDINGS:END -->", open(path, encoding="utf-8").read(), re.S)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return FINDINGS_PLACEHOLDER
findings = extract_findings(OUT)

L = []
L.append("# Multidimensional Geometric Evaluation — Summary\n")
L.append("**Generated**: 2026-06-26  |  **decoding**: greedy + repetition_penalty=1.0  |  **vLLM**: 0.23.0  |  **GPU**: 4× NVIDIA A100 80GB PCIe\n")
L.append("評価データ: `data/questions_augmented.csv` + `data/numeric_augmented.csv`（**numeric / numeric_mc を含む**）。次元 2D/3D/4D。\n")

L.append("\n## 考察 / Findings\n")
L.append("<!-- FINDINGS:START -->")
L.append(findings)
L.append("<!-- FINDINGS:END -->\n")

L.append("\n## スコア定義\n")
L.append("| 指標 | 定義 |")
L.append("|---|---|")
L.append("| **Acc%**（正答率） | 正解数 ÷ 出題数。各 rotation 変種も 1 問として計上。 |")
L.append("| **Empty%**（空率） | 解析可能な解答を抽出できなかった割合（`extract_answer`/`extract_numeric` が None）。思考漏れ・冗長出力で増える。 |")
L.append("| **Conf**（信頼度） | モデルが出力した「自分の答えトークン」に割り当てた確率 = exp(logprob)。MC はレター、numeric は数値先頭トークン。vLLM `logprobs_mode=\"raw_logprobs\"`（既定）= 元 logits の log-softmax で、temperature・repetition_penalty 適用前の生確率。選択トークンには decoding/penalty が影響しうるが conf 値自体は不変。HF/Azure 経路は非対応（空欄）。 |")
L.append("| **Conf✓ / Conf✗** | 正答時 / 誤答時の平均 Conf。Conf✓>Conf✗ なら「自信と正誤」が整合（弁別的）。差が小さい=過信。 |")
L.append("| **提示スロット別 acc/conf** | 正解がレター位置 L に提示された変種での Acc% と平均 Conf。位置バイアス測定用。 |")
L.append("| **出力レター別 conf(n)** | モデルが実際に出力したレター別の平均 Conf と選択数 n。出力位置に内在する信頼度・選好の偏り。 |")
L.append("| **A→C 差** | A 位置と C 位置の Acc / Conf の差。負（A<C）= 先頭レター回避（primacy aversion）。 |")

L.append("\n## 実行条件 / Reproduction\n")
L.append("sampling 系は全ラン共通でデフォルト値（コマンドで上書きせず）。\n")
L.append("| 項目 | 値 |")
L.append("|---|---|")
for k, v in [("decoding", "greedy (do_sample=False; vLLM temperature=0)"),
             ("repetition_penalty", "1.0"),
             ("top_p / top_k", "1.0 / -1 (greedy 下で無効)"),
             ("seed", "0 (vLLM 既定)"), ("batch_size", "8"),
             ("dtype", "bfloat16"), ("gpu_memory_utilization", "0.85"),
             ("max_new_tokens", "16"), ("dims", "2,3,4")]:
    L.append(f"| {k} | {v} |")
L.append("\n### モデル別の prompt / TP / max_new_tokens\n")
L.append("| モデル | prompt_type | TP | max_new_tokens |")
L.append("|---|---|---|---|")
for name, d, p, tp, mt in MODELS:
    L.append(f"| {name} | `{p}` | {tp} | {mt} |")

L.append("\n## 問題種別\n")
L.append("| 略称 | 種別 | 選択肢/形式 |")
L.append("|---|---|---|")
L.append("| PPC | 平行/垂直分類 | A=Parallel B=Perpendicular C=Neither D=Cannot be inferred |")
L.append("| IC | 交差分類 | A=Intersecting B=Not intersecting C=Cannot be inferred |")
L.append("| CC | 共線性判定 | A=Yes B=No C=Cannot be inferred |")
L.append("| Numeric | 数値直接回答 | 数値出力 |")
L.append("| Numeric-MC | 数値の多肢選択 | 正解+誤答モード3択をrotate（A–D） |")

# ---- データセット基本情報（ローテーション前の正準GT） ----
qa = list(csv.DictReader(open(os.path.join(ROOT, "data/questions_augmented.csv"), encoding="utf-8")))
na = list(csv.DictReader(open(os.path.join(ROOT, "data/numeric_augmented.csv"), encoding="utf-8")))
ROT = {"1": 4, "2": 3, "3": 3, "numeric": 1, "numeric_mc": 4}
dsdim, dsans = {}, {}
for t in ("1", "2", "3"):
    tr = [r for r in qa if r["type"] == t]
    dsdim[t] = {d: sum(1 for r in tr if (r.get(d) or "").strip() not in ("", "-")) for d in ("2D", "3D", "4D")}
    a = Counter()
    for r in tr:
        for d in ("2D", "3D", "4D"):
            if (r.get(d) or "").strip() not in ("", "-"):
                a[(r["answer"] or "").strip()] += 1
    dsans[t] = a
nperdim = Counter(r["dimension"] for r in na)

L.append("\n## データセット基本情報\n")
L.append(f"`questions_augmented.csv`: {len(qa)} 行（ラベル置換 augment）。`numeric_augmented.csv`: {len(na)} 行。"
         "数値は**ローテーション前（=データセットの正準GT）**。\n")

L.append("### 出題数（ローテーション前）と評価時 N\n")
L.append("評価時は各問の全 rotation 変種を 1 問として計上するため、**評価 N = pre × rotation**。\n")
L.append("| Type | 2D | 3D | 4D | 計(pre) | ×rotation |")
L.append("|---|---|---|---|---|---|")
labelmap = {"1": "PPC", "2": "IC", "3": "CC"}
for t in ("1", "2", "3"):
    d = dsdim[t]; tot = sum(d.values())
    L.append(f"| {labelmap[t]} | {d['2D']} | {d['3D']} | {d['4D']} | {tot} | ×{ROT[t]} |")
L.append(f"| Numeric | {nperdim.get('2',0)} | {nperdim.get('3',0)} | {nperdim.get('4',0)} | {len(na)} | ×1 |")
L.append(f"| Numeric-MC | {nperdim.get('2',0)} | {nperdim.get('3',0)} | {nperdim.get('4',0)} | {len(na)} | ×4 |")

L.append("\n### ローテーション前 正解ラベル分布（正準GT）\n")
L.append("素の accuracy はこの分布に依存（多数派ラベルを当てやすい）。位置バイアスは rotation 集計で相殺されるが、"
         "**ラベル自体の不均衡は残る**ため解釈時に留意。\n")
L.append("| Type | A | B | C | D | 偏り |")
L.append("|---|---|---|---|---|---|")
NOTE = {"1": "**D=Cannot be inferred** が最多", "2": "**A=Intersecting** が優勢", "3": "A/B 均衡・**C=Cannot** が少数"}
for t in ("1", "2", "3"):
    a = dsans[t]; tot = sum(a.values()) or 1
    def cell(k):
        return f"{a[k]} ({100*a[k]/tot:.0f}%)" if k in a else "—"
    L.append(f"| {labelmap[t]} | {cell('A')} | {cell('B')} | {cell('C')} | {cell('D')} | {NOTE[t]} |")
L.append("\n- **Numeric-MC**: 正解は構成上すべて canonical **A**（正解値を index0 に置く）。rotation で提示位置を A–D に均等化。")
L.append("- PPC は4択(A–D)、IC/CC は3択(A–C)。\n")

L.append("\n## 総合正答率（次元別 Overall, acc% / empty%）\n")
L.append("| モデル | 2D | 3D | 4D | empty% |")
L.append("|---|---|---|---|---|")
rank = []
for name, d, _, _, _ in MODELS:
    by = data[name]; per = {}; te = tt = 0
    for dim in ["2", "3", "4"]:
        if dim in by:
            o = overall(by[dim]); per[dim] = acc(o); te += o[2]; tt += o[0]
    rank.append((name, per.get("2"), per.get("3"), per.get("4"), 100 * te / tt if tt else None))
for name, a2, a3, a4, em in rank:  # MODELS order (系列ごと); not score-sorted
    L.append(f"| {name} | {_f(a2)} | {_f(a3)} | {_f(a4)} | {_f(em)} |")

L.append("\n## 信頼度 confidence（モデルが選んだ答えトークンの平均確率）\n")
L.append("Conf = 全体平均、Conf✓ = 正答時、Conf✗ = 誤答時（全次元集計）。Conf✓>Conf✗ なら自信と正誤が整合。\n")
L.append("| モデル | 2D Conf | 3D Conf | 4D Conf | Conf✓ | Conf✗ |")
L.append("|---|---|---|---|---|---|")
for name, d, _, _, _ in MODELS:
    by = data[name]
    dconf = {dim: conf(overall(by[dim])) for dim in ["2", "3", "4"] if dim in by}
    allo = _new()
    for dim in ["2", "3", "4"]:
        if dim in by:
            o = overall(by[dim])
            for i in range(9): allo[i] += o[i]
    L.append(
        f"| {name} | {_f(dconf.get('2'),3)} | {_f(dconf.get('3'),3)} | {_f(dconf.get('4'),3)} "
        f"| {_f(conf_ok(allo),3)} | {_f(conf_no(allo),3)} |"
    )

L.append("\n## 次元×種別 詳細（acc% / empty% / conf）\n")
for name, d, _, _, _ in MODELS:
    by = data[name]
    L.append(f"\n### {name}\n")
    L.append("| Dim | " + " | ".join(TYPELABEL[t] for t in TYPES) + " | Overall |")
    L.append("|---|" + "---|" * (len(TYPES) + 1))
    for dim in ["2", "3", "4"]:
        if dim not in by: continue
        cells = []
        for t in TYPES:
            a = by[dim].get(t)
            if a and a[0] > 0:
                cells.append(f"{_f(acc(a))}% / {_f(emp(a),0)}% / {_f(conf(a),2)}")
            else:
                cells.append("—")
        o = overall(by[dim])
        L.append(f"| {dim}D | " + " | ".join(cells) + f" | **{_f(acc(o))}** |")
L.append("\n(セル = `acc% / empty% / conf`。例 `80.6% / 0% / 0.83` は 正答80.6%・空0%・平均信頼度0.83)\n")

with open(OUT, "w") as f:
    f.write("\n".join(L))
print("wrote", OUT, f"({len(L)} lines)")