#!/usr/bin/env python
"""位置バイアス分析を confidence 重点で計算し、summary.md に追記する。
MC分類(PPC/IC/CC = type 1/2/3, 全次元合算)について、ローテーションで各問の正解は
全スロットに均等配置されるので、提示スロット(letter)に残る差は純粋な位置効果。
 (1) 提示スロット別 = 正答率 + 平均confidence（位置バイアス）
 (2) 出力レター別   = 平均confidence + 選択数（出力位置の信頼度偏り）
意味軸(正準ラベル別)の分析は削除した。
"""
import csv, os, glob

csv.field_size_limit(10**7)

ROOT = "/home/tota_abe/Multidimensional_Geometric_Evaluation"
BASE = "results/final_20260625"
SUMMARY = os.path.join(ROOT, BASE, "summary.md")

MODELS = [
 ("Qwen3.5-9B",                 f"{BASE}/Qwen_Qwen3.5-9B"),
 ("Qwen3.5-27B",                f"{BASE}/Qwen_Qwen3.5-27B"),
 ("Qwen3.5-35B-A3B",            f"{BASE}/Qwen_Qwen3.5-35B-A3B"),
 ("Qwen3.5-122B-A10B",          f"{BASE}/Qwen_Qwen3.5-122B-A10B"),
 ("Qwen3-30B-A3B",              f"{BASE}/Qwen_Qwen3-30B-A3B"),
 ("Qwen3-32B",                  f"{BASE}/Qwen_Qwen3-32B"),
 ("Qwen3-Next-80B-A3B-Instruct",f"{BASE}/Qwen_Qwen3-Next-80B-A3B-Instruct"),
 ("Qwen3-235B-A22B-FP8",        f"{BASE}/Qwen_Qwen3-235B-A22B-FP8"),
 ("gemma-4-12B-it",             f"{BASE}/google_gemma-4-12B-it"),
 ("gemma-4-31B-it",             f"{BASE}/google_gemma-4-31B-it"),
 ("gpt-oss-120b",               f"{BASE}/openai_gpt-oss-120b"),
]


def li(s):
    s = (s or "").strip().upper()
    return ord(s) - ord("A") if len(s) == 1 and s.isalpha() else None


def analyze(model_dir):
    # slot[L]  = [ok, tot, conf_sum, conf_n]  (L = presented position of correct answer)
    # chosen[L]= [conf_sum, conf_n, count]    (L = letter the model actually output)
    slot, chosen = {}, {}
    for f in glob.glob(os.path.join(ROOT, model_dir, "dim_*_per_question.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if (r.get("type") or "").strip() not in ("1", "2", "3"):
                continue
            gi = li(r.get("ground_truth_normalized"))
            if gi is None:
                continue
            corr = int(r.get("is_correct", "0") or 0)
            cstr = (r.get("confidence") or "").strip()
            cv = float(cstr) if cstr else None
            s = slot.setdefault(chr(ord("A") + gi), [0, 0, 0.0, 0])
            s[0] += corr; s[1] += 1
            if cv is not None:
                s[2] += cv; s[3] += 1
            pj = li(r.get("predicted_normalized"))
            if pj is not None:
                ch = chosen.setdefault(chr(ord("A") + pj), [0.0, 0, 0])
                ch[2] += 1
                if cv is not None:
                    ch[0] += cv; ch[1] += 1
    return slot, chosen


def acc(d, k):  return 100 * d[k][0] / d[k][1] if k in d and d[k][1] else None
def sconf(d, k):return d[k][2] / d[k][3] if k in d and d[k][3] else None
def cconf(d, k):return d[k][0] / d[k][1] if k in d and d[k][1] else None
def cnt(d, k):  return d[k][2] if k in d else 0
def fa(v):      return f"{v:.1f}" if v is not None else "—"
def fc(v):      return f"{v:.2f}" if v is not None else "—"


res = {name: analyze(d) for name, d in MODELS}

L = []
L.append("\n\n## 位置バイアス分析（confidence 重点）\n")
L.append("MC分類（PPC/IC/CC = type 1/2/3）のみ、全次元(2D/3D/4D)合算。numeric / numeric_mc は対象外。")
L.append("ローテーションで各問の正解は全スロットに均等配置されるため、提示スロット(letter)に残る差は"
         "**純粋な位置効果**。confidence は「モデルが選んだ答えトークンの確率」。\n")

L.append("### (1) 提示スロット別：正答率 + 平均confidence（位置バイアス）\n")
L.append("各 letter = 「正解がその変種で提示されたスロット位置」（意味ではない）。"
         "セル = `acc% / conf`。slot A の acc・conf がともに低い = letter \"A\" を出し渋る **primacy aversion**。\n")
L.append("| モデル | A acc/conf | B | C | D | A→C acc差 | A→C conf差 |")
L.append("|---|---|---|---|---|---|---|")
for name, _ in MODELS:
    slot, _ch = res[name]
    cells = " | ".join(f"{fa(acc(slot, k))}/{fc(sconf(slot, k))}" for k in "ABCD")
    ad = (acc(slot, "A") - acc(slot, "C")) if acc(slot, "A") is not None and acc(slot, "C") is not None else None
    cd = (sconf(slot, "A") - sconf(slot, "C")) if sconf(slot, "A") is not None and sconf(slot, "C") is not None else None
    L.append(f"| {name} | {cells} | {('%+.1f pp' % ad) if ad is not None else '—'} "
             f"| {('%+.2f' % cd) if cd is not None else '—'} |")

L.append("\n### (2) 出力レター別：平均confidence + 選択数（出力位置の信頼度偏り）\n")
L.append("モデルが実際に出力した letter ごとの平均confidenceと選択数。特定スロットで conf が系統的に高い/低い "
         "= **出力位置に依存した信頼度バイアス**。セル = `conf (n)`。D は PPC のみ（4択）に出現。\n")
L.append("| モデル | A conf(n) | B | C | D |")
L.append("|---|---|---|---|---|")
for name, _ in MODELS:
    _slot, ch = res[name]
    cells = " | ".join(f"{fc(cconf(ch, k))} ({cnt(ch, k)})" for k in "ABCD")
    L.append(f"| {name} | {cells} |")

# 考察・解釈は別途 FINDINGS.md（手書き）に記載。本スクリプトは集計表のみを出力する。

txt = open(SUMMARY, encoding="utf-8").read().rstrip() + "\n" + "\n".join(L) + "\n"
open(SUMMARY, "w", encoding="utf-8").write(txt)
print("appended confidence-weighted position-bias analysis to", SUMMARY)
print("\n".join(L))