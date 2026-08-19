#!/usr/bin/env python3
"""对账:API 判别输出 vs 人工三档标注。

用法: python score_pilot.py <pred.jsonl> [<pred2.jsonl> ...]
标注固定读 data/tutu_task1_annotations_1233.csv。
报:覆盖率/无效率、三档混淆矩阵、accuracy、bad 召回、bad→good 灾难错误、
二分口径(bad vs 其余)的召回与放行、理由标签重合率。
"""
import csv
import json
import sys
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = {r["filename"]: r for r in csv.DictReader(
    open(ROOT / "data/tutu_task1_annotations_1233.csv", encoding="utf-8-sig"))}


def score(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["filename"] in ANN]
    n = len(rows)
    err = [r for r in rows if "error" in r]
    ok = [r for r in rows if "result" in r]
    print(f"\n===== {path}")
    print(f"总 {n} | 有效 {len(ok)} | 错误 {len(err)}",
          collections.Counter(r["error"][:40] for r in err) if err else "")
    cm = collections.Counter()
    reason_hit = reason_tot = 0
    catastrophic = []
    for r in ok:
        gt = ANN[r["filename"]]["grade"]
        pd = r["result"].get("grade", "?")
        cm[(gt, pd)] += 1
        if gt == "bad" and pd == "good":
            catastrophic.append(r["filename"])
        gt_reasons = set(x.strip() for x in (ANN[r["filename"]]["reasons"] or "").split(";") if x.strip())
        pd_reasons = set(r["result"].get("reason_labels", []))
        if gt == "bad" and gt_reasons and gt_reasons != {"不属于以上"}:
            reason_tot += 1
            if gt_reasons & pd_reasons:
                reason_hit += 1
    grades = ["good", "normal", "bad"]
    print("混淆矩阵 (行=人工, 列=模型; abstain 单列):")
    hdr = ["      "] + [f"{g:>7}" for g in grades] + [f"{'abstain':>8}"]
    print("".join(hdr))
    for gt in grades:
        line = [f"{gt:>6}"] + [f"{cm[(gt, p)]:>7}" for p in grades] + [f"{cm[(gt,'abstain')]:>8}"]
        print("".join(line))
    tot = sum(cm.values())
    acc = sum(cm[(g, g)] for g in grades) / max(1, tot)
    bad_n = sum(cm[("bad", p)] for p in grades + ["abstain"])
    bad_rec = cm[("bad", "bad")] / max(1, bad_n)
    # 二分口径:bad vs 放行(good+normal);abstain 计为拦截(保守)
    rel_n = sum(cm[(g, p)] for g in ("good", "normal") for p in grades + ["abstain"])
    released = sum(cm[(g, p)] for g in ("good", "normal") for p in ("good", "normal"))
    bad_blocked = sum(cm[("bad", p)] for p in ("bad", "abstain"))
    print(f"三档 accuracy={acc:.3f} | bad召回(三档)={bad_rec:.3f}")
    print(f"二分口径: bad拦截率={bad_blocked/max(1,bad_n):.3f} | 可放行样本放行率={released/max(1,rel_n):.3f}")
    print(f"灾难错误 bad→good: {len(catastrophic)} 条 {catastrophic[:6]}")
    if reason_tot:
        print(f"理由重合(判对轴即算): {reason_hit}/{reason_tot} = {reason_hit/reason_tot:.1%}")
    us = [r.get("usage", {}) for r in ok]
    tt = sum(u.get("totalTokenCount", u.get("total_tokens", 0)) for u in us)
    print(f"tokens 合计 {tt:,} | 均值 {tt/max(1,len(ok)):,.0f}/条")
    # bad_score 连续口径(若有)
    scored = [(r["result"]["bad_score"], 1 if ANN[r["filename"]]["grade"] == "bad" else 0)
              for r in ok if isinstance(r["result"].get("bad_score"), (int, float))]
    if len(scored) >= 20:
        import numpy as np
        s = np.array([x[0] for x in scored], float)
        yy = np.array([x[1] for x in scored])
        gn_s = np.sort(s[yy == 0])
        outs = []
        for rel in (0.70, 0.80, 0.90):
            k = int(np.floor(rel * len(gn_s)))
            T = gn_s[k - 1] if k > 0 else -1
            removed = float((s[yy == 1] > T).mean())
            outs.append(f"br@{int(rel*100)}%={removed:.3f}")
        # AUC
        from itertools import product
        pos = s[yy == 1]; neg = s[yy == 0]
        auc = float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0)
                             for a, b in product(pos, neg)]))
        print(f"bad_score 连续口径: {' | '.join(outs)} | AUC={auc:.3f} (n={len(scored)})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        score(p)
