import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
rows = []
for k in range(3):
    ev = [json.loads(l) for l in open(f"/workspace/r2/up/splits/fold{k}_eval.jsonl")]
    for e in ev:
        stem = Path(e["video"]).stem
        d = json.load(open(f"/workspace/r2/pred_fold{k}/{e['label']}/{stem}.json"))
        rows.append((e["video"], e["label"], d["p_bad"], k))
print("total OOF:", len(rows))
y = np.array([1 if r[1] == "bad" else 0 for r in rows])
s = np.array([r[2] for r in rows])
r = rankdata(s); pos = r[y == 1]
auc = (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())
def br(y, s, rel):
    quota = rel * (y == 0).sum(); rg = rb = 0.; nb = (y == 1).sum()
    for v in np.unique(np.sort(s)):
        g = ((y == 0) & (s == v)).sum(); bb = ((y == 1) & (s == v)).sum()
        if rg + g <= quota:
            rg += g
        else:
            f = (quota - rg) / max(1e-9, g) if g else 0.
            rb += bb * (1 - f); rb += ((y == 1) & (s > v)).sum(); break
    else:
        return 0.
    return rb / max(1, nb)
print(f"LoRA 3-fold OOF n={len(y)} AUC={auc:.4f} br@70={br(y,s,.7):.4f} br@80={br(y,s,.8):.4f} br@90={br(y,s,.9):.4f}")
import csv
with open("/workspace/r2/lora_oof_1233.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["video", "label", "p_bad", "fold"])
    w.writerows(rows)
