import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
fold = sys.argv[1]
ev = [json.loads(l) for l in open(f"/workspace/r2/up/splits/fold{fold}_eval.jsonl")]
ys, ps = [], []
for e in ev:
    stem = Path(e["video"]).stem
    p = Path(f"/workspace/r2/pred_fold{fold}/{e['label']}/{stem}.json")
    if not p.exists():
        continue
    d = json.load(open(p))
    ys.append(1 if e["label"] == "bad" else 0)
    ps.append(d["p_bad"])
y = np.array(ys); s = np.array(ps)
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
print(f"fold{fold} OOF n={len(y)} bad={y.sum()} AUC={auc:.4f} "
      f"br@70={br(y,s,.7):.4f} br@80={br(y,s,.8):.4f} br@90={br(y,s,.9):.4f}")
