import csv, json, os, pickle
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())
def auc(x, y):
    m = np.isfinite(x)
    if m.sum() < 100: return np.nan
    r = rankdata(x[m]); n1 = y[m].sum()
    return float((r[y[m] == 1].sum() - n1*(n1+1)/2) / (n1*(m.sum()-n1)))

oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float); y_tr = np.asarray(y_tr, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz")
X_tr = z["X_tr"].astype(float)
md = np.nanmedian(X_tr, axis=0)
ii = np.where(~np.isfinite(X_tr)); X_tr[ii] = np.take(md, ii[1])
tr = [json.loads(l) for l in open("splits/train_v3.jsonl")]
def load(name):
    rd = list(csv.reader(open(f"data/s3/{name}.csv")))
    cols = rd[0][1:]
    m = {os.path.basename(r[0]): [float(v) if v not in ("", "nan") else np.nan for v in r[1:]] for r in rd[1:]}
    return cols, np.array([m.get(r["video"], [np.nan]*len(cols)) for r in tr], float)
c_d, D = load("w5_dover"); c_v, V = load("w5_vs2")
print("覆盖 DOVER", f"{np.isfinite(D).all(1).mean():.0%}", "| VS2", f"{np.isfinite(V).all(1).mean():.0%}", flush=True)
for cols, A, tag in ((c_d, D, "DOVER"), (c_v, V, "VS2")):
    for j, c in enumerate(cols):
        print(f"  {tag}.{c:<12} AUC {auc(A[:, j], y_tr):.3f}", flush=True)
W = np.hstack([D, V])
for A in (D, V, W):
    wmd = np.nanmedian(A, axis=0); wi = np.where(~np.isfinite(A)); A[wi] = np.take(wmd, wi[1])
B = np.hstack([oof15, X_tr])
champ = json.load(open("data/s3/e18_champion.json"))["params"]
def mk():
    return lgb.LGBMClassifier(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
        min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"], feature_fraction=champ["ff"],
        bagging_fraction=champ["bf"], bagging_freq=1, random_state=42, verbose=-1)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
for name, TR in (("冠军基准", B), ("冠军+DOVER", np.hstack([B, D])), ("冠军+VS2", np.hstack([B, V])), ("冠军+W5全6列", np.hstack([B, W]))):
    o = np.zeros(len(y_tr))
    for a, b in folds:
        m = mk(); m.fit(TR[a], y_tr[a]); o[b] = m.predict_proba(TR[b])[:, 1]
    print(f"{name}: train-OOF gn@95 = {gn(o, y_tr):.4f}", flush=True)
print("W5_VERDICT_DONE", flush=True)
