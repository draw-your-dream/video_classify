import json, os, pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1]); T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())
oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float); y_tr = np.asarray(y_tr, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz"); X_tr = z["X_tr"].astype(float)
md = np.nanmedian(X_tr, axis=0); ii = np.where(~np.isfinite(X_tr)); X_tr[ii] = np.take(md, ii[1])
tr = [json.loads(l) for l in open("splits/train_v3.jsonl")]
B = np.hstack([oof15, X_tr])
champ = json.load(open("data/s3/e18_champion.json"))["params"]
def mk():
    return lgb.LGBMClassifier(num_leaves=champ["leaves"], n_estimators=champ["est"], learning_rate=champ["lr"],
        min_child_samples=champ["mcs"], scale_pos_weight=champ["spw"], feature_fraction=champ["ff"],
        bagging_fraction=champ["bf"], bagging_freq=1, random_state=42, verbose=-1)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
cols = {}
for t in ("e34a", "e34c"):
    o = np.load(f"data/s3/{t}_oof.npy")
    rels = json.load(open(f"data/s3/{t}_rels.json"))
    m = {os.path.basename(r): v for r, v in zip(rels, o)}
    c = np.array([m.get(r["video"], np.nan) for r in tr], float)
    c[~np.isfinite(c)] = np.nanmedian(c)
    cols[t] = c
    from scipy.stats import rankdata
    rk = rankdata(c); n1 = y_tr.sum()
    print(f"{t}: 对齐后 AUC {(rk[y_tr==1].sum()-n1*(n1+1)/2)/(n1*(len(y_tr)-n1)):.4f}", flush=True)
for name, TR in (("冠军基准", B),
                 ("冠军+e34a", np.hstack([B, cols['e34a'][:, None]])),
                 ("冠军+e34c", np.hstack([B, cols['e34c'][:, None]])),
                 ("冠军+两者", np.hstack([B, cols['e34a'][:, None], cols['e34c'][:, None]]))):
    o = np.zeros(len(y_tr))
    for a, b in folds:
        m2 = mk(); m2.fit(TR[a], y_tr[a]); o[b] = m2.predict_proba(TR[b])[:, 1]
    print(f"{name}: train-OOF gn@95 = {gn(o, y_tr):.4f}", flush=True)
print("E35_DONE", flush=True)
