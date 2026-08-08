"""E39:直接优化放行率的损失(2026-08-08 预注册)。基准 0.3218。
A 尾部 pairwise:只对(低分bad × 低分good)样本对做排序损失(局部化,区别于崩掉的全局lambdarank)。
B soft-quantile 代理:用 sigmoid 近似 P(good < T_5%bad),对 T 用软分位数。
实现:PyTorch MLP over [oof15⊕X320](标准化),5折OOF,与冠军同折。
"""
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def gn(p, y, rec=0.95):
    b = np.sort(p[y == 1])
    T = b[len(b) - int(np.ceil(rec * len(b)))]
    return float((p[y == 0] < T).mean())


oof15, ev15, y_tr, y_ev, *_ = pickle.load(open("upstream/cache_v3/_stack_15expert.pkl", "rb"))
oof15 = np.asarray(oof15, float)
y_tr = np.asarray(y_tr, int)
z = np.load("upstream/cache_v3/_full_raw_v2.npz")
X_tr = z["X_tr"].astype(float)
md = np.nanmedian(X_tr, axis=0)
ii = np.where(~np.isfinite(X_tr))
X_tr[ii] = np.take(md, ii[1])
B = StandardScaler().fit_transform(np.hstack([oof15, X_tr]))
base = np.load("data/s3/e18_champion_train_oof.npy")
print(f"基准 {gn(base, y_tr):.4f}", flush=True)
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr))
dev = "cuda" if torch.cuda.is_available() else "cpu"


def make_net(d):
    return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)).to(dev)


def soft_gn_loss(s, y, tau=0.1, q=0.05):
    """s: 分数(越高越bad)。软分位数取 bad 的 5% 分位 T,再软计 good<T 的比例(取负作损失)。"""
    sb = s[y == 1]
    sg = s[y == 0]
    if len(sb) < 5 or len(sg) < 5:
        return s.sum() * 0
    k = max(1, int(q * len(sb)))
    T = torch.sort(sb)[0][k - 1]
    return -torch.sigmoid((T - sg) / tau).mean()


def tail_pair_loss(s, y, frac=0.4, tau=1.0):
    """只对低分区的 (bad, good) 对做 pairwise:希望 bad 高于 good。"""
    thr = torch.quantile(s.detach(), frac)
    lb = (y == 1) & (s.detach() <= thr)
    lg = (y == 0) & (s.detach() <= thr)
    if lb.sum() < 2 or lg.sum() < 2:
        return s.sum() * 0
    diff = s[lb].unsqueeze(1) - s[lg].unsqueeze(0)
    return torch.nn.functional.softplus(-diff / tau).mean()


CFG = [("bce", None), ("soft_gn", 1.0), ("tail_pair", 1.0), ("bce+soft_gn", 0.5), ("bce+tail_pair", 0.5)]
for name, w in CFG:
    oof = np.zeros(len(y_tr))
    for a, b in folds:
        torch.manual_seed(42)
        net = make_net(B.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Xa = torch.tensor(B[a]).float().to(dev)
        ya = torch.tensor(y_tr[a]).float().to(dev)
        pw = torch.tensor([(len(a) - y_tr[a].sum()) / max(1, y_tr[a].sum())]).to(dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(300):
            net.train()
            opt.zero_grad()
            s = net(Xa).squeeze(1)
            if name == "bce":
                loss = bce(s, ya)
            elif name == "soft_gn":
                loss = soft_gn_loss(s, ya)
            elif name == "tail_pair":
                loss = tail_pair_loss(s, ya)
            elif name == "bce+soft_gn":
                loss = bce(s, ya) + w * soft_gn_loss(s, ya)
            else:
                loss = bce(s, ya) + w * tail_pair_loss(s, ya)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            oof[b] = net(torch.tensor(B[b]).float().to(dev)).squeeze(1).cpu().numpy()
    print(f"[E39 {name}] train-OOF gn@95 = {gn(oof, y_tr):.4f}", flush=True)
    np.save(f"data/s3/e39_{name.replace('+','_')}_oof.npy", oof)
print("E39_DONE", flush=True)
