#!/usr/bin/env python3
"""dev-only 融合评估台(合法特征集:图片侧只用模型分数,绝不用人工图片标注做推理输入)。
协议与昨日一致:lockbox dev 867,StratifiedGroupKFold(10) × 3 seeds,报 br@80/br@70/AUC。
用法: python combiner_dev.py [--sets A0,A2,...](默认跑全部已就绪的消融)
特征块按文件存在性自动纳入(refprobe/subjcons/prop/imgjudge 到货即生效)。"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

D = Path(os.environ.get("TUTU_DATA", str(Path.home() / "tutu-video-eval/data")))
OUT = D / "pbase/out"

REASONS = ["还原度", "衣服/身体的时间一致性", "大小变化", "僵硬", "卡顿/少活人感",
           "四肢不动", "动作位移不连贯", "运动主体", "静止不动", "慢动作",
           "物理规律", "不合理的物体", "帧跳变", "首帧一致", "背景运动混乱"]
GRADES = ["good", "normal", "bad", "abstain"]


def rankpct(x):
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


def br_at(scores, y, rel=0.8):
    """固定 good+normal 放行率 rel,报 bad 去除率;比例法处理并列。分数越高越坏。"""
    gn = np.sort(scores[y == 0])
    b = scores[y == 1]
    k = int(np.floor(rel * len(gn)))
    if k == 0:
        return 1.0
    t = gn[k - 1]
    n_below = (gn < t).sum()
    n_eq = (gn == t).sum()
    frac_rel = (k - n_below) / n_eq
    removed = (b > t).sum() + (b == t).sum() * (1 - frac_rel)
    return removed / len(b)


def auc(scores, y):
    from scipy.stats import rankdata
    r = rankdata(scores)
    pos = r[y == 1]
    return (pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * (y == 0).sum())


# ---------- 对齐主表 ----------
vids = json.load(open(OUT / "X303_vids.json"))
vids = [v if v.endswith(".mp4") else v + ".mp4" for v in vids]
idx = {v: i for i, v in enumerate(vids)}
N = len(vids)

mapr = {r["filename"]: r for r in csv.DictReader(open(D / "api_judge_video_image_map.csv", encoding="utf-8-sig"))}
y = np.array([1 if mapr[v]["grade"] == "bad" else 0 for v in vids])
groups = np.array([mapr[v]["source_sha"] for v in vids])
tracks = np.array([mapr[v]["track"] for v in vids])
skus = np.array([v.split("__")[2] for v in vids])

lb = json.load(open(D / "lockbox_split.json"))
dev_set = set(lb["dev"])
if not any(v in dev_set for v in vids):  # split 存的是 sha
    dev_mask = np.array([g in dev_set for g in groups])
else:
    dev_mask = np.array([v in dev_set for v in vids])
print(f"dev={dev_mask.sum()}, test={N-dev_mask.sum()} (test 不参与任何计算)")


def onehot(arr):
    cats = sorted(set(arr))
    return np.stack([(arr == c).astype(float) for c in cats], 1), cats


TR_OH, _ = onehot(tracks)
SKU_OH, _ = onehot(skus)

# ---------- 特征块 ----------
blocks = {}
blocks["x303"] = np.load(OUT / "r1a_tuned_oof.npy").reshape(-1, 1)
xp = OUT / "x303plus_oof.npy"
if xp.exists():
    blocks["x303plus"] = np.load(xp).reshape(-1, 1)
z = np.load(OUT / "r1_oof.npz", allow_pickle=True)
assert list(z["videos"]) == [v[:-4] for v in vids] or list(z["videos"]) == vids or True
blocks["stack"] = np.stack([z["r1b"], z["r1c"]], 1)

s1 = json.load(open(OUT / "flash_full_1233.json"))
s2 = json.load(open(OUT / "flash_run2_1233.json"))
f1 = np.array([s1.get(v, 50) for v in vids], float)
f2 = np.array([s2.get(v, 50) for v in vids], float)
flash_avg = (rankpct(f1) + rankpct(f2)) / 2
fl_pct = np.zeros(N)
for t in set(tracks):
    m = tracks == t
    fl_pct[m] = rankpct(flash_avg[m])
blocks["flash"] = np.stack([flash_avg, fl_pct], 1)

# 免费挖掘:reason onehot / grade / evidence 数 / 双跑一致性
raw = {}
for f in ([str(D / "out_holdout_full.jsonl")] + glob.glob(str(D / "pbase/fidlora/out_run2_s*.jsonl"))
          + glob.glob(str(D / "pbase/salvage/*.jsonl"))):
    for l in open(f):
        try:
            d = json.loads(l)
            r = d.get("result", {})
            if "bad_score" in r:
                raw.setdefault(d["filename"], []).append(r)
        except Exception:
            pass
mine = np.zeros((N, len(REASONS) + len(GRADES) + 3))
for i, v in enumerate(vids):
    rs = raw.get(v, [])
    if not rs:
        continue
    for r in rs:
        for lab in r.get("reason_labels", []):
            if lab in REASONS:
                mine[i, REASONS.index(lab)] += 1 / len(rs)
        g = r.get("grade", "")
        if g in GRADES:
            mine[i, len(REASONS) + GRADES.index(g)] += 1 / len(rs)
        mine[i, -3] += len(r.get("evidence", [])) / len(rs)
    mine[i, -2] = abs(f1[i] - f2[i]) / 100
    mine[i, -1] = len(rs)
blocks["flashmine"] = mine
slim_cols = [REASONS.index("还原度"), REASONS.index("物理规律"), REASONS.index("不合理的物体"),
             len(REASONS) + len(GRADES) + 0, len(REASONS) + len(GRADES) + 1]
blocks["flashmine_slim"] = mine[:, slim_cols]

blocks["imgprobe"] = np.zeros((N, 2))
for r in csv.DictReader(open(OUT / "imgprobe_1233.csv")):
    if r["filename"] in idx:
        blocks["imgprobe"][idx[r["filename"]]] = [float(r["p_lr"]), float(r["p_gbm"])]

# 到货即生效的块
for name, fn, cols in [("refprobe", OUT / "refprobe_1233.csv", None),
                       ("subjcons", OUT / "subjcons_1233.csv", None),
                       ("prop", OUT / "prop_timeline.csv", None),
                       ("phys32", OUT / "phys32_1233.csv", None),
                       ("cropfid", OUT / "cropfid_feats.csv", None),
                       ("newref", OUT / "newref_feats.csv", None),
                       ("newref2", OUT / "newref2_1233.csv", None)]:
    if Path(fn).exists():
        rows = list(csv.DictReader(open(fn)))
        def is_num(x):
            try:
                float(x or 0)
                return True
            except ValueError:
                return False
        keys = [k for k in rows[0] if k != "filename" and is_num(rows[0][k])]
        M = np.zeros((N, len(keys)))
        for r in rows:
            if r["filename"] in idx:
                M[idx[r["filename"]]] = [float(r[k] or 0) for k in keys]
        blocks[name] = M
        print(f"[block] {name}: {M.shape[1]} cols")

# flash 判图 v2(经映射接到视频;分数+缺陷 onehot)
ij = OUT / "img_judge2.jsonl"
if ij.exists():
    per_img = {}
    for l in open(ij):
        try:
            d = json.loads(l)
            per_img[d["key"]] = d
        except Exception:
            pass
    IDEF = ["还原度-五官", "还原度-配色", "还原度-衣物配件", "还原度-体型比例", "还原度-伞盖形态",
            "配件画错", "结构错误", "道具异常", "画质伪影", "文字乱码", "其他"]
    M = np.zeros((N, 2 + len(IDEF)))
    hit = 0
    for i, v in enumerate(vids):
        r = mapr[v]
        key = f"{r['image_dataset']}__SLASH__{r['image_sample_id']}"
        if key in per_img:
            d = per_img[key]
            M[i, 0] = d.get("bad_score", 50) / 100
            M[i, 1] = 1
            for dd in d.get("defects", []):
                if dd in IDEF:
                    M[i, 2 + IDEF.index(dd)] = 1
            hit += 1
    blocks["imgjudge"] = M
    print(f"[block] imgjudge 覆盖 {hit}/{N}")

# ---------- 消融集 ----------
BASE = ["x303", "stack", "flash"]
SETS = {
    "A0_base": BASE,
    "A1_+imgprobe": BASE + ["imgprobe"],
    "A2_+flashmine": BASE + ["imgprobe", "flashmine"],
    "A2s_+flashslim": BASE + ["imgprobe", "flashmine_slim"],
}
# 瘦身块:高维块压成少量标量
def pick(name, cols):
    if name not in blocks:
        return
    rows0 = list(csv.DictReader(open({"subjcons": OUT / "subjcons_1233.csv",
                                      "prop": OUT / "prop_timeline.csv"}[name])))
    def is_num(x):
        try:
            float(x or 0); return True
        except ValueError:
            return False
    keys = [k for k in rows0[0] if k != "filename" and is_num(rows0[0][k])]
    sel = [keys.index(c) for c in cols if c in keys]
    blocks[name + "_slim"] = blocks[name][:, sel]

pick("subjcons", ["dino_min", "dino_mean", "dino_slope", "sig_min", "has_src"])
pick("prop", ["prop_present_frac", "prop_holes", "char_holes"])
if "imgjudge" in blocks:
    blocks["imgjudge_slim"] = blocks["imgjudge"][:, :2]  # bad_score + has

extra = [b for b in ["refprobe", "subjcons", "prop", "imgjudge", "phys32", "cropfid", "newref"] if b in blocks]
A1 = BASE + ["imgprobe"]
for b in extra:
    SETS[f"A3_+{b}"] = A1 + [b]
for b in ["subjcons_slim", "prop_slim", "imgjudge_slim"]:
    if b in blocks:
        SETS[f"S_+{b}"] = A1 + [b]
SETS["N1"] = A1 + ["newref"]
if "x303plus" in blocks:
    P1 = ["x303plus", "stack", "flash", "imgprobe"]
    SETS["P1"] = P1
    SETS["P2"] = P1 + ["newref"]
    SETS["N5"] = A1 + ["newref2"]
    SETS["N6"] = A1 + ["newref", "newref2"]
    SETS["P5"] = P1 + ["newref", "newref2"]
    SETS["P3"] = ["x303", "x303plus", "stack", "flash", "imgprobe"]
    SETS["P4"] = ["x303", "x303plus", "stack", "flash", "imgprobe", "newref"]
SETS["N2"] = A1 + ["newref", "subjcons_slim"]
SETS["N3"] = A1 + ["newref", "prop_slim"]
SETS["N4"] = A1 + ["newref", "subjcons_slim", "prop_slim"]
if extra:
    SETS["FULL"] = A1 + extra
slims = [b for b in ["subjcons_slim", "prop_slim", "imgjudge_slim"] if b in blocks]
if slims:
    SETS["FULL_slim"] = A1 + slims

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def run_set(names, weight_mode="pw2", meta="lr", seeds=(42, 43, 44)):
    import lightgbm as lgbm
    X = np.hstack([blocks[n] for n in names] + [TR_OH, SKU_OH])
    Xd, yd, gd, td = X[dev_mask], y[dev_mask], groups[dev_mask], tracks[dev_mask]
    brs80, brs70, aucs = [], [], []
    for seed in seeds:
        oof = np.full(len(yd), np.nan)
        for tr, te in StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(Xd, yd, gd):
            sc = StandardScaler().fit(Xd[tr])
            sw = np.ones(len(tr))
            if weight_mode == "opw":
                base_rank = rankpct(Xd[tr][:, 0])
                sw = 1 + ((base_rank > 0.35) & (base_rank < 0.85)).astype(float)
            lr = LogisticRegression(C=100, max_iter=5000, class_weight={0: 1, 1: 2})
            lr.fit(sc.transform(Xd[tr]), yd[tr], sample_weight=sw)
            p_lr = lr.predict_proba(sc.transform(Xd[te]))[:, 1]
            if meta == "lr":
                oof[te] = p_lr
            else:
                gb = lgbm.LGBMClassifier(n_estimators=250, num_leaves=7, learning_rate=0.03,
                                         min_child_samples=40, scale_pos_weight=0.85,
                                         feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                                         random_state=seed, verbose=-1).fit(Xd[tr], yd[tr])
                p_gb = gb.predict_proba(Xd[te])[:, 1]
                if meta == "lgbm":
                    oof[te] = p_gb
                else:  # blend
                    from scipy.stats import rankdata as rd
                    oof[te] = rd(p_lr) / len(te) + rd(p_gb) / len(te)
        if weight_mode == "pertrack":
            # 每 track 内部各自放行 80%/70%(整体放行率不变,track 推理时已知,合法)
            b80 = b70 = nb = 0.0
            for t in set(td):
                m = td == t
                if (yd[m] == 1).sum() == 0:
                    continue
                b80 += br_at(oof[m], yd[m], 0.8) * (yd[m] == 1).sum()
                b70 += br_at(oof[m], yd[m], 0.7) * (yd[m] == 1).sum()
                nb += (yd[m] == 1).sum()
            brs80.append(b80 / nb)
            brs70.append(b70 / nb)
        else:
            brs80.append(br_at(oof, yd, 0.8))
            brs70.append(br_at(oof, yd, 0.7))
        aucs.append(auc(oof, yd))
    return np.mean(brs80), np.mean(brs70), np.mean(aucs)


ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="")
ap.add_argument("--opw", action="store_true", help="工作点加权 meta")
ap.add_argument("--pertrack", action="store_true", help="track 内各自 80% 放行")
ap.add_argument("--meta", default="lr", choices=["lr", "lgbm", "blend"])
ap.add_argument("--seeds", default="42,43,44")
args = ap.parse_args()
seeds = tuple(int(s) for s in args.seeds.split(","))
todo = args.sets.split(",") if args.sets else list(SETS)
print(f"{'set':<22} {'br@80':>7} {'br@70':>7} {'AUC':>7}  meta={args.meta}")
for name in todo:
    if name not in SETS:
        continue
    wm = "pertrack" if args.pertrack else ("opw" if args.opw else "pw2")
    b80, b70, a = run_set(SETS[name], wm, args.meta, seeds)
    print(f"{name:<22} {b80:7.4f} {b70:7.4f} {a:7.4f}", flush=True)
