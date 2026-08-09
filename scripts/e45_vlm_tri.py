#!/usr/bin/env python
"""E45:三档序数 VLM 微调 + 尾部诊断门(2026-08-09 预注册)。

两个改动相对 E36(AUC 0.649 / fold0 0.677):
 ①目标格式:答案首 token 从二元「是/否」换成三档「好/中/差」,读 3 个 logit 归一化。
   动机:E36 用单个「是」token 概率打分,分辨率天生粗且饱和;三档 simplex 给连续分数。
   且序数监督用的是标注者的真实边界(bad 判成 good 跨两档,判成 normal 跨一档),
   与 E39(人为定义尾部,已失败)有本质区别。
 ②底座升级:Qwen3-VL-8B → 30B-A3B(MoE,3B 激活,容量近 32B 而速度近 8B)。

**验收标准放在正确层级(E34/E36 教训)**:AUC 不是标准——E34 的 0.665、E36 的 0.649
都不差,入栈后无一过 0.3218。本实验主指标是 **tail-AUC**:
  取 E18 train-OOF 分数最低的那批 bad(tailB,即顶住阈值、决定放行率的那些),
  对全部可放行样本(good+normal, REL)算 AUC(tailB vs REL)。
  E18 在此指标上按构造接近 0。VLM 若 tail-AUC > 0.55,说明它看到了 E18 看不到的东西。
判准(发车前冻结):tail-AUC ≤ 0.55 即判该配置对放行率无用,**立即停,不付满折的钱**;
> 0.55 才跑满折 OOF 并入栈验 gn@95 > 0.3218。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from scipy.stats import rankdata

ROOT = Path("/root/mech")

Q_BIN = """这是AI生成的毛绒蘑菇角色「蘑菇TUTU」短视频的{n}帧画面(按时间顺序)。
判断标准:形象须与首帧一致(无眉毛/尾巴/手指,款式质感不变),动作须自然连贯有活物感(不僵硬刚体、不卡顿、四肢随动),
与场景交互正确,符合物理常识(有支撑不悬空),画面连续无跳变。
任一维度出现一眼可见、令人出戏的缺陷即为有缺陷。
问题:这条视频有明显缺陷吗?"""

Q_TRI = """这是AI生成的毛绒蘑菇角色「蘑菇TUTU」短视频的{n}帧画面(按时间顺序)。
判断标准:形象须与首帧一致(无眉毛/尾巴/手指,款式质感不变),动作须自然连贯有活物感(不僵硬刚体、不卡顿、四肢随动),
与场景交互正确,符合物理常识(有支撑不悬空),画面连续无跳变。
质量分三档:好=各维度自然可直接使用;中=存在轻微瑕疵但整体可接受;差=有一眼可见、令人出戏的缺陷。
问题:这条视频属于哪一档?"""

GROUPS = {          # 与 E36 同表:把标注 reasons 归并成 6 类缺陷词,作为多任务文本信号
    "僵硬": ["僵硬"], "卡顿": ["卡顿/少活人感", "动作位移不连贯"],
    "少动": ["四肢不动", "静止不动", "运动主体", "慢动作"],
    "还原": ["还原度", "衣服/身体的时间一致性", "大小变化"],
    "物理": ["物理规律", "不合理的物体"], "画面": ["帧跳变", "首帧一致", "背景运动混乱"],
}
TRI_CH = ["好", "中", "差"]
TRI_TXT = {0: "好,各维度自然,可直接使用。", 1: "中,存在轻微瑕疵但整体可接受。",
           2: "差,存在一眼可见的明显缺陷。"}
BIN_TXT = {0: "否,未见明显缺陷。", 1: "是,存在明显缺陷。"}
TIER = {"good": 0, "normal": 1, "bad": 2}


_G = {}          # 模块级共享态,供 DataLoader worker(fork)继承


class _DS(torch.utils.data.Dataset):
    """把 cv2 解码 + Qwen image processor 这段 CPU 重活丢给多进程预取,
    否则单进程预处理会让 GPU 空转(实测 GPU 0% / load 8.3)。"""

    def __init__(self, idxs, train):
        self.idxs, self.train = idxs, train

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = self.idxs[k]
        rel, t, _e, tgt = _G["items"][i]
        bd = _G["build"](rel, tgt if self.train else None)
        if bd is None:
            return (i, None, 0)
        enc, plen = bd
        return (i, dict(enc), plen)


def _first(b):
    return b[0]


def make_loader(idxs, train, workers):
    return torch.utils.data.DataLoader(
        _DS(idxs, train), batch_size=1, shuffle=False, num_workers=workers,
        collate_fn=_first, prefetch_factor=(6 if workers else None))


def load_frames(rel, n, px=336):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    jp = sorted(d.glob("f*.jpg"))
    if len(jp) < 4:
        return None
    idx = np.linspace(0, len(jp) - 1, n).round().astype(int)
    ims = []
    for i in idx:
        im = cv2.imread(str(jp[i]))
        if im is None:
            return None
        H, W = im.shape[:2]
        s = px / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(im, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)))
    return ims


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def e18_train_oof():
    """现算 E18 冠军的 train-OOF,用于定义 tailB(顶住阈值的那批 bad)。"""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    oof15, _ev, y_tr, *_ = pickle.load(open(ROOT / "upstream/cache_v3/_stack_15expert.pkl", "rb"))
    oof15 = np.asarray(oof15, float)
    y_tr = np.asarray(y_tr, int)
    z = np.load(ROOT / "upstream/cache_v3/_full_raw_v2.npz")
    X = z["X_tr"].astype(float)
    md = np.nanmedian(X, axis=0)
    ii = np.where(~np.isfinite(X))
    X[ii] = np.take(md, ii[1])
    B = np.hstack([oof15, X])
    c = json.load(open(ROOT / "data/s3/e18_champion.json"))["params"]
    o = np.zeros(len(y_tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(B, y_tr):
        m = lgb.LGBMClassifier(num_leaves=c["leaves"], n_estimators=c["est"], learning_rate=c["lr"],
                               min_child_samples=c["mcs"], scale_pos_weight=c["spw"],
                               feature_fraction=c["ff"], bagging_fraction=c["bf"], bagging_freq=1,
                               random_state=42, verbose=-1)
        m.fit(B[a], y_tr[a])
        o[b] = m.predict_proba(B[b])[:, 1]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--target", choices=["binary", "tri"], default="tri")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--px", type=int, default=336)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mlp-lora", action="store_true", help="LoRA 也加到 MLP 投影")
    ap.add_argument("--bf16", action="store_true",
                    help="不做 4bit 量化,直接 bf16(8B 仅 16GB,H100 80GB 放得下,比 nf4 快 2-3 倍)")
    ap.add_argument("--lam", type=float, default=0.5, help="score = P(差) + lam*P(中)")
    ap.add_argument("--hard-weight", type=float, default=0.0,
                    help="尾部加权 α:bad 样本 loss 权重 w=1+α*(1-pct),pct 为其 E18 分数在 bad 内的分位。"
                         "α>0 时 E18 判得最低(最难)的 bad 权重最高,直接把表征学习压向顶阈值那批。")
    ap.add_argument("--multitask", action="store_true",
                    help="目标文本附缺陷类型词(E36 做法),单独测多任务文本信号的价值")
    ap.add_argument("--workers", type=int, default=24, help="数据预取进程数(机器 112 核)")
    ap.add_argument("--tag", default="e45b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-fold0", action="store_true", help="只跑 fold0,过尾部门再谈满折")
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[os.path.basename(rel)] = rel

    e18 = e18_train_oof()
    print(f"E18 train-OOF 就绪 (n={len(e18)})", flush=True)

    items = []
    for i, r in enumerate(tr):
        rel = rel_of.get(r["video"])
        if not rel:
            continue
        t = TIER[r["label"]]
        items.append((rel, t, e18[i]))
    if args.limit:
        # 分层截断,保持三档比例
        rng = np.random.RandomState(0)
        keep = []
        for t in (0, 1, 2):
            idx = [j for j, it in enumerate(items) if it[1] == t]
            keep += list(rng.choice(idx, min(len(idx), args.limit // 3), replace=False))
        items = [items[j] for j in sorted(keep)]
    t_all = np.array([it[1] for it in items])
    y_all = (t_all == 2).astype(int)
    e18_all = np.array([it[2] for it in items])
    print(f"样本 {len(items)}  good {int((t_all==0).sum())} normal {int((t_all==1).sum())} "
          f"bad {int((t_all==2).sum())}", flush=True)

    # tailB:bad 中 E18 分数最低的 20%(顶住阈值那批);REL:全部 good+normal
    bad_idx = np.where(y_all == 1)[0]
    k = max(5, int(0.20 * len(bad_idx)))
    tailB = bad_idx[np.argsort(e18_all[bad_idx])[:k]]
    REL = np.where(y_all == 0)[0]
    print(f"tailB={len(tailB)} 条(E18 分数 {e18_all[tailB].min():.3f}~{e18_all[tailB].max():.3f}), "
          f"REL={len(REL)} 条;E18 自身 tail-AUC={auc(e18_all[tailB], e18_all[REL]):.4f}", flush=True)

    # 尾部加权:E18 判得越低的 bad,训练权重越高(α=0 时退化为等权)
    w_all = np.ones(len(items))
    if args.hard_weight > 0:
        pct = rankdata(e18_all[bad_idx]) / len(bad_idx)
        w_all[bad_idx] = 1.0 + args.hard_weight * (1.0 - pct)
        print(f"尾部加权 α={args.hard_weight}: bad 权重 {w_all[bad_idx].min():.2f}~"
              f"{w_all[bad_idx].max():.2f},tailB 均权 {w_all[tailB].mean():.2f}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(args.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    if args.target == "tri":
        ids = [proc.tokenizer.encode(c)[0] for c in TRI_CH]
        for c, i in zip(TRI_CH, ids):
            assert len(proc.tokenizer.encode(c)) == 1, f"{c} 不是单 token: {proc.tokenizer.encode(c)}"
        print(f"档位 token: 好={ids[0]} 中={ids[1]} 差={ids[2]}", flush=True)
        Q, TXT = Q_TRI, TRI_TXT
    else:
        ids = [proc.tokenizer.encode("否")[0], proc.tokenizer.encode("是")[0]]
        print(f"否={ids[0]} 是={ids[1]}", flush=True)
        Q, TXT = Q_BIN, {0: BIN_TXT[0], 1: BIN_TXT[0], 2: BIN_TXT[1]}

    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=42)
                 .split(np.zeros(len(items)), t_all))          # 按三档分层,好过按二分层
    if args.only_fold0:
        folds = folds[:1]
    score = np.full(len(items), np.nan)

    def build(rel, tgt=None):
        ims = load_frames(rel, args.frames, args.px)
        if ims is None:
            return None
        content = [{"type": "image", "image": im} for im in ims]
        content.append({"type": "text", "text": Q.format(n=len(ims))})
        msgs = [{"role": "user", "content": content}]
        p_enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt")
        if tgt is None:
            return p_enc, int(p_enc["input_ids"].shape[1])
        msgs2 = msgs + [{"role": "assistant", "content": [{"type": "text", "text": tgt}]}]
        enc = proc.apply_chat_template(msgs2, tokenize=True, return_dict=True, return_tensors="pt")
        return enc, int(p_enc["input_ids"].shape[1])

    # 组装每样本的目标文本。--multitask 时给非 good 档附缺陷类型词(E36 的做法,
    # 本实验单独测其价值:我们的纯二分变体 0.6464 低于 E36 fold0 0.677,差别正在于此)
    reasons = {}
    if args.multitask:
        import csv as _csv
        reasons = {r["path"]: r.get("reasons", "") for r in _csv.DictReader(
            open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
        print(f"多任务缺陷词已启用(reasons 表 {len(reasons)} 条)", flush=True)

    def mk_tgt(rel, t):
        base = TXT[t]
        if not args.multitask or t == 0:
            return base
        rs = reasons.get(os.path.basename(rel), "")
        gs = [g for g, tags in GROUPS.items() if any(x in rs for x in tags)]
        if not gs:
            return base
        w = "、".join(gs)
        if args.target == "tri":
            return f"差,存在{w}问题。" if t == 2 else f"中,轻微{w}问题。"
        return f"是,存在{w}问题。" if t == 2 else base

    items = [(rel, t, e, mk_tgt(rel, t)) for rel, t, e in items]
    _G.update(items=items, TXT=TXT, build=build)   # 供 worker 进程 fork 继承

    tmods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.mlp_lora:
        tmods += ["gate_proj", "up_proj", "down_proj"]

    for fi, (a_idx, b_idx) in enumerate(folds):
        # 预量化仓库(名含 4bit/bnb)自带 quantization_config,再传一次会冲突
        prequant = ("4bit" in args.model.lower()) or ("bnb" in args.model.lower())
        kw = dict(dtype=torch.bfloat16, device_map="cuda")
        if not args.bf16 and not prequant:
            kw["quantization_config"] = bnb
        if fi == 0:
            print(f"加载模式: {'预量化(仓库自带4bit)' if prequant else ('bf16' if args.bf16 else 'nf4量化')}",
                  flush=True)
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kw)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
            target_modules=tmods, task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        order = list(a_idx)
        for ep in range(args.epochs):
            random.shuffle(order)
            model.train()
            tot, nb = 0.0, 0
            for kk, (i, enc, plen) in enumerate(make_loader(order, True, args.workers)):
                if enc is None:
                    continue
                enc = {k2: v.to("cuda", non_blocking=True) for k2, v in enc.items()}
                labels = enc["input_ids"].clone()
                labels[:, :plen] = -100
                labels[labels == proc.tokenizer.pad_token_id] = -100
                out = model(**enc, labels=labels)
                (out.loss * float(w_all[i]) / 4).backward()
                if (kk + 1) % 4 == 0:
                    opt.step(); opt.zero_grad()
                tot += float(out.loss.detach()); nb += 1
                if (kk + 1) % 50 == 0:
                    print(f"  fold{fi} ep{ep} [{kk+1}/{len(order)}] loss {tot/max(1,nb):.4f}", flush=True)
        model.eval()
        with torch.inference_mode():
            for cnt, (i, enc, _p) in enumerate(make_loader(list(b_idx), False, args.workers)):
                if enc is None:
                    continue
                enc = {k2: v.to("cuda", non_blocking=True) for k2, v in enc.items()}
                lg = model(**enc).logits[0, -1]
                pr = torch.softmax(lg[ids].float(), 0).cpu().numpy()
                score[i] = pr[2] + args.lam * pr[1] if args.target == "tri" else pr[1]
                if (cnt + 1) % 300 == 0:
                    print(f"  fold{fi} 推理 [{cnt+1}/{len(b_idx)}]", flush=True)
        ok = np.isfinite(score)
        print(f"fold{fi} 累计 AUC {auc(score[ok & (y_all==1)], score[ok & (y_all==0)]):.4f} "
              f"(覆盖 {int(ok.sum())})", flush=True)
        del model
        torch.cuda.empty_cache()

    ok = np.isfinite(score)
    tb = [i for i in tailB if ok[i]]
    rl = [i for i in REL if ok[i]]
    A = auc(score[y_all == 1][np.isfinite(score[y_all == 1])],
            score[y_all == 0][np.isfinite(score[y_all == 0])])
    TA = auc(score[tb], score[rl])
    print(f"\n=== [{args.tag}] {args.model.split('/')[-1]} target={args.target} "
          f"frames={args.frames} ===", flush=True)
    print(f"  整体 AUC   = {A:.4f}   (E36 历史:full 0.649 / fold0 0.677)", flush=True)
    print(f"  **tail-AUC = {TA:.4f}**  (主指标;判准 >0.55 才跑满折)  n_tail={len(tb)}", flush=True)
    print(f"  判决:{'✔ 过尾部门,可跑满折 OOF' if TA > 0.55 else '✘ 未过尾部门,停,不付满折的钱'}",
          flush=True)
    np.save(ROOT / f"data/{args.tag}_score.npy", score)
    json.dump({"tag": args.tag, "model": args.model, "target": args.target, "frames": args.frames,
               "px": args.px, "epochs": args.epochs, "hard_weight": args.hard_weight, "multitask": args.multitask,
               "auc": A, "tail_auc": TA, "n_tail": len(tb), "lam": args.lam},
              open(ROOT / f"data/{args.tag}_summary.json", "w"), ensure_ascii=False, indent=1)
    print("E45_DONE", flush=True)


if __name__ == "__main__":
    main()
