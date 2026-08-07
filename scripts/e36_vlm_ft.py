#!/usr/bin/env python
"""E36:VLM QLoRA 微调直判(2026-08-07 预注册)。首次尝试大底座+多任务标签+密帧。
底座 Qwen3-VL(8B 验证管线 → 32B 主实验),4bit QLoRA,视觉塔冻结,只训语言侧 LoRA。
任务:回答"这条视频是否有明显缺陷"+缺陷类型词(多任务由文本目标承载)。
读分数:比较 token "是"/"否" 的 logit 差(不采样,确定性)。
协议:train_v3 内 2 折 OOF(算力约束,如实标注),报 AUC 与 gn@95;过门才谈并栈与 eval。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")
GROUPS = {
    "僵硬": ["僵硬"], "卡顿": ["卡顿/少活人感", "动作位移不连贯"],
    "少动": ["四肢不动", "静止不动", "运动主体", "慢动作"],
    "还原": ["还原度", "衣服/身体的时间一致性", "大小变化"],
    "物理": ["物理规律", "不合理的物体"], "画面": ["帧跳变", "首帧一致", "背景运动混乱"],
}
Q = """这是AI生成的毛绒蘑菇角色「蘑菇TUTU」短视频的{n}帧画面(按时间顺序)。
判断标准:形象须与首帧一致(无眉毛/尾巴/手指,款式质感不变),动作须自然连贯有活物感(不僵硬刚体、不卡顿、四肢随动),
与场景交互正确,符合物理常识(有支撑不悬空),画面连续无跳变。
任一维度出现一眼可见、令人出戏的缺陷即为有缺陷。
问题:这条视频有明显缺陷吗?"""


def load_frames(rel, n):
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
        s = 336 / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(cv2.resize(im, (int(W*s), int(H*s))), cv2.COLOR_BGR2RGB)))
    return ims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tag", default="e36a")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(ROOT / "splits/train_v3.jsonl")]
    reasons = {r["path"]: r["reasons"] for r in csv.DictReader(
        open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}
    rel_of = {}
    for l in (ROOT / "manifest_all.tsv").read_text().splitlines():
        if l.strip():
            rel = l.split("\t")[0]
            rel_of[os.path.basename(rel)] = rel
    items = []
    for r in tr:
        rel = rel_of.get(r["video"])
        if not rel:
            continue
        y = 1 if r["label"] == "bad" else 0
        rs = reasons.get(r["video"], "")
        gs = [g for g, tags in GROUPS.items() if y == 1 and any(t in rs for t in tags)]
        tgt = ("是,存在" + "、".join(gs) + "问题。") if y == 1 else "否,未见明显缺陷。"
        if y == 1 and not gs:
            tgt = "是,存在明显缺陷。"
        items.append((rel, y, tgt))
    if args.limit:
        items = items[:args.limit]
    y_all = np.array([y for _, y, _ in items])
    print(f"样本 {len(items)},bad {y_all.sum()}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(args.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    yes_id = proc.tokenizer.encode("是")[0]
    no_id = proc.tokenizer.encode("否")[0]
    print(f"是={yes_id} 否={no_id}", flush=True)

    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=42).split(np.zeros(len(items)), y_all))
    oof = np.full(len(items), np.nan)

    def build(rel, tgt=None):
        """返回 (enc, prompt_len)。prompt_len 用于把提示段 mask 掉,只对回答算 loss。"""
        ims = load_frames(rel, args.frames)
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

    for fi, (a_idx, b_idx) in enumerate(folds):
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        lcfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        order = list(a_idx)
        for ep in range(args.epochs):
            random.shuffle(order)
            model.train()
            tot, nb = 0.0, 0
            for k, i in enumerate(order):
                rel, y, tgt = items[i]
                bd = build(rel, tgt)
                if bd is None:
                    continue
                enc, plen = bd
                enc = {k2: v.to("cuda") for k2, v in enc.items()}
                labels = enc["input_ids"].clone()
                labels[:, :plen] = -100          # 只对 assistant 回答算 loss
                labels[labels == proc.tokenizer.pad_token_id] = -100
                out = model(**enc, labels=labels)
                (out.loss / 4).backward()
                if (k + 1) % 4 == 0:
                    opt.step(); opt.zero_grad()
                tot += float(out.loss); nb += 1
                if (k + 1) % 20 == 0:
                    print(f"  fold{fi} ep{ep} [{k+1}/{len(order)}] loss {tot/max(1,nb):.4f}", flush=True)
        model.eval()
        with torch.inference_mode():
            for cnt, i in enumerate(b_idx):
                rel, y, _ = items[i]
                bd = build(rel)
                if bd is None:
                    continue
                enc, _plen = bd
                enc = {k2: v.to("cuda") for k2, v in enc.items()}
                logits = model(**enc).logits[0, -1]
                oof[i] = float(torch.softmax(logits[[no_id, yes_id]].float(), 0)[1])
                if (cnt + 1) % 300 == 0:
                    print(f"  fold{fi} 推理 [{cnt+1}/{len(b_idx)}]", flush=True)
        ok = np.isfinite(oof)
        from scipy.stats import rankdata
        r = rankdata(oof[ok]); n1 = y_all[ok].sum()
        a2 = (r[y_all[ok] == 1].sum() - n1*(n1+1)/2) / (n1*(ok.sum()-n1))
        print(f"fold{fi} 累计 AUC {a2:.4f} (覆盖{ok.sum()})", flush=True)
        del model
        torch.cuda.empty_cache()

    ok = np.isfinite(oof)
    def gn(p, y, rec=0.95):
        b = np.sort(p[y == 1]); T = b[len(b) - int(np.ceil(rec*len(b)))]
        return float((p[y == 0] < T).mean())
    from scipy.stats import rankdata
    r = rankdata(oof[ok]); n1 = y_all[ok].sum()
    print(f"[{args.tag}] AUC = {(r[y_all[ok]==1].sum()-n1*(n1+1)/2)/(n1*(ok.sum()-n1)):.4f} "
          f"gn@95 = {gn(oof[ok], y_all[ok]):.4f}", flush=True)
    np.save(ROOT / f"data/{args.tag}_oof.npy", oof)
    json.dump([it[0] for it in items], open(ROOT / f"data/{args.tag}_rels.json", "w"))
    print("E36_DONE", flush=True)


if __name__ == "__main__":
    main()
