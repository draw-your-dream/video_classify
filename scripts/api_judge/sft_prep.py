#!/usr/bin/env python3
"""新批(sft_task1,901 条)清单构建:解析文件名 → 子集/款式/玩法/道具,与 1233 语料对齐字段。
输出 data/sft901_manifest.csv,列:filename, rel(相对路径), subset, ckpt, sku, track, prop, grade, reasons, y
款式对齐:2.0 子集第二段即款式(去掉尾部「款」字后与 1233 的八款对齐);1.0 子集款式未知,置空待补。
玩法对齐 1233 的轨:exact→exact-r1 同义、noprop→base-noprop、generic→generic。"""
import csv, collections
from pathlib import Path
SRC=Path("/mnt/c/Users/Lenovo/Downloads/sft_task1_annotations.csv")
D=Path.home()/"tutu-video-eval/data"
SKU8={"爆炸菇（爆炸前）","爆炸菇（爆炸后）","粽子菇","蘑菇力","吐司菇","毒蘑菇","汉堡菇","炸虾菇"}
rows=[]
for r in csv.DictReader(open(SRC,encoding="utf-8-sig")):
    rel=r["filename"]; parts=rel.split("/"); base=parts[-1][:-4] if parts[-1].endswith(".mp4") else parts[-1]
    seg=base.split("__")
    subset=parts[0]                       # base / 1.0 / 2.0
    vid=seg[0]
    second=seg[1] if len(seg)>1 else ""
    track=seg[2] if len(seg)>2 else ""
    prop=seg[3] if len(seg)>3 else ""
    ckpt=second if subset=="1.0" else ""
    sku=""
    if subset=="2.0":
        s=second[:-1] if second.endswith("款") else second
        sku=s if s in SKU8 else s
    trmap={"exact":"exact","noprop":"base-noprop","generic":"generic"}
    rows.append({"filename":base+".mp4","rel":rel,"subset":subset,"ckpt":ckpt,"sku":sku,
                 "track":trmap.get(track,track),"prop":prop,"vid":vid,
                 "grade":r["grade"],"reasons":r["reasons"] or "",
                 "y":1 if r["grade"]=="bad" else 0})
out=D/"sft901_manifest.csv"
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"→ {out}  {len(rows)} 行")
print("子集:",dict(collections.Counter(r["subset"] for r in rows)))
print("轨:",dict(collections.Counter(r["track"] for r in rows)))
print("款式(2.0):",dict(collections.Counter(r["sku"] for r in rows if r["subset"]=="2.0")))
print("款式能否对上 1233 八款:",{s:(s in SKU8) for s in sorted({r["sku"] for r in rows if r["sku"]})})
print("bad 率 总体 %.3f"%(sum(r["y"] for r in rows)/len(rows)))
for sub in ("1.0","2.0","base"):
    g=[r for r in rows if r["subset"]==sub]
    if g: print(f"  {sub}: n={len(g)} bad率 {sum(x['y'] for x in g)/len(g):.3f}")
# 文件名唯一性 + 与 1233 是否重名
names=collections.Counter(r["filename"] for r in rows)
dup=[k for k,v in names.items() if v>1]
print("重名:",len(dup))
old={r["filename"] for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
print("与 1233 文件名交集:",len({r["filename"] for r in rows}&old))
