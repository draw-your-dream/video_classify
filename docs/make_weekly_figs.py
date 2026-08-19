#!/usr/bin/env python
"""周报数据图(深色报告版):FIG1 各路信号与融合判断的拦截率;FIG2 交付版的放行-拦截曲线。
数字取自 2.0 数据集建模集(867 条)十折分组交叉验证,复算脚本 scripts/api_judge/combiner_dev.py。
"""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

D = Path.home()/"tutu-video-eval/data"; OUT = D/"pbase/out"
DOC = Path(__file__).resolve().parent

def cjk_font(bold=False):
    for p in ["/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(p).exists(): return FontProperties(fname=p)
    return FontProperties()
F, FB = cjk_font(), cjk_font(True)
BG, INK, MUT, GRID = "#171a20", "#e9e7e2", "#98a0ac", "#2b3039"
C_HI, C_LO = "#7aa2f7", "#454d5c"

def br_at(s, y, rel=0.8):
    gn = np.sort(s[y==0]); b = s[y==1]; k = int(np.floor(rel*len(gn)))
    if k == 0: return 1.0
    t = gn[k-1]; nb=(gn<t).sum(); ne=(gn==t).sum(); fr=(k-nb)/ne
    return ((b>t).sum() + (b==t).sum()*(1-fr))/len(b)

def rankpct(x): return rankdata(x)/len(x)

def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=10.5)
    ax.xaxis.label.set_color(MUT); ax.yaxis.label.set_color(MUT)

vids=[v if v.endswith(".mp4") else v+".mp4" for v in json.load(open(OUT/"X303_vids.json"))]
idx={v:i for i,v in enumerate(vids)}; N=len(vids)
mapr={r["filename"]:r for r in csv.DictReader(open(D/"api_judge_video_image_map.csv",encoding="utf-8-sig"))}
y=np.array([1 if mapr[v]["grade"]=="bad" else 0 for v in vids])
groups=np.array([mapr[v]["source_sha"] for v in vids])
lb=json.load(open(D/"lockbox_split.json")); dev=set(lb["dev"])
dm=np.array([(g in dev) or (v in dev) for g,v in zip(groups,vids)])

z=np.load(OUT/"r1_oof.npz",allow_pickle=True)
s1=json.load(open(OUT/"flash_full_1233.json")); s2=json.load(open(OUT/"flash_run2_1233.json"))
f1=np.array([s1.get(v,50) for v in vids],float); f2=np.array([s2.get(v,50) for v in vids],float)
flash=(rankpct(f1)+rankpct(f2))/2
ip=np.zeros(N)
for r in csv.DictReader(open(OUT/"imgprobe_1233.csv")):
    if r["filename"] in idx: ip[idx[r["filename"]]]=float(r["p_gbm"])
bag=np.load(OUT/"weekly_bag_oof.npy"); yd=np.load(OUT/"weekly_bag_y.npy")

# ---------- FIG1 ----------
items=[("官方形象图对照", 0.3220, False),
       ("Gemini Flash 判官直接检测", br_at(flash[dm],y[dm]), False),
       ("专家模型集合直接检测", br_at(z["r1c"].astype(float)[dm],y[dm]), False),
       ("源图检测模型(只看第一帧)", br_at(ip[dm],y[dm]), False),
       ("多路信号融合判断", br_at(bag,yd), True)]
fig, ax = plt.subplots(figsize=(9.0,4.0), dpi=150, facecolor=BG)
ys=np.arange(len(items))[::-1]
for (lab,v,hi),yy in zip(items,ys):
    ax.barh(yy, v*100, height=0.54, color=(C_HI if hi else C_LO))
    ax.text(v*100+0.9, yy, f"{v*100:.1f}%", va="center", fontsize=11.5,
            color=(C_HI if hi else MUT), fontproperties=FB if hi else F)
ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items], fontproperties=F, fontsize=11.5)
for t,(lab,v,hi) in zip(ax.get_yticklabels(), items):
    t.set_color(INK if hi else MUT)
ax.set_xlabel("放行 80% 合格视频时,被拦下的问题视频比例", fontproperties=F, fontsize=11.5)
ax.set_xlim(0,62)
style(ax)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
fig.tight_layout(); fig.savefig(DOC/"wk_fig1_signals.png", facecolor=BG); plt.close(fig)

# ---------- FIG2 ----------
rels=np.arange(0.50,0.99,0.02)
fig, ax = plt.subplots(figsize=(8.6,4.2), dpi=150, facecolor=BG)
ax.plot(rels*100, [br_at(bag,yd,r)*100 for r in rels], color=C_HI, lw=2.6)
ax.axvline(80, color="#3a4150", lw=1, ls=(0,(4,4)))
ax.scatter([80],[br_at(bag,yd)*100], color=C_HI, zorder=5, s=38)
ax.annotate(f"放行 80% 时拦下 {br_at(bag,yd)*100:.1f}%", (80, br_at(bag,yd)*100),
            textcoords="offset points", xytext=(-14,-34), fontproperties=FB, fontsize=11.5,
            color=INK, ha="right")
ax.set_xlabel("自动放行的合格视频比例", fontproperties=F, fontsize=11.5)
ax.set_ylabel("被拦下的问题视频比例", fontproperties=F, fontsize=11.5)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(0, 88)
style(ax)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(DOC/"wk_fig2_workpoint.png", facecolor=BG); plt.close(fig)
print("ok", [(l, round(v,4)) for l,v,_ in items])
