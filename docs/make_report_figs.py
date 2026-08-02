#!/usr/bin/env python
"""汇报页数据图(报告版):FIG1 双域头条对比柱状图;FIG2 语料召回-放行 pareto 折线。
数字一字不改取自 FACTOR_PREREG 判决:
  语料 eval_v3:基线 ev@95=18.8 / ev@100=1.1 / ev@90=29.7 / ev@80=51.2
             本工作 C5    26.5 / 12.1 / 35.7 / 52.4
  产线 prod500(27/27 全召回):前人上线 0.0,前人最优组合 5.9,本工作 26.4
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent


def cjk_font():
    for p in ("/mnt/c/Windows/Fonts/msyh.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if Path(p).exists():
            return FontProperties(fname=p)
    return FontProperties()


F = cjk_font()
INK, MUT = "#22262e", "#5c6470"
C_OLD, C_NEW = "#9ca3af", "#1d4ed8"

# ---------- FIG1 双域头条 ----------
fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
groups = ["训练语料\n拦住95%坏视频时", "训练语料\n拦住100%坏视频时", "生产抽检\n拦住100%坏视频时"]
old = [18.8, 1.1, 5.9]
new = [26.5, 12.1, 26.4]
x = range(3)
w = 0.32
b1 = ax.bar([i - w / 2 for i in x], old, w, color=C_OLD, label="前人方法(同一口径下的最好成绩)")
b2 = ax.bar([i + w / 2 for i in x], new, w, color=C_NEW, label="本工作")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.7, f"{r.get_height():.1f}",
                ha="center", fontsize=10.5, color=INK, fontproperties=F)
ax.set_xticks(list(x))
ax.set_xticklabels(groups, fontproperties=F, fontsize=11.5)
ax.set_ylabel("自动放行的好视频比例(%)", fontproperties=F, fontsize=11.5)
ax.set_ylim(0, 33)
ax.legend(prop=F, fontsize=11, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "fig1_headline.png")
plt.close(fig)

# ---------- FIG2 语料 pareto ----------
fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=150)
rec = [80, 90, 95, 100]
old_p = [51.2, 29.7, 18.8, 1.1]
new_p = [52.4, 35.7, 26.5, 12.1]
ax.plot(rec, old_p, "-o", color=C_OLD, lw=2, ms=6, label="前人方法(要求拦住的坏视频比例越高,能放行的越少)")
ax.plot(rec, new_p, "-o", color=C_NEW, lw=2, ms=6, label="本工作(在高拦截要求下优势最大)")
for xx, yy in zip(rec, old_p):
    ax.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=10, color=MUT, fontproperties=F)
for xx, yy in zip(rec, new_p):
    ax.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=10, color=C_NEW, fontproperties=F)
ax.set_xticks(rec)
ax.set_xticklabels([f"{r}%" for r in rec], fontproperties=F)
ax.set_xlabel("要求拦住的坏视频比例(召回率)", fontproperties=F, fontsize=11.5)
ax.set_ylabel("自动放行的好视频比例(%)", fontproperties=F, fontsize=11.5)
ax.set_ylim(-4, 60)
ax.legend(prop=F, fontsize=10.5, frameon=False, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=10.5)
fig.tight_layout()
fig.savefig(OUT / "fig2_pareto.png")
plt.close(fig)
print("figs written:", OUT / "fig1_headline.png", OUT / "fig2_pareto.png")
