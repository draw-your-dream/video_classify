"""周报 2026-08-12 报告版配图(黑底版):图1 放行率演进;图2 尾部构成。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

def cjk_font():
    for p in ("/mnt/c/Windows/Fonts/msyh.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(p):
            return FontProperties(fname=p)
    return FontProperties()

F = cjk_font()
BG = "#1a1a19"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUT = "#898781"
BLUE = "#3987e5"
RED = "#e66767"
AMBER = "#d9a83a"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- 图1:放行率演进 ----------
stages = ["之前的系统", "更换组合器", "组合器系统搜索", "加入眉毛检测门"]
vals = [18.8, 26.5, 29.1, 30.9]
cols = ["#33517a", "#33517a", "#33517a", "#33517a"]

fig, ax = plt.subplots(figsize=(8.6, 4.1), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.bar(range(4), vals, width=0.58, color=cols, edgecolor="none")
for i, v in enumerate(vals):
    ax.text(i, v + 0.55, f"{v}%", ha="center", fontsize=12.5,
            color=INK, fontproperties=F)
ax.set_xticks(range(4))
ax.set_xticklabels(stages, fontproperties=F, fontsize=11, color=INK2)
ax.set_ylabel("自动放行的好视频比例(%)", fontproperties=F, fontsize=11.5, color=INK2)
ax.set_ylim(0, 37)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(MUT)
ax.tick_params(colors=MUT)
for t in ax.get_yticklabels():
    t.set_color(INK2)
ax.set_title("在拦住 95% 坏视频的前提下,自动放行比例的演进", fontproperties=F,
             fontsize=13, color=INK, pad=12)
fig.tight_layout()
fig.savefig(f"{OUT}/wfig1_headline.png", facecolor=BG)
print("figs done")
