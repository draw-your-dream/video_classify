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

# ---------- 图2:尾部构成 ----------
fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fams = [("运动类(僵硬、卡顿、无生命感):11 条", 11, RED),
        ("还原度类(形象与原设不符:大小、细节、衣着):8 条", 8, BLUE),
        ("物理类(悬空等违反物理):1 条", 1, AMBER)]
left = 0
for name, n, c in fams:
    ax.barh(0, n, left=left, height=0.52, color=c, edgecolor=BG, linewidth=1.5)
    ax.text(left + n / 2, 0, str(n), ha="center", va="center", color="#111",
            fontsize=13, fontweight="bold", fontproperties=F)
    left += n
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in fams]
leg = ax.legend(handles, [n1 for n1, _, _ in fams], loc="upper center",
                bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False, prop=F)
for t in leg.get_texts():
    t.set_fontproperties(F); t.set_fontsize(11); t.set_color(INK2)
ax.set_xlim(0, 20)
ax.set_ylim(-0.6, 0.6)
ax.axis("off")
ax.set_title("决定放行率的 20 条最难判坏视频的构成(每修复一条约 +0.9 个百分点)",
             fontproperties=F, fontsize=12.5, color=INK, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/wfig2_tail.png", facecolor=BG, bbox_inches="tight")
print("figs done")
