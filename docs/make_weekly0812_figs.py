"""周报 2026-08-12 报告版配图:图1 放行率演进阶梯;图2 尾部杠杆结构。"""
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
INK = "#22262e"
MUT = "#5c6470"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- 图1:放行率演进 ----------
stages = ["之前的系统", "更换组合器\n(上次汇报)", "组合器系统搜索\n(上次汇报)", "本周:眉毛否决门\n+双钥匙修正"]
vals = [18.8, 26.5, 29.1, 30.9]
cols = ["#b9b3a6", "#9aa7c4", "#7d90bd", "#1d4ed8"]

fig, ax = plt.subplots(figsize=(8.6, 4.1), dpi=160)
bars = ax.bar(range(4), vals, width=0.58, color=cols, edgecolor="none")
for i, v in enumerate(vals):
    ax.text(i, v + 0.55, f"{v}%", ha="center", fontsize=12.5,
            color=INK if i < 3 else "#1d4ed8",
            fontweight="bold" if i == 3 else "normal", fontproperties=F)
d = vals[3] - vals[2]
ax.annotate("", xy=(2.72, vals[3] + 3.1), xytext=(2, vals[2] + 3.1),
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.4))
ax.text(2.36, vals[3] + 3.7, f"本周 +{d:.1f} 个百分点", ha="center", fontsize=11,
        color="#1d4ed8", fontproperties=F)
ax.set_xticks(range(4))
ax.set_xticklabels(stages, fontproperties=F, fontsize=11)
ax.set_ylabel("自动放行的好视频比例(%)", fontproperties=F, fontsize=11.5)
ax.set_ylim(0, 37)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUT)
ax.set_title("在拦住 95% 坏视频的前提下,自动放行比例的演进", fontproperties=F,
             fontsize=13, color=INK, pad=12)
fig.tight_layout()
fig.savefig(f"{OUT}/wfig1_headline.png", facecolor="white")

# ---------- 图2:尾部杠杆结构 ----------
# 评估集阈值由分数最低的 20 条坏视频决定;构成:运动类 11 / 还原度类 8 / 物理类 1
fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=160)
fams = [("运动类(僵硬、卡顿、无生命感)", 11, "#c2410c"),
        ("还原度类(形象与原设不符:大小、细节、衣着)", 8, "#1d4ed8"),
        ("物理类(悬空等违反物理)", 1, "#7c3aed")]
left = 0
for name, n, c in fams:
    ax.barh(0, n, left=left, height=0.52, color=c, edgecolor="white", linewidth=1.5)
    ax.text(left + n / 2, 0, str(n), ha="center", va="center", color="white",
            fontsize=13, fontweight="bold", fontproperties=F)
    left += n
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in fams]
ax.legend(handles, [f"{n1}:{n2} 条" for n1, n2, _ in fams], loc="upper center",
          bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False,
          prop=F, fontsize=11)
for t in ax.get_legend().get_texts():
    t.set_fontproperties(F); t.set_fontsize(11)
ax.set_xlim(0, 20)
ax.set_ylim(-0.6, 0.6)
ax.axis("off")
ax.set_title("决定放行率的 20 条最难判坏视频的构成(每修复一条约 +0.9 个百分点)",
             fontproperties=F, fontsize=12.5, color=INK, pad=10)
fig.tight_layout()
fig.savefig(f"{OUT}/wfig2_tail.png", facecolor="white", bbox_inches="tight")
print("figs done")
