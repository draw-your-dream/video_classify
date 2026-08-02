# -*- coding: utf-8 -*-
"""从《视频数据分类和prompt校对标准.html》拆出三档分级独立文档(v3 重构版)。
结构:任务说明 → 三档定义 → 判定流程 → 五个维度 → 标签表 → 示范。
单任务文档,去掉全部「任务一/任务二」框架;速查表不独立成节,其实质并入
判定流程第二步(轻微/明显分界基准)与第一部分常见不合格点(眉毛出现即bad)。"""
import re, sys

SRC = "/mnt/c/Users/Lenovo/Downloads/视频数据分类和prompt校对标准.html"
raw = open(SRC, encoding="utf-8").read()

def cut(start_marker, end_marker):
    a = raw.index(start_marker)
    b = raw.index(end_marker, a)
    return raw[a:b]

# ---------- 原样切片 ----------
taskbox1 = cut('<div class="taskbox">\n<h3>任务一 视频三档分级</h3>', '<div class="taskbox">\n<h3>任务二')
grades_tbl = cut('<h3>任务一 三档定义</h3>', '<h3>badcase 标准标签表</h3>').replace(
    "<h3>任务一 三档定义</h3>", "").strip()
dims  = cut('<h2 id="sec1">', '<h2 id="sec6">')
demos = cut('<h2 id="good">', '<h2 id="pv">')

aims = re.findall(r'<p class="aim">.*?</p>', taskbox1, re.S)
steps_tbl = taskbox1[taskbox1.index("<table>"):taskbox1.index("</table>") + 8]

# ---------- 并入1:第二步补「轻微/明显」分界基准 ----------
old = "存在轻微瑕疵（头身比略失调、手指略分明、表情略呆滞、帽子略小等）"
new = old + "——「轻微」的基准是须暂停与首帧对比才能确认、整体不出戏；一眼可见的失衡或崩坏属明显缺陷，应在第一步判 bad——"
assert old in steps_tbl
steps_tbl = steps_tbl.replace(old, new)

# ---------- 清理:单任务文档删去任务归属句 ----------
dims = dims.replace("本部分及第二至第五部分适用于任务一。", "")

# ---------- 并入2:第一部分常见不合格点补「出现即bad」与五官崩坏 ----------
old = "长出眉毛、尾巴，手指变得根根分明，头身比失调，质感变塑料。"
new = ("长出眉毛、尾巴（官方形象没有这些部件，出现即判bad；眉毛是既往还原度问题中最高频的子特征，重点排查），"
       "手指变得根根分明，头身比失调，五官错位或融化，质感变塑料。")
assert old in dims
dims = dims.replace(old, new)

# ---------- 标签表(三列判据版) ----------
D = {"d1": "第一部分 IP还原", "d2": "第二部分 动作特征", "d4": "第四部分 物理与物体", "d5": "第五部分 画面稳定"}
LABELS = [
    ("d1", "还原度", "TUTU本体形象不符或走样：款式/斑点错乱、长出眉毛尾巴、手指分明、头身比失调、质感变塑料等"),
    ("d1", "衣服/身体的时间一致性", "衣物与身体的穿着关系前后不一致，中途变化、融合或消失"),
    ("d1", "大小变化", "相对场景参照物的体型无理由膨大或缩小（以首帧为准）"),
    ("d2", "僵硬", "呈整块硬物状被翻动或平移，无自然弯曲与缓冲（最高频）"),
    ("d2", "卡顿/少活人感", "动作一顿一顿地挪动，缺少连续生命感"),
    ("d2", "四肢不动", "身体在动而四肢完全固定，摆件式平移"),
    ("d2", "动作位移不连贯", "姿态或位置在相邻时刻跳跃，衔接断裂"),
    ("d2", "运动主体", "应当TUTU动，实际是镜头推拉或背景物体在动"),
    ("d2", "静止不动", "主体全程或过长时间静止，无有效动态"),
    ("d2", "慢动作", "动作速度异常缓慢、拖影（少见）"),
    ("d4", "物理规律", "违反重力与支撑逻辑：悬空、漂浮、无支撑滑动；穿模、嵌入物体也计入此类"),
    ("d4", "不合理的物体", "物体凭空出现、凭空消失或无理由变形"),
    ("d5", "帧跳变", "相邻帧画面内容突变：背景物体瞬间出现、消失或替换"),
    ("d5", "首帧一致", "视频起始画面与给定首帧不一致，开场即偏离"),
    ("d5", "背景运动混乱", "背景人物或物体混乱运动、扭曲、崩坏（少见）"),
]
rows = []
from itertools import groupby
for dim, grp in groupby(LABELS, key=lambda x: x[0]):
    grp = list(grp)
    for i, (_, name, crit) in enumerate(grp):
        dimcell = (f'<td rowspan="{len(grp)}"><span class="ptag {dim}">{D[dim]}</span></td>' if i == 0 else "")
        rows.append(f"<tr>{dimcell}<td style=\"width:190px\"><b>{name}</b></td><td>{crit}</td></tr>")
label_sec = f"""<h2 id="labels">badcase 标准标签表</h2>
<p class="desc">判 bad 时 reasons 列只能使用下表中的标签原文（与我方内部数据字段一致，用于合并统计）。一条视频可命中多个标签，用分号分隔，如「卡顿/少活人感;四肢不动;僵硬」。</p>
<table>
<tr><th style="width:150px">所属维度</th><th style="width:190px">标签（原文使用）</th><th>一句话判据</th></tr>
{chr(10).join(rows)}
</table>
<p class="desc">「还原度」与「衣服/身体的时间一致性」的分工：TUTU 本体形象特征不对（无论首帧就错还是中途走样）记「还原度」；仅衣物与身体的穿着关系前后变化记「衣服/身体的时间一致性」。</p>
<p class="desc">交互维度（第三部分）无独立标签：穿模、接触错误按情形归入「物理规律」或「还原度」。</p>
<p class="desc">note 列自由填写，建议注明具体部位或现象（如「眉毛」「头身比」）；每批提交时建议附几句整体汇总：本批 bad 主要集中在哪些标签，有无标签表未覆盖的新现象或拿不准的案例。</p>"""

# ---------- 组装 ----------
head = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>蘑菇TUTU 视频三档分级标准</title>
""" + cut("<style>", "</style>") + """</style>
</head>
<body>
<div class="wrap">
<h1>蘑菇TUTU 视频三档分级标准</h1>
<p class="sub">适用任务：对AI图生视频模型生成的TUTU短视频逐条分级（good / normal / bad 三档）。视频约5秒，横竖版均有。判定存在争议时，以该视频首帧画面与各款官方形象为最终依据。</p>
<nav>
<a href="#task">任务说明</a><a href="#def">三档定义</a><a href="#flow">判定流程</a><a href="#sec1">第一部分 IP还原</a><a href="#sec2">第二部分 动作特征</a><a href="#sec3">第三部分 交互</a><a href="#sec4">第四部分 物理与物体</a><a href="#sec5">第五部分 画面稳定</a><a href="#labels">标签表</a><a href="#good">good示范</a><a href="#normal">normal示范</a><a href="#bad">bad示范</a>
</nav>

<h2 id="task">任务说明</h2>
""" + "\n".join(aims) + """

<h2 id="def">三档定义</h2>
""" + grades_tbl + """

<h2 id="flow">判定流程</h2>
<p class="desc">每条视频按两步判定：第一步排查明显缺陷（命中即 bad），第二步在无缺陷的视频中区分 good 与 normal。</p>
""" + steps_tbl + """

""" + dims + label_sec + "\n\n" + demos + "</div>\n</body>\n</html>\n"

doc = head
OUT = sys.argv[1]
open(OUT, "w", encoding="utf-8").write(doc)
n_vid = len(re.findall(r"data:video/", doc))
print(f"written {OUT}: {len(doc)/1e6:.2f} MB, videos: {n_vid} (expect 21), aims: {len(aims)} (expect 2)")
