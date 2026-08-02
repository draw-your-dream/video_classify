# -*- coding: utf-8 -*-
"""从《视频数据分类和prompt校对标准.html》拆出任务一独立文档。
原文全部沿用;改动:①判定流程两个单元格追加一句 ②标签表升级三列(加一句话判据)
③新增「高频瑕疵分档速查」「填写与提交汇总」 ④删任务二与prompt示范。"""
import re, sys

SRC = "/mnt/c/Users/Lenovo/Downloads/视频数据分类和prompt校对标准.html"
raw = open(SRC, encoding="utf-8").read()

def cut(start_marker, end_marker):
    a = raw.index(start_marker)
    b = raw.index(end_marker, a)
    return raw[a:b]

# ---------- 原样切片 ----------
taskbox1 = cut('<div class="taskbox">\n<h3>任务一 视频三档分级</h3>', '<div class="taskbox">\n<h3>任务二')
grades   = cut('<h3>任务一 三档定义</h3>', '<h3>badcase 标准标签表</h3>')
dims     = cut('<h2 id="sec1">', '<h2 id="sec6">')
demos    = cut('<h2 id="good">', '<h2 id="pv">')

# ---------- 修改1:排查顺序(CG/实拍) ----------
old1 = "排查顺序建议：先看动作（僵硬、卡顿最高频），再暂停对比首帧核对形象，最后核对物理与画面。"
new1 = old1 + "此顺序适用于CG风格；实拍风格中物理与形象类问题相对更高发，建议先核对物理与形象，再看动作。"
assert old1 in taskbox1
taskbox1 = taskbox1.replace(old1, new1)

# ---------- 修改2:边界处理加分布锚点 ----------
old2 = "看了会觉得怪、出戏则判 bad，不会则判 normal。"
new2 = old2 + "参考基准：既往同类已标注数据（4473 条）中 good 约 28%、normal 约 27%、bad 约 45%；若你的 good 比例明显过半，通常意味着尺度偏松。"
assert old2 in taskbox1
taskbox1 = taskbox1.replace(old2, new2)

# ---------- 修改3:标签表升级三列 ----------
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
label_table = f"""<h3>badcase 标准标签表</h3>
<p class="desc">判 bad 时 reasons 列只能使用下表中的标签原文（与我方内部数据字段一致，用于合并统计）。一条视频可命中多个标签，用分号分隔，如「卡顿/少活人感;四肢不动;僵硬」。</p>
<table>
<tr><th style="width:150px">所属维度</th><th style="width:190px">标签（原文使用）</th><th>一句话判据</th></tr>
{chr(10).join(rows)}
</table>
<p class="desc">「还原度」与「衣服/身体的时间一致性」的分工：TUTU 本体形象特征不对（无论首帧就错还是中途走样）记「还原度」；仅衣物与身体的穿着关系前后变化记「衣服/身体的时间一致性」。</p>
<p class="desc">交互维度（第三部分）无独立标签：穿模、接触错误按情形归入「物理规律」或「还原度」。</p>"""

# ---------- 新增1:高频瑕疵分档速查 ----------
quick = """<h3 id="quick">高频瑕疵分档速查</h3>
<p class="desc">同一现象常因程度不同跨档，以下为既往标注中最常见、也最容易打错档的几类。先对照本表，再回到三档定义；表中未覆盖的情形按「边界处理」执行。</p>
<table>
<tr><th style="width:190px">现象</th><th>判 normal（轻微）</th><th>判 bad（明显）</th></tr>
<tr><td><b>多出部件</b>（眉毛、尾巴、牙齿、舌头）</td><td>无轻微档——官方形象没有这些部件</td><td>画面中出现即判 bad，记「还原度」。眉毛是既往还原度问题中最高频的子特征，重点排查</td></tr>
<tr><td><b>头身比</b></td><td>略失调，须暂停与首帧对比才能确认，整体不出戏</td><td>一眼可见头或身明显失衡，观感像换了角色</td></tr>
<tr><td><b>手指</b></td><td>略分明，但仍是短爪圆润轮廓</td><td>根根分明呈人手状</td></tr>
<tr><td><b>帽子 / 伞盖</b></td><td>帽径略偏小或略偏大，款式与斑点正确</td><td>款式变为另一款、斑点重排，或大小失控破坏形象</td></tr>
<tr><td><b>表情五官</b></td><td>表情略呆滞、欠缺灵气</td><td>五官错位、融化、眼睛变写实等崩坏</td></tr>
<tr><td><b>动态量</b></td><td>动作平淡、幅度小，但流畅有活物感</td><td>全程静止（静止不动）或身体平移四肢不摆（四肢不动/僵硬）</td></tr>
</table>"""

# ---------- 新增2:填写与提交汇总 ----------
submit = """<h3 id="submit">填写与提交汇总</h3>
<table>
<tr><th>note 列</th><td>自由填写，建议注明具体部位或现象（如「眉毛」「头身比」「手中卡片凭空出现 3s」），便于问题归类统计。标签表未覆盖的新问题必须在 note 中说明。</td></tr>
<tr><th>批次汇总</th><td>每完成一批提交时，附几句话的整体汇总：① 本批 bad 主要集中在哪些标签；② 还原度类问题集中在哪些部位（如眉毛、头身比）；③ 是否出现标签表未覆盖的新现象或拿不准的案例（列出文件名）。不要求逐条展开，按批说明即可。</td></tr>
</table>"""

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
<p class="sub">适用任务：对AI图生视频模型生成的TUTU短视频逐条分级（good / normal / bad 三档）。视频约5秒，横竖版均有。判定存在争议时，以该视频首帧画面与各款官方形象为最终依据。本文档由《蘑菇TUTU 视频标注与筛选标准》任务一部分独立而来，供数据标注参考。</p>
<nav>
<a href="#sec0">任务说明</a><a href="#quick">分档速查</a><a href="#sec1">第一部分 IP还原</a><a href="#sec2">第二部分 动作特征</a><a href="#sec3">第三部分 交互</a><a href="#sec4">第四部分 物理与物体</a><a href="#sec5">第五部分 画面稳定</a><a href="#good">good示范</a><a href="#normal">normal示范</a><a href="#bad">bad示范</a>
</nav>

<h2 id="sec0">任务说明</h2>
<p class="desc">本任务为视频三档分级，对象是AI图生视频模型生成的TUTU短视频，依据第一至第五部分标准执行。</p>

"""
doc = (head + taskbox1 + "\n" + grades + label_table + "\n\n" + quick + "\n\n" + submit
       + "\n\n" + dims + demos + "</div>\n</body>\n</html>\n")

OUT = sys.argv[1]
open(OUT, "w", encoding="utf-8").write(doc)
n_vid = len(re.findall(r"data:video/", doc))
print(f"written {OUT}: {len(doc)/1e6:.2f} MB, videos: {n_vid} (expect 21)")
