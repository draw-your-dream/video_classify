#!/usr/bin/env python3
"""F3:五轴分解判官(codex M1 架构)。每轴一次独立 flash 调用,只审自己负责的维度;
程序聚合:任一轴 major → bad;任一轴 minor 或整体平淡 → normal;全 pass → good。

用法: python axis_judge.py --videos videos --manifest pilot50.csv --refs refs --sku-refs sku_refs --out out/p50_axis.jsonl
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pilot import sku_ref_paths, sku_context, load_done  # noqa: E402

HEAD = """你是「蘑菇TUTU」AI 生成短视频的单维度证据审计员,只审下述一个维度,其他维度一律忽略。
第一张图(若有官方参考则前几张)展示该款式的标准形象与本视频首帧;之后是完整视频。
判定档位:pass=该维度无可确认瑕疵;minor=须暂停或仔细对照才能发现、正常播放不出戏的轻微瑕疵;
major=正常播放一眼可见、令普通观众出戏的明显缺陷;uncertain=无法可靠判断。
major 必须给出时间区间和具体可见现象。只输出 JSON:
{"severity":"pass|minor|major|uncertain","evidence":[{"start_sec":0,"end_sec":0,"observation":""}],"note":""}"""

AXES = {
    "ip": """【维度:IP还原】全程对照首帧/官方参考:伞盖款式与斑点、五官、头身比、体型大小、毛绒质感不得改变;
不得长出眉毛/尾巴/分明的手指脚趾/脖子;身体四肢不得拉长抻细;质感不得变塑料;
衣物与身体的穿着关系不得中途变化融合消失;相对场景的大小不得无理由变化。
注意:眯眼/闭眼时黑豆眼变成弧线是正常表情,不算缺陷。缺陷常在中段短暂出现后恢复,须逐段扫描。""",
    "motion": """【维度:动作特征】核心是"活物感":动作与位移须平滑连续,不得一顿一顿;
身体四肢须柔软有缓冲,不得呈整块硬物被翻动平移;身体动时四肢须相应摆动;
主体须有有效动态(全程静止、或只有镜头/背景在动 = major);速度正常不得慢动作拖影。
动作幅度小本身不扣分,只要流畅有生命力。""",
    "interact": """【维度:交互】TUTU 与场景物体的接触须贴合:无穿模、不嵌入物体;
前后遮挡关系符合相机视角;与物体的相对大小稳定。仅站立、无任何交互不算缺陷(记 pass)。""",
    "physics": """【维度:物理与物体】站坐须有真实支撑面,不得悬空、漂浮、无支撑滑动;
画面中物体不得凭空出现、凭空消失或无理由变形;雪地/纯白背景下支撑面可能不可见,此时勿轻判悬空。""",
    "stability": """【维度:画面稳定】相邻帧内容不得突变(背景物体瞬间出现/消失/替换);
视频起始画面须与首帧一致;背景不得混乱运动、扭曲、崩坏。只审 TUTU 以外的画面。""",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["pass", "minor", "major", "uncertain"]},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "start_sec": {"type": "number"}, "end_sec": {"type": "number"},
            "observation": {"type": "string"}},
            "required": ["start_sec", "end_sec", "observation"]}},
        "note": {"type": "string"},
    },
    "required": ["severity", "evidence"],
}


def call_axis(video_b64, ref_parts, axis_text, ctx, model, key):
    parts = list(ref_parts)
    parts.append({"inlineData": {"mimeType": "video/mp4", "data": video_b64},
                  "videoMetadata": {"fps": 5}})
    parts.append({"text": ctx + HEAD + "\n" + axis_text})
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseJsonSchema": SCHEMA,
                                 "thinkingConfig": {"thinkingLevel": "medium"},
                                 "mediaResolution": "MEDIA_RESOLUTION_LOW"}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    return json.loads(resp["candidates"][0]["content"]["parts"][-1]["text"])


def aggregate(ax):
    sevs = {k: v.get("severity", "uncertain") for k, v in ax.items()}
    if any(s == "major" for s in sevs.values()):
        return "bad"
    if any(s == "uncertain" for s in sevs.values()):
        # 不确定不强猜,但只有不确定而无缺陷时按 normal 保守放行档
        return "normal"
    if any(s == "minor" for s in sevs.values()):
        return "normal"
    return "good"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--refs", default=None)
    ap.add_argument("--sku-refs", default=None)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    key = os.environ["GOOGLE_API_KEY"]
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    done = load_done(args.out)
    f = open(args.out, "a")
    t0, n = time.time(), 0
    for r in rows:
        fn = r["filename"]
        if fn in done:
            continue
        vp = Path(args.videos) / fn
        row = {"filename": fn, "model": args.model}
        try:
            vb = base64.b64encode(open(vp, "rb").read()).decode()
            ref_parts = []
            for sp in sku_ref_paths(fn, args.sku_refs):
                ref_parts.append({"inlineData": {"mimeType": "image/png",
                                                 "data": base64.b64encode(open(sp, "rb").read()).decode()}})
            if args.refs:
                rp = Path(args.refs) / (fn[:-4] + ".png")
                if rp.exists():
                    ref_parts.append({"inlineData": {"mimeType": "image/png",
                                                     "data": base64.b64encode(open(rp, "rb").read()).decode()}})
            ctx = sku_context(fn)
            ax = {}
            for aname, atext in AXES.items():
                for attempt in range(3):
                    try:
                        ax[aname] = call_axis(vb, ref_parts, atext, ctx, args.model, key)
                        break
                    except Exception as e:
                        if attempt == 2:
                            ax[aname] = {"severity": "uncertain", "evidence": [],
                                         "note": f"error:{repr(e)[:60]}"}
                        else:
                            time.sleep(4 * (attempt + 1))
            row["axes"] = ax
            row["result"] = {"grade": aggregate(ax),
                             "reason_labels": [],
                             "evidence": sum((v.get("evidence", []) for v in ax.values()), []),
                             "normal_play_visible": any(v.get("severity") == "major" for v in ax.values())}
        except Exception as e:
            row["error"] = repr(e)[:150]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        n += 1
        if n % 10 == 0:
            print(f"[{n}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    print("AXIS_DONE", flush=True)


if __name__ == "__main__":
    main()
