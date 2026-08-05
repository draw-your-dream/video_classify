#!/usr/bin/env python
"""E25:把《三档分级标准》判定方法整理后直喂 Qwen3-VL-32B,四变体试点(2026-08-04 预注册+调研修订)。
P1 标准整理版+三档(pointwise对照) | P2 检查单二元分解(13标签yes/no)
P3 锚定成对(vs同源最自信good,双向问序) | P4 = P1 + 原生视频模式(24帧+时间元数据)
试点150条(train侧),输出 e25_p{1..4}.jsonl,可续跑。"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/mech")

STD = """你要按照「蘑菇TUTU官方标注标准」评估一条约5秒的AI图生视频(可能是CG动画风格,也可能是实拍场景中的玩偶摆件风格)。
角色官方设定:蓬松毛绒蘑菇伞盖(基础款橙红底白圆点,另有冰晶/星月等款式,以首帧为准);奶白色圆脸,两颗实心黑豆眼,腮红,无眉、无牙、无舌;身体矮胖梨形,头身比约1:1,四肢短小圆润,无手指、无脚趾、无尾巴;全身短毛绒哑光质感。首帧即标准答案:款式、大小、质感与场景关系全程不得走样。
五个维度的标准:
一、IP还原:形象须与首帧一致且符合官方特征。常见不合格:长出眉毛或尾巴(官方没有,出现即bad,眉毛最高频须重点排查)、手指根根分明、头身比失调、伞盖款式或斑点中途改变、衣物与身体穿着关系中途变化融合消失、体型无理由膨大缩小、五官错位融化、质感变塑料。
二、动作特征(bad最高频来源,重点排查):核心是「活物感」。不得整块硬物状被翻动或平移(僵硬);不得一顿一顿挪动(卡顿);身体动时四肢须相应摆动,不得摆件式平移(四肢不动);不得姿态或位置在相邻时刻跳跃(位移不连贯);应当TUTU动时不得以镜头推拉或背景移动代替(运动主体错误);不得全程或过长时间静止;不得慢动作拖影。
三、交互:与场景物体接触须贴合,无穿模不嵌入;遮挡关系符合视角;相对大小与首帧一致且稳定。
四、物理与物体:站坐须有真实支撑面,不得悬空漂浮无支撑滑动;物体不得凭空出现、凭空消失或无理由变形。
五、画面稳定:相邻帧画面内容不得突变;背景物体不得瞬间出现消失替换;起始画面须与首帧一致;背景不得混乱运动扭曲崩坏。
判定流程:第一步排查明显缺陷——任一维度出现「一眼可见、令人出戏」程度的缺陷即判bad,一票否决,不与优点相抵。第二步,无明显缺陷者:有需仔细看才能发现的轻微瑕疵、或整体平淡缺乏活物感亮点,判normal;形象全程稳定、动作自然流畅有活物感、交互与物理均正确、无可挑剔,判good。边界:good与normal拿不准判normal;normal与bad拿不准以普通观众观感为准,看了会觉得怪、出戏则bad。"""

LABELS = [
    ("还原度", "TUTU本体形象不符或走样:款式/斑点错乱、长出眉毛尾巴、手指分明、头身比失调、质感变塑料等"),
    ("衣服/身体的时间一致性", "衣物与身体的穿着关系前后不一致,中途变化、融合或消失"),
    ("大小变化", "相对场景参照物的体型无理由膨大或缩小(以首帧为准)"),
    ("僵硬", "呈整块硬物状被翻动或平移,无自然弯曲与缓冲"),
    ("卡顿/少活人感", "动作一顿一顿地挪动,缺少连续生命感"),
    ("四肢不动", "身体在动而四肢完全固定,摆件式平移"),
    ("动作位移不连贯", "姿态或位置在相邻时刻跳跃,衔接断裂"),
    ("运动主体", "应当TUTU动,实际是镜头推拉或背景物体在动"),
    ("静止不动", "主体全程或过长时间静止,无有效动态"),
    ("物理规律", "违反重力与支撑逻辑:悬空、漂浮、无支撑滑动;穿模、嵌入物体"),
    ("不合理的物体", "物体凭空出现、凭空消失或无理由变形"),
    ("帧跳变", "相邻帧画面内容突变:背景物体瞬间出现、消失或替换"),
    ("首帧一致", "视频起始画面与给定首帧不一致,开场即偏离"),
]

P1_TAIL = """
以下按时间顺序给出这条视频均匀抽取的16帧。请按上述标准与流程判定,只输出一行JSON:
{"grade": "good"或"normal"或"bad", "bad_prob": 0到100整数(这条视频是bad的概率), "reasons": [命中的标签,从下表原文选取,没有则空数组], "note": "一句话"}
标签表:""" + "、".join(l for l, _ in LABELS)

P2_TAIL = ("""
以下按时间顺序给出这条视频均匀抽取的16帧。请对下面13个问题逐一回答,每个问题独立判断,是=1,否=0。只报你确定看到的,不确定answer 0。
""" + "\n".join(f"{i+1}. {l}:{d}?" for i, (l, d) in enumerate(LABELS)) + """
最后综合按上述判定流程给档位。只输出一行JSON:
{"hits": [13个0或1], "grade": "good"或"normal"或"bad"}""")

P3_PROMPT = STD + """
下面是两条TUTU视频:视频A(前8帧)与视频B(后8帧),各自按时间顺序均匀抽取。
按上述标准判断:哪一条更可能是「bad」(存在一眼可见、令人出戏的明显缺陷)?若两条都无明显缺陷或无法区分,答same。
只输出一行JSON:{"worse": "A"或"B"或"same", "reason": "一句话"}"""


def read_frames(vp, n):
    cap = cv2.VideoCapture(str(vp))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if tot <= 1:
        cap.release(); return []
    idxs = [int(round(i * (tot - 1) / (n - 1))) for i in range(n)]
    want = set(idxs); out = {}; k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            out[k] = fr
        k += 1
    cap.release()
    ims = []
    for i in sorted(out):
        fr = out[i]
        H, W = fr.shape[:2]
        s = 448 / max(H, W)
        ims.append(Image.fromarray(cv2.cvtColor(
            cv2.resize(fr, (int(W * s), int(H * s))), cv2.COLOR_BGR2RGB)))
    return ims


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(ROOT / "data/corpus_videos"))
    ap.add_argument("--cfg", default=str(ROOT / "e25_pilot.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    ap.add_argument("--variants", default="1,2,3,4")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.cfg))
    pilot = cfg["pilot"]
    if args.limit:
        pilot = pilot[:args.limit]

    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print("vlm loaded", flush=True)

    def ask(content, max_new=160):
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    # P4 原生视频模式可用性探测
    video_mode = False
    try:
        dummy = [Image.new("RGB", (448, 448)) for _ in range(4)]
        msgs = [{"role": "user", "content": [{"type": "video", "video": dummy},
                                             {"type": "text", "text": "一句话描述"}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)
        video_mode = True
    except Exception as e:
        print(f"video-mode不可用,P4退化为24帧图列: {repr(e)[:120]}", flush=True)
    print(f"video_mode={video_mode}", flush=True)

    variants = [int(v) for v in args.variants.split(",")]
    anchors = {s: read_frames(Path(args.videos_dir) / r, 8) for s, r in cfg["anchors"].items()}

    for v in variants:
        out_p = ROOT / f"data/e25_p{v}.jsonl"
        done = set()
        if out_p.exists():
            done = {json.loads(l)["rel"] for l in out_p.read_text().splitlines()
                    if l.strip() and "error" not in json.loads(l)}
        todo = [p for p in pilot if p["rel"] not in done]
        print(f"== P{v}: todo {len(todo)} ==", flush=True)
        f = open(out_p, "a"); t0 = time.time()
        for k, p in enumerate(todo):
            row = {"rel": p["rel"], "label": p["label"]}
            try:
                if v in (1, 2):
                    ims = read_frames(Path(args.videos_dir) / p["rel"], 16)
                    if len(ims) < 8:
                        raise RuntimeError("too few frames")
                    tail = P1_TAIL if v == 1 else P2_TAIL
                    content = [{"type": "image", "image": im} for im in ims]
                    content.append({"type": "text", "text": STD + tail})
                    j = parse_json(ask(content, 200 if v == 1 else 260))
                    row.update(j or {"parse_error": 1})
                elif v == 3:
                    cand = read_frames(Path(args.videos_dir) / p["rel"], 8)
                    anc = anchors.get(p["src"]) or []
                    if len(cand) < 6 or len(anc) < 6:
                        raise RuntimeError("too few frames")
                    for tag, pair in (("first", (cand, anc)), ("second", (anc, cand))):
                        content = [{"type": "image", "image": im} for im in pair[0]]
                        content += [{"type": "image", "image": im} for im in pair[1]]
                        content.append({"type": "text", "text": P3_PROMPT})
                        txt = ask(content, 96)
                        j = parse_json(txt) or {}
                        row[f"worse_{tag}"] = j.get("worse")
                        if j.get("worse") is None:
                            row[f"raw_{tag}"] = txt[:120]
                else:
                    ims = read_frames(Path(args.videos_dir) / p["rel"], 24)
                    if len(ims) < 12:
                        raise RuntimeError("too few frames")
                    tail = STD + P1_TAIL.replace("16帧", "24帧")
                    j = None
                    if video_mode:
                        try:
                            content = [{"type": "video", "video": ims,
                                        "metadata": {"fps": len(ims) / 5.0, "duration": 5.0,
                                                     "total_num_frames": len(ims)}},
                                       {"type": "text", "text": tail}]
                            j = parse_json(ask(content, 200))
                        except Exception:
                            content = [{"type": "video", "video": ims},
                                       {"type": "text", "text": tail}]
                            j = parse_json(ask(content, 200))
                    else:
                        content = [{"type": "image", "image": im} for im in ims]
                        content.append({"type": "text", "text": tail})
                        j = parse_json(ask(content, 200))
                    row.update(j or {"parse_error": 1})
            except Exception as e:
                row["error"] = repr(e)[:120]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (k + 1) % 20 == 0:
                f.flush()
                print(f"  [{k+1}/{len(todo)}] {(time.time()-t0)/(k+1):.1f}s/vid", flush=True)
        f.close()
        print(f"P{v}_DONE", flush=True)
    print("E25_DONE", flush=True)


if __name__ == "__main__":
    main()
