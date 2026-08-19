#!/usr/bin/env python3
"""API 判别试点 runner(B1 单轮直判基线)。

用法:
  export $(cat .env.api | xargs)   # 或上箱后手动 export 两把 key
  python run_pilot.py --backend gemini --videos <dir> --manifest <csv> --out out.jsonl
  python run_pilot.py --backend openai --videos <dir> --manifest <csv> --out out.jsonl

manifest csv 需含 filename 列(其余列原样带过,便于事后对齐 grade);
标签列不会进入任何 prompt。输出 jsonl 逐条追加,可断点续跑。

Gemini:原生视频输入(inline base64,<20MB;5s 短视频足够),videoMetadata.fps=5,
structured output(responseSchema)。模型默认 gemini-3.6-flash(codex 方案推荐的固定稳定版)。
OpenAI:无原生视频,抽 16 帧(约 3fps)按时间顺序作为图片序列输入,同一 JSON schema。
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import urllib.request

THINK = "high"
MEDIA_RES = "MEDIA_RESOLUTION_HIGH"

SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["good", "normal", "bad", "abstain"]},
        "bad_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason_labels": {"type": "array", "items": {"type": "string", "enum": [
            "还原度", "衣服/身体的时间一致性", "大小变化",
            "僵硬", "卡顿/少活人感", "四肢不动", "动作位移不连贯", "运动主体", "静止不动", "慢动作",
            "物理规律", "不合理的物体",
            "帧跳变", "首帧一致", "背景运动混乱"]}},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "start_sec": {"type": "number"}, "end_sec": {"type": "number"},
            "observation": {"type": "string"}},
            "required": ["start_sec", "end_sec", "observation"]}},
        "other_issue": {"type": "string"},
        "normal_play_visible": {"type": "boolean"},
    },
    "required": ["grade", "bad_score", "reason_labels", "evidence", "normal_play_visible"],
}

RUBRIC = """你是「蘑菇TUTU」AI 生成短视频的质量证据审计员。TUTU 是一个蘑菇角色 IP,有多种款式(经典毛绒款、爆炸菇、汉堡菇、炸虾菇、粽子菇等)。
【重要】第一张图片是该视频的首帧参考图,它定义了本条视频中角色与场景的标准形态。判断"画得对不对"一律以这张参考图为准绳,不要用你对"蘑菇角色应该长什么样"的先验知识。
【正常表情说明】角色眯眼或闭眼时,豆状眼睛本身会变成一条弧线/横线,这是正常表情,不算五官变化缺陷。
请对照参考图完整观看视频后,依次检查五个维度:
1. IP还原:款式/五官/比例/衣物/体型是否全程与首帧参考图一致(五官结构改变、衣物配件融合消失、无理由缩放均为缺陷);
2. 动作:姿态与位移是否连续,身体是否柔软有生命感(僵硬、卡顿、四肢锁死、位移跳跃、过长静止为缺陷);
3. 交互:接触/遮挡/抓握/支撑是否正确(穿模、嵌入、接触错位为缺陷);
4. 物理与物体:是否符合重力与支撑,物体是否连续存在(漂浮、凭空出现消失为缺陷);
5. 画面稳定:开场是否与首帧一致,相邻帧/背景/文字是否稳定(帧跳变、背景扭曲为缺陷)。
判档规则(严格执行):
- 任一缺陷在正常播放速度下明显可见、足以令普通观众出戏 → bad;
- 只有暂停或仔细对照才能发现的轻微瑕疵,或整体表现平淡缺乏动态 → normal;
- 全维度无可确认瑕疵且有流畅自然的动态 → good;
- 输入不足或证据冲突无法判断 → abstain,不要强猜。
每个缺陷必须给出时间区间和具体可见现象,不接受"感觉不自然"这类空泛描述;reason_labels 只能从给定枚举中选。动作幅度小本身不扣分。只输出 JSON。"""


def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["filename"])
            except Exception:
                pass
    return done


def sku_context(filename):
    parts = filename[:-4].split("__")
    sku = parts[2] if len(parts) >= 3 else ""
    if not sku:
        return ""
    prop = parts[4] if len(parts) >= 6 else ""
    c = f"背景信息:角色款式「{sku}」"
    if prop:
        c += f",场景道具「{prop}」"
    return c + "(仅作任务背景,不构成质量判断依据)。\n"


def sku_ref_paths(filename, sku_dir, n=2):
    if not sku_dir:
        return []
    sku = filename[:-4].split("__")[2]
    import glob as _g
    # 新版官方立绘(sku_ref_v2/views):<款式名>_v0..v4.jpg,v0=正面 v1=微侧 v4=正背
    views = sorted(_g.glob(os.path.join(sku_dir, f"{sku}_v*.jpg")))
    if views:
        pick = [v for i, v in enumerate(views) if i in (0, 1, len(views) - 1)]
        return pick[:max(n, 3)]
    cands = sorted(_g.glob(os.path.join(sku_dir, f"{sku}*_[12].png")))
    if not cands:  # 名称变体(全角括号/无道具后缀)
        cands = sorted(_g.glob(os.path.join(sku_dir, f"{sku.split('（')[0].split('(')[0]}*_[12].png")))
    return cands[:n]


def call_gemini(video_path, model, key, fps=5, ref_path=None, sku_dir=None):
    data = base64.b64encode(open(video_path, "rb").read()).decode()
    parts = []
    for sp in sku_ref_paths(os.path.basename(str(video_path)), sku_dir):
        mt = "image/jpeg" if sp.lower().endswith((".jpg", ".jpeg")) else "image/png"
        parts.append({"inlineData": {"mimeType": mt,
                                     "data": base64.b64encode(open(sp, "rb").read()).decode()}})
    if ref_path:
        parts.append({"inlineData": {"mimeType": "image/png",
                                     "data": base64.b64encode(open(ref_path, "rb").read()).decode()}})
    parts.append({"inlineData": {"mimeType": "video/mp4", "data": data},
                  "videoMetadata": {"fps": fps}})
    n_sku = len(sku_ref_paths(os.path.basename(str(video_path)), sku_dir))
    pre = (f"前 {n_sku} 张图是该款式的官方标准形象(多视角,姿态可不同,只用于核对长相/比例/伞盖/配色/材质);"
           f"第 {n_sku+1} 张图是本视频的首帧参考图。\n") if n_sku else ""
    parts.append({"text": pre + sku_context(os.path.basename(str(video_path))) + RUBRIC})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
            "thinkingConfig": {"thinkingLevel": THINK},
            "mediaResolution": MEDIA_RES,
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    txt = resp["candidates"][0]["content"]["parts"][-1]["text"]
    return json.loads(txt), resp.get("modelVersion", ""), resp.get("usageMetadata", {})


def extract_frames(video_path, n=16):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video_path),
                        "-vf", f"select='not(mod(n\\,max(1\\,floor(n_frames/{n}))))',scale=448:-2",
                        "-frames:v", str(n), "-vsync", "vfr", f"{td}/f%02d.jpg"],
                       check=False, capture_output=True)
        fs = sorted(Path(td).glob("f*.jpg"))
        if not fs:  # 兜底:按时间均匀采样
            subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video_path),
                            "-vf", "fps=3,scale=448:-2", "-frames:v", str(n),
                            f"{td}/g%02d.jpg"], check=False, capture_output=True)
            fs = sorted(Path(td).glob("g*.jpg"))
        return [base64.b64encode(open(f, "rb").read()).decode() for f in fs]


def call_openai(video_path, model, key, ref_path=None, sku_dir=None):
    frames = extract_frames(video_path)
    if not frames:
        raise RuntimeError("no_frames")
    sku_refs = sku_ref_paths(os.path.basename(str(video_path)), sku_dir)
    intro = ""
    if sku_refs:
        intro += f"以下前 {len(sku_refs)} 张图是该款式的官方标准形象(多视角,只用于核对长相/比例/配色);"
    if ref_path:
        intro += ("紧接着一张" if sku_refs else "以下第 1 张图") + "是首帧参考图;"
    intro += f"其后 {len(frames)} 张图是同一条约5秒视频按时间顺序的抽帧。\n"
    content = [{"type": "text",
                "text": intro + sku_context(os.path.basename(str(video_path))) + RUBRIC}]
    for sp in sku_refs:
        mt = "image/jpeg" if sp.lower().endswith((".jpg", ".jpeg")) else "image/png"
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mt};base64," + base64.b64encode(open(sp, "rb").read()).decode(),
            "detail": "high"}})
    if ref_path:
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(open(ref_path, "rb").read()).decode(),
            "detail": "high"}})
    for f in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"}})
    body = {"model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "tutu_judge", "schema": SCHEMA, "strict": False}}}
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    txt = resp["choices"][0]["message"]["content"]
    return json.loads(txt), resp.get("model", ""), resp.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemini", "openai"], required=True)
    ap.add_argument("--videos", required=True, help="mp4 目录")
    ap.add_argument("--manifest", required=True, help="含 filename 列的 csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refs", default=None, help="参考图目录(<视频名去.mp4>.png);缺图时用视频第0帧兜底")
    ap.add_argument("--rubric", default=None, help="rubric 文本文件路径(默认用内置 v2)")
    ap.add_argument("--media-res", default="MEDIA_RESOLUTION_HIGH")
    ap.add_argument("--sku-refs", default=None, help="官方款式参考图目录(<款式名>_N.png)")
    ap.add_argument("--think", default="high")
    args = ap.parse_args()
    if args.rubric:
        global RUBRIC
        RUBRIC = open(args.rubric, encoding="utf-8").read()
    global MEDIA_RES, THINK
    MEDIA_RES, THINK = args.media_res, args.think

    model = args.model or ("gemini-3.6-flash" if args.backend == "gemini" else "gpt-5.2")
    key = os.environ["GOOGLE_API_KEY" if args.backend == "gemini" else "OPENAI_API_KEY"]
    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8-sig")))
    if args.limit:
        rows = rows[:args.limit]
    done = load_done(args.out)
    vdir = Path(args.videos)
    f = open(args.out, "a")
    t0, n = time.time(), 0
    for r in rows:
        fn = r["filename"]
        if fn in done:
            continue
        vp = vdir / fn
        row = {"filename": fn, "backend": args.backend, "model": model, "fps": args.fps}
        ref = None
        if args.refs:
            rp = Path(args.refs) / (fn[:-4] + ".png")
            kind = "exact"
            if not rp.exists():  # 兜底:视频第0帧(i2v 首帧≈源图)
                kind = "frame0"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(vp),
                                "-frames:v", "1", str(rp)], check=False, capture_output=True)
            if rp.exists():
                ref = rp
                row["ref"] = kind
        if not vp.exists():
            row["error"] = "missing_video"
        else:
            for attempt in range(3):
                try:
                    j, mv, usage = (call_gemini(vp, model, key, args.fps, ref, args.sku_refs)
                                    if args.backend == "gemini" else call_openai(vp, model, key, ref, args.sku_refs))
                    row.update(result=j, model_version=mv, usage=usage)
                    break
                except Exception as e:
                    if attempt == 2:
                        row["error"] = repr(e)[:200]
                    else:
                        time.sleep(5 * (attempt + 1))
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        n += 1
        if n % 10 == 0:
            print(f"[{n}] {(time.time()-t0)/n:.1f}s/条", flush=True)
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
