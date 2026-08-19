#!/usr/bin/env python3
"""flash 帧级判定:视频末帧 vs 源图+官方三视角——判还原度退化(不看运动)。
输入顺序:官方视图×3 → 源图(该视频的首帧参考) → 视频末帧。
输出 /workspace/r2/vidframe_judge.jsonl,marker=VIDFRAME_DONE。"""
import argparse
import base64
import csv
import glob
import io
import json
import os
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from PIL import Image

R2 = Path("/workspace/r2")
VIEW_DIR = R2 / "data/sku_ref_v2/views"

for line in open(R2 / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
KEY = os.environ["GOOGLE_API_KEY"]

SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["good", "normal", "bad", "abstain"]},
        "bad_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "defects": {"type": "array", "items": {"type": "string", "enum": [
            "五官退化", "配色漂移", "衣物配件丢失或变形", "体型比例失真", "伞盖形态改变",
            "结构错误", "道具异常", "画质退化", "无退化"]}},
        "evidence": {"type": "string"},
    },
    "required": ["grade", "bad_score", "defects"],
}

RUBRIC = """你是「蘑菇TUTU」AI 生成视频的还原度审计员。图片顺序:前 {n} 张为款式「{sku}」官方标准形象(姿态可不同);第 {m} 张为本视频的源图(首帧参考,画面应与它一致地开始);最后一张为该视频的最后一帧。
任务:只判「最后一帧相对源图的角色退化」——AI 生成视频常在播放过程中让角色走形。
逐项对照源图检查最后一帧中的角色:五官结构、配色、衣物配件(源图中有的配件是否还在、形状是否保持)、体型比例、伞盖形态;并对照官方形象确认退化方向。
注意:姿态/表情/位置/朝向的变化是正常表演,绝不算退化;只有"角色长相/配件/比例变了"才算。
判档:退化一眼可见令观众出戏 → bad;仔细对照才能察觉 → normal;无退化 → good。
bad_score 0-100(退化严重度,拉开区分度)。evidence 一句话。只输出 JSON。"""


def sku_views(sku):
    vs = sorted(glob.glob(str(VIEW_DIR / f"{sku}_v*.jpg")))
    return [v for i, v in enumerate(vs) if i in (0, 1, len(vs) - 1)]


def b64_jpg(pil, q=90):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def call_one(views, src_pil, last_pil, sku, model):
    parts = []
    for v in views:
        parts.append({"inlineData": {"mimeType": "image/jpeg",
                                     "data": base64.b64encode(open(v, "rb").read()).decode()}})
    parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64_jpg(src_pil)}})
    parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64_jpg(last_pil)}})
    parts.append({"text": RUBRIC.format(n=len(views), sku=sku, m=len(views) + 1)})
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": SCHEMA,
            "thinkingConfig": {"thinkingLevel": "high"},
            "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
        },
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.load(r)
    return json.loads(resp["candidates"][0]["content"]["parts"][-1]["text"]), resp.get("usageMetadata", {})


def frames_of(vp):
    cap = cv2.VideoCapture(str(vp))
    first = last = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if first is None:
            first = im
        last = im
    cap.release()
    if first is None:
        return None, None
    def pil(im):
        p = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        p.thumbnail((640, 640))
        return p
    return pil(first), pil(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--out", default=str(R2 / "vidframe_judge.jsonl"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(R2 / "data/api_judge_video_image_map.csv", encoding="utf-8-sig")))
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["filename"])
            except Exception:
                pass
    todo = [r for r in rows if r["filename"] not in done and (R2 / "videos" / r["filename"]).exists()]
    print(f"todo {len(todo)}", flush=True)

    lock = threading.Lock()
    n_ok = [0]
    t0 = time.time()

    def work(r):
        fn = r["filename"]
        sku = fn.split("__")[2]
        views = sku_views(sku)
        if not views:
            return
        first, last = frames_of(R2 / "videos" / fn)
        if first is None:
            return
        src = first
        sfp = R2 / "qcimgs" / f"{r['image_dataset']}__SLASH__{r['image_sample_id']}.png"
        has_src = 0
        if r["image_sample_id"] and sfp.exists() and sfp.stat().st_size > 0:
            try:
                src = Image.open(sfp).convert("RGB")
                src.thumbnail((640, 640))
                has_src = 1
            except Exception:
                src = first
        for attempt in range(4):
            try:
                res, usage = call_one(views, src, last, sku, args.model)
                rec = {"filename": fn, "grade_h": r["grade"], "has_src": has_src, **res,
                       "tokens": usage.get("totalTokenCount", 0)}
                with lock:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok[0] += 1
                    if n_ok[0] % 50 == 0:
                        print(f"[{n_ok[0]}/{len(todo)}] {(time.time()-t0)/n_ok[0]:.1f}s/条", flush=True)
                return
            except urllib.error.HTTPError as e:
                time.sleep((30 if e.code == 429 else 10) * (attempt + 1))
            except Exception:
                time.sleep(10 * (attempt + 1))
        with lock:
            print("FAIL", fn, flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    print("VIDFRAME_DONE", flush=True)


if __name__ == "__main__":
    main()
