#!/usr/bin/env python3
"""flash 判图:4553 张质检图,每张挂 8 款官方正面图,让模型先识别款式再对照评分。
输出 /workspace/r2/img_judge.jsonl(可断点续跑);--limit N 先跑分层试点。
用法: python img_judge.py [--limit 240] [--workers 8] [--model gemini-3.6-flash]
"""
import argparse
import base64
import csv
import glob
import json
import os
import random
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R2 = Path("/workspace/r2")

# ---- key ----
for line in open(R2 / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
KEY = os.environ["GOOGLE_API_KEY"]

# ---- 官方正面图(8款) ----
VIEW_DIR = R2 / "data/sku_ref_v2/views"
ref_v0 = sorted(glob.glob(str(VIEW_DIR / "*_v0.jpg")))
SKUS = [Path(p).name[:-len("_v0.jpg")] for p in ref_v0]
assert len(SKUS) == 8, f"expect 8 skus, got {SKUS}"
REF_PARTS = []
for p in ref_v0:
    REF_PARTS.append({"inlineData": {"mimeType": "image/jpeg",
                                     "data": base64.b64encode(open(p, "rb").read()).decode()}})

SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string", "enum": SKUS + ["无法识别"]},
        "grade": {"type": "string", "enum": ["good", "normal", "bad", "abstain"]},
        "bad_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "defects": {"type": "array", "items": {"type": "string", "enum": [
            "还原度-五官", "还原度-配色", "还原度-衣物配件", "还原度-体型比例", "还原度-伞盖形态",
            "结构错误", "道具异常", "画质伪影", "文字乱码", "其他"]}},
        "evidence": {"type": "string"},
    },
    "required": ["sku", "grade", "bad_score", "defects"],
}

RUBRIC = """你是「蘑菇TUTU」AI 生成图片的质量审计员。前 8 张图是 8 个款式的官方标准形象(正面),依次为:{names}。最后一张是待审的 AI 生成图片。
步骤:
1) 判断待审图片中的角色是哪个款式(以伞盖形态/配色/标志性配件为准);若角色不属于这 8 款(其他系列/其他角色),sku 填「无法识别」,并跳过第 2 步的官方形象比对,只按第 3 步通用标准判档——绝不因"不在官方图中"或"与某款式不像"扣分;
2) 对照该款式官方形象逐项核对:五官结构、伞盖形态、配色、衣物配件、体型比例——只判"与官方形象不一致",姿态/表情/视角不同不算缺陷;
3) 检查结构错误(肢体缺失/多余/融合/断裂)、道具异常(悬浮、穿模、残缺、凭空多物)、画质(伪影噪点、局部崩坏)、文字乱码。
判档(严格执行):
- 任一缺陷一眼可见、足以令普通观众出戏 → bad;
- 需仔细对照才能发现的轻微瑕疵 → normal;
- 无可确认瑕疵 → good;
- 图片内容无法判断(空图/严重遮挡)→ abstain。
bad_score 为 0-100 连续分,越高越差,用于排序,请拉开区分度。evidence 用一句话写最关键的可见缺陷。只输出 JSON。"""


def call_gemini_img(img_path, model):
    parts = list(REF_PARTS)
    parts.append({"inlineData": {"mimeType": "image/png",
                                 "data": base64.b64encode(open(img_path, "rb").read()).decode()}})
    parts.append({"text": RUBRIC.format(names="、".join(SKUS))})
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
    txt = resp["candidates"][0]["content"]["parts"][-1]["text"]
    return json.loads(txt), resp.get("usageMetadata", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--out", default=str(R2 / "img_judge.jsonl"))
    args = ap.parse_args()

    rows = []
    for f in ["tutu_image_annotations_2962.csv", "tutu_image_annotations_0813.csv"]:
        for r in csv.DictReader(open(R2 / "data" / f, encoding="utf-8-sig")):
            fp = R2 / "qcimgs" / f"{r['dataset']}__SLASH__{r['sample_id']}.png"
            if fp.exists() and fp.stat().st_size > 0:
                rows.append((str(fp), r["label"]))
    if args.limit:
        rnd = random.Random(42)
        by_lab = {}
        for fp, l in rows:
            by_lab.setdefault(l, []).append((fp, l))
        picked = []
        for l, grp in sorted(by_lab.items()):
            k = max(1, round(args.limit * len(grp) / len(rows)))
            picked += rnd.sample(grp, min(k, len(grp)))
        rows = picked
    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["filename"])
            except Exception:
                pass
    todo = [(fp, l) for fp, l in rows if Path(fp).name not in done]
    print(f"total {len(rows)}, done {len(done)}, todo {len(todo)}", flush=True)

    lock = threading.Lock()
    n_ok = [0]
    t0 = time.time()

    def work(item):
        fp, lab = item
        for attempt in range(4):
            try:
                res, usage = call_gemini_img(fp, args.model)
                rec = {"filename": Path(fp).name, "label": lab, **res,
                       "tokens": usage.get("totalTokenCount", 0)}
                with lock:
                    with open(args.out, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_ok[0] += 1
                    if n_ok[0] % 25 == 0:
                        print(f"[{n_ok[0]}/{len(todo)}] {(time.time()-t0)/n_ok[0]:.1f}s/条", flush=True)
                return
            except urllib.error.HTTPError as e:
                wait = 30 if e.code == 429 else 10
                time.sleep(wait * (attempt + 1))
            except Exception:
                time.sleep(10 * (attempt + 1))
        with lock:
            print("FAIL", Path(fp).name, flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    print("IMG_JUDGE_DONE", flush=True)


if __name__ == "__main__":
    main()
