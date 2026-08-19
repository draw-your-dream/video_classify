#!/usr/bin/env python3
"""flash 判图 v2:意图条件化——从 sample_id 解析款式+配件,挂该款式 3 视角官方图,
只判"是否符合本图意图"(配件是有意添加,存在不扣分,画错才扣)。
目标集 = 938 张视频源图(进融合层) + 0813 抽样(对人工标签算 AUC)。
输出 /workspace/r2/img_judge2.jsonl,marker=IMG_JUDGE2_DONE。"""
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
VIEW_DIR = R2 / "data/sku_ref_v2/views"

for line in open(R2 / ".env"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
KEY = os.environ["GOOGLE_API_KEY"]

SKUS = sorted({Path(p).name.rsplit("_v", 1)[0] for p in glob.glob(str(VIEW_DIR / "*.jpg"))})

SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["good", "normal", "bad", "abstain"]},
        "bad_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "defects": {"type": "array", "items": {"type": "string", "enum": [
            "还原度-五官", "还原度-配色", "还原度-衣物配件", "还原度-体型比例", "还原度-伞盖形态",
            "配件画错", "结构错误", "道具异常", "画质伪影", "文字乱码", "其他"]}},
        "evidence": {"type": "string"},
    },
    "required": ["grade", "bad_score", "defects"],
}

RUBRIC = """你是「蘑菇TUTU」AI 生成图片的质量审计员。前 {n} 张图是款式「{sku}」的官方标准形象(多视角,姿态可不同)。最后一张是待审的 AI 生成图片。
本图的生成意图:款式「{sku}」{prop_line}
审查规则:
1) 还原度:对照官方形象核对 五官结构/伞盖形态/配色/体型比例/自带衣物配件——只判"与官方形象不一致";姿态、表情、视角、场景与官方图不同一律不算缺陷;
2) 意图配件:{prop_rule}
3) 结构错误(肢体缺失/多余/融合/断裂)、道具异常(悬浮、穿模、残缺)、画质(伪影噪点、局部崩坏)、文字乱码。
判档(严格执行):
- 任一缺陷一眼可见、足以令普通观众出戏 → bad;
- 需仔细对照才能发现的轻微瑕疵 → normal;
- 无可确认瑕疵 → good。
bad_score 为 0-100 连续分,越高越差,请拉开区分度。evidence 一句话写最关键的可见缺陷。只输出 JSON。"""


def sku_views(sku):
    vs = sorted(glob.glob(str(VIEW_DIR / f"{sku}_v*.jpg")))
    return [v for i, v in enumerate(vs) if i in (0, 1, len(vs) - 1)]


def parse_intent(sample_id):
    p = sample_id.split("__")
    sku = p[1] if len(p) >= 2 else ""
    prop = p[3] if len(p) == 6 else ""
    return sku, prop


def call_one(img_path, sku, prop, model):
    views = sku_views(sku)
    parts = []
    for v in views:
        parts.append({"inlineData": {"mimeType": "image/jpeg",
                                     "data": base64.b64encode(open(v, "rb").read()).decode()}})
    parts.append({"inlineData": {"mimeType": "image/png",
                                 "data": base64.b64encode(open(img_path, "rb").read()).decode()}})
    prop_line = f",并有意为角色添加了配件/道具「{prop}」。" if prop else ",无额外配件。"
    prop_rule = (f"「{prop}」是有意添加的,它的存在本身绝不扣分;只有当它画错(形状崩坏/与角色融合/残缺/穿模)时才记「配件画错」。"
                 if prop else "本图不应有额外配件;若出现凭空多出的物体记「道具异常」。")
    parts.append({"text": RUBRIC.format(n=len(views), sku=sku, prop_line=prop_line, prop_rule=prop_rule)})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval813", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--out", default=str(R2 / "img_judge2.jsonl"))
    args = ap.parse_args()

    # 图片标签表(有则带上,便于对标)
    lab = {}
    for f in ["tutu_image_annotations_2962.csv", "tutu_image_annotations_0813.csv"]:
        for r in csv.DictReader(open(R2 / "data" / f, encoding="utf-8-sig")):
            lab[f"{r['dataset']}__SLASH__{r['sample_id']}"] = r["label"]

    targets = {}
    # A. 视频源图(融合层用)
    for r in csv.DictReader(open(R2 / "data/api_judge_video_image_map.csv", encoding="utf-8-sig")):
        s = r["image_sample_id"]
        if not s:
            continue
        sku, prop = parse_intent(s)
        if sku not in SKUS:
            continue
        key = f"{r['image_dataset']}__SLASH__{s}"
        fp = R2 / "qcimgs" / f"{key}.png"
        if fp.exists() and fp.stat().st_size > 0:
            targets[key] = (str(fp), sku, prop, lab.get(key, ""), "src")
    n_src = len(targets)
    # B. 0813 标注图抽样(AUC 验证)
    pool = []
    for r in csv.DictReader(open(R2 / "data/tutu_image_annotations_0813.csv", encoding="utf-8-sig")):
        sku, prop = parse_intent(r["sample_id"])
        key = f"{r['dataset']}__SLASH__{r['sample_id']}"
        fp = R2 / "qcimgs" / f"{key}.png"
        if sku in SKUS and key not in targets and fp.exists() and fp.stat().st_size > 0:
            pool.append((key, (str(fp), sku, prop, r["label"], "eval")))
    rnd = random.Random(42)
    for key, v in rnd.sample(pool, min(args.eval813, len(pool))):
        targets[key] = v
    print(f"targets: {len(targets)} (src {n_src} + eval {len(targets)-n_src})", flush=True)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done.add(json.loads(l)["key"])
            except Exception:
                pass
    todo = [(k, *v) for k, v in targets.items() if k not in done]
    print(f"todo {len(todo)}", flush=True)

    lock = threading.Lock()
    n_ok = [0]
    t0 = time.time()

    def work(item):
        key, fp, sku, prop, label, role = item
        for attempt in range(4):
            try:
                res, usage = call_one(fp, sku, prop, args.model)
                rec = {"key": key, "sku": sku, "prop": prop, "label": label, "role": role, **res,
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
            print("FAIL", key, flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    print("IMG_JUDGE2_DONE", flush=True)


if __name__ == "__main__":
    main()
