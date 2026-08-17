#!/usr/bin/env python3
"""本地版批量打分(箱子已销毁,缓存已回传):
23 组特征缓存 → 15 专家 → meta+boost = p_base;导出 full 专家 320 维原始特征。

替补说明:
- expert_hint 用 HF v1 权重(v3 已丢失);
- asr 缓存缺失(箱上 torchaudio 未装完)→ 用"无音轨"默认向量兜底,
  只影响 hint / per_src_hint 两个专家(TUTU 生成视频本就无音轨,近似忠实)。

输出: data/pbase/out/pbase_1233.csv + full_raw_1233.npz
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UP_CODE = ROOT / "upstream"
CACHE = ROOT / "data/pbase/upstream/data/cache"
ART = ROOT / "inference/artifacts_v3"
SPLIT = ROOT / "data/pbase/upstream/splits/train_v2.jsonl"
OUT = ROOT / "data/pbase/out"
sys.path.insert(0, str(UP_CODE))

from training import feature_loader as FL  # noqa: E402
from training.expert_definitions import EXPERTS  # noqa: E402

# asr 兜底:缓存缺失时按"无音轨"默认值构造同维向量
# load_asr 对空字典 d={} 的输出正是全零向量("none" 不在任何枚举中)
_dim = len(FL.ASR_LANGS) + len(FL.ASR_EVENTS) + len(FL.ASR_EMOTIONS) + 3
_asr_default = np.zeros(_dim, dtype=np.float32)
FL.FEATURE_REGISTRY["asr"] = lambda c, l, s, a: _asr_default


def load_art(name):
    p = ART / f"expert_{name}.pkl"
    if not p.exists() and name == "hint":
        p = ART / "expert_hint_v1.pkl"
    return pickle.load(open(p, "rb"))


def main():
    targets = [json.loads(l) for l in open(SPLIT)]
    arts = {s["name"]: load_art(s["name"]) for s in EXPERTS}
    meta = pickle.load(open(ART / "meta_lr.pkl", "rb"))
    boost = pickle.load(open(ART / "boost_lgbm.pkl", "rb"))

    def predict_expert(art, x, src):
        X = x.reshape(1, -1)
        models = (art["models_per_src"].get(src, art["models"])
                  if art.get("per_source") else art["models"])
        return float(np.mean([m.predict_proba(X)[0, 1] for m in models]))

    out_rows, raws = [], {}
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(targets):
        stem = Path(e["video"]).stem
        row = {"video": e["video"]}
        try:
            src = FL.get_source(e["abs_path"])
            stack = np.zeros(len(EXPERTS))
            for j, spec in enumerate(EXPERTS):
                x = FL.featurize_one(CACHE, e["label"], stem, e["abs_path"], spec["features"])
                if spec["name"] == "full":
                    raws[e["video"]] = x.copy()
                stack[j] = predict_expert(arts[spec["name"]], x, src)
                row[f"p_{spec['name']}"] = round(stack[j], 6)
            X = stack.reshape(1, -1)
            p = float(np.clip(meta.predict_proba(X)[0, 1] + boost.predict(X)[0], 0, 1))
            row["p_base"] = round(p, 6)
        except Exception as ex:
            row["error"] = repr(ex)[:150]
        out_rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(targets)}] {(time.time()-t0)/(i+1):.2f}s/条", flush=True)

    cols = ["video", "p_base"] + [f"p_{s['name']}" for s in EXPERTS] + ["error"]
    with open(OUT / "pbase_1233.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    vids = sorted(raws)
    np.savez_compressed(OUT / "full_raw_1233.npz",
                        videos=np.array(vids), X=np.stack([raws[v] for v in vids]))
    nerr = sum(1 for r in out_rows if "error" in r)
    dim = len(next(iter(raws.values()))) if raws else 0
    print(f"done: {len(out_rows)} rows | errors {nerr} | full_raw dim {dim}")


if __name__ == "__main__":
    main()
