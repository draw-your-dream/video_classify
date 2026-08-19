#!/usr/bin/env python3
"""批量打分:23 组特征缓存 → 15 专家(artifacts_v3) → meta+boost = p_base;
同时导出 full 专家原始特征向量(E18 的 X_raw)。

在箱上跑: python3 batch_score_pbase.py
输出: /workspace/pbase/out/pbase_1233.csv + full_raw_1233.npz
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/pbase")
UP = ROOT / "upstream"
sys.path.insert(0, str(ROOT))

from training.expert_definitions import EXPERTS  # noqa: E402
from training.feature_loader import featurize_one, get_source  # noqa: E402

CACHE = UP / "data/cache"
ART = ROOT / "inference/artifacts_v3"


def main():
    targets = [json.loads(l) for l in open(UP / "splits/train_v2.jsonl")]
    arts = {s["name"]: pickle.load(open(ART / f"expert_{s['name']}.pkl", "rb")) for s in EXPERTS}
    meta = pickle.load(open(ART / "meta_lr.pkl", "rb"))
    boost = pickle.load(open(ART / "boost_lgbm.pkl", "rb"))

    def predict_expert(art, x, src):
        X = x.reshape(1, -1)
        models = (art["models_per_src"].get(src, art["models"])
                  if art.get("per_source") else art["models"])
        return float(np.mean([m.predict_proba(X)[0, 1] for m in models]))

    out_rows = []
    raws = {}
    t0 = time.time()
    (ROOT / "out").mkdir(exist_ok=True)
    for i, e in enumerate(targets):
        stem = Path(e["video"]).stem
        row = {"video": e["video"]}
        try:
            src = get_source(e["abs_path"])
            stack = np.zeros(len(EXPERTS))
            for j, spec in enumerate(EXPERTS):
                x = featurize_one(CACHE, e["label"], stem, e["abs_path"], spec["features"])
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
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(targets)}] {(time.time()-t0)/(i+1):.2f}s/条", flush=True)

    cols = ["video", "p_base"] + [f"p_{s['name']}" for s in EXPERTS] + ["error"]
    with open(ROOT / "out/pbase_1233.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    vids = sorted(raws)
    np.savez_compressed(ROOT / "out/full_raw_1233.npz",
                        videos=np.array(vids), X=np.stack([raws[v] for v in vids]))
    nerr = sum(1 for r in out_rows if "error" in r)
    dim = len(next(iter(raws.values()))) if raws else 0
    print(f"done: {len(out_rows)} rows | errors {nerr} | full_raw dim {dim}")
    print("BATCH_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
