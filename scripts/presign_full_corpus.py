#!/usr/bin/env python
"""全语料 4473 上机:比对盒上已有,签发缺口视频的 12h 预签名 URL(curl 配置)。

铁律:AWS 凭证绝不上外部盒子;桶 trash-in-picaa 在 us-east-2,显式 region+s3v4。
布局:skus → corpus_videos/<款式>/<id>.mp4;ti2i2v → corpus_videos/ti2i2v/<id>.mp4
产物:
  scratch/full_urls.curl      盒侧 curl -K 批量下载配置(url+output 对)
  data/s3/corpus_full.tsv     全 4473 manifest(rel\tlabel)
  data/s3/manifest_new.tsv    本次新下载的 3554 manifest(提取增量用)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "trash-in-picaa"
DPREF = "Datasets/tutu-video-eval"
EXPIRES = 12 * 3600


def main():
    out_curl = Path(sys.argv[1])
    labels = {r["path"]: r["label"] for r in csv.DictReader(
        open(ROOT / "data/s3/merged_labels.csv", encoding="utf-8-sig"))}

    key_by_base = {}
    scratch = out_curl.parent
    for f in ("skus_ls.txt", "ti2i2v_ls.txt"):
        for line in open(scratch / f, encoding="utf-8"):
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            key = parts[3].strip()
            base = key.rsplit("/", 1)[-1]
            if base.endswith(".mp4"):
                key_by_base[base] = key

    onbox = set()
    for l in open(ROOT / "data/prod500/mech_subset.tsv", encoding="utf-8"):
        if l.strip():
            onbox.add(l.split("\t")[0].rsplit("/", 1)[-1])

    def rel_of(key: str) -> str:
        sub = key[len(DPREF) + 1:]          # v-0430-skus/款式/id.mp4 | v-0430-ti2i2v/id.mp4
        parts = sub.split("/")
        return "/".join(parts[1:]) if parts[0] == "v-0430-skus" else "ti2i2v/" + parts[-1]

    full, new = [], []
    for base, lab in sorted(labels.items()):
        key = key_by_base.get(base)
        assert key, f"label {base} not found in S3 listings"
        rel = rel_of(key)
        full.append((rel, lab, key))
        if base not in onbox:
            new.append((rel, lab, key))
    print(f"full manifest: {len(full)}  new downloads: {len(new)}")

    (ROOT / "data/s3/corpus_full.tsv").write_text(
        "".join(f"{r}\t{l}\n" for r, l, _ in full), encoding="utf-8")
    (ROOT / "data/s3/manifest_new.tsv").write_text(
        "".join(f"{r}\t{l}\n" for r, l, _ in new), encoding="utf-8")

    s3 = boto3.client("s3", region_name="us-east-2",
                      config=Config(signature_version="s3v4"))
    with open(out_curl, "w", encoding="utf-8") as f:
        for rel, _, key in new:
            url = s3.generate_presigned_url(
                "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=EXPIRES)
            f.write(f'url = "{url}"\noutput = "data/corpus_videos/{rel}"\n')
    print("wrote", out_curl)


if __name__ == "__main__":
    main()
