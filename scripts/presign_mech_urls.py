#!/usr/bin/env python
"""为机制验证子集批量签发 12h 预签名 URL,输出 curl 配置文件。

铁律:AWS 凭证绝不上外部盒子,盒侧只拿预签名 URL 下载。
桶 trash-in-picaa 在 us-east-2,必须显式指定 region + s3v4,否则 400;
预签名 URL 只对 GET 有效,验证连通性用 GET(range),HEAD 会 403。

用法(本地):
  python scripts/presign_mech_urls.py > /tmp/mech_urls.curl
  # 抽检: curl -sS -r 0-0 -o /dev/null -w "%{http_code}\n" "<某条url>"  期望 206
盒侧下载:
  curl --parallel --parallel-max 16 --retry 3 -sS -K mech_urls.curl
"""
from __future__ import annotations

import sys
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "trash-in-picaa"
PREFIX = "Datasets/tutu-video-eval/v-0430-skus"
EXPIRES = 12 * 3600


def main():
    man = ROOT / "data/prod500/mech_subset.tsv"
    s3 = boto3.client("s3", region_name="us-east-2",
                      config=Config(signature_version="s3v4"))
    print("create-dirs")
    n = 0
    for line in man.read_text().splitlines():
        if not line.strip():
            continue
        rel = line.split("\t")[0]
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/{rel}"},
            ExpiresIn=EXPIRES)
        print(f'url = "{url}"')
        print(f'output = "data/corpus_videos/{rel}"')
        n += 1
    print(f"signed {n} urls (12h)", file=sys.stderr)


if __name__ == "__main__":
    main()
