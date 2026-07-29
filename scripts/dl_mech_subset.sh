#!/bin/bash
# 下载机制验证子集(skus 还原度 bad 464 + skus good 455)到 data/corpus_videos/
# 来源 s3://trash-in-picaa/Datasets/tutu-video-eval/<款>/<file>(默认 profile)
set -e
ROOT="${TUTU_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC=s3://trash-in-picaa/Datasets/tutu-video-eval
DST=$ROOT/data/corpus_videos
MAN=$ROOT/data/prod500/mech_subset.tsv
mkdir -p "$DST"
n=0
while IFS=$'\t' read -r rel label; do
  out="$DST/$rel"
  [ -f "$out" ] && continue
  mkdir -p "$(dirname "$out")"
  echo "$SRC/$rel|$out"
  n=$((n+1))
done < "$MAN" > /tmp/mech_dl_list.txt
echo "todo $(wc -l < /tmp/mech_dl_list.txt) files"
cat /tmp/mech_dl_list.txt | xargs -P 4 -I{} bash -c '
  IFS="|" read -r src out <<< "{}"
  aws s3 cp "$src" "$out" --only-show-errors || echo "FAIL $src"
'
echo "done. $(find $DST -name '*.mp4' | wc -l) videos present"
