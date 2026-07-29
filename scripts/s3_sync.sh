#!/bin/bash
# 同步 ~/tutu-video-eval 产出到 s3://sowii-reward-model/tutu/video_reward/
# 用法: s3_sync.sh docs|data|corpus|scripts|results <dir>|models <dir>|videos|hfcache|status
set -e
ROOT=/home/lenovo/tutu-video-eval
DST=s3://sowii-reward-model/tutu/video_reward
P="--profile reward-model-s3"

case "${1:-status}" in
  docs)
    aws s3 cp $ROOT/S3_LAYOUT.md $DST/README.md $P
    for f in data/prod500/FAILURE_CATALOG.md data/prod500/FACTOR_PREREG.md RETRAIN.md ITER_LOG.md; do
      [ -f $ROOT/$f ] && aws s3 cp $ROOT/$f $DST/docs/$(basename $f) $P
    done ;;
  data)
    aws s3 sync $ROOT/data/prod500/ $DST/data/prod500/ $P \
      --exclude "videos/*" --exclude "sheets/*" --exclude "*.tar" \
      --exclude "FAILURE_CATALOG.md" --exclude "FACTOR_PREREG.md" ;;
  corpus)
    aws s3 cp $ROOT/data/s3/merged_labels.csv $DST/data/corpus/merged_labels.csv $P
    aws s3 sync $ROOT/splits/ $DST/data/corpus/splits/ $P ;;
  scripts)
    aws s3 sync $ROOT/scripts/ $DST/scripts/ $P --exclude "*.pyc" --exclude "__pycache__/*" ;;
  results)
    [ -z "$2" ] && { echo "usage: s3_sync.sh results <local_dir> (目录名建议 YYYYMMDD_<topic>)"; exit 1; }
    aws s3 sync "$2" $DST/results/$(basename "$2")/ $P ;;
  models)
    [ -z "$2" ] && { echo "usage: s3_sync.sh models <local_dir>"; exit 1; }
    aws s3 sync "$2" $DST/models/$(basename "$2")/ $P ;;
  videos)
    T=$(mktemp -d)/videos.tar
    tar -cf $T -C $ROOT/data/prod500 videos
    aws s3 cp $T $DST/data/prod500/videos.tar $P
    rm -f $T ;;
  hfcache)
    aws s3 sync $ROOT/.hf_cache/ $DST/hf_cache/ $P --exclude "*.incomplete" --exclude "*.lock" ;;
  status)
    aws s3 ls $DST/ --recursive $P | tail -40 ;;
  *) echo "unknown: $1"; exit 1 ;;
esac
echo "sync [$1] done -> $DST"
