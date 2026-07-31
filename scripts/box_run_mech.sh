#!/bin/bash
# 盒侧一键管线:下载 919 条 -> patch 特征提取 -> 裁剪核验 v2 -> 全局库评估 -> 分款对照。
# 盒侧布局 /root/mech/{scripts/*.py, data/...};python 用 /venv/main/bin/python。
#
# 发车流程(本地):
#   1) python scripts/presign_mech_urls.py > /tmp/mech_urls.curl        # 签 12h URL
#   2) ssh box "mkdir -p /root/mech/scripts /root/mech/data/prod500"
#   3) scp scripts/{extract_patch_feat,patch_bank_eval,patch_bank_eval_style,verify_crops2}.py \
#          box:/root/mech/scripts/
#      scp scripts/box_run_mech.sh box:/root/mech/
#      scp data/prod500/mech_subset.tsv box:/root/mech/data/prod500/
#      scp /tmp/mech_urls.curl box:/root/mech/
#   4) ssh -f box "cd /root/mech && nohup bash box_run_mech.sh > mech_run.log 2>&1 < /dev/null"
#      (ssh -f + </dev/null,否则 ssh 挂住;完成标记见日志尾部 ALL_DONE)
set -e
cd "$(dirname "$0")"
PY=/venv/main/bin/python
mkdir -p data/corpus_videos data/corpus_patch_feat

echo "== 下载 =="
curl --parallel --parallel-max 16 --retry 3 -sS -K mech_urls.curl
echo "videos: $(find data/corpus_videos -name '*.mp4' | wc -l)"

echo "== 依赖 =="
$PY - <<'EOF'
import torch, transformers, cv2, pandas
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
EOF

echo "== 提取(GroundingDINO+DINOv2-base,16帧/条)=="
$PY scripts/extract_patch_feat.py --cache-dir /root/mech/.hf_cache
echo "npz: $(find data/corpus_patch_feat -name '*.npz' | wc -l)"

echo "== 裁剪核验 v2(图像侧离群,蒙太奇人工终审)=="
$PY scripts/verify_crops2.py

echo "== 全局库评估 =="
$PY scripts/patch_bank_eval.py

echo "== 分款对照 =="
$PY scripts/patch_bank_eval_style.py

echo "ALL_DONE"
