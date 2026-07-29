# video_classify — 蘑菇TUTU 视频 badcase 判别

AI 生成毛毡蘑菇 IP 短视频(~5s)的质检漏斗:**bad 近零漏检前提下,尽量多地自动放行 good**。
唯一指标:**prod500 上 27/27 bad 全召回时的 good 自动放行率**。线上部署基线 = 0%。

## 当前成果(2026-07-29)

| 方案 | 全召回下 good 放行 |
|---|---|
| 线上现状 | 0% |
| 语料训练模型直接迁移(最优) | ≤5% |
| dim_fidelity 单轴(保守底线) | 13.3% |
| **dim_fidelity ⊕ c_first_last 双轴 OR 门** | **26.4%**(t=0.468,LOO 漏 1/27=8fdc8f08,bootstrap CI 下界 26.4%) |

- `dim_fidelity`:线上 VLM rubric 的还原度劣化分(越高越差;感知层可用,其决策层 p_bad 池内 AUC 0.526 不可用)。
- `c_first_last`:GroundingDINO 裁角色 → 首末帧 SigLIP2 嵌入余弦(自参照设计,跨域不携带绝对尺度)。
- 门构造:两轴各自域内分位,OR 取最大;阈值压在最难 bad 的分位上(全召回由构造保证,泛化看 LOO)。

## 复现 26.4%(不需要 S3、不需要 GPU)

```bash
git clone https://github.com/pkucaoyuan/video_classify && cd video_classify
pip install pandas numpy                  # 仅有的依赖
python scripts/eval_or_gate.py            # 基线 26.4%(t=0.468, LOO 漏 1/27)
python scripts/eval_or_gate.py --table    # 逐 bad 分位表(看门的盲区)
```

复现所需的两张数据表已在仓库内:`data/prod500/prod500.csv`(GT 与线上评分,已脱敏)
和 `data/prod500/prod_crop.csv`(裁剪特征含 c_first_last)。已验证全新 clone 直接出 26.4%。

## 大文件走 S3(视频/模型/语料)

桶 `s3://sowii-reward-model/tutu/video_reward/`(us-east-1),访问 key 不入库、向维护者索取,配置:

```bash
aws configure --profile reward-model-s3   # 填 access key / secret
aws s3 ls s3://sowii-reward-model/tutu/video_reward/ --profile reward-model-s3
```

| S3 路径 | 内容 | 用途 |
|---|---|---|
| `data/prod500/videos.tar` | 500 条视频(386MB) | 跑 `extract_f1f5.py` 全流程 |
| `hf_cache/` | grounding-dino-base + siglip2-base 权重镜像 | 免翻墙拉底模 |
| `data/corpus/` | 语料 4473 标签与划分 | 机制验证 |
| `results/` `models/` | 每轮实验产出 / 训练权重 | 只增不改 |

全流程复现(需 GPU,~12GB 显存):

```bash
aws s3 cp s3://sowii-reward-model/tutu/video_reward/data/prod500/videos.tar . --profile reward-model-s3
tar -xf videos.tar -C data/prod500/       # 解出 data/prod500/videos/*.mp4
pip install torch transformers opencv-python-headless pillow
python scripts/extract_f1f5.py            # 产出 factors_f1f5.jsonl(底模默认从 HF 拉,或先从 S3 hf_cache/ 同步到 .hf_cache)
python scripts/eval_or_gate.py --sweep    # 新因子并门评估
```

语料机制验证子集(464 还原度 bad + 455 good)清单在 `data/prod500/mech_subset.tsv`,
视频在语料桶 `s3://trash-in-picaa/Datasets/tutu-video-eval/`(需该桶权限),`scripts/dl_mech_subset.sh` 下载。

## 为什么不用语料 4473 训练(一句话版)

语料(45% bad,82% 运动缺陷,近景棚拍)与产线(5.4% bad,还原度/物理缺陷,生活场景)是不相交的两个域:
域分类器 AUC 0.9997,同协议大样本对照里语料内 0.83 的还原度信号跨域跌到 0.512。
**测量机制可迁移,权重与阈值不可迁移**。语料唯一正当用法 = 机制标定台(详见 `docs/RETRAIN.md` 审计与 `docs/FAILURE_CATALOG.md`)。

## 进行中

- F1–F5 部件级/自参照因子(`scripts/extract_f1f5.py`,预注册于 `docs/FACTOR_PREREG.md`,先登记后评估):
  菌盖漂移 / 首帧锚定 min-over-t / 脸部漂移 / 镜头补偿尺寸趋势 / 非角色残余运动。
- F6 定向 VLM 问题轴 + F7 IP 模板符合度(Qwen3-VL-30B,H100)。
- 机制验证子集:语料 skus 域还原度 bad 464 + good 455(`data/prod500/mech_subset.tsv`)。

## 目录

```
docs/     失败模式精读目录 / 因子预注册 / 前人体系审计 / 迭代日志 / S3 公约
scripts/  评估门 / 因子提取 / S3 同步 / 子集下载
data/prod500/  GT 表(脱敏,无内部 URL)+ 裁剪特征表 + 机制子集清单
```

大文件(500 条视频、模型权重、语料)不入 git,走 `s3://sowii-reward-model/tutu/video_reward/`(公约见 `docs/S3_LAYOUT.md`)。
