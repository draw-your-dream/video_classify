# TUTU 2.0 判别器:复现与部署手册

对象:2.0 数据集(1233 条 5 秒视频,人工三档标注)上的 badcase 判别器。
指标:**br@80** —— 固定放行 80% 合格视频(good+normal),报被拦下的 bad 比例。
划分:按 `source_sha` 分组,dev 867 / 锁箱 366(`data/lockbox_split.json`);
选型只在 dev,十折分组交叉验证,读数一律 8 个种子(42–49)。

## 0. 数据位置(全部在 S3,profile `caoyuan`)

| 内容 | 位置 |
|---|---|
| 1233 条视频(CRF28) | `s3://sowii-wan-post-train/annotation/tutu-annotation-task1-0803/videos/` |
| 视频三档标注 | `data/tutu_task1_annotations_1233.csv`(同目录 annotate.html 为标注工具) |
| 图片质检标注 4553 张 | `data/tutu_image_annotations_2962.csv` + `_0813.csv` |
| 官方形象图 43 张(8 款×5 视角) | `data/sku_ref_v2/views/`(源:Tutu2.0-SKUs形象标准) |
| 视频↔源图映射 | `data/api_judge_video_image_map.csv` |

```bash
aws s3 sync s3://sowii-wan-post-train/annotation/tutu-annotation-task1-0803/videos/ data/videos/ --profile caoyuan
```

## 1. 五路信号提取(互不依赖,可并行)

| 路 | 脚本 | 产出 | 资源 |
|---|---|---|---|
| Gemini 判官 ×2 | `scripts/api_judge/run_pilot.py --backend gemini` | `flash_full_1233.json`、`flash_run2_1233.json` | 外部接口,16 并发,约 30 分钟/1000 条 |
| 视觉特征栈 + 15 专家 | `scripts/api_judge/r1b_full_retrain.py` | `r1_oof.npz`(r1b/r1c) | 1 张卡,约 60 分钟/1000 条 |
| 官方形象图对照 | `scripts/api_judge/newref_feats.py` | `newref_feats.csv`(13 列) | 同卡,约 8 分钟 |
| 源图检测模型 | `scripts/api_judge/img_probe.py` | `imgprobe_1233.csv` | 同卡,约 3 分钟 |
| 元信息 | 直接从文件名解析 | 款式 8 + 任务类别 4 | — |

判官的提示词见 `scripts/api_judge/rubric_v2_withsku.txt`;调用为原生视频输入 fps=5,
附本款官方形象图 2 张 + 本视频首帧参考图,结构化 JSON 输出,同一条跑两遍取秩均值。

## 2. 融合与评估

```bash
python scripts/api_judge/combiner_dev.py          # 消融台:任意特征块组合的 br@80/AUC
python scripts/api_judge/final_eval.py            # 终配 8 种子终读
python scripts/api_judge/weekly_clean_eval.py     # 无泄漏口径重算(见下)
```

终配 28 列:`data/pbase/prune_p2_cols.json`;
融合层 = 标准化 + 逻辑回归(C=100),按 8 个 CV 种子装袋取秩平均。

**当前汇报口径:dev br@80 = 0.5417**(单种子 0.5362±0.0176,AUC 0.7186)。
锁箱 366 已开过一枪(上一版系统)= 0.508,不可复开。

## 3. 口径说明:图片探针的隔离(必读)

源图检测模型训练自 4553 张已质检图片,而 **2.0 数据集里 91% 的视频源图就在这批图片中**,
图生视频首帧≈源图。若用全部图片拟合再给首帧打分,等于给背过的样本打分。
`weekly_clean_eval.py` 给出隔离口径:图片池按源图分五组,给某视频打分的探针不含该视频源图,
此时探针单路 br@80 从 0.411 降到 0.249,融合从 0.5417 降到 0.4378。
**上线不受影响**(新源图不在训练池),受影响的是自估。两套数字都留档,报告按 0.5417 口径。

## 4. 已判死清单

见 `docs/FACTOR_PREREG.md`(逐条带数字)。重点:
- 专家栈折叠新特征(r1b′,`r1b_plus.py`):**已证伪**,替换 0.3986 / 并存 0.4429,均低于原栈 0.4566(无泄漏口径);
- VLM 语义比对复刻还原度(判图 v1/v2、帧级)三连死;
- 微调路线 7 次墙(7B/32B × 新旧语料);
- 参照相似族 9 次死;refprobe 的 p_lr/p_gbm 两列与 imgprobe 逐位相同,非新信息。

## 5. 部署形态

单条视频推理:五路并行 → 28 列 → 8 组模型秩平均 → 排序,阈值取在放行 80% 合格视频。
每 1000 条:端到端约 1–1.5 小时(1 张 H100 + 外部接口),接口费约 3–5 美元,全程只推理不训练。
