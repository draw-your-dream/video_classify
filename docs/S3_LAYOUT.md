# s3://sowii-reward-model/tutu/video_reward/ 目录公约

维护方:曹源(本地 WSL,`~/tutu-video-eval`)。同步入口:`scripts/s3_sync.sh`(profile `reward-model-s3`)。
原则:**S3 是交付与交接面,不是工作区**——本地跑完、验证过的东西才上传;每轮实验产出带日期目录,只增不改。

```
tutu/video_reward/
├── README.md                  # 本公约(每次改动同步)
├── docs/                      # 分析与协议文档
│   ├── FAILURE_CATALOG.md     #   prod500 27条bad失败模式精读目录
│   ├── FACTOR_PREREG.md       #   因子预注册清单(F1-F7,先登记后评估)
│   ├── RETRAIN.md             #   前人23组特征/15专家体系审计
│   └── ITER_LOG.md            #   前人79轮迭代日志
├── data/
│   ├── prod500/               # 产线500条:GT表 prod500.csv、既有因子表(prod_crop.csv等)
│   │   └── videos.tar         #   500条视频打包(386MB,远端机器一次拉取)
│   └── corpus/                # 语料4473:merged_labels.csv + splits/*.jsonl
├── results/
│   └── YYYYMMDD_<topic>/      # 每轮实验:因子表jsonl/csv + 评估报告md(含结论与LOO数字)
├── models/
│   └── <name>/                # 训练产出:权重 + 训练配置 + 指标md
├── hf_cache/                  # 公共底模镜像(grounding-dino-base、siglip2-base-patch16-224)
└── scripts/                   # 提取/评估脚本快照(与本地 scripts/ 同步)
```

## 关键数字(截至 2026-07-29)
- 唯一指标:27/27 bad 全召回下 good 放行率。线上基线 0%;当前双轴门(dim_fidelity ⊕ c_first_last)= **26.4%**(LOO 漏 1/27=b7291aac,bootstrap CI [23.0%, 32.3%])。
- 进行中:F1-F5 部件级/自参照因子(本地 5070Ti),F6/F7 VLM 轴待大机器。

## 同步约定
- 文档/脚本/小表:改完即 `s3_sync.sh docs|scripts|data`。
- 实验产出:每轮结束 `s3_sync.sh results <本地结果目录>`,目录名 `YYYYMMDD_<topic>`。
- 模型权重:训练收敛且过验证后 `s3_sync.sh models <本地模型目录>`。
- 大文件(videos.tar、hf_cache)一次性上传,不重复同步。
