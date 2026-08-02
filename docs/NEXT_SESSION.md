# 下次开机清单(2026-08-03 凌晨定稿)

机器状态:vast 实例已 stop(磁盘保留,/root/mech 数据全在:4845 视频、sam3_feat、
cutouts、face_crops、32B/8B 权重缓存)。start 后 ssh 端口可能变,以面板为准。

## 优先级 1:视频原生编码器专家(预计 2-3h 盒时)
- 动机:语料 bad 六成运动类;现栈唯一运动原生组件 V-JEPA2(专家 0.63)未换代,
  是"换底模"里唯一未被证伪的一格。
- 做法:VideoMAE-v2 或 InternVideo2 级编码器(下载优先走 HF,box 网速快)对全 4845 抽视频级
  嵌入 → 本地训头(参照 E7 协议:train 侧 OOF 选配置,eval 单发)→ 并入 C5 输入矩阵重训。
- 预注册后执行;判准沿用 Δev@95 ≥ +0.03 且 ev@100 不降。

## 优先级 2(可选):E4 train 侧补齐
- 已叫停(eval 判决 0.546 抛硬币)。仅当想给汇报补"VLM 运动分并栈也无增量"的完整句时
  才值得花 3h,默认不跑。

## 零盒时事项(随时)
- 尾部盲重标:data/s3/relabel_candidates.csv(60条打乱无标签)交标注,
  依据 docs/TUTU视频三档分级标准.html;key 在 relabel_key_private.csv(勿给标注方)。
- 召回点讨论:90% 召回 → 35.7% 放行(C5 已有数,零研发)。
- 汇报页 docs/report_20260803.html 用户过目后再外发。

## 部署交付物
- 推荐模型:data/s3/c5_lgbm_final.pkl(输入=[15专家分, X320 中位数填],单视频打分)。
- 语料头条:ev@95=26.5%(基线18.8)/ ev@100=12.1%(基线1.1);产线:26.4% @27/27。
- 全部判决与协议:docs/FACTOR_PREREG.md;S3 备份 codes/code/tutu-video-eval/ours-backup-20260803/。
