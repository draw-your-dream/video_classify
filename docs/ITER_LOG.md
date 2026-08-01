
## run (eval_v3 held-out cal/test, 20 seeds)
- base (15 experts): ev@95=0.301±0.035  AUC=0.742
- augmented (+motion/reason): ev@95=0.342±0.049  AUC=0.802
- full (+full_raw+centroid): ev@95=0.336±0.055  AUC=0.798

## iteration 2 — DEPLOYMENT protocol (fit train_v3 -> predict eval_v3, real new-video number)
- base (15 experts):              ev@95=0.226 [0.18,0.29]  AUC=0.748
- augmented (+motion/reason):     ev@95=0.389 [0.32,0.46]  AUC=0.802   <- current best
- augmented2 (+fidelity/identity):ev@95=0.393 [0.30,0.45]  AUC=0.803   (no gain; redundant w/ experts)
Finding: motion/reason signals (per_reason,q3vl_motion,qwen25vl,lm_sft,dense_flow) are the
generalizable lift (+0.16 ev@95, +0.05 AUC). Fidelity/identity already covered by experts.
Gap to target (0.50): ~0.11 ev@95 (~AUC 0.80->0.85). Next: motion-signal interactions,
recall-region-focused objective, denser-frame motion re-extraction, label cleaning.

## iteration 3 — modeling tricks exhausted; data is the lever
- augmented2 (+fidelity/identity):  ev@95=0.393  AUC=0.803  (no gain)
- augmented_inter (+motion interactions): ev@95=0.389  AUC=0.804  (no gain — interactions help only in-sample cascade)
Conclusion: no cheap modeling win beats augmented (0.389/0.802). q3vl_motion was missing
~1150 videos (MOSTLY BADS: bad probed only 1030/2026) -> motion signal blind on half the bads.
ACTION (running): complete q3vl_motion for all 4845 (probe_q3vl_motion.py, Qwen3-VL-8B).
Next iteration: rebuild _q3_motion_oof on full coverage, re-measure deployment ev@95.

## iteration 4-5 — completed q3vl_motion (all 4845, bad 2026/2026) and CAUGHT A LEAKAGE ARTIFACT
- Before (partial motion coverage): augmented ev@95=0.389 AUC=0.802  <- INFLATED
- After (full coverage, honest):    augmented ev@95=0.288 AUC=0.754  <- corrected
- Cause: q3vl_motion was missing ~1000 BADS; col-mean fill made missingness (∝ label) leak.
  Full-coverage q3m alone AUC=0.543 (near-chance) — the motion VLM signal is genuinely weak.
- clean (+per_reason+q3 only): ev@95=0.247 AUC=0.745  (per_reason adds real lift over base 0.226)
- Per-expert held-out AUC: hint=0.751, per_reason_max=0.734, full_lean=0.720, lm_sft=0.706,
  siglip=0.697, sigb=0.684; WEAK/anti: qwen25vl_unnatural=0.42, drift=0.44, vlm_judgment=0.50,
  fidelity_sku=0.50, tia=0.47.
HONEST baseline for a NEW video: ev@95 ≈ 0.29, AUC ≈ 0.75. Target 0.50 needs ~AUC 0.85 (+0.10).
Reweighting existing signals won't close it. Next levers: (1) label cleaning (noisy bads),
(2) stronger learned discriminator — improve the LoRA SFT (best single signal, AUC 0.71;
denser frames / better targets), (3) denser-frame motion model.

## iteration 6 — label cleaning (meta level): no gain
- Q=0.00 ev@95=0.288 AUC=0.754 | Q=0.05 0.295/0.756 | Q=0.10 0.284 | Q=0.15 0.275 | Q=0.20 0.275
Labels are not the bottleneck. Honest new-video baseline holds: ev@95≈0.29, AUC≈0.75.
Ruled out (no held-out gain): extra fidelity signals, motion interactions, full raw kitchen-sink,
label cleaning, completing motion coverage (that only removed a leakage artifact).
Remaining real levers require a STRONGER DISCRIMINATOR (not reweighting): improved LoRA SFT
(denser frames/better targets — best single signal 0.71), denser-frame motion model, end-to-end FT.

## iteration 7 — temporal-motion features from embeddings: weak, no gain
- temporal-only AUC=0.564; base+per_reason(+temporal) ev@95 0.233->0.236, AUC 0.746->0.747
CORE FINDING across iters: motion defects (67% of bads) are weakly captured by ALL current reps:
  q3vl_motion VLM=0.54, temporal embedding deltas=0.56, V-JEPA2 video expert=0.63 (< static siglip 0.70).
Cheap+medium levers exhausted at AUC≈0.75 / ev@95≈0.29. Closing to 0.50 (AUC~0.85) needs a
materially stronger motion discriminator. Heavy levers left: dense-frame LoRA SFT, dense optical-flow
temporal model, end-to-end video FT. These are GPU-hours with uncertain payoff.

## iteration 7b — SECOND leakage found + heavy lever launched
- lm_sft_v3 (best learned signal) was trained on train_v2; 765/968 eval_v3 videos leaked into it.
  AUC: leaked subset 0.719 vs HELD-OUT subset 0.656. So the honest baseline (0.75) is itself
  slightly inflated by this. -> retrain SFT on train_v3 (eval_v3 truly held out).
- LAUNCHED: LoRA SFT on train_v3, 16 frames @224 (motion focus), -> lm_sft_v3b_pred (new cache,
  frozen model untouched). ~3h. Doubles as honesty fix + denser-frame improvement attempt.
- Honest trajectory so far: every leakage removed LOWERS the number; cheap/medium levers flat at
  AUC~0.75. Motion-defect detection weak across ALL reps (VLM 0.54, temporal 0.56, V-JEPA2 0.63).

## iteration 8 — ROOT-CAUSE diagnostic of the ev@95 bottleneck
The 20 hardest bads (bound ev@95) are character-motion-quality defects:
  reasons: 僵硬×7, 还原度×5, 卡顿×5, 动作不连贯×3, 四肢不动×2.
  per_reason_max on them = 0.231 (LOWER than goods' 0.436) — invisible to all signals.
Motion signals on these 20 vs goods (higher=more bad-like, want hard>good):
  frozen_ratio   hard 0.149 < good 0.181   (NOT globally frozen)
  mean_mag       hard 0.186 > good 0.166   (MORE flow than goods)
  stuck_p ~0; rigid_p 0.059 vs 0.028; limb_p 0.134 vs 0.083 (faint, noisy)
=> Defect is the CHARACTER moving stiffly/unnaturally while global flow looks normal.
   Global optical flow + sparse-frame VLM + image embeddings cannot separate it.
   This is the structural reason ev@95 caps ~0.29; SFT(16-frame) is the attempt to learn it
   end-to-end. If it can't, the realistic levers are: pose/keypoint tracking of the character
   over dense frames, OR more labeled data on these subtle cases, OR revisiting label subjectivity.

## iteration 9 — character-region motion: PROMISING lead
Hypothesis test on the 20 hard bads vs 20 goods (dense-frame center-region optical flow):
  center motion: HARD 2.73 vs good 4.06 | cen/glob ratio: HARD 1.29 vs good 1.63
=> stiff-character bads move LESS in the character region (even relative to scene) — a DIRECT
   measure of 僵硬/四肢不动 that global flow misses. Building char_motion_v2 for all 4845 (CPU)
   then testing held-out deployment gain.

## iteration 11 — char_motion population sanity (done subset ~2878 vids): signal confirmed
  cen_glob_ratio  good 1.56 / normal 1.60 / bad 1.13   (bad clearly lower)
  inner_mean      good 3.33 / normal 3.84 / bad 2.44
  cen_frozen_frac good 0.21 / normal 0.20 / bad 0.31   (bad more frozen)
Bads move ~0.7x goods in the character region — population-level signal, not just the 20 hard bads.
Full held-out test (iter_charmotion.py) pending full 4845 coverage.

## iteration 12 — char_motion held-out result: MARGINAL (no real gain)
- char_motion SOLO held-out AUC = 0.576 (high variance; center-region is a crude character proxy)
- base+per_reason: ev@95=0.233 AUC=0.746 -> +char_motion: ev@95=0.242 AUC=0.749 (+0.003, noise)
Why: population separation is real (bad 0.7x good motion) but individual-level too noisy + partly
redundant with dense_flow/motion experts. Precise character segmentation might do better but the
proxy ceiling (~0.58 solo) suggests diminishing returns.
STATE after 12 iters: held-out ev@95≈0.24-0.29, AUC≈0.75. Every motion feature (VLM 0.54,
temporal 0.56, char-region 0.58) is weak — the character-motion-quality defect resists crude detection.
Last in-flight lever: 16-frame leakage-free SFT (training, ~1.5h left).

## iteration 13 — recall-region-focused loss: small operating-point gain, far from target
- surrogate-loss MLP (directly optimizes ev@95): ev@95=0.268 AUC=0.736 (vs ensemble 0.233/0.746)
=> tail-focused loss squeezes the operating point up ~0.03 by trading AUC, but AUC~0.74 bounds it.
MODELING ANGLE EXHAUSTED: standard ensemble, recall-focused loss, every feature add (motion/fidelity/
interactions/char-region/temporal), label cleaning -> all land ev@95≈0.23-0.29, AUC≈0.75.
Only remaining in-flight new-signal lever: 16-frame leakage-free SFT (~1.7h left).

## iteration 30 — LEAKAGE-FREE SFT result + complete honest verdict
- NEW 16-frame SFT (trained train_v3, eval_v3 truly held out): solo AUC=0.623
- best leakage-free deployment ensemble: base+per_reason+NEW_sft  ev@95=0.274  AUC=0.746
  (+char_motion -> 0.256: hurts operating point; per_reason+sft is the clean best)
HONEST VERDICT after ~30 iters + a 3h dedicated motion SFT + 2 caught leakages:
  Best fully-leakage-free NEW-VIDEO number = ev@95 ≈ 0.27, AUC ≈ 0.75. Target = 0.50 (AUC~0.85).
  Every lever (15 experts, per_reason, VLM motion, char-region motion, temporal, recall-loss,
  16-frame end-to-end SFT, label cleaning) lands ev@95 0.23-0.29. The bottleneck (subtle
  character-motion-quality defects) resists all available ≤10B representations.
  SFT direction gave the biggest single gain (+0.04) -> launched stronger 2-epoch/r32 SFT as the
  next bet. Honest expectation: incremental, unlikely alone to reach 0.50 — that likely needs more
  labeled motion-defect data or a fundamentally stronger dense-frame video-motion model.

## iterations 31-66 — THREE SFT variants; ensemble firmly bottlenecked
SFT solo held-out AUC: 16-frame=0.623, 32-frame=0.670 (denser frames DID help discrimination +0.047),
  16f-2ep=pending. But ENSEMBLE ev@95 (deployment) regardless of SFT:
  base+per_reason 0.233 | +16f 0.274 | +32f 0.252 | +BOTH 0.258  — all AUC~0.74.
=> More frames improves the SFT's own AUC but does NOT move the ensemble operating point.
ROBUST VERDICT (66 iters, 3 SFTs/many GPU-h, all feature+modeling levers, 2 leakages caught):
  honest NEW-VIDEO ev@95 ≈ 0.27, AUC ≈ 0.74-0.75. Does not approach 0.50.
  Reaching 0.50 (AUC~0.85) needs a DIFFERENT investment, not more feature/SFT tuning:
  (a) more labeled data on the subtle motion-quality cases, or
  (b) a fundamentally stronger dense-frame video-motion model (>10B / video-native, end-to-end).

## iteration 73 — 16f-2ep result; best honest ensemble so far
- 16f-2ep solo held-out AUC=0.577 (2 epochs OVERFIT, < 1-epoch 0.623) but ensemble ev@95=0.282 AUC=0.754 (best)
- SFT-variant ensemble ev@95: 16f-1ep 0.274 | 32f-1ep 0.252 | 16f-2ep 0.282  (all AUC~0.74-0.75)
Best honest NEW-VIDEO number to date: ev@95 ≈ 0.282, AUC ≈ 0.754. Still far from 0.50.
Last in-flight lever: motion-specialized SFT (2ep/32f, motion-defect bads vs good/normal, ~4h).

## iteration (final lever) — motion-specialized SFT
- solo all-bad AUC=0.635; MOTION-bad vs good/normal AUC=0.732 (focused supervision DID help the subproblem)
- ensemble base+per_reason+motionSFT: ev@95=0.245 AUC=0.735 (does NOT beat best 0.282)
INTERPRETATION: a motion detector CAN be built (0.73 on motion-bad-vs-good), but ev@95 is bounded by
the hardest ~5% of bads across ALL types (incl. 还原度/fidelity), so the overall operating point stays ~0.28.

## FINAL VERDICT (exhaustive: 4 SFTs [8/16/32-frame + motion-specialized], all features, recall-loss,
## label-cleaning, char-motion, 2 leakages caught)
Best honest NEW-VIDEO number: ev@95 ≈ 0.282, AUC ≈ 0.754 (base + per_reason + 16f-2ep SFT).
Split-variance band over 15 random splits: 0.26 ± 0.03 (range 0.22-0.31).
Target = 0.50 (needs AUC~0.85). NOT reachable by feature/model tuning on the current data.
The motion subproblem reaches 0.73 — so the route to 0.50 is: (1) more labeled data on the subtle
hard cases (motion AND fidelity), and/or (2) a stronger dense-frame video model — NOT more tuning.
Single-video inference: inference/predict.py is self-contained (video -> p_bad).

## NEW DIRECTION — specialist union (evidence-backed)
Diagnostic: the 20 hardest bads (base rank 0.083) are caught much better by the motion-specialist
(rank 0.399; 8/20 above good-median). Specialists ARE complementary. Blending base+motion lifted a
weak base 0.208->0.242. => one detector per defect FAMILY, unioned, can catch more hard bads.
Motion family covered (0.73 on motion-bad-vs-good). LAUNCHED fidelity-specialist SFT (还原度/塑料/
不是TUTU/身材变形, 404 fid-bad, 2ep) to cover the other hard-bad family. Will union motion+fidelity+base.

## specialist-union result — does NOT beat best single addition
- base+per_reason 0.233 | +16f2ep 0.284 (BEST) | +16f2ep+motion 0.224 | +union(motion+fid) 0.266
- fid-SFT solo AUC=0.574. The complementarity on hard bads is real but cancels at ev@95 because the
  specialists also rank some goods high -> meta/MAX-union can't exploit it at 95% recall.

## DEFINITIVE VERDICT (everything exhausted)
Levers tried & measured leakage-free on a fresh held-out split (eval_v3):
  15-expert stack; per_reason; 4 SFTs (8/16/32-frame generic + motion-specialist + fidelity-specialist);
  char-region motion; temporal-embedding features; fidelity/identity signals; motion interactions;
  recall-region-focused loss; label cleaning; specialist-union (meta & MAX). Caught 2 leakages.
BEST HONEST NEW-VIDEO NUMBER: ev@95 = 0.284, AUC = 0.754. Split-variance band 0.22-0.31.
This matches the project's original OOF analysis (AUC~0.727 -> g+n@95~0.27).
Target = 0.50 (AUC~0.85) is NOT reachable by feature/model tuning on the current 4845-video dataset.
The only remaining paths require new resources, not tuning:
  (1) more labeled data on the subtle hardest cases (motion + fidelity),
  (2) a fundamentally stronger dense-frame video model.
Deliverable: inference/predict.py — self-contained single-video -> p_bad.

## FINAL — Qwen3-VL-8B (video-native, <=10B) SFT
solo held-out AUC = 0.6916 (BEST single discriminator; Qwen2.5-VL: 16f 0.623 / 32f 0.670 / 16f-2ep 0.577).
Ensemble: base+per_reason 0.233 | +16f2ep 0.284 | +Qwen3VL8B 0.272 | +16f2ep+Qwen3VL8B 0.281.
=> video-native model gives the best single AUC but does NOT lift ensemble ev@95 past 0.284.
CONCLUSION (stopping per user request): best honest new-video ev@95 = 0.284, AUC = 0.754, stable
across 5 SFTs (incl. video-native + 2 specialists), all features, specialist-union, recall-loss,
label-cleaning, 2 leakages caught. Reaching 0.50 needs more targeted data or a >10B/stronger video model.

## Qwen3.5-9B SFT (user-suggested path; Qwen3.5 is a real public native-multimodal model, 2026-03)
solo held-out AUC = 0.6753 (2nd-best single discriminator; beats all Qwen2.5-VL SFTs, ~ Qwen3-VL-8B 0.692).
Ensemble: base+per_reason 0.233 | +16f2ep 0.284 | +Qwen3.5-9B 0.275 | +16f2ep+Qwen3.5-9B 0.281.
=> Qwen3.5-9B is a strong judge but, like every SFT (incl. Qwen3-VL-8B), does NOT lift ensemble ev@95
   past 0.284. The ev@95 ceiling is set by the hardest ~5% of bads that no single model reliably catches.

## Part1 (engineering) — wire Qwen3.5-9B as the `sft` expert signal: NO improvement
Rebuilt 15 experts+stack+meta with sft = Qwen3.5-9B SFT (vs old Qwen2.5-VL). Base eval Pareto:
  pre-q35 (Qwen2.5-VL sft): ev@95=0.188 / ev90=0.297 / ev80=0.512
  q35     (Qwen3.5-9B sft): ev@95=0.167 / ev90=0.286 / ev80=0.478   (slightly WORSE at 95%)
=> consistent with all prior results: stronger single discriminator (Qwen3.5-9B solo AUC 0.675)
   does NOT lift the ensemble. NOT promoted; kept the current artifacts_v3. predict.py single-video
   verified working with --artifacts.

## 2026-07-30 | 单类 patch 异常检测机制验证(语料 919 子集,H100)

假设:身份属性不可变/状态属性自由变 -> good 定义"正常轨道",patch 级异常检测(AnomalyDINO 范式)
可分开还原度 bad/good。预注册 FACTOR_PREREG.md P1(v2:款分层划分 + s_hist 逻辑分支 + 裁剪核验)。

流水:919 条(还原度 bad 464 + good 455)预签名 URL 直拉 H100(密钥不上机)->
GroundingDINO 原版配置裁剪 -> DINOv2-base 37x37 patch 特征(2.4s/vid,全 919 零错误)->
good 款分层 299 建库 / 156 评估 -> 1-NN 余弦 + k-means64 直方图,三口径聚合。

结果:**未过晋级关(AUC>=0.70)**。最好 s_hist_mean=0.568 / s_top5_fg_mean=0.543;
剔跑偏裁剪敏感性 0.564-0.571 稳健;分款分库对照池化 0.574,款级收紧无效。

机理:bank_good P50=0.19 vs eval_good P50=0.42 —— 开放内容域 novelty 噪声地板淹没缺陷信号,
MVTec 封闭场景前提不成立。**参照必须收紧到本视频 -> i2v 首帧原图 = 头号数据请求。**
副产品:P2 角色在场性轴(1-cos_centroid)AUC 0.584,prod 可直接复算,待预注册评估。

工件:scripts/{extract_patch_feat,patch_bank_eval,patch_bank_eval_style,verify_crops2}.py;
data/prod500/{patch_bank_scores,patch_bank_scores_style,verify_crops2}.csv;
crop_montage.jpg / flag_montage.jpg(人工复核通过,跑偏尾部 ~20 条)。
v1 文本塔核验(transformers 5.x 下损坏)作废,见 PREREG 附注。

## 2026-07-31 第0步逐簇AUC分解:0达标簇,bank范式关闭
- 旧盒GPU被占无法重启;新盒H100全链路重跑,与原机千分位复现(0.567/0.575,verify2逐字一致)。
- 逐簇分解(64 k-means簇=免费部件分割,预注册判读门 AUC>=0.65+覆盖+语义):
  外观轴 P50=0.502 P90=0.552 满覆盖max=0.587;占比轴最偏0.613/0.380 -> 0达标。
- 结论:novelty地板在簇内部,三级参照系收紧(全身->款->部位)全失败;部件线弃bank,
  转 线C比例轨迹(测量)/线B关键点计数(标注)/线A合成缺陷监督;i2v首帧原图仍头号数据请求。
- 产物:cluster_auc.csv、cluster_montage_parts.jpg(待人工过目簇语义)、cluster_auc_decomp.py。

## 2026-08-01 P3 战役:SAM3底座+参考图三轴+VLM判官,一晚三范式全判
- 资源落地:tutu-renders-2026-04(152张:基础款kf 98+六款各9,含SAM3掩码);SAM3权重
  sha256对官方核验后取自公开镜像(盒上零凭证);前人图片侧IP-eval管线消化(抠像+LPIPS/DreamSim/so400m)。
- SAM3提取:919语料+500产线,1.2s/条,抠像蒙太奇人工核验通过(道具正确排除,实例数稳定=1)。
- R轴(DreamSim/so400m vs 参照池):语料最好r_ds_mean=0.568;G轴几何0.53/漂移0.47;C轴计数0.46-0.47。
- prod500探索性集成:基线26.4%精确复现;任一新轴并入OR门均降低放行(13.7-21.1%),无增量。
- V轴(Qwen3-VL-8B参考图对照问答):AUC=0.500退化,261条全9分,关闭。
- 结论:款级参照+外观度量+语义判官全线穷尽,0.57天花板横跨五方法族;存活方向=
  i2v首帧原图(头号数据请求)/合成缺陷监督(线A,唯一未测)/缺陷子类型标注(裁决标签异质假设)。
- 产物:axis_rgc_scores.csv、axis_rgc_prod.csv、vlm_v_axis.jsonl、ref_embeds、
  extract_sam3/prep_ref_embeds/axis_rgc_eval/axis_rgc_prod/vlm_v_axis.py。

## 2026-08-01(晚)V2/R-self/线A 三连判 + 标签异质假设成型
- V2(Qwen3-VL-32B):排序 0.571 关闭;登记发现=旗标零误报(bad 8/56, good 0/57)。
- R-self(c_first_last 语料版):0.41-0.48 反向——自漂移量的是动作幅度;语料/产线缺陷构成不同再证。
- 线A v0(合成缺陷监督):真 bad 0.518;仪器自检 held-out 合成 0.876(黑点0.98/尾0.996/拉长0.97)。
- 三方互证 => 主解释:语料「还原度 bad」静帧可见缺陷占比 ~15%,0.57=1-口径天花板。
- 裁决:39 例盲审页已交用户(旗标bad8+未旗标bad23+good10,乱序匿名)。
- 资产:synth_defects.py(7族)、synth_head_v0.pt、audit_page、axis_rself.csv、vlm_v2.jsonl。

## 2026-08-01(深夜)线F判决,语料还原度轴收档
- 线F脸部高分辨率:SAM3认脸12/12;919条原生分辨率脸裁剪;五族合成(眉毛族用户点名)。
- 真实bad 0.529 vs 仪器自检0.974(eyebrows 1.000)——测量链闭合:清单型静态缺陷检测器有效,
  真bad中无此信号。语料还原度标签非静态外观驱动,轴收档。
- E集成0.574无增量;V3三变体0.527-0.598只移工作点。
- 转向:prod域+i2v原图+子标签数据请求。资产:双检测头(0.876/0.974)+双合成管线+SAM3提取链。
