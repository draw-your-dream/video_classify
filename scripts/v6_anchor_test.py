#!/usr/bin/env python
"""v6 合并版锚测:Pass1 头部放大(眉毛宽措辞+眼睛异常) | Pass2 下身放大(尾巴中性开放)。
7锚:3眉毛目标+1尾巴目标+3对照。全过才解锁全量。"""
import json, re
from pathlib import Path
import cv2, torch
from PIL import Image
ROOT = Path("/root/mech")
Q1 = """以下最多16张图是同一条AI生成视频的逐帧画面,已放大到角色头部区域(按时间顺序,第0张起)。只判断毛绒蘑菇角色「蘑菇TUTU」本体。
官方设定:TUTU 无眉毛;两颗实心黑豆眼。被遮挡或看不清的部位一律默认正常。
两个问题,独立判断:
1. eyebrows(眉毛):位于眼睛上方、与眼睛明显分离的独立弧线、斜线或条状痕迹(颜色可深可浅)。眯起的窄眼本身不算;但同一帧里既有眼睛、上方又另有分离线条/痕迹,就是眉毛。
2. eye_anomaly(眼睛异常):眼睛数量不对、位置错乱、形状明显崩坏、两眼明显不对称(一大一小/一高一低)。眼内高光反光、画质模糊、被遮挡看不清的不算。
输出一行JSON:{"eyebrows": [帧号], "eye_anomaly": [帧号], "note": "一句话"}"""
Q2 = """以下最多16张图是同一条AI生成视频的逐帧画面(已放大到角色身体下半部分,按时间顺序)。主角是毛绒蘑菇角色「蘑菇TUTU」(浅黄色身体、红色伞盖)。
官方设定:TUTU 没有尾巴,身体上除了四肢和伞盖不应有任何长出来的结构。
问题:请逐帧检查 TUTU 的身体(尤其臀部两侧、背后、下后方),是否有任何长在身体上的多余凸起、球状物或附属结构?如果有,报告颜色、位置和帧号;没有报空。
排除:穿戴物、手持物、腮红(脸颊上的粉色圆斑)、其他物体、被遮挡处。
输出一行JSON:{"extra": [帧号], "color": "颜色或无", "position": "位置或无", "note": "一句话"}"""
def views(rel):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    up, lo = [], []
    for p in sorted(d.glob("f*.jpg"))[:16]:
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        u = im[0:int(H*0.62), :]
        up.append(Image.fromarray(cv2.cvtColor(cv2.resize(u, (W*2, int(H*0.62*2)), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)))
        l = im[int(H*0.40):, :]
        lo.append(Image.fromarray(cv2.cvtColor(cv2.resize(l, (W*2, (H-int(H*0.40))*2), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)))
    return up, lo
def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None
from transformers import AutoModelForImageTextToText, AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-32B-Instruct", dtype=torch.bfloat16, device_map="cuda").eval()
def ask(ims, q):
    content = [{"type": "image", "image": im} for im in ims] + [{"type": "text", "text": q}]
    inputs = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                      tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=170, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
import time
CASES = [("眉毛靶H086", "基础款/5719.mp4"), ("眉毛靶H089", "基础款/6565.mp4"), ("眉毛靶H101", "基础款/6228.mp4"),
         ("尾巴靶H099", "基础款/5822.mp4"), ("对照+7060", "基础款/7060.mp4"),
         ("对照-7241", "基础款/7241.mp4"), ("对照-6472", "花花款/6472.mp4")]
t0 = time.time()
f = open(ROOT / "data/v6_anchor.jsonl", "w")
for tag, rel in CASES:
    up, lo = views(rel)
    r1 = parse(ask(up, Q1)) or {}
    r2 = parse(ask(lo, Q2)) or {}
    row = {"tag": tag, "rel": rel, "head": r1, "lower": r2}
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{tag}: 眉{r1.get('eyebrows')} 眼{r1.get('eye_anomaly')} | 尾{r2.get('extra')} {r2.get('color','')}@{r2.get('position','')}", flush=True)
f.close()
print(f"每条耗时 {(time.time()-t0)/len(CASES):.1f}s", flush=True)
print("V6_ANCHOR_DONE", flush=True)
