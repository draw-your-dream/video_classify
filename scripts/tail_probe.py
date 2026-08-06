#!/usr/bin/env python
"""尾巴精确锚定探针:H099(5822)。两视图:全身裁剪448 + 左下半区放大;锚定问法(左臀/粉色/小)。"""
import json, re
from pathlib import Path
import cv2, torch
from PIL import Image
ROOT = Path("/root/mech")
Q = """以下图片是同一条AI生成视频的逐帧画面(按时间顺序)。主角是毛绒蘑菇角色「蘑菇TUTU」(浅黄色身体、红色伞盖)。
官方设定 TUTU 没有尾巴。已知线索:这条视频里,TUTU 的臀部左侧可能长了一个粉色的小凸起(小尾巴),它比较小、颜色偏粉,可能一直存在。
问题:请逐帧仔细检查 TUTU 的臀部两侧和身体下后方,是否有这个粉色小凸起/小尾巴?
排除:穿戴物、手持物、腮红(在脸上)、其他物体。
输出一行JSON:{"tail": [看到的帧号], "position": "位置描述", "note": "一句话"}"""
def load_crops(rel):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    return [Image.open(p).convert("RGB") for p in sorted(d.glob("f*.jpg"))[:16]]
def lower_left_zoom(rel):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    ims = []
    for p in sorted(d.glob("f*.jpg"))[:16]:
        im = cv2.imread(str(p)); H, W = im.shape[:2]
        crop = im[int(H*0.35):, :int(W*0.72)]
        crop = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2), interpolation=cv2.INTER_CUBIC)
        ims.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    return ims
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
        gen = model.generate(**inputs, max_new_tokens=140, do_sample=False)
    return proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
rel = "基础款/5822.mp4"
for name, ims in (("全身裁剪448", load_crops(rel)), ("左下放大x2", lower_left_zoom(rel))):
    j = parse(ask(ims, Q))
    print(f"{name}: {json.dumps(j, ensure_ascii=False)[:200]}", flush=True)
# 负对照:7060(无尾巴)同问法防"顺着线索点头"
for name, ims in (("负对照7060全身", load_crops("基础款/7060.mp4")),):
    j = parse(ask(ims, Q))
    print(f"{name}: {json.dumps(j, ensure_ascii=False)[:200]}", flush=True)
print("TAIL_PROBE_DONE", flush=True)
