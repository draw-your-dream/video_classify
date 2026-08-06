#!/usr/bin/env python
"""尾巴去暗示复测:中性问法,不提颜色/位置,3条(目标5822 + 负对照7060/6472)。"""
import json, re
from pathlib import Path
import torch
from PIL import Image
ROOT = Path("/root/mech")
Q = """以下图片是同一条AI生成视频的逐帧画面(按时间顺序)。主角是毛绒蘑菇角色「蘑菇TUTU」(浅黄色身体、红色伞盖)。
官方设定:TUTU 没有尾巴,身体上除了四肢和伞盖不应有任何长出来的结构。
问题:请逐帧检查 TUTU 的身体(尤其臀部两侧、背后、下后方),是否有任何长在身体上的多余凸起、球状物或附属结构?
如果有,报告它的颜色、位置和出现帧号;没有就报空。排除:穿戴物、手持物、腮红(脸颊上的粉色圆斑)、其他物体、被遮挡处。
输出一行JSON:{"extra": [帧号], "color": "颜色或无", "position": "位置或无", "note": "一句话"}"""
def load_crops(rel):
    d = ROOT / "data/crops_v3" / rel.replace(".mp4", "")
    return [Image.open(p).convert("RGB") for p in sorted(d.glob("f*.jpg"))[:16]]
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
for tag, rel in (("目标5822", "基础款/5822.mp4"), ("负对照7060", "基础款/7060.mp4"), ("负对照6472", "花花款/6472.mp4")):
    j = parse(ask(load_crops(rel), Q))
    print(f"{tag}: {json.dumps(j, ensure_ascii=False)[:220]}", flush=True)
print("TAIL_PROBE2_DONE", flush=True)
