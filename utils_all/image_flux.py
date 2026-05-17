DEFAULT_IMG_DIR = '/ori_img'
import argparse
import pdb
import torch
import hashlib
import random
from io import BytesIO
from diffusers import FluxPipeline
import os
# def calc_img_md5(img):
#     byte_io = BytesIO()
#     img.save(byte_io, format='PNG')
#     byte_io.seek(0)
#     # 读取字节流并计算MD5
#     md5 = hashlib.md5(byte_io.read()).hexdigest()
#     return md5

def get_pipe_slow(model_path,cuda="cuda:0"):
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,local_files_only=True)
    print("Loading model")
    pipe.to(cuda)
    return pipe

def t2i_slow(pipe, prompt, idx, width=1024, height=1024, output_dir="/ori_img"):
    image = pipe(
        prompt,
        width=width,
        height=height,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=torch.Generator("cpu").manual_seed(random.randint(1, 1000))
        # generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]
    # md5_fn = calc_img_md5(image)
    # image.save(f"{output_dir}/{idx}_{md5_fn}.png")
    image.save(f"{output_dir}/{idx}.png")
    return image

def t2i_slow_batch(pipe, prompt,  width=1024, height=1024, output_dir="/ori_img"):
    image = pipe(
        prompt,
        width=width,
        height=height,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=torch.Generator("cpu").manual_seed(random.randint(1, 1000))
        # generator=torch.Generator("cpu").manual_seed(42)
    ).images
    # md5_fn = calc_img_md5(image)
    # image.save(f"{output_dir}/{idx}_{md5_fn}.png")
    return image
