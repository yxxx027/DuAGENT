DEFAULT_IMG_DIR = '/ori_img'
import argparse
import pdb
import torch
import hashlib
import random
from io import BytesIO
from diffusers import StableDiffusion3Pipeline
import os
import requests
import openai
from PIL import Image
import io

def get_pipe_slow(model_path,cuda="cuda:0"):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16,local_files_only=True)
    pipe = pipe.to(cuda)

    return pipe

def t2i_slow(pipe, prompt, idx, width=1024, height=1024, output_dir="/ori_img"):
    image = pipe(prompt, width=width, height=height, num_inference_steps=50, 
                  generator=torch.Generator("cpu").manual_seed(random.randint(1, 1000))).images[0]

    # md5_fn = calc_img_md5(image)
    # image.save(f"{output_dir}/{idx}_{md5_fn}.png")
    image.save(f"{output_dir}/{idx}.png")
    return image

def t2i_slow_batch(pipe, prompt,  width=1024, height=1024, output_dir="/ori_img"):
    image = pipe(prompt, 
                 width=width, 
                 height=height, 
                 num_inference_steps=50, 
                generator=torch.Generator("cpu").manual_seed(random.randint(1, 1000))).images
    # md5_fn = calc_img_md5(image)
    # image.save(f"{output_dir}/{idx}_{md5_fn}.png")
    return image

