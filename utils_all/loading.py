import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from PIL import Image
import base64
from openai import OpenAI
# from gpt_proxy import OpenAIApiProxy
import os
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from numpy import mean
import random
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import threading
import torch
from PIL import Image
import random
import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_json(file_path):
    """ 加载 JSON 文件 """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_txt(input_file):
    # 逐行读取 prompts 并处理
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]  # 去除空行
    return prompts

def logger_print_txt(output_file,content):
    with open(output_file, "a", encoding="utf-8") as file:
        file.write(str(content)+ "\n") 
    print(content)

# def load_jsonl(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         return [json.loads(line.strip()) for line in f]
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
# 按顺序加载图像
def load_images(folder_path):
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[0])  # 按文件名数字排序
    )
    return [Image.open(os.path.join(folder_path, img_file)) for img_file in image_files]
# 获取图片文件路径
def get_image_paths(folder_path):
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0].split("_")[0])  # 按文件名数字排序
    )
    return [os.path.join(folder_path, img_file) for img_file in image_files]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
