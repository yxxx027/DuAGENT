import argparse
from tqdm import tqdm
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from openai import OpenAI
import base64
from utils_all.api import APIClient

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 定义处理单个图像的函数
def process_image(img):
    vision_prompt = """
    Given the image, provide the following information:
    - A detailed description of the image
    """
    caption = client.request_gpt_with_image(vision_prompt,"image/png",encode_image(f"{input_folder}/ori_img/{img}"))
    # print(caption)
    # assert False
    return json.dumps(caption, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input file') 
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output file') 
    parser.add_argument('--api_key', type=str, required=True, help='Api_key for API request')
    parser.add_argument('--url', type=str, required=True, help='Base_url for API request')
    parser.add_argument('--api_model', type=str, required=True, help='Model for API request')
    # 解析命令行参数
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    api_key = args.api_key
    url = args.url
    api_model = args.api_model
    client = APIClient(api_key ,url, api_model)

    output_path = f"{output_folder}/captions.jsonl"

    imgs = os.listdir(f"{input_folder}/ori_img")
    sorted_imgs = sorted(imgs, key=lambda x: int(x.split('.')[0]))

    # 使用 ThreadPoolExecutor 进行并发处理
    with open(output_path, 'w') as out:
        with ThreadPoolExecutor() as executor:
            # 使用进度条处理结果
            for caption in tqdm(executor.map(process_image, sorted_imgs), total=len(sorted_imgs)):
                out.write(f"{caption}\n")
                out.flush()
