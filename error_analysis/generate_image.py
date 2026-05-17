import argparse
from tqdm import tqdm
from PIL import Image
import requests
import os
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from utils_all.api import APIClient
import os
import importlib
def load_model_module(name):
    if name == "flux":
        module_name = "utils_all.image_flux"
    elif name in ["sd1", "sd2", "sd3"]:
        module_name = f"utils_all.image_{name}"
    elif name == "sana":
        module_name = f"utils_all.image_{name}"
    else:
        raise ValueError(f"Unknown model name: {name}")

    return importlib.import_module(module_name)


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(15))
def request_image(prompt,idx,output_dir):
    response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    n=1,
    size="1024x1024"
    )
    # print(response)
    # print(response)
    image_url = response.data[0].url
    # 下载图像
    img_response = requests.get(image_url)

    img = Image.open(io.BytesIO(img_response.content))
    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img.save(f"{output_dir}/{idx}.png")
    print(idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate image")
    parser.add_argument('--cuda', type=str, required=True, help='cuda devices') 
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input file') 
    parser.add_argument('--model_name', type=str, required=True, help='Model name: flux, sd1, sd2, sd3')
    parser.add_argument('--model_path', type=str, required=True, help='Model path: flux, sd1, sd2, sd3')
    parser.add_argument('--api_key', type=str, required=True, help='Api_key for API request')
    parser.add_argument('--url', type=str, required=True, help='Base_url for API request')
    parser.add_argument('--api_model', type=str, required=True, help='Model for API request')
    # 解析命令行参数
    args = parser.parse_args()
    input_folder = args.input_folder
    api_key = args.api_key
    url = args.url
    api_model = args.api_model
    model_path = args.model_path

    width = height = 1024
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 5  # 重试间隔时间（秒）
    output_dir = f"{input_folder}/ori_img"
    prompt_file = f"{input_folder}/original_prompts.txt"

    if args.model_name == "dalle":
        client = APIClient(api_key ,url, api_model)
    else:
        model_module = load_model_module(args.model_name)

        pipe = model_module.get_pipe_slow(model_path,args.cuda)

    with open(prompt_file) as fd:
        idx = 0
        for line in tqdm(fd):
            # if idx >=160:
            prompt = line.strip()
            if args.model_name == "dalle":
                request_image(prompt,idx,output_dir)
            else:
                model_module.t2i_slow(pipe, prompt, idx,output_dir=output_dir)
            idx += 1
    print("Done!")

