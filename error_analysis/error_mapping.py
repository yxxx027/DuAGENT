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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from utils_all.loading import logger_print_txt,load_json,load_txt,load_jsonl,get_image_paths
from utils_all.api import APIClient
import argparse
def match_image_with_text(image_folder, caption_file, prompt_file):
    bugs = load_jsonl(caption_file)
    prompt = load_txt(prompt_file)
    images = get_image_paths(image_folder)

    # 验证匹配并打印结果
    if len(images) == len(bugs) == len(prompt):
        print("Images and JSONL files are correctly matched by length.")
    else:
        print(f"Mismatch: {len(images)} images, {len(bugs)} entries in JSONL caption, {len(prompt)} entries in JSONL prompt.")
    image_data = [
        {
            "prompt": prompt[i],
            "bugs": bugs[i]
        }
        for i in range(len(images))
    ]
    return image_data


def process_case(i, image_data, template):
    result_row = {}  # 为每个i初始化一个字典，用于存储该i的结果
    for j in range(len(image_data[i]["bugs"])):
        user_textprompt=f"Input: \n Original Prompt: \n {image_data[i]['prompt']} \n Identified Errors: \n {image_data[i]['bugs'][str(j+1)]} \n "
        textprompt= f"{' '.join(template)} \n {user_textprompt}"
        gen_text = client.request_gpt(textprompt)
        while gen_text is None:
            gen_text = client.request_gpt(textprompt)
        sentences = [sentence.strip() for sentence in gen_text.split('\n') if sentence.strip()]
        result = {str(i + 1): sentence for i, sentence in enumerate(sentences)}
    # 将每个j的结果添加到result_row中
        result_row[str(j + 1)] = result
        # print(f"==============={i}-{j}-result===============")
        # print(gen_text)
    return i, result_row  # 返回索引和结果

def process_wrapper(i):
    _, result_row = process_case(i, image_data, template)
    return json.dumps(result_row, ensure_ascii=False)  # 返回 JSON 格式的字符串

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="error mapping")
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input file') 
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output file') 
    parser.add_argument('--api_key', type=str, required=True, help='Api_key for API request')
    parser.add_argument('--url', type=str, required=True, help='Base_url for API request')
    parser.add_argument('--api_model', type=str, required=True, help='Model for API request')
    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    api_key = args.api_key
    url = args.url
    api_model = args.api_model
    client = APIClient(api_key ,url, api_model)

    # 定义文件夹和JSONL文件路径
    image_folder = f"{input_folder}/ori_img"
    jsonl_caption = f'{output_folder}/errors_reformed.jsonl'
    ori_prompt = f"{input_folder}/original_prompts.txt"
    output_file = f'{output_folder}/find_error.jsonl'

    with open('prompts/find_bug.txt', 'r', encoding='utf-8') as f:
        template=f.readlines()
    
    image_data = match_image_with_text(image_folder, jsonl_caption, ori_prompt)
    # print(image_data[0]['prompt'])
    output_data = [None] * len(image_data)  
    with open(output_file, 'w', encoding='utf-8') as out:
        with ThreadPoolExecutor(max_workers=20) as executor:
            # 使用 executor.map() 并行执行任务，并保证按索引顺序输出
            for json_str in tqdm(executor.map(process_wrapper, range(len(image_data))), total=len(image_data)):
                out.write(json_str + "\n")  # 直接写入 JSONL
                out.flush()  # 立即写入磁盘
    print(f"生成的 JSONL 文件已保存到 {output_file}")
