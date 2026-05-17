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
from utils_all.loading import logger_print_txt,load_json,load_txt,load_jsonl,get_image_paths,encode_image
from utils_all.api import APIClient
import argparse


def match_image_with_text(image_folder,  prompt_file):
    # captions = load_jsonl(caption_file)
    prompt = load_jsonl(prompt_file)
    images = get_image_paths(image_folder)
    image_data = [
        {
            "img_type" : "image/png",
            "image": images[i],
            "prompt": prompt[i],
        }
        for i in range(len(images))
    ]
    return image_data

def process_image_data(i, image_data, template):
    result_row = {}  # 为每个i初始化一个字典，用于存储该i的结果
    for j in range(len(image_data[i]["prompt"])):
        # print(j)
        user_textprompt=f"Input: \n Prompt breakdown \n {image_data[i]['prompt'][str(j+1)]} \n "
        textprompt= f"{' '.join(template)} \n {user_textprompt}"
        img_type = image_data[i]['img_type']
        img_b64_str = encode_image(image_data[i]['image'])
        gen_text = client.request_gpt_with_image(textprompt, img_type, img_b64_str)
        sentences = [sentence.strip() for sentence in gen_text.split('\n') if sentence.strip()]
        result = {str(i + 1): sentence for i, sentence in enumerate(sentences)}
    # 将每个j的结果添加到result_row中
        result_row[str(j + 1)] = result
        print(f"==============={i}-{j}-result===============")
        print(gen_text)
    return i,result_row

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="question generation")
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
    jsonl_prompt = f"{output_folder}/decomposed_prompt_reformed.jsonl"
    output_file = f'{output_folder}/questions.jsonl'

    with open('prompts/ques.txt', 'r', encoding='utf-8') as f:
        template=f.readlines()
    
    image_data = match_image_with_text(image_folder, jsonl_prompt)
    output_data = []
    # print(image_data[0]['prompt'])
# 使用 ThreadPoolExecutor 并行处理最外层的i
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        output_data = [None] * len(image_data)  # 用于存储结果，保持顺序不变

        # 提交每个任务
        for i in range(len(image_data)):
            futures.append(executor.submit(process_image_data, i, image_data, template))
        # 获取并处理结果，确保结果按i的顺序排列
        for future in concurrent.futures.as_completed(futures):
            i, result_row = future.result()
            output_data[i] = result_row
            print(f"=================================={i} result==================================")
    # 将结果写入 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"生成的 JSONL 文件已保存到 {output_file}")
