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
from tqdm import tqdm
# 加载JSONL文件内容
from utils_all.loading import logger_print_txt,load_json,load_txt,load_jsonl,get_image_paths,encode_image
from utils_all.api import APIClient
import argparse

def match_image_with_text(image_folder, caption_file, prompt_file):
    question = load_jsonl(caption_file)
    prompt = load_jsonl(prompt_file)
    images = get_image_paths(image_folder)

    # 验证匹配并打印结果
    if len(images) == len(question) == len(prompt):
        print("Images and JSONL files are correctly matched by length.")
    else:
        print(f"Mismatch: {len(images)} images, {len(question)} entries in JSONL caption, {len(prompt)} entries in JSONL prompt.")

    # 示例：关联图像和JSONL内容
    image_data = [
        {
            "img_type" : "image/png",
            "image": images[i],
            "prompt": prompt[i],
            "question": question[i]
        }
        for i in range(len(images))
    ]
    return image_data

def process_image_data(i, image_data, client, template):
    result_row = {}  # 为每个i初始化一个字典，用于存储该i的结果
    for j in range(len(image_data[i]["prompt"])):
        gen_text_li = ""
        for k in range(len(image_data[i]["question"][str(j+1)])):
            if "YES" in image_data[i]["question"][str(j+1)][str(k+1)] or "Yes" in image_data[i]["question"][str(j+1)][str(k+1)] or "yes" in image_data[i]["question"][str(j+1)][str(k+1)]:
                continue
            user_textprompt=f"Input: \n Breakdown prompt: \n {image_data[i]['prompt'][str(j+1)]} \n Question: {image_data[i]['question'][str(j+1)][str(k+1)]} \n "

            textprompt= f"{' '.join(template)} \n {user_textprompt}"

            img_type = image_data[i]['img_type']
            img_b64_str = encode_image(image_data[i]['image'])
            gen_text = client.request_gpt_with_image(textprompt, img_type, img_b64_str)
            gen_text_li += gen_text
        result_row[str(j + 1)] = gen_text_li
        print(f"==============={i}-{j}-result===============")
        print(gen_text_li)
    return i, result_row

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="qa")
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

    image_folder = f"{input_folder}/ori_img"
    jsonl_question = f'{output_folder}/questions.jsonl'
    ori_prompt = f"{output_folder}/decomposed_prompt_reformed.jsonl"
    output_file = f'{output_folder}/check_image_qa_answer.jsonl'

    with open('prompts/qa.txt', 'r', encoding='utf-8') as f:
        template=f.readlines()
    print(jsonl_question)
    image_data = match_image_with_text(image_folder, jsonl_question, ori_prompt)
    # 创建 ThreadPoolExecutor 来进行高并发处理
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        output_data = [None] * len(image_data)  # 用于存储结果，并确保顺序一致

        for i in tqdm(range(len(image_data)), desc="Processing Images"):
            futures.append(executor.submit(process_image_data, i, image_data, client, template))

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
