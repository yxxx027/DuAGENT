# from gpt_proxy import OpenAIApiProxy
import json
import argparse
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from utils_all.api import APIClient

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="decompose prompt")
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
    # prompt = "The metallic picture frame and glass stand display the rectangular photo on the black table."
    # 文件路径
    prompt_file = f"{input_folder}/original_prompts.txt"

    # 逐行读取 prompts 并处理
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]  # 去除空行

    output_data = []
    with open('prompts/decom.txt', 'r', encoding='utf-8') as f:
        template=f.readlines()
    # for i in range(len(prompts)):
    for i in tqdm(range(len(prompts)), desc="Processing Decomposing Prompts"):
        user_textprompt=f"Input: {prompts[i]}"
        textprompt= f"{' '.join(template)} \n {user_textprompt}"
        gen_text = client.request_gpt(textprompt)
        # 分句处理生成的文本 (假设句子以句号或换行符结束)
        sentences = [sentence.strip() for sentence in gen_text.split('\n') if sentence.strip()]
        result = {str(i + 1): sentence for i, sentence in enumerate(sentences)}
        # 将结果存储到 JSON 行
        output_data.append(result)
    # 将结果写入 JSONL 文件
    with open(f"{output_folder}/decomposed_prompt.jsonl", 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"生成的 JSONL 文件已保存到 {output_folder}/decomposed_prompt.jsonl")
