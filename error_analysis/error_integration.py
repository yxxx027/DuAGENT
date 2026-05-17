

import json
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
from utils_all.loading import logger_print_txt,load_json,load_txt,load_jsonl,get_image_paths
from utils_all.api import APIClient

def match_image_with_text(image_folder, caption_file, prompt_file,ori_pro):
    captions = load_jsonl(caption_file)
    prompt = load_jsonl(prompt_file)
    images = get_image_paths(image_folder)
    ori_pro = load_txt(ori_pro)
    # 验证匹配并打印结果
    if len(images) == len(captions) == len(prompt) == len(ori_pro):
        print("Images and JSONL files are correctly matched by length.")
    else:
        print(f"Mismatch: {len(images)} images, {len(captions)} entries in JSONL caption, {len(prompt)} entries in JSONL prompt.")


    image_data = [
        {
            "img_type" : "image/png",
            "image": images[i],
            "prompt": prompt[i],
            "caption": captions[i],
            "ori_pro": ori_pro[i]
        }
        for i in range(len(images))
    ]
    return image_data


def filter_text(text):
    if "error" in text or "ERROR" in text or "Error" in text or "wrong" in text or "Wrong" in text:
            return text.replace("YES","").replace("\n","")
    else:
        return "None"
    
def process_image_data(i, image_data, template):
    result_row = {}  # 为每个i初始化一个字典，用于存储该i的结果
    for j in range(len(image_data[i]["caption"])):
        # print(i,image_data[i]['ori_pro'])
        prompt_filter = filter_text(image_data[i]['prompt'][str(j+1)])
        caption_filter = filter_text(image_data[i]['caption'][str(j+1)])
        user_textprompt=f"Input: \n Text 1: \n {prompt_filter} \n Text 2: \n {caption_filter} \n Original full prompt: \n {image_data[i]['ori_pro']} \n"
        textprompt= f"{' '.join(template)} \n {user_textprompt}"
        gen_text = client.request_gpt(textprompt)
        sentences = [sentence.strip() for sentence in gen_text.split('\n') if sentence.strip()]
        result = {str(i + 1): sentence for i, sentence in enumerate(sentences)}
    # 将每个j的结果添加到result_row中
        result_row[str(j + 1)] = result
        print(f"==============={i}-{j}-result===============")
        # # print(image_data[i]['ori_pro'])
        print(gen_text)
    return i, result_row  # 返回索引和结果

def process_wrapper(i):
    k, result_row = process_image_data(i, image_data, template)
    return json.dumps(result_row, ensure_ascii=False)  # 返回 JSON 字符串，避免在多线程中直接写文件

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="error integration")
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

    # 文件路径
    image_folder = f"{input_folder}/ori_img"
    jsonl_caption = f'{output_folder}/check_captions_reformed.jsonl'
    jsonl_prompt = f"{output_folder}/check_image_qa_answer.jsonl"
    output_file = f'{output_folder}/errors.jsonl'
    ori_prompt = f"{input_folder}/original_prompts.txt"
    with open('prompts/combine.txt', 'r', encoding='utf-8') as f:
        template=f.readlines()
    
    image_data = match_image_with_text(image_folder, jsonl_caption, jsonl_prompt,ori_prompt)
    output_data = [None] * len(image_data)

    with open(output_file, 'w', encoding='utf-8') as out:
        with ThreadPoolExecutor() as executor:
            for json_str in tqdm(executor.map(process_wrapper, range(len(image_data))), total=len(image_data)):
                out.write(json_str + "\n")  # 写入 JSONL 格式数据
                out.flush()  # 立即写入磁盘
    print(f"生成的 JSONL 文件已保存到 {output_file}")
