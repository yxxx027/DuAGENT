import json
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="question generation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file') 
    args = parser.parse_args()
# 定义输入和输出文件路径
    input_file = args.input_file
    output_file = input_file.replace(".jsonl","") + "_reformed.jsonl"  # 替换为你的输出文件路径

    # 打开输入文件并处理
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析每行 JSON
            data = json.loads(line.strip())
            
            # 新的合并结构
            merged_data = {}
            
            # 遍历每个键值对
            for key, value in data.items():
                # 在每个值之间添加换行符
                if type(value) == str:
                    merged_data[key] = value
                    continue
                if len(value) == 1:
                    merged_data[key] = value["1"]
                    continue
                str_tmp = ""
                # print(value)
                # print(len(value))
                for j in range(len(value)):
                    str_tmp += value[str(j+1)]
                    str_tmp += "\n"
                # 更新新数据结构
                merged_data[key] = str_tmp
            
            # 写入到输出文件
            outfile.write(json.dumps(merged_data, ensure_ascii=False) + "\n")

    print(f"处理完成，结果保存在 {output_file}")
