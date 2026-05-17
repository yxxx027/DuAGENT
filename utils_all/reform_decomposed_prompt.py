import json
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="question generation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file') 
    args = parser.parse_args()
    # 读取原始 jsonl 文件
    input_file = args.input_file
    output_file = input_file.replace(".jsonl","") + "_reformed.jsonl"  # 替换为你的输出文件路径

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析 JSON
            data = json.loads(line.strip())
            
            if data.get("1") == "Answer:":
                # 删除"1": "Answer:" 键值对
                data.pop("1")
                
                # 重新编码键值对，从 "1" 开始
                new_data = {}
                for i, (key, value) in enumerate(data.items(), start=1):
                    new_data[str(i)] = value
                
                # 写入修改后的内容
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            else:
                # 如果不是"Answer:"，直接写入原数据
                outfile.write(line)
