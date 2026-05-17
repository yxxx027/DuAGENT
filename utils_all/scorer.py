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
import functools
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def qa_check(client,questions,prompt,img,template):
    img_b64_str = encode_image(img)
    errors_all = ""
    ans_list = []
    for q in range(len(questions)):
        # print(f"==============={q}-{questions[str(q+1)]}==============")
        qa_textprompt=f"Input: \n prompt: \n {prompt} \n Question: {questions[str(q+1)]} \n "
        qaprompt= f"{' '.join(template)} \n {qa_textprompt}"
        ans = client.request_gpt_with_image(qaprompt, "image/png", img_b64_str)
        # print(f"==============={q}-{ans}==============")
        ans_list.append(ans)
        if "NO" in ans:
            errors_all += ans.replace("NO.","")
    return errors_all, ans_list

def qa_batch_check(client,questions,prompt,img_path,template):
    # with lock_b:
        error_list = []
        ans_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(qa_check, client,questions,prompt,img,template): img for img in img_path
            }

            # 等待所有线程完成并收集结果
            for future in concurrent.futures.as_completed(futures):
                qa_result, ans_result = future.result()  # 获取每个图像的处理结果

                ans_list.append(ans_result)
                if qa_result != "" and qa_result != " " and qa_result != "\n":
                    error_list.append("NO ERROR")  # 添加错误信息
                else:
                    error_list.append(qa_result)
                    # error_list.append(qa_result)  # 添加错误信息
        # for img in img_path:
        #     print(f"==============={img}==============")
        #     qa_result, ans_result = qa_check(questions,prompt,img,template)
        #     ans_list.append(ans_result)
        #     if qa_result != "" or qa_result != " " or qa_result != "\n":
        #         error_list.append(qa_check(questions,prompt,img,template))
        return error_list, ans_list

def rate_ori(client,prompt_before,error_before,propmpt_after,error_after,img,template):
    img_b64_str = encode_image(img)
    user_textprompt_rate=f"""Input: \n original prompt: \n {prompt_before} \n errors in round 1: \n {error_before} \n modified prompt: \n {propmpt_after} \n errors in round 2: \n {error_after} """
    # user_textprompt_rate=f"""Input: \n original prompt: \n {prompt_init_init} \n errors in round 1: \n {image_data[i]['errors'][str(j+1)]} \n modified prompt: \n {prompt_init} \n """
    textprompt_rate= f"{' '.join(template)} \n {user_textprompt_rate}"
    scores = client.request_gpt_with_image(textprompt_rate, "image/png", img_b64_str)
    # 将字典转化为JSON格式
    if scores is None:
        rate_result = None
        return rate_result
    scores_start = scores.find("```json")
    score_end = scores.rfind("}")
    if scores_start != -1 and score_end != -1:
        scores_post_process = scores[scores_start+len("```json"):score_end+1]
        # print("scores_post_process")
        # print(scores_post_process)
        # print("!!!!!!!!!!!!!!!!!!")
        try:
            rate_result = json.loads(scores_post_process)
        except Exception:
            rate_result = None
            print("score format wrong!!!!!!!!!")
    else:
        print(scores)
        print("WARNING!!!!!!!!!!!!!!!!!! score format wrong")
        rate_result = None
    return rate_result

def rate(client,prompt_before, error_before, propmpt_after, error_after, img, template, max_retries=5, retry_delay=2):
    """请求 GPT 评分，失败时重试"""
    img_b64_str = encode_image(img)
    user_textprompt_rate = f"""Input: \n original prompt: \n {prompt_before} \n errors in round 1: \n {error_before} \n modified prompt: \n {propmpt_after} \n errors in round 2: \n {error_after} """
    textprompt_rate = f"{' '.join(template)} \n {user_textprompt_rate}"

    for attempt in range(max_retries):
        scores = client.request_gpt_with_image(textprompt_rate, "image/png", img_b64_str)

        if scores is None:
            print(f"Attempt {attempt+1}: No response, retrying...")
            time.sleep(retry_delay)
            continue  # 重新请求

        scores_start = scores.find("```json")
        score_end = scores.rfind("}")

        if scores_start != -1 and score_end != -1:
            scores_post_process = scores[scores_start+len("```json"):score_end+1]
            try:
                rate_result = json.loads(scores_post_process)
                return rate_result  # 解析成功，直接返回
            except json.JSONDecodeError:
                print(f"Attempt {attempt+1}: JSON format error, retrying...")
                time.sleep(retry_delay)
                continue  # 解析失败，重新请求

        print(f"Attempt {attempt+1}: WARNING! refused answer, retrying...")
        time.sleep(retry_delay)

    print("Max retries reached, returning None")
    return None  # 达到最大重试次数，返回 None

def safe_to_int(val):
    """将值安全地转为 int，根据类型处理"""
    if isinstance(val, int):
        return val
    elif isinstance(val, str):
        return int(val)
    elif isinstance(val, list):
        if len(val) > 0:
            return int(val[0])
        else:
            return int(str(val))
    else:
        raise TypeError(f"无法转换类型为 {type(val)} 的值：{val}")
    
def avg_rate(client,rate_list,template):
    final_result = {
        "scores" : {
            "Attribute-Binding": [],
            "Object-Relationship": [],
            "Background-Consistency": []
        },
        "reasons" : {
            "Attribute-Binding": [],
            "Object-Relationship": [],
            "Background-Consistency": []
        },
        "score_average":{
            "Attribute-Binding": 0,
            "Object-Relationship": 0,
            "Background-Consistency": 0
        } ,
        "reasons-in-short":{
            "Attribute-Binding": "",
            "Object-Relationship": "",
            "Background-Consistency": ""
        }
    }

    for rate_i in rate_list:
        print(rate_i["scores"]["Attribute-Binding"])
        final_result["scores"]["Attribute-Binding"].append(safe_to_int(rate_i["scores"]["Attribute-Binding"]))
        final_result["scores"]["Object-Relationship"].append(safe_to_int(rate_i["scores"]["Object-Relationship"]))
        final_result["scores"]["Background-Consistency"].append(safe_to_int(rate_i["scores"]["Background-Consistency"]))
        final_result["reasons"]["Attribute-Binding"].append(rate_i["reasons"]["Attribute-Binding"])
        final_result["reasons"]["Object-Relationship"].append(rate_i["reasons"]["Object-Relationship"])
        final_result["reasons"]["Background-Consistency"].append(rate_i["reasons"]["Background-Consistency"])

    final_result["score_average"]["Attribute-Binding"] = mean(final_result['scores']['Attribute-Binding'])
    final_result["score_average"]["Object-Relationship"] = mean(final_result['scores']['Object-Relationship'])
    final_result["score_average"]["Background-Consistency"] = mean(final_result['scores']['Background-Consistency'])
    
    for i in range(len(final_result["reasons-in-short"])):
        text = ""
        if i == 0:
            text = "Attribute-Binding"
        elif i == 1:
            text = "Object-Relationship"
        else:
            text = "Background-Consistency"
        user_textprompt=f"""Input: \n Errors: \n {final_result["reasons"][str(text)]}  \n """
        textprompt= f"""{' '.join(template)} \n {user_textprompt}"""
        final_result["reasons-in-short"][str(text)] = client.request_gpt(textprompt)

    return final_result

def batch_rate(client,img_path,prompt_before,error_before,prompt_after,error_list,template,template_reason):
    # with lock_c:
        rate_list = [None] * len(img_path)  # 预先创建一个与 img_path 相同大小的空列表，用于存储结果

        def process_rate(i):
            # 对每个索引 i 执行 rate 操作
            
            rate_result = rate(client,prompt_before, error_before, prompt_after, error_list[i], img_path[i], template)
            count_num = 0
            while rate_result is None or count_num < 2:
                rate_result = rate(client,prompt_before, error_before, prompt_after, error_list[i], img_path[i], template)
                count_num += 1
            return i, rate_result  # 返回索引 i 和对应的结果

        # 使用 ThreadPoolExecutor 并发执行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(process_rate, i): i for i in range(len(img_path))}

            for future in concurrent.futures.as_completed(futures):
                i, rate_result = future.result()  # 获取每个任务的结果
                rate_list[i] = rate_result  # 将结果按原始顺序放入 rate_list 中
        return avg_rate(client,rate_list,template_reason)
    # rate_list = []
    # for i in range(len(img_path)):
    #     rate_result = rate(prompt_before,error_before,propmpt_after,error_list[i],img_path[i],template)
    #     count_num = 0
    #     while rate_result is None or count_num < 2:
    #         rate_result = rate(prompt_before,error_before,propmpt_after,error_list[i],img_path[i],template)
    #         count_num += 1
    #     rate_list.append(rate_result)

# 定义评分函数，衡量生成图像与 prompt 的匹配程度
def scoring_function(client,image_data,i,j,modified_prompt, img_list,prompt_origin,question_list,template_ques,template_rate,template_reason):
    # 这里简单模拟匹配程度，实际需要更复杂的图像分析
    # 假设 prompt 中的关键词与图像特征有某种关联
    error_list, ans_list = qa_batch_check(client,question_list,modified_prompt,img_list,template_ques)
    rate_result = batch_rate(client,img_list,prompt_origin,image_data[i]['errors'][str(j+1)],modified_prompt,error_list,template_rate,template_reason)
    # rate_sentence.append(rate_result["score_average"])

    return rate_result["score_average"]["Attribute-Binding"] + rate_result["score_average"]["Object-Relationship"] + rate_result["score_average"]["Background-Consistency"],rate_result

def find_max_score_index(rate_result):
    """
    找到rate_result列表中score_average最大的那个index。

    :param rate_result: 列表，每个元素是一个json数据格式。
    :return: 最大score_average的index。
    """
    def compare_scores(a, b):
        """
        比较两个score_average字典。
        返回True如果a > b，否则返回False。
        """
        keys = ["Attribute-Binding", "Object-Relationship", "Background-Consistency"]
        for key in keys:
            if a[key] > b[key]:
                return True
            elif a[key] < b[key]:
                return False
        return False  # 如果所有字段都相等，返回False

    max_index = 0
    if len(rate_result) ==0:
        return None
    # max_score = rate_result[0]["score_average"]
    max_score = rate_result[0]
    for i in range(1, len(rate_result)):
        # current_score = rate_result[i]["score_average"]
        current_score = rate_result[i]
        if compare_scores(current_score, max_score):
            max_index = i
            max_score = current_score

    return max_index

def update_history(sentence,final_result):
    history_avg = sentence + str(final_result["score_average"]) + str(final_result["reasons-in-short"])
    return history_avg
