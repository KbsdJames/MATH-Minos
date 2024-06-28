import argparse
import json
import re
import jsonlines
from fraction import Fraction
# from vllm import LLM, SamplingParams
import sys
import os
import pdb
import util

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"
MAX_SEED = 256

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def extract_answer(completion):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        try:
            ss1 = util.strip_string(extract_ans)
            return ss1
        except Exception:
            return extract_ans
    else:
        return INVALID_ANS

def find_most_frequent_answers(answer_list):
    # 确保输入列表的尺寸是MAX_SEED x 500
    assert len(answer_list) == MAX_SEED and all(len(sublist) == 500 for sublist in answer_list)
  
    # 初始化最终答案列表
    final_answers = [None] * 500
    
    # 遍历所有问题
    for question_idx in range(500):
        # 用于存储每个有效答案出现的次数，[invalid]答案不计入
        answer_frequency = {}
        
        # 遍历每一个答案
        for answer_idx in range(MAX_SEED):
            answer = answer_list[answer_idx][question_idx]
            # 忽略[invalid]答案
            if answer != "[invalid]":
                if answer in answer_frequency:
                    answer_frequency[answer] += 1
                else:
                    answer_frequency[answer] = 1

        # 判断这道题的答案是否全为[invalid]
        if not answer_frequency:
            final_answers[question_idx] = "[invalid]"
        else:
            # 选出出现次数最多的答案
            final_answers[question_idx] = max(answer_frequency, key=answer_frequency.get)
    
    return final_answers

def find_most_frequent_answers_with_reward(answer_list, total_reward):
    # pdb.set_trace()
    # 确保输入列表的尺寸是MAX_SEED x 500
    assert len(answer_list) == MAX_SEED and all(len(sublist) == 500 for sublist in answer_list)
  
    # 初始化最终答案列表
    final_answers = [None] * 500
    
    # 遍历所有问题
    for question_idx in range(500):
        # 用于存储每个有效答案出现的次数，[invalid]答案不计入
        answer_frequency = {}
        answer_accu_reward = {}
        
        # 遍历每一个答案
        for answer_idx in range(MAX_SEED):
            answer = answer_list[answer_idx][question_idx]
            # 忽略[invalid]答案
            if answer != "[invalid]":
                if answer in answer_accu_reward:
                    answer_accu_reward[answer] += total_reward[answer_idx][question_idx]
                else:
                    answer_accu_reward[answer] = total_reward[answer_idx][question_idx]
        
        # 判断这道题的答案是否全为[invalid]
        if not answer_accu_reward:
            final_answers[question_idx] = "[invalid]"
        else:
            # 选出出现次数最多的答案
            final_answers[question_idx] = max(answer_accu_reward, key=answer_accu_reward.get)
    
    # pdb.set_trace()
    return final_answers

def outcome_reward_model_result(path_list, MATH_answers):
    total_ans = []
    total_reward = {}
    total_response = []
    ans_len = len(MATH_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        file_response = []
        with open(file_path) as f:
            for line_idx, line in enumerate(f.readlines()[:ans_len]):
                n = extract_answer(eval(line)[0][1])
                file_response.append(eval(line)[0][1])
                reward = eval(line)[1][0]
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(MATH_answers)
        assert len(file_response) == len(ans)
        assert len(total_reward[file_path_idx]) == len(ans)
        total_ans.append(ans) 
        total_response.append(file_response)

    # 初始化一个长度为500的列表，
    # 用于存储每个位置最大分数对应的答案
    final_ans = [''] * 500

    # 初始化一个长度为500的列表，
    # 用于存储每个位置的最大分数，初始化为非常小的值
    max_scores = [-float('inf')] * 500
    idx_j = ['x'] * 500
    invalid_response = []
    # 遍历每一个位置
    for i in range(500):
        # 遍历所有MAX_SEED个子列表
        for j in range(MAX_SEED):
            # 检查当前位置的分数是否大于已记录的最大分数
            if total_reward[j][i] > max_scores[i] and total_ans[j][i] != INVALID_ANS:
                # 更新最大分数
                max_scores[i] = total_reward[j][i]
                # 更新对应的答案
                final_ans[i] = total_ans[j][i]
                idx_j[i] = j

    # max_score_ans 现在包含了每个位置得分最高的答案
    result = []
    invalid_outputs = []
    invalid_idx = 0
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        # if y_pred == INVALID_ANS:
        #     invalid_idx += 1
        #     continue
        if y_pred != None and y_pred != INVALID_ANS:
             result.append(util.is_equiv(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('MATH length====', len(result), ', MATH acc====', acc)

    print(invalid_idx)
    return acc

def self_consistency_and_reward(path_list, MATH_answers):
    # TODO
    total_ans = []
    total_reward = {}
    total_response = []
    ans_len = len(MATH_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        file_response = []
        with open(file_path) as f:
            for line_idx, line in enumerate(f.readlines()[:ans_len]):
                n = extract_answer(eval(line)[0][1])
                file_response.append(eval(line)[0][1])
                reward = eval(line)[1][0]
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(MATH_answers)
        assert len(file_response) == len(ans)
        assert len(total_reward[file_path_idx]) == len(ans)
        total_ans.append(ans) 
        total_response.append(file_response)
    

    final_ans = find_most_frequent_answers_with_reward(total_ans, total_reward)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(util.is_equiv(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('MATH length====', len(result), ', MATH acc====', acc)
    return acc


def self_consistency(path_list, MATH_answers):
    total_ans = []
    ans_len = len(MATH_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                n = extract_answer(eval(line)[0][1])
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(MATH_answers)
        total_ans.append(ans)

    final_ans = find_most_frequent_answers(total_ans)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, MATH_answers)):
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(util.is_equiv(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('MATH length====', len(result), ', MATH acc====', acc)
    return acc

def MATH_test(data_path, result_dir, max_seed, mode):
    MATH_answers = []
    MATH_responses = []
    MATH_ins = []
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = item["query"]
            MATH_ins.append(temp_instr)
            temp_res = item['response']
            temp_ans = item['answer']
            MATH_responses.append(temp_res)
            MATH_answers.append(temp_ans)

    print('lenght ====', len(MATH_answers))

    folder = result_dir
    print(folder)
    path_list = [f'{folder}/raw_generation_0.7_{idx}.json' for idx in range(1,max_seed+1,1)]
    path_list = [f for f in path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(path_list)}')

    if mode == 'self-consistency':
        acc = self_consistency(path_list, MATH_answers)
    elif mode == 'ensemble':
        acc = self_consistency_and_reward(path_list, MATH_answers)
    else:
        acc = outcome_reward_model_result(path_list, MATH_answers)

    return acc




if __name__ == '__main__':
    MATH_test('/test_set_path.jsonl', '/path/to/your/result/dir', MAX_SEED, 'rm')
    