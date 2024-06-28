import argparse
import json
import re
import jsonlines
from fraction import Fraction
# from vllm import LLM, SamplingParams
import sys
import os
import pdb

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"
MAX_SEED = 128
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def find_most_frequent_answers(answer_list):
    # 确保输入列表的尺寸是MAX_SEED x 1319
    assert len(answer_list) == MAX_SEED and all(len(sublist) == 1319 for sublist in answer_list)
  
    # 初始化最终答案列表
    final_answers = [None] * 1319
    
    # 遍历所有问题
    for question_idx in range(1319):
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

def find_most_frequent_answers_with_filter(answer_list, pred_list):
    # 确保输入列表的尺寸是MAX_SEED x 1319
    assert len(answer_list) == MAX_SEED and all(len(sublist) == 1319 for sublist in answer_list)
    assert len(pred_list) == MAX_SEED and all(len(sublist) == 1319 for sublist in pred_list)
  
    # 初始化最终答案列表
    final_answers = [None] * 1319
    
    # 遍历所有问题
    for question_idx in range(1319):
        # 用于存储每个有效答案出现的次数，[invalid]答案不计入
        answer_frequency = {}
        
        # 遍历每一个答案
        for answer_idx in range(MAX_SEED):
            answer = answer_list[answer_idx][question_idx]
            # 忽略[invalid]答案
            if answer != "[invalid]" and pred_list[answer_idx][question_idx] == 'True':
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
    # 确保输入列表的尺寸是MAX_SEED x 1319
    assert len(answer_list) == MAX_SEED and all(len(sublist) == 1319 for sublist in answer_list)
  
    # 初始化最终答案列表
    final_answers = [None] * 1319
    
    # 遍历所有问题
    for question_idx in range(1319):
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


def outcome_reward_model_result(path_list, gsm8k_answers):
    total_ans = []
    total_reward = {}
    total_response = []
    ans_len = len(gsm8k_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        file_response = []
        with open(file_path) as f:
            for line_idx, line in enumerate(f.readlines()[:ans_len]):
                n = extract_answer_number(eval(line)[0][1])
                file_response.append(eval(line)[0][1])
                reward = eval(line)[1][0]
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(gsm8k_answers)
        assert len(file_response) == len(ans)
        assert len(total_reward[file_path_idx]) == len(ans)
        total_ans.append(ans) 
        total_response.append(file_response)

    # 假设 total_ans 和 total_reward 已经被正确定义和赋值
    # total_ans 是一个包含MAX_SEED个子列表的列表，每个子列表包含1319个字符串
    # total_reward 是一个包含MAX_SEED个子列表的列表，每个子列表包含1319个浮点数

    # 初始化一个长度为1319的列表，
    # 用于存储每个位置最大分数对应的答案
    final_ans = [''] * 1319

    # 初始化一个长度为1319的列表，
    # 用于存储每个位置的最大分数，初始化为非常小的值
    max_scores = [-float('inf')] * 1319
    idx_j = ['x'] * 1319
    invalid_response = []
    # 遍历每一个位置
    for i in range(1319):
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
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, gsm8k_answers)):
        # if y_pred == INVALID_ANS:
            # invalid_idx += 1
            # continue
        if y_pred != '' and y_pred != None and y_pred != INVALID_ANS:
            try:
                result.append(float(y_pred) == float(prompt_answer))
            except:
                pdb.set_trace()
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)

    print(invalid_idx)
    return acc

def self_consistency_and_reward(path_list, gsm8k_answers):
    # TODO
    total_ans = []
    total_reward = {}
    total_response = []
    ans_len = len(gsm8k_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        file_response = []
        with open(file_path) as f:
            for line_idx, line in enumerate(f.readlines()[:ans_len]):
                n = extract_answer_number(eval(line)[0][1])
                file_response.append(eval(line)[0][1])
                reward = eval(line)[1][0]
                if file_path_idx in total_reward.keys():
                    total_reward[file_path_idx].append(reward)
                else:
                    total_reward[file_path_idx] = [reward]
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(gsm8k_answers)
        assert len(file_response) == len(ans)
        assert len(total_reward[file_path_idx]) == len(ans)
        total_ans.append(ans) 
        total_response.append(file_response)
    

    final_ans = find_most_frequent_answers_with_reward(total_ans, total_reward)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, gsm8k_answers)):
        # doc = {'question': prompt}
        # y_pred = extract_answer_number(completion)
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    # pdb.set_trace()
    return acc

def text_rm_filter(path_list, gsm8k_answers, preds_path_list):
    total_preds = []
    total_ans = []
    ans_len = len(gsm8k_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                n = extract_answer_number(eval(line)[0][1])
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(gsm8k_answers)
        total_ans.append(ans)

    for file_path_idx in range(len(preds_path_list)):
        file_path = preds_path_list[file_path_idx]

        preds = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                preds_text = eval(line)[0][1]
                try:
                    pred_text = preds_text.split('final answer is')[1]
                    if 'True' in pred_text:
                        pred = 'True'
                    else:
                        pred = 'False'
                except:
                    pred = 'False'
                
                preds.append(pred)
        assert len(preds) == ans_len
        total_preds.append(preds)

    # pdb.set_trace()
    final_ans = find_most_frequent_answers_with_filter(total_ans, total_preds)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, gsm8k_answers)):
        # doc = {'question': prompt}
        # y_pred = extract_answer_number(completion)
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'output': y_pred, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    return acc

def self_consistency(path_list, gsm8k_answers):
    total_ans = []
    ans_len = len(gsm8k_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                n = extract_answer_number(eval(line)[0][1])
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(gsm8k_answers)
        total_ans.append(ans)

    final_ans = find_most_frequent_answers(total_ans)

    result = []
    invalid_outputs = []
    for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, gsm8k_answers)):
        # doc = {'question': prompt}
        # y_pred = extract_answer_number(completion)
        if y_pred != None and y_pred != INVALID_ANS:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    return acc


def eval_all(path_list, gsm8k_answers):
    total_ans = []
    ans_len = len(gsm8k_answers)
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]

        ans = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                n = extract_answer_number(eval(line)[0][1])
                ans.append(INVALID_ANS if n == None else n)
        assert len(ans) == len(gsm8k_answers)
        total_ans.append(ans)
    
    for file_idx, final_ans in enumerate(total_ans):
        result_json = []
        result = []
        invalid_outputs = []
        for idx, (y_pred, prompt_answer) in enumerate(zip(final_ans, gsm8k_answers)):
            if y_pred != None and y_pred != INVALID_ANS:
                result.append(float(y_pred) == float(prompt_answer))
                result_json.append({'y_pred': y_pred, 'acc': str(float(y_pred) == float(prompt_answer))})
            else:
                result.append(False)
                temp = {'output': y_pred, 'answer': prompt_answer}
                result_json.append({'y_pred': y_pred, 'acc': 'False'})
                invalid_outputs.append(temp)
        acc = sum(result) / len(result)

        file = 'raw_generation_0.7_{}.json'.format(file_idx + 1)
        with open('/path/to/your/result/dir'+file, 'w') as f:
            json.dump(result_json,f)

        print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
        print('gsm8k length====', len(result), ', gsm8k acc====', acc)

    return acc

def gsm8k_test(data_path, result_dir, max_seed, mode, filter_path = None):
    gsm8k_answers = []
    gsm8k_ins = []
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = item["query"]
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    print('lenght ====', len(gsm8k_answers))

    folder = result_dir
    print(folder)
    path_list = [f'{folder}/raw_generation_0.7_{idx}.json' for idx in range(1,max_seed+1,1)]
    path_list = [f for f in path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(path_list)}')


    if filter_path:
        filter_path_list = [f'{filter_path}/{idx}.json' for idx in range(1,max_seed+1,1)]
        filter_path_list = [f for f in filter_path_list if os.path.exists(f)]

    if mode == 'self-consistency':
        acc = self_consistency(path_list, gsm8k_answers)
    elif mode == 'ensemble':
        acc = self_consistency_and_reward(path_list, gsm8k_answers)
    elif mode == 'text_rm':
        acc = text_rm_filter(path_list, gsm8k_answers, filter_path_list)
    else:
        acc = outcome_reward_model_result(path_list, gsm8k_answers)




if __name__ == '__main__':
    gsm8k_test('/test_set_path.jsonl', '/path/to/your/result/dir', MAX_SEED, 'rm')