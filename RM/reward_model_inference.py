import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from typing import Optional, Dict, Sequence
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import pdb
import os
import re
import json
import argparse
import torch.nn.functional as F

def prepare_model_input(instruction, response, tokenizer):
    template = "Human: {q}\nAssistant: {r}"
    # if gsm8k
    assistance_response = re.sub(r'\n?#### \d+', '', response)
    inputs = template.format(q=instruction, r=assistance_response)
    tokenized_inputs = tokenizer(inputs, return_tensors='pt')
    return tokenized_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_new", type=str, required=True, help="The new folder path to save the results.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The new folder path to save the results.")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    folder = 'path/to/generator_result'
    folder_new = args.folder_new
    max_seed = 256
    ans_len = 1319

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=1
    ).to(torch.bfloat16).to('cuda')

    # 假设你的reward_model已经加载并设置为evaluation模式
    reward_model.to(device)
    reward_model.eval()

    test_set = []
    with open('path/to/test_gsm8k.jsonl') as f:
        for line in f.readlines():
            test_set.append(json.loads(line))


    print(folder)
    path_list = [f'{folder}/raw_generation_0.7_{idx}.json' for idx in range(1,max_seed+1,1)]
    path_list = [f for f in path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(path_list)}')
    for file_path_idx in range(len(path_list)):
        print("processing: {}".format(file_path_idx))
        file_path = path_list[file_path_idx]
    
        data_per_seed = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                data_per_seed.append(json.loads(line))

        assert len(data_per_seed) == len(test_set)

        writer = open(f'{folder_new}/raw_generation_0.7_{file_path_idx+1}.json', 'a+')

        for idx, data in tqdm(enumerate(data_per_seed), total=ans_len):
            # 准备一个空列表，稍后用来保存带有reward的数据
            data_with_rewards = []

            item = data[0]
            instruction, response = item[0], item[1]
            instruction = test_set[idx]['query']
            # 将输入数据转换为适合模型的格式
            # 注意：根据你的模型输入需求，你可能需要对数据进行适当的预处理
            input_data = prepare_model_input(instruction, response, tokenizer)  # 这是一个虚构的函数
            input_data.to(device)
            inputs = input_data
            # pdb.set_trace()
            # 使用reward_model计算reward inference
            with torch.no_grad():
                reward = reward_model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],return_dict=True,)["logits"]
            
            # 将reward接输出成scalar值。如果reward本身就是一个scalar，这一步可以省略。
            reward = F.sigmoid(input=torch.FloatTensor([reward.item()])).item()
            # 将原始数据和计算出的reward保存到列表
            data_with_rewards.append(item)
            data_with_rewards.append([reward])
            writer.write(str(data_with_rewards)+'\n')
        
        writer.close()
