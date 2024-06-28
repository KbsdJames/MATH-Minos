#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Zheng Yuan and Hongyi Yuan

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import wandb
import pdb
import json
import os
import random
import numpy as np
from tqdm import tqdm

# wandb.login()
# wandb.init(mode='online', project='textual_orm_training')

def f1_micro(y_true, y_pred):
    # 将y_true和y_pred转换为列表以防传入的是NDArray
    y_true = list(y_true)
    y_pred = list(y_pred)

    # 确保y_true和y_pred有相同数量的数据点
    assert len(y_true) == len(y_pred), "The number of predictions must match the number of labels"
    
    tp = 0
    fp = 0
    fn = 0
    
    # 遍历所有的类别
    unique_classes = set(y_true).union(set(y_pred))
    for cls in unique_classes:
        for t, p in zip(y_true, y_pred):
            if t == cls and p == cls:
                tp += 1
            elif t != cls and p == cls:
                fp += 1
            elif t == cls and p != cls:
                fn += 1
    
    precision_micro = tp / (tp + fp) if tp + fp != 0 else 0
    recall_micro = tp / (tp + fn) if tp + fn != 0 else 0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) != 0 else 0
    
    return f1_micro

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": [],
    "prompt_no_input": (
        "### Instruction:\n"
        "You are given answer A and question Q. Please evaluate whether the given answer A correctly solve the question Q by giving a step-by-step evaluation, and judge whether the final answer is correct.\n"
        # "Write a response that appropriately completes the request.\n\n"
        "\n{query}\n\n"
        "### Response:"
        # "Please evaluate whether the given answer A correctly solve the question Q by giving a step-by-step evaluation, and judge whether the final answer is correct.\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the evaluation data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# class CustomTrainer(Trainer):
#     def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
#         # 调用父类的评估方法获取评估结果
#         eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
#         if self.is_world_process_zero():
#             # 获取评估数据集
            
#             eval_dataloader = self.get_eval_dataloader(eval_dataset)
            
#             # 下面演示如何保存模型的生成输出
#             self.model.eval()  # 确保模型在评估模式
#             outputs_texts = []
#             total_label = []
#             for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
#                 inputs = self._prepare_inputs(batch)
#                 print(inputs.keys())
#                 # process labels
#                 labels = batch['labels']
#                 decoded_labels = []
#                 for label_sequence in labels:
#                     # 过滤出有效的token IDs (大于等于0的值)
#                     filtered_label_sequence = [label for label in label_sequence if label >= 0]
#                     # 使用tokenizer.decode解码有效的token IDs
#                     decoded_sequence = self.tokenizer.decode(filtered_label_sequence, skip_special_tokens=True)
#                     # 将解码的文本添加到结果列表中
#                     decoded_labels.append(decoded_sequence)
#                 total_label.extend(decoded_labels)
                
#                 with torch.no_grad():  # 不跟踪梯度
#                     # inputs = {k: v.to(self.args.device) for k, v in batch.items()}
#                     outputs = self.model.generate(**inputs, max_new_tokens=512)
#                     # 解码并保存文本
#                     for output in outputs:
#                         text = self.tokenizer.decode(output, skip_special_tokens=True)
#                         text = text.split('### Response:')[-1]
#                         outputs_texts.append(text)
            
#             # 保存输出文本到文件（这里简单打印示例）
#             assert len(outputs_texts) == len(total_label)

#             ## test
#             y_pred_lst = []
#             ans_lst = []
#             for js_c, js_a in zip(outputs_texts, total_label):
#                 y_pred = js_c.split('final answer is ')[-1]
#                 ans = js_a
#                 ans_lst.append(ans)

#                 if 'True' in y_pred:
#                     y_pred = 'True'
#                 else:
#                     y_pred = 'False'
                
#                 y_pred_lst.append(y_pred)

#             f1_micro_score = f1_micro(ans_lst, y_pred_lst)
#             eval_result['f1_micro'] = f1_micro_score
#             print(eval_result)
#         return eval_result

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]
        
        def preprocess(list_data_dict):
            for js in list_data_dict:
                js['query'] = js['query'].replace('Human:', 'Q:').replace('Assistant:', "\nA:")

            random.shuffle(list_data_dict)
            return list_data_dict

        list_data_dict = preprocess(list_data_dict)
        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        # if 'instruction' in list_data_dict[0]:
        #     pass
        # else:
        #     def get_input(query):
        #         if query.find('\n') == -1:
        #             return ''
        #         return '\n'.join(query.split('\n')[1:])
        # pdb.set_trace()
        list_data_dict = [{'query':data['query'], 'output':data['response']} for data in list_data_dict]
        
        # import ipdb; ipdb.set_trace()
        sources = [
            prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

        # pdb.set_trace()
        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])


class SupervisedEvalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedEvalDataset, self).__init__()
        logging.warning("Loading data...")
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]
        
        # list_data_dict = list_data_dict[:10]

        sources = [
            example['query'] for example in list_data_dict
        ]

        targets = [f"{example['label']['acc']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    # eval_dataset = SupervisedEvalDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    # pdb.set_trace()
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # else:
    #     last_checkpoint = None
    # print(last_checkpoint)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    ).to(torch.bfloat16)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()