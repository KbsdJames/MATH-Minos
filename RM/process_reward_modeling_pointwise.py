# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --eval_steps=500 \
    --max_length=512 \
"""
import warnings

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import transformers
from transformers.modeling_outputs import SequenceClassifierOutputWithPast,TokenClassifierOutput
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer
from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel, MistralModel

from typing import Optional, Dict, Sequence
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


from torch import nn
import pdb
import wandb

import os
import random
import functools
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, field
import json
from torch.nn import  CrossEntropyLoss
from typing import List, Optional, Tuple, Union
wandb.login(key='WANDB_KEY')
tqdm.pandas()

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_PRED_TOKEN = "и"

@dataclass
class CustomArgs:
    data_path: str = field(default=None, metadata={"help": "The path to the dataset directory."})


# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->Mistral, LLAMA->MISTRAL
class MistralForTokenClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
        # output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg


# define custom batch preprocessor
def collate_fn(batch, tokenizer):
    dict_keys = ['input_ids', 'attention_mask', 'labels']
    d = {k: [dic[k] for dic in batch] for k in dict_keys}
    d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
    )
    d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
        d['attention_mask'], batch_first=True, padding_value=0
    )
    d['labels'] = torch.nn.utils.rnn.pad_sequence(
        d['labels'], batch_first=True, padding_value=-100
    )
    return d


# define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    
    # 初始化空的Tensor来保存最小predictions值和对应的label值
    valid_predictions_min = []
    valid_labels = []
    
    # 遍历每行，提取不等于-100的predictions的最小值，以及对应的label值
    for i in range(labels.shape[0]):
        valid_mask = labels[i] != -100  # 找到当前行中不等于-100的位置
        valid_predictions_row = predictions[i][valid_mask]  # 提取有效的predictions
        
        if valid_predictions_row.size > 0:  # 确保至少存在一个有效prediction
            min_pred = np.min(valid_predictions_row)  # 获取最小prediction
            valid_predictions_min.append(min_pred.item())  # .item()将单个值tensor转换为标量
            
            # 因为每一行的label都是一样的，我们只需要取其中一个，放入valid_labels中即可
            valid_labels.append(labels[i][valid_mask][0])  # 取得一个有效标签的值
            
    # 使用Numpy转换列表为数组，适合f1_score函数
    valid_predictions_min = np.array(valid_predictions_min)
    valid_labels = np.array(valid_labels)
    
    # 计算F1分数
    f1_micro = f1_score(valid_labels, valid_predictions_min > 0, average='micro')
    f1_macro = f1_score(valid_labels, valid_predictions_min > 0, average='macro')
    f1_weighted = f1_score(valid_labels, valid_predictions_min > 0, average='weighted')
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def generate_training_labels(encoded_text, encoded_label):
    # 创建一个与input_ids形状相同但全部填充为-100的标签列表
    training_labels = [-100] * len(encoded_text)
    
    # 遍历encoded_text中的每个token的id
    # 如果当前token对应于特殊字符
    special_token_id = 839
    positive_id = 648
    negative_id = 387
    for i, token_id in enumerate(encoded_text):
        if token_id == special_token_id:
            # 检查encoded_label在相同位置的token_id来决定该位置是正(1)还是负(0)
            if encoded_label[i] == positive_id:
                training_labels[i] = 1
            elif encoded_label[i] == negative_id:
                training_labels[i] = 0
            # 对于不匹配正负标识的特殊token位置，我们已经默认设置为-100，这里不做额外处理
                
    return training_labels

def preprocess_labels(text, labels, tokenizer):
    text = [t.replace('ки', 'и') for t in text]
    training_labels = [-100] * len(text)
    
    # 遍历encoded_text中的每个token的id
    # 如果当前token对应于特殊字符
    special_token_id = 839
    positive_id = 648
    negative_id = 387

    input_ids = []
    r_labels = []
    attention_mask = tokenizer(text, truncation=True, padding='max_length', max_length=512)['attention_mask']
    for t_line, l_line in zip(text, labels):
        t_e = tokenizer.encode(t_line, truncation=True, padding='max_length', max_length=512)
        l_e = tokenizer.encode(l_line, truncation=True, padding='max_length', max_length=512)
        
        assert len(t_e) == len(l_e)
        
        l = generate_training_labels(t_e, l_e)
        r_labels.append(l)
        input_ids.append(t_e)
        
        assert len(l) == len(t_e)
        assert l.count(1) + l.count(0) == t_e.count(839)

    # 确保labels是tensor
    labels_tensor = torch.tensor(labels)
    pass

def label_tensor_everywhere(input_tensor, labels, tokenizer):
    # 确保labels是tensor
    labels_tensor = labels
    
    # 创建一个与input_tensor形状相同，但全部填充为-100的tensor
    output_tensor = torch.full(input_tensor.shape, -100, dtype=input_tensor.dtype)
    
    # 遍历每一行，寻找所有值为839的位置
    for i, row in enumerate(input_tensor):
        # 得到一个与row形状相同的tensor，其中839位置是True，其他位置是False
        mask = (row == 839)
        # 用mask来选择对应位置，然后用label填充这些位置
        output_tensor[i][mask] = labels_tensor[i]

    return output_tensor


# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    labels = label_tensor_everywhere(examples['input_ids'], examples['labels'], tokenizer)
    input_ids = examples['input_ids']
    attention_mask = examples['attention_mask']
    return {
        'input_ids': input_ids, 
        'labels': labels, 
        'attention_mask': attention_mask
    }

# create custom trainer class to be able to pass label weights and calculate mutilabel loss
class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 处理labels和logits以便于计算损失，-100位置的标签将不被计算损失
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(active_logits, active_labels.to(torch.bfloat16))
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig, CustomArgs))
    config, model_config, data_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    model = MistralForTokenClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    ).to(torch.bfloat16).to('cuda')

    # tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        # 
        # pred_token=DEFAULT_PRED_TOKEN
    # try:
    #     _ = tokenizer.pred_token
    # except:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict= {'additional_special_tokens': [' ки']},
    #         tokenizer=tokenizer,
    #         model=model,
    #     )
    #     tokenizer.pred_token = ' ки'


    model.config.pad_token_id = tokenizer.pad_token_id
    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    tokenized_ds = load_from_disk(data_config.data_path)
    ################
    # Training
    ################
    trainer = CustomTrainer(
        model = model,
        args = config,
        train_dataset = tokenized_ds['train'],
        eval_dataset = tokenized_ds['eval'],
        tokenizer = tokenizer,
        data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
