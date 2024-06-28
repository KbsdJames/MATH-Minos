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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer
from typing import Optional, Dict, Sequence
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config

import pdb
import wandb

import os
import random
import functools
import csv
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, field
wandb.login(key='WANDB_KEY')
tqdm.pandas()

DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class CustomArgs:
    data_path: str = field(default=None, metadata={"help": "The path to the dataset directory."})


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
    d['labels'] = torch.stack(d['labels'])
    d['labels'] = d['labels'].view(-1, 1)
    return d

# define which metrics to compute for evaluation
def compute_metrics(p):
    predictions, labels = p
    f1_micro = f1_score(labels, predictions > 0, average = 'micro')
    f1_macro = f1_score(labels, predictions > 0, average = 'macro')
    f1_weighted = f1_score(labels, predictions > 0, average = 'weighted')
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# preprocess dataset with tokenizer
def tokenize_examples(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['text'])
    tokenized_inputs['labels'] = examples['label']
    return tokenized_inputs


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
        
        # compute custom loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.to(torch.bfloat16))
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    ).to(torch.bfloat16).to('cuda')

    # tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        # pdb.set_trace()
        
    model.config.pad_token_id = tokenizer.pad_token_id
    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    raw_datasets = load_from_disk(data_config.data_path)
    tokenized_ds = raw_datasets.map(functools.partial(tokenize_examples, tokenizer=tokenizer), batched=True)
    tokenized_ds = tokenized_ds.with_format('torch')
    train_dataset = tokenized_ds["train"]
    eval_dataset = tokenized_ds["eval"]
    ################
    # Training
    ################
    trainer = CustomTrainer(
        model = model,
        args = config,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        data_collator = functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics = compute_metrics,
    )

    trainer.train()
    trainer.save_model(config.output_dir)