export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"

PBS=32
GAS=2
EVAL_STEPS=$((440000 / (8 * $PBS * $GAS * 15)))

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env reward_modeling_pointwise.py \
    --model_name_or_path="/path/to/textrm" \
    --output_dir="./ckpts/$2" \
    --data_path="/path/to/orm_training_data" \
    --per_device_train_batch_size=$PBS \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=$GAS \
    --bf16 \
    --gradient_checkpointing=True \
    --learning_rate=$1 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --lr_scheduler_type "cosine" \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --save_steps=$EVAL_STEPS \
    --eval_steps=$EVAL_STEPS \
    --max_length=512 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True

