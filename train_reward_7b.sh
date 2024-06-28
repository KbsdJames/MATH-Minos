export MODEL_PATH= # MODEL_PATH
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export DATA_PATH=$1
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_API_KEY= # YOUR_WANDB_KEY_HERE
export WANDB_MODE=online
EVAL_STEPS=$((30000 / (8 * 4 * 4 * 15)))

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train_reward.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "none" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    --tf32 True
