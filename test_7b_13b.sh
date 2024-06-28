export MASTER_ADDR=localhost
export MASTER_PORT=2131
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_DIR=$1
DATA_DIR=$2
OUT_DIR=$3

torchrun --nproc_per_node 8 --master_port 7834 test_text_orm.py \
                        --base_model $MODEL_DIR \
                        --data_path $DATA_DIR \
                        --out_path $OUT_DIR \
                        --batch_size 12