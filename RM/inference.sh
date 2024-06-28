MODEL=$1
FOLDER_GSM8K=$2

mkdir -p $FOLDER_GSM8K
python reward_model_inference.py \
    --model_name_or_path $MODEL\
    --folder_new $FOLDER_GSM8K

FOLDER_MATH=$3
mkdir -p $FOLDER_MATH
python reward_model_inference_MATH.py \
    --model_name_or_path $MODEL\
    --folder_new $FOLDER_MATH