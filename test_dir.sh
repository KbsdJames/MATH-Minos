#!/bin/bash

# 指定模型路径
MODEL_PATH=

# 数据目录路径
DATA_DIR=

# 输出目录路径
OUTPUT_DIR=

# 检查输出目录是否存在，如果不存在则创建
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# 遍历数据目录下的所有json文件
for i in {1..256}
do
  echo "proceeding $i..."
  # 定义输入文件路径
  DATA_PATH="$DATA_DIR/$i.json"
  
  # 定义输出文件路径
  OUTPUT_PATH="$OUTPUT_DIR/$i.json"
  
  # 调用test脚本进行处理
  bash ./test_7b_13b.sh "$MODEL_PATH" "$DATA_PATH" "$OUTPUT_PATH"
done

echo "All files processed."