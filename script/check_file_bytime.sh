#!/bin/bash

# 目标路径
TARGET_PATH="/data1/zilun/ImageRAG0226/data/eval"

# 检查目标路径是否存在
if [ ! -d "$TARGET_PATH" ]; then
    echo "目录不存在: $TARGET_PATH"
    exit 1
fi

# 遍历目录中的所有文件
for file in "$TARGET_PATH"/*; do
    if [ -f "$file" ]; then
        # 使用 stat 获取文件的创建时间
        create_time=$(stat -c %y "$file" | cut -d ' ' -f 1-2)
        echo "文件: $(basename "$file") 创建时间: $create_time"
    fi
done
