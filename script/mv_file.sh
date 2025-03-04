#!/bin/bash

# 源路径
SOURCE_PATH="/data1/zilun/ImageRAG0226/data/eval"
# 目标路径
DESTINATION_PATH="/data1/zilun/ImageRAG0226/data/eval/old"

# 获取当天 14:00 的时间戳
REFERENCE_TIME=$(date -d "$(date +%Y-%m-%d) 14:00" +%s)

# 检查目标路径是否存在，如果不存在则创建
if [ ! -d "$DESTINATION_PATH" ]; then
    mkdir -p "$DESTINATION_PATH"
fi

# 使用 find 列出文件并判断修改时间
find "$SOURCE_PATH" -type f -exec sh -c '
    # 获取文件的修改时间戳
    file_mtime=$(date -r "$1" +%s)
    # 检查文件的修改时间是否在14:00之后
    if [ "$file_mtime" -lt "$2" ]; then
        # 移动文件
        echo "Moving $1 to $3"
        mv "$1" "$3"
    fi
' sh {} "$REFERENCE_TIME" "$DESTINATION_PATH" \;
