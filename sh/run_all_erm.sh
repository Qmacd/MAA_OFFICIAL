#!/bin/bash

# 运行脚本: run_baseframe.py
# 遍历指定的CSV数据文件
# 遍历指定的 generator 和对应的 window_size 组合
# 输出目录结构: ./output/<dataset_basename>/<generator_name>

DATA_DIR="../database/kline_processed_data"
PYTHON_SCRIPT="../run_multi_gan.py"

# 定义 generator 和对应的 window_size
declare -a generators=("gru" "lstm" "transformer")
declare -a window_sizes=(5 10 15)


# 默认的 start_timestamp <source_id data="0" title="run_all_maa.sh" />
DEFAULT_START=31
DEFAULT_END=-1

# 遍历数据文件 <source_id data="0" title="run_all_maa.sh" />
for FILE in "$DATA_DIR"/processed_*_day.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}" # 例如: processed_原油_day

    START_TIMESTAMP=$DEFAULT_START
    END_TIMESTAMP=$DEFAULT_END


    echo "Processing data file: $FILENAME"
    echo "-------------------------------------"

    # 遍历 generator 和 window_size 组合
    for i in "${!generators[@]}"; do
        generator=${generators[$i]}
        window_size=${window_sizes[$i]}

        # === 修改点: 定义特定于此组合的输出目录 ===
        # 结构: ./output/<dataset_basename>/<generator_name>
        # 例如: ./output/processed_原油_day/gru
        OUTPUT_DIR_COMBINED="../output/erm/${BASENAME}/${generator}"

        # 确保目录存在 (如果不存在则创建)
        mkdir -p "$OUTPUT_DIR_COMBINED"

        echo "Running with generator=$generator, window_size=$window_size, start=$START_TIMESTAMP..."
        echo "Output directory: $OUTPUT_DIR_COMBINED" # 打印确认信息

        # === 修改点: 使用新的 OUTPUT_DIR_COMBINED ===
        # 执行 Python 脚本，传入所有参数，使用新的输出目录
        python "$PYTHON_SCRIPT" \
            --data_path "$FILE" \
            --output_dir "$OUTPUT_DIR_COMBINED" \
            --start_timestamp "$START_TIMESTAMP" \
            --end_timestamp "$END_TIMESTAMP" \
            --generator "$generator" \
            --feature_columns 1 19 \
            --window_size "$window_size" \
            --N_pairs 1 \
            --distill_epochs 0 \
            --cross_finetune_epochs 0 \
            --num_epochs 9999 \
            --ERM True

        echo "Finished run for generator=$generator, window_size=$window_size."
        echo "" # 添加空行以便区分不同的运行日志

    done
    echo "-------------------------------------"
    echo "Finished processing file: $FILENAME"
    echo ""
done

echo "All tasks completed."