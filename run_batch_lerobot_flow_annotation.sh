#!/bin/bash

# 批量处理所有数据集的光流标注
# 处理所有temp_*数据集，每个数据集的cam_high和cam_side（如果可用）

DATA_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/robochallenge_lerobot_v3_0/robochallenge_all_temp"
# DATA_ROOT="/home/unitree/remote_jensen2/Galaxea-Open-World-Dataset/lerobot_v3_0"
OUTPUT_ROOT="/home/unitree/remote_jensen2/oxe_lerobot/robochallenge_lerobot_v3_0/optical_flow_annotation"

echo "=========================================="
echo "Batch Optical Flow Annotation"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Model: refine"
echo "Time offset: 1.0s (30 frames)"
echo "Cameras: cam_high, cam_side"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 运行批处理

CUDA_VISIBLE_DEVICES=2 python batch_annotate_flow_lerobot_v3.py \
    --data_root "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model refine \
    --batch_size 8 \
    --time_offset 0.1 \
    --min_threshold 3.0 \
    --top_percentile 10 \
    --noise_threshold 5.0 \
    --device cuda \
    --save_vis \
    --max_vis_frames 500 \
    --datasets temp_arrange_flowers


# CUDA_VISIBLE_DEVICES=1 python batch_annotate_flow_lerobot_v3.py \
#     --data_root "$DATA_ROOT" \
#     --output_root "$OUTPUT_ROOT" \
#     --model refine \
#     --batch_size 8 \
#     --time_offset 0.1 \
#     --min_threshold 3.0 \
#     --top_percentile 10 \
#     --noise_threshold 5.0 \
#     --device cuda \
#     --datasets temp_stack_bowls

echo ""
echo "=========================================="
echo "Batch processing completed!"
echo "Results saved to: $OUTPUT_ROOT"
echo "Check processing_summary.json for details"
echo "=========================================="
