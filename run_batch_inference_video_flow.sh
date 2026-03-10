#!/bin/bash

# 批量推理视频序列的flow - Bash脚本
# 用法: ./run_batch_inference_flow.sh [选项]

# 默认参数
DATA_ROOT="/home/unitree/remote_jensen/unitree_vla/wma_dataset/av_mixed/train/videos"
OUTPUT_ROOT="/home/unitree/remote_jensen/huangjianxin/open-x/train/flows"
MODEL="base"
DEVICE="cuda:1"
NO_VIS=true
BATCH_SIZE=8
SKIP_EXISTING=false
FPS=15.0
DIRECT_MASK=true
MIN_THRESHOLD=1.0
TOP_PERCENTILE=10.0
NOISE_THRESHOLD=5.0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        -o|--output_root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no_vis)
            NO_VIS=true
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --direct_mask)
            DIRECT_MASK=true
            shift
            ;;
        --min_threshold)
            MIN_THRESHOLD="$2"
            shift 2
            ;;
        --top_percentile)
            TOP_PERCENTILE="$2"
            shift 2
            ;;
        --noise_threshold)
            NOISE_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "必需参数:"
            echo "  -d, --video_root PATH        数据根目录路径（包含video目录的路径）"
            echo "  -o, --output_root PATH      输出根目录路径"
            echo ""
            echo "可选参数:"
            echo "  --model MODEL               模型版本 (base|refine, 默认: base)"
            echo "  --device DEVICE             计算设备 (默认: cuda)"
            echo "  --no_vis                    不保存可视化结果（默认保存）"
            echo "  --batch_size SIZE           批量推理的大小（默认: 8）"
            echo "  --skip_existing             如果输出文件已存在则跳过（默认强制覆盖）"
            echo "  --fps FPS                   视频的实际帧率（Hz），用于下采样到5Hz（默认: 5.0）"
            echo "  --direct_mask               直接生成mask而不保存光流文件（默认保存光流）"
            echo "  --min_threshold THRESHOLD   mask生成的最小模长阈值（像素），用于direct_mask模式（默认: 1.0）"
            echo "  --top_percentile PERCENT    mask生成的分位数（0-100），用于direct_mask模式（默认: 10.0）"
            echo "  --noise_threshold THRESHOLD mask生成的噪声阈值（像素），用于direct_mask模式（默认: 5.0）"
            echo "  -h, --help                  显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 -d /path/to/data -o /path/to/output"
            echo "  $0 -d /path/to/data -o /path/to/output --model refine --batch_size 16"
            echo "  $0 -d /path/to/data -o /path/to/output --direct_mask --fps 30"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [[ -z "$DATA_ROOT" ]]; then
    echo "错误: 必须指定 --data_root 参数"
    echo "使用 -h 或 --help 查看帮助信息"
    exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "错误: 必须指定 --output_root 参数"
    echo "使用 -h 或 --help 查看帮助信息"
    exit 1
fi

# 检查数据目录是否存在
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "错误: 数据目录不存在: $DATA_ROOT"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/batch_inference_video_flow.py"

# 检查Python脚本是否存在
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 构建命令
CMD="python $PYTHON_SCRIPT"
CMD="$CMD --video_root \"$DATA_ROOT\""
CMD="$CMD --output_root \"$OUTPUT_ROOT\""
CMD="$CMD --model $MODEL"
CMD="$CMD --device $DEVICE"

if [[ "$NO_VIS" == true ]]; then
    CMD="$CMD --no_vis"
fi

CMD="$CMD --batch_size $BATCH_SIZE"

if [[ "$SKIP_EXISTING" == true ]]; then
    CMD="$CMD --skip_existing"
fi

# CMD="$CMD --fps $FPS"

if [[ "$DIRECT_MASK" == true ]]; then
    CMD="$CMD --direct_mask"
    CMD="$CMD --min_threshold $MIN_THRESHOLD"
    CMD="$CMD --top_percentile $TOP_PERCENTILE"
    CMD="$CMD --noise_threshold $NOISE_THRESHOLD"
fi

# 显示配置信息
echo "=========================================="
echo "批量推理视频序列的flow"
echo "=========================================="
echo "视频根目录: $DATA_ROOT"
echo "输出根目录: $OUTPUT_ROOT"
echo "模型版本: $MODEL"
echo "计算设备: $DEVICE"
echo "批量大小: $BATCH_SIZE"
echo "帧率: ${FPS}Hz"
echo "跳过已存在: $SKIP_EXISTING"
echo "保存可视化: $([ "$NO_VIS" == true ] && echo "否" || echo "是")"
if [[ "$DIRECT_MASK" == true ]]; then
    echo "模式: 直接生成mask (RLE压缩)"
    echo "  - 最小阈值: $MIN_THRESHOLD"
    echo "  - 分位数: $TOP_PERCENTILE"
    echo "  - 噪声阈值: $NOISE_THRESHOLD"
else
    echo "模式: 保存光流"
fi
echo "=========================================="
echo ""

# 执行命令
echo "执行命令:"
echo "$CMD"
echo ""
eval $CMD

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "处理完成！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "处理失败！"
    echo "=========================================="
    exit 1
fi
