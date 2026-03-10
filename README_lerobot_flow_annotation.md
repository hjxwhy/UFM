# LeRobot v3.0 光流标注工具

批量标注LeRobot v3.0数据集的1秒间隔光流，生成运动mask并保存为RLE压缩格式。

## 功能特性

1. **智能相机选择**：仅处理 `cam_high` 和 `cam_side`，自动根据机器人类型过滤黑帧相机
   - ARX5: cam_high ✓, cam_side ✓
   - UR5: cam_high ✓, cam_side ✗ (黑帧)
   - FRANKA: cam_high ✓, cam_side ✓
   - ALOHA: cam_high ✓, cam_side ✗ (黑帧)

2. **时间戳索引**：从合并视频中精确提取每个episode的帧序列

3. **1秒间隔光流**：计算当前帧到1秒后（30帧@30fps）的光流

4. **RLE压缩**：大幅减少存储空间（通常>10x压缩率）

5. **批量处理**：支持GPU批量推理，提高处理效率

6. **可选可视化**：生成调试可视化，验证光流质量

## 数据格式

### 输入格式（LeRobot v3.0）

```
robochallenge_all_temp/
├── temp_arrange_flowers/
│   ├── meta/
│   │   ├── info.json              # 数据集元信息（fps, robot_type等）
│   │   └── episodes/
│   │       └── chunk-000/
│   │           └── file-000.parquet  # episode元数据（时间戳等）
│   └── videos/
│       ├── cam_high/
│       │   └── chunk-000/
│       │       └── file-000.mp4   # 合并视频（多个episode）
│       └── cam_side/
│           └── chunk-000/
│               └── file-000.mp4
├── temp_arrange_fruits_in_basket/
└── ...（共30个数据集）
```

### 输出格式

```
optical_flow_annotations_1s/
├── temp_arrange_flowers/
│   ├── cam_high/
│   │   ├── episode_000000.json    # RLE压缩的mask
│   │   ├── episode_000001.json
│   │   └── ...
│   ├── cam_side/
│   │   └── ...
│   ├── visualizations/            # 可选
│   │   └── episode_000000_cam_high/
│   │       ├── frame_000000.jpg
│   │       └── ...
│   └── processing_stats.json      # 处理统计
├── temp_arrange_fruits_in_basket/
└── processing_summary.json        # 全局统计
```

### JSON文件格式

每个episode的JSON文件包含RLE压缩的mask：

```json
{
  "episode_index": 0,
  "dataset": "temp_arrange_flowers",
  "camera_key": "cam_high",
  "robot_type": "arx5",
  "num_frames": 1641,
  "time_offset": 1.0,
  "fps": 30,
  "metadata": {
    "image_size": [480, 640],
    "from_timestamp": 0.0,
    "to_timestamp": 55.7,
    "video_file": "videos/cam_high/chunk-000/file-000.mp4",
    "episode_length": 1671
  },
  "frames": {
    "0": {"size": [480, 640], "counts": [123, 456, ...]},
    "1": {"size": [480, 640], "counts": [789, 234, ...]},
    ...
  }
}
```

## 使用方法

### 1. 测试单个数据集

```bash
# 测试temp_arrange_flowers数据集
./test_lerobot_flow_annotation.sh
```

这会处理第一个数据集并生成可视化，用于验证流程。

### 2. 批量处理所有数据集

```bash
# 处理所有temp_*数据集
./run_batch_lerobot_flow_annotation.sh
```

### 3. 自定义参数

```bash
python batch_annotate_flow_lerobot_v3.py \
    --data_root /path/to/robochallenge_all_temp/ \
    --output_root /path/to/output \
    --model refine \
    --batch_size 8 \
    --time_offset 1.0 \
    --min_threshold 1.0 \
    --top_percentile 10 \
    --noise_threshold 5.0 \
    --skip_existing \
    --save_vis \
    --datasets temp_arrange_flowers temp_fold_dishcloth
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | LeRobot数据集根目录 | **必需** |
| `--output_root` | 输出目录 | **必需** |
| `--model` | UFM模型版本 (`base`/`refine`) | `refine` |
| `--device` | 计算设备 | `cuda` |
| `--batch_size` | 批量推理大小 | `8` |
| `--time_offset` | 时间偏移（秒） | `1.0` |
| `--min_threshold` | Mask最小阈值（像素） | `1.0` |
| `--top_percentile` | Mask分位数阈值 | `10` |
| `--noise_threshold` | 噪声过滤阈值（像素） | `5.0` |
| `--skip_existing` | 跳过已存在的文件 | `False` |
| `--save_vis` | 保存可视化 | `False` |
| `--max_vis_frames` | 每个episode可视化帧数 | `10` |
| `--datasets` | 指定处理的数据集列表 | 全部 |

## 加载RLE压缩的Mask

使用 `generate_mask.py` 中的函数加载：

```python
from generate_mask import load_compressed_masks, decode_rle
import json

# 方法1: 使用加载函数
masks_dict, metadata = load_compressed_masks('episode_000000.json')
# masks_dict: {frame_idx: mask_array}

# 方法2: 手动解码
with open('episode_000000.json', 'r') as f:
    data = json.load(f)

mask_0 = decode_rle(data['frames']['0'])  # (H, W) numpy数组
```

## 处理流程

1. **扫描数据集**：读取 `meta/info.json` 获取robot_type和fps
2. **过滤相机**：根据robot_type选择有效的cam_high和cam_side
3. **遍历episodes**：读取 `meta/episodes/*.parquet` 获取时间戳
4. **解码视频**：从合并视频中提取episode的帧（基于from_timestamp和to_timestamp）
5. **批量推理**：使用UFM模型计算1秒间隔的光流
6. **生成mask**：根据光流幅值生成运动区域mask
7. **RLE压缩**：压缩mask并保存为JSON
8. **可选可视化**：生成前N帧的可视化图像

## 性能

- **处理速度**：约5-10 episodes/分钟（取决于GPU和episode长度）
- **压缩率**：通常>10x（例如：100MB原始mask压缩为<10MB JSON）
- **内存占用**：每个episode约2-5GB GPU内存（batch_size=8）

## 故障排除

### 问题1：CUDA out of memory

**解决方案**：减小 `--batch_size`

```bash
python batch_annotate_flow_lerobot_v3.py ... --batch_size 4
```

### 问题2：视频解码失败

**可能原因**：
- 视频文件损坏
- 时间戳超出视频范围

**检查方法**：
```python
import cv2
cap = cv2.VideoCapture('file-000.mp4')
print(f"Total frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
```

### 问题3：Episode太短被跳过

**说明**：如果episode长度 < time_offset * fps + 1帧，会被自动跳过

**解决方案**：减小 `--time_offset` 或接受跳过

## 输出统计

处理完成后检查统计信息：

```bash
# 全局统计
cat processing_summary.json

# 单个数据集统计
cat temp_arrange_flowers/processing_stats.json
```

统计信息包括：
- 成功/失败episode数量
- 总帧数
- 原始/压缩文件大小
- 压缩率
- 错误日志

## 依赖项

- Python 3.8+
- PyTorch
- OpenCV (cv2)
- NumPy
- Pandas
- tqdm
- UniFlowMatch (UFM模型)

## 示例输出

```
================================================================================
Processing dataset: temp_arrange_flowers
================================================================================
Robot type: arx5
Active cameras: ['cam_high', 'cam_side']
Total episodes: 973
FPS: 30

  Processing camera: cam_high
  cam_high: 100%|██████████| 973/973 [15:23<00:00,  1.05it/s]

  cam_high Summary:
    Successful: 973/973
    Total frames: 1589903
    Compression: 3048.23MB -> 287.45MB
    Compression ratio: 10.60x

  Processing camera: cam_side
  cam_side: 100%|██████████| 973/973 [14:52<00:00,  1.09it/s]

  cam_side Summary:
    Successful: 973/973
    Total frames: 1589903
    Compression: 3048.23MB -> 291.12MB
    Compression ratio: 10.47x
```

## 相关文件

- `batch_annotate_flow_lerobot_v3.py`：主处理脚本
- `generate_mask.py`：Mask生成和RLE编解码
- `batch_inference_flow.py`：光流推理函数
- `test_lerobot_flow_annotation.sh`：测试脚本
- `run_batch_lerobot_flow_annotation.sh`：批处理脚本

## 注意事项

1. **存储空间**：虽然使用RLE压缩，但处理30个数据集仍需要约10-20GB存储空间
2. **处理时间**：完整处理30个数据集约需10-15小时（单GPU）
3. **中断恢复**：使用 `--skip_existing` 可以从中断处继续
4. **黑帧相机**：UR5和ALOHA的cam_side会被自动跳过

## 引用

如使用此工具，请引用：
- UniFlowMatch: https://github.com/example/uniflowmatch
- LeRobot: https://github.com/huggingface/lerobot
