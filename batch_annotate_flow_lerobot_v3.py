"""
批量标注LeRobot v3.0数据集的光流（1秒间隔）

该脚本遍历LeRobot v3.0格式的数据集，对每个episode计算当前帧到1秒后的光流，
生成运动mask并保存为RLE压缩格式。

数据格式：
- 输入：LeRobot v3.0格式数据集（包含meta/info.json和videos/）
- 输出：每个episode每个相机独立的JSON文件（包含RLE压缩的mask）

特性：
1. 智能相机选择：仅处理cam_high和cam_side（根据robot_type过滤黑帧相机）
2. 时间戳索引：从合并视频中提取特定episode的帧
3. 1秒间隔光流：计算当前帧到1秒后（30帧）的光流
4. RLE压缩：大幅减少存储空间（>10x压缩率）
5. 可选可视化：支持生成调试可视化

Usage:
    python batch_annotate_flow_lerobot_v3.py \
        --data_root /path/to/robochallenge_all_temp/ \
        --output_root /path/to/output \
        --model refine \
        --batch_size 8 \
        --time_offset 1.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import traceback

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

# 导入UFM模型
sys.path.append(os.path.join(os.path.dirname(__file__), "UniCeption"))
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

# 导入mask生成和光流预测函数
from generate_mask import generate_mask_magnitude_threshold, encode_rle, generate_robust_motion_mask
from batch_inference_flow import predict_correspondences

# 导入LeRobot的视频解码工具（支持AV1编码）
lerobot_path = Path(__file__).parent.parent / "lerobot" / "src"
if lerobot_path.exists():
    sys.path.insert(0, str(lerobot_path))
from lerobot.datasets.video_utils import decode_video_frames_torchvision

# 机器人类型与有效相机映射
VALID_CAMERAS = {
    'arx5': ['cam_high', 'cam_side'],
    'ur5': ['cam_high'],  # cam_side是黑帧
    'franka': ['cam_high', 'cam_side'],
    'aloha': ['cam_high'],  # cam_side是黑帧
    'r1lite': ['observation.images.head_rgb']
}

# 目标相机列表（仅处理这两个）
TARGET_CAMERAS = ['cam_high', 'cam_side', 'observation.images.head_rgb']


def load_dataset_info(dataset_path):
    """
    加载数据集元信息和有效相机列表
    
    Args:
        dataset_path: 数据集根目录
    
    Returns:
        dict: 包含info, robot_type, active_cameras, episodes, fps等信息
    """
    dataset_path = Path(dataset_path)
    
    # 读取info.json
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    robot_type = info.get('robot_type', 'unknown')
    fps = info.get('fps', 30)
    
    # 确定有效相机（仅保留TARGET_CAMERAS中的）
    valid_cams = VALID_CAMERAS.get(robot_type, ['cam_high'])
    active_cameras = [cam for cam in valid_cams if cam in TARGET_CAMERAS]
    
    # 读取所有episodes
    episodes = []
    episodes_dir = dataset_path / "meta" / "episodes"
    for parquet_file in sorted(episodes_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        episodes.extend(df.to_dict('records'))
    
    return {
        'info': info,
        'robot_type': robot_type,
        'active_cameras': active_cameras,
        'episodes': episodes,
        'fps': fps,
        'video_path_template': info.get('video_path', 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    }


def decode_video_frames_by_time(video_path, from_ts, to_ts, fps=30, backend="pyav"):
    """
    按时间戳范围解码视频帧（使用LeRobot的PyAV解码器，支持AV1编码）
    
    Args:
        video_path: 视频文件路径
        from_ts: 起始时间戳（秒）
        to_ts: 结束时间戳（秒）
        fps: 视频帧率
        backend: 视频解码后端（pyav或video_reader）
    
    Returns:
        frames: (T, H, W, 3) numpy数组，RGB格式
    """
    # 计算需要的帧数
    expected_num_frames = int((to_ts - from_ts) * fps)
    
    # 生成时间戳列表（每帧的时间戳）- 确保是Python float类型
    timestamps = [float(from_ts + i / fps) for i in range(expected_num_frames)]
    
    try:
        # 使用LeRobot的PyAV解码器（支持AV1）
        frames_tensor = decode_video_frames_torchvision(
            video_path=video_path,
            timestamps=timestamps,
            tolerance_s=1.0 / fps,  # 容差为1帧的时间
            backend=backend,
            log_loaded_timestamps=False
        )
        
        # 转换为numpy数组 (T, C, H, W) -> (T, H, W, C)
        frames = frames_tensor.permute(0, 2, 3, 1).numpy()
        
        # 转换为uint8 [0, 255]
        if frames.dtype == np.float32 and frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        elif frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        
        if len(frames) == 0:
            raise ValueError(f"No frames decoded from {video_path} (ts: {from_ts}-{to_ts})")
        
        return frames
        
    except Exception as e:
        # 如果PyAV失败，尝试使用OpenCV（可能对某些格式有效）
        print(f"    Warning: PyAV decoding failed, falling back to OpenCV: {e}")
        return decode_video_frames_opencv(video_path, from_ts, to_ts, fps)


def decode_video_frames_opencv(video_path, from_ts, to_ts, fps=30):
    """
    使用OpenCV解码视频帧（备用方案）
    
    Args:
        video_path: 视频文件路径
        from_ts: 起始时间戳（秒）
        to_ts: 结束时间戳（秒）
        fps: 视频帧率
    
    Returns:
        frames: (T, H, W, 3) numpy数组，RGB格式
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # 获取视频帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = fps  # 使用默认fps
    
    # 计算帧索引范围
    start_frame_idx = int(from_ts * video_fps)
    end_frame_idx = int(to_ts * video_fps)
    num_frames = end_frame_idx - start_frame_idx
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path} (ts: {from_ts}-{to_ts})")
    
    return np.array(frames)


def extract_episode_frames(dataset_root, episode, camera_key, fps=30):
    """
    从合并视频中提取单个episode的所有帧
    
    Args:
        dataset_root: 数据集根目录
        episode: episode元数据字典
        camera_key: 相机名称
        fps: 帧率
    
    Returns:
        frames: (T, H, W, 3) numpy数组
    """
    # 获取视频文件路径信息
    chunk_idx = episode[f'videos/{camera_key}/chunk_index']
    file_idx = episode[f'videos/{camera_key}/file_index']
    from_ts = episode[f'videos/{camera_key}/from_timestamp']
    to_ts = episode[f'videos/{camera_key}/to_timestamp']
    
    # 构建视频文件完整路径
    video_path = dataset_root / "videos" / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # 解码视频
    frames = decode_video_frames_by_time(video_path, from_ts, to_ts, fps)
    
    return frames


def process_episode_optical_flow(model, frames, fps=30, time_offset=1.0, 
                                  batch_size=8, device='cuda',
                                  min_threshold=1.0, top_percentile=10, noise_threshold=5.0,
                                  inference_size=(520, 460), save_flow_for_vis=False,
                                  roi=None):
    """
    处理单个episode的所有帧对，计算光流并生成mask
    
    Args:
        model: UFM模型
        frames: (T, H, W, 3) numpy数组
        fps: 帧率
        time_offset: 时间偏移（秒）
        batch_size: 批量大小
        device: 计算设备
        min_threshold: mask生成的最小阈值
        top_percentile: mask生成的分位数
        noise_threshold: mask生成的噪声阈值
        inference_size: 推理时的图像尺寸 (width, height)
        save_flow_for_vis: 是否保存光流数据用于可视化
        roi: ROI区域 (x1, y1, x2, y2)，只在此区域计算光流，其他区域默认为无运动
    
    Returns:
        masks: {frame_idx: mask} 字典
        flows_dict: {frame_idx: (flow, covisibility)} 字典（如果save_flow_for_vis=True）
    """
    offset_frames = int(time_offset * fps)
    num_valid_frames = len(frames) - offset_frames
    
    if num_valid_frames <= 0:
        if save_flow_for_vis:
            return {}, {}
        return {}
    
    # 获取原始尺寸
    original_h, original_w = frames.shape[1:3]
    inference_w, inference_h = inference_size
    
    # 解析ROI
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_w, x2), min(original_h, y2)
        roi_w, roi_h = x2 - x1, y2 - y1
        print(f"    Using ROI: ({x1}, {y1}) to ({x2}, {y2}), size: {roi_w}x{roi_h}")
    else:
        x1, y1, x2, y2 = 0, 0, original_w, original_h
        roi_w, roi_h = original_w, original_h
    
    masks = {}
    flows_dict = {} if save_flow_for_vis else None
    
    # 批量处理
    for start_idx in tqdm(range(0, num_valid_frames, batch_size), 
                          desc="    Processing frames", leave=False):
        end_idx = min(start_idx + batch_size, num_valid_frames)
        
        # 准备批量数据
        source_imgs = []
        target_imgs = []
        for i in range(start_idx, end_idx):
            src = frames[i]
            tgt = frames[i + offset_frames]
            
            # 裁剪ROI区域
            src_roi = src[y1:y2, x1:x2]
            tgt_roi = tgt[y1:y2, x1:x2]
            
            # Resize到推理尺寸
            src_resized = cv2.resize(src_roi, (inference_w, inference_h))
            tgt_resized = cv2.resize(tgt_roi, (inference_w, inference_h))
            
            source_imgs.append(src_resized)
            target_imgs.append(tgt_resized)
        
        try:
            # 批量推理光流
            results = predict_correspondences(model, source_imgs, target_imgs, device)
            
            # 处理结果（results是列表）
            for i, (flow, covis) in enumerate(results):
                # 生成mask（在推理尺寸上）
                # mask, _ = generate_mask_magnitude_threshold(
                #     flow,
                #     min_threshold=min_threshold,
                #     top_percentile=top_percentile,
                #     noise_threshold=noise_threshold
                # )
                mask, mask_info = generate_robust_motion_mask(
                    flow, 
                    min_threshold=min_threshold,
                    top_percentile=top_percentile,
                    noise_threshold=noise_threshold
                )
                
                # Resize mask回ROI尺寸
                mask_roi = cv2.resize(
                    mask.astype(np.uint8), 
                    (roi_w, roi_h), 
                    interpolation=cv2.INTER_NEAREST
                )
                
                # 创建完整的mask（原始尺寸），初始化为0（无运动）
                mask_full = np.zeros((original_h, original_w), dtype=np.uint8)
                # 将ROI区域的mask放回
                mask_full[y1:y2, x1:x2] = mask_roi
                
                frame_idx = start_idx + i
                masks[frame_idx] = mask_full
                
                # 保存光流数据（用于可视化）- 需要放回完整尺寸
                if save_flow_for_vis:
                    # Resize光流回ROI尺寸
                    flow_roi = np.zeros((2, roi_h, roi_w), dtype=np.float32)
                    flow_roi[0] = cv2.resize(flow[0], (roi_w, roi_h))
                    flow_roi[1] = cv2.resize(flow[1], (roi_w, roi_h))
                    
                    covis_roi = cv2.resize(covis, (roi_w, roi_h))
                    
                    # 创建完整的光流和covisibility
                    flow_full = np.zeros((2, original_h, original_w), dtype=np.float32)
                    flow_full[:, y1:y2, x1:x2] = flow_roi
                    
                    covis_full = np.zeros((original_h, original_w), dtype=np.float32)
                    covis_full[y1:y2, x1:x2] = covis_roi
                    
                    flows_dict[frame_idx] = (flow_full, covis_full)
                    
        except Exception as e:
            print(f"\n    Error processing batch {start_idx}-{end_idx}: {e}")
            # 尝试单个处理
            for i in range(start_idx, end_idx):
                try:
                    src = frames[i]
                    tgt = frames[i + offset_frames]
                    
                    # 裁剪ROI区域
                    src_roi = src[y1:y2, x1:x2]
                    tgt_roi = tgt[y1:y2, x1:x2]
                    
                    # Resize到推理尺寸
                    src_resized = cv2.resize(src_roi, (inference_w, inference_h))
                    tgt_resized = cv2.resize(tgt_roi, (inference_w, inference_h))
                    
                    flow, covis = predict_correspondences(
                        model, src_resized, tgt_resized, device
                    )
                    
                    mask, _ = generate_mask_magnitude_threshold(
                        flow,
                        min_threshold=min_threshold,
                        top_percentile=top_percentile,
                        noise_threshold=noise_threshold
                    )
                    
                    # Resize mask回ROI尺寸
                    mask_roi = cv2.resize(
                        mask.astype(np.uint8), 
                        (roi_w, roi_h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # 创建完整的mask
                    mask_full = np.zeros((original_h, original_w), dtype=np.uint8)
                    mask_full[y1:y2, x1:x2] = mask_roi
                    
                    masks[i] = mask_full
                    
                    # 保存光流数据（用于可视化）
                    if save_flow_for_vis:
                        flow_roi = np.zeros((2, roi_h, roi_w), dtype=np.float32)
                        flow_roi[0] = cv2.resize(flow[0], (roi_w, roi_h))
                        flow_roi[1] = cv2.resize(flow[1], (roi_w, roi_h))
                        
                        covis_roi = cv2.resize(covis, (roi_w, roi_h))
                        
                        flow_full = np.zeros((2, original_h, original_w), dtype=np.float32)
                        flow_full[:, y1:y2, x1:x2] = flow_roi
                        
                        covis_full = np.zeros((original_h, original_w), dtype=np.float32)
                        covis_full[y1:y2, x1:x2] = covis_roi
                        
                        flows_dict[i] = (flow_full, covis_full)
                        
                except Exception as e2:
                    print(f"    Error processing frame {i}: {e2}")
                    continue
    
    if save_flow_for_vis:
        return masks, flows_dict
    return masks


def save_episode_optical_flow(masks, output_path, episode_info):
    """
    保存单个episode的光流mask（RLE格式）
    
    Args:
        masks: {frame_idx: mask} 字典
        output_path: 输出文件路径
        episode_info: episode信息字典
    
    Returns:
        stats: 统计信息
    """
    if not masks:
        return None
    
    # 构建RLE数据
    rle_data = {
        'episode_index': episode_info['episode_index'],
        'dataset': episode_info['dataset'],
        'camera_key': episode_info['camera_key'],
        'robot_type': episode_info['robot_type'],
        'num_frames': len(masks),
        'time_offset': episode_info['time_offset'],
        'fps': episode_info['fps'],
        'metadata': {
            'image_size': list(list(masks.values())[0].shape),
            'from_timestamp': episode_info['from_timestamp'],
            'to_timestamp': episode_info['to_timestamp'],
            'video_file': episode_info['video_file'],
            'episode_length': episode_info['episode_length']
        },
        'frames': {}
    }
    
    # 压缩所有mask
    for frame_idx, mask in sorted(masks.items()):
        rle = encode_rle(mask)
        rle_data['frames'][str(frame_idx)] = rle
    
    # 保存JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(rle_data, f, indent=2)
    
    # 计算压缩率
    original_size = sum(m.nbytes for m in masks.values())
    compressed_size = output_path.stat().st_size
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    return {
        'original_size_mb': original_size / 1024 / 1024,
        'compressed_size_mb': compressed_size / 1024 / 1024,
        'compression_ratio': compression_ratio,
        'num_frames': len(masks)
    }


def save_episode_visualization(frames, masks, flows_dict, output_dir, camera_key, 
                                episode_idx, time_offset_frames=30, max_vis_frames=10):
    """
    保存前N帧的可视化结果（包含光流可视化）
    
    Args:
        frames: 所有帧
        masks: mask字典
        flows_dict: 光流字典 {frame_idx: (flow, covisibility)}
        output_dir: 输出目录
        camera_key: 相机名称
        episode_idx: episode索引
        time_offset_frames: 时间偏移帧数
        max_vis_frames: 最多可视化的帧数
    """
    import flow_vis
    
    vis_dir = output_dir / "visualizations" / f"episode_{episode_idx:06d}_{camera_key}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(max_vis_frames, len(masks))):
        if i not in masks:
            continue
        
        source = frames[i]
        target_idx = i + time_offset_frames
        target = frames[target_idx] if target_idx < len(frames) else frames[-1]
        mask = masks[i]
        
        # 获取光流数据
        flow = None
        covis = None
        if i in flows_dict:
            flow, covis = flows_dict[i]
        
        # 创建可视化图像
        vis_images = []
        
        # 1. Source图像
        vis_images.append(source)
        
        # 2. Target图像
        vis_images.append(target)
        
        # 3. Mask叠加到source
        overlay = source.copy().astype(np.float32)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32)
        overlay = overlay * (1 - mask_3ch * 0.5) + np.array([255, 0, 0], dtype=np.float32) * mask_3ch * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        vis_images.append(overlay)
        
        if flow is not None:
            # 4. 光流可视化（彩色编码）
            flow_vis_img = flow_vis.flow_to_color(flow.transpose(1, 2, 0))
            vis_images.append(flow_vis_img)
            
            # 5. 光流幅值热力图
            flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)
            magnitude_normalized = (flow_magnitude / (flow_magnitude.max() + 1e-6) * 255).astype(np.uint8)
            magnitude_colormap = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
            magnitude_colormap = cv2.cvtColor(magnitude_colormap, cv2.COLOR_BGR2RGB)
            vis_images.append(magnitude_colormap)
            
            # 6. Covisibility可视化
            if covis is not None:
                covis_vis = (covis * 255).astype(np.uint8)
                covis_vis_rgb = np.stack([covis_vis, covis_vis, covis_vis], axis=2)
                vis_images.append(covis_vis_rgb)
        
        # 调整所有图像到相同高度
        target_h = source.shape[0]
        vis_images_resized = []
        for img in vis_images:
            h, w = img.shape[:2]
            if h != target_h:
                new_w = int(w * target_h / h)
                img_resized = cv2.resize(img, (new_w, target_h))
            else:
                img_resized = img
            vis_images_resized.append(img_resized)
        
        # 水平拼接所有图像
        combined = np.hstack(vis_images_resized)
        
        # 添加标签
        labels = ['Source', 'Target (1s later)', 'Mask Overlay']
        if flow is not None:
            labels.extend(['Flow (color)', 'Flow Magnitude', 'Covisibility'])
        
        label_y = 30
        label_x = 10
        for label in labels:
            cv2.putText(combined, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            label_x += vis_images_resized[0].shape[1]
        
        # 保存
        vis_path = vis_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(vis_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    print(f"    Visualizations saved to: {vis_dir}")


def load_ufm_model(model_type='base', device='cuda'):
    """加载UFM模型"""
    print(f"Loading UFM model ({model_type})...")
    
    if model_type == 'refine':
        model = UniFlowMatchClassificationRefinement.from_pretrained("/mnt/raid0/data/UFM-Refine")
    else:
        model = UniFlowMatchConfidence.from_pretrained("/mnt/raid0/data/UFM-Base")
    
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    return model


def process_dataset(dataset_path, output_root, model, args):
    """
    处理单个数据集
    
    Args:
        dataset_path: 数据集路径
        output_root: 输出根目录
        model: UFM模型
        args: 命令行参数
    """
    dataset_name = dataset_path.name
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # 加载数据集信息
        dataset_info = load_dataset_info(dataset_path)
        
        print(f"Robot type: {dataset_info['robot_type']}")
        print(f"Active cameras: {dataset_info['active_cameras']}")
        print(f"Total episodes: {len(dataset_info['episodes'])}")
        print(f"FPS: {dataset_info['fps']}")
        
        # 创建输出目录
        dataset_output_dir = output_root / dataset_name
        
        # 统计信息
        dataset_stats = {
            'dataset': dataset_name,
            'robot_type': dataset_info['robot_type'],
            'total_episodes': len(dataset_info['episodes']),
            'cameras': {},
            'errors': []
        }
        
        # 处理每个相机
        for camera_key in dataset_info['active_cameras']:
            print(f"\n  Processing camera: {camera_key}")
            
            camera_stats = {
                'total_episodes': 0,
                'successful_episodes': 0,
                'failed_episodes': 0,
                'total_frames': 0,
                'total_original_size_mb': 0,
                'total_compressed_size_mb': 0
            }
            
            # 处理每个episode
            for episode in tqdm(dataset_info['episodes'], desc=f"  {camera_key}", leave=True):
                episode_idx = episode['episode_index']
                episode_length = episode['length']
                
                camera_stats['total_episodes'] += 1
                
                # 检查是否已处理
                output_path = dataset_output_dir / camera_key / f"episode_{episode_idx:06d}.json"
                if output_path.exists() and args.skip_existing:
                    camera_stats['successful_episodes'] += 1
                    continue
                
                # 检查episode长度（至少需要time_offset秒的数据）
                min_frames = int(args.time_offset * dataset_info['fps']) + 1
                if episode_length < min_frames:
                    print(f"    Skipping episode {episode_idx}: too short ({episode_length} < {min_frames} frames)")
                    camera_stats['failed_episodes'] += 1
                    dataset_stats['errors'].append({
                        'episode_index': episode_idx,
                        'camera': camera_key,
                        'error': f'Episode too short: {episode_length} frames'
                    })
                    continue
                
                try:
                    # 解码视频
                    frames = extract_episode_frames(
                        dataset_path,
                        episode,
                        camera_key,
                        dataset_info['fps']
                    )
                    
                    # 光流推理
                    result = process_episode_optical_flow(
                        model, frames,
                        fps=dataset_info['fps'],
                        time_offset=args.time_offset,
                        batch_size=args.batch_size,
                        device=args.device,
                        min_threshold=args.min_threshold,
                        top_percentile=args.top_percentile,
                        noise_threshold=args.noise_threshold,
                        inference_size=(args.inference_width, args.inference_height),
                        save_flow_for_vis=args.save_vis,
                        roi=(args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2) if args.use_roi else None
                    )
                    
                    # 解包结果
                    if args.save_vis:
                        masks, flows_dict = result
                    else:
                        masks = result
                        flows_dict = None
                    
                    if not masks:
                        print(f"    Warning: No masks generated for episode {episode_idx}")
                        camera_stats['failed_episodes'] += 1
                        continue
                    
                    # 保存结果
                    stats = save_episode_optical_flow(masks, output_path, {
                        'episode_index': episode_idx,
                        'dataset': dataset_name,
                        'camera_key': camera_key,
                        'robot_type': dataset_info['robot_type'],
                        'time_offset': args.time_offset,
                        'fps': dataset_info['fps'],
                        'from_timestamp': episode[f'videos/{camera_key}/from_timestamp'],
                        'to_timestamp': episode[f'videos/{camera_key}/to_timestamp'],
                        'video_file': f"videos/{camera_key}/chunk-{episode[f'videos/{camera_key}/chunk_index']:03d}/file-{episode[f'videos/{camera_key}/file_index']:03d}.mp4",
                        'episode_length': episode_length
                    })
                    
                    if stats:
                        camera_stats['successful_episodes'] += 1
                        camera_stats['total_frames'] += stats['num_frames']
                        camera_stats['total_original_size_mb'] += stats['original_size_mb']
                        camera_stats['total_compressed_size_mb'] += stats['compressed_size_mb']
                    
                    # 可选：保存可视化
                    if args.save_vis:
                        save_episode_visualization(
                            frames, masks, flows_dict, dataset_output_dir,
                            camera_key, episode_idx,
                            time_offset_frames=int(args.time_offset * dataset_info['fps']),
                            max_vis_frames=args.max_vis_frames
                        )
                    
                    # 释放内存
                    del frames, masks
                    
                except Exception as e:
                    print(f"    Error processing episode {episode_idx}: {e}")
                    camera_stats['failed_episodes'] += 1
                    dataset_stats['errors'].append({
                        'episode_index': episode_idx,
                        'camera': camera_key,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    continue
            
            # 计算压缩率
            if camera_stats['total_original_size_mb'] > 0:
                camera_stats['compression_ratio'] = (
                    camera_stats['total_original_size_mb'] / 
                    camera_stats['total_compressed_size_mb']
                )
            else:
                camera_stats['compression_ratio'] = 0
            
            dataset_stats['cameras'][camera_key] = camera_stats
            
            # 打印相机统计
            print(f"\n  {camera_key} Summary:")
            print(f"    Successful: {camera_stats['successful_episodes']}/{camera_stats['total_episodes']}")
            print(f"    Total frames: {camera_stats['total_frames']}")
            print(f"    Compression: {camera_stats['total_original_size_mb']:.2f}MB -> {camera_stats['total_compressed_size_mb']:.2f}MB")
            print(f"    Compression ratio: {camera_stats['compression_ratio']:.2f}x")
        
        # 保存数据集统计
        stats_path = dataset_output_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        return dataset_stats
        
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="批量标注LeRobot v3.0数据集的光流")
    parser.add_argument(
        "--data_root", "-d",
        required=True,
        help="数据根目录路径（包含多个temp_*数据集）"
    )
    parser.add_argument(
        "--output_root", "-o",
        required=True,
        help="输出根目录路径"
    )
    parser.add_argument(
        "--model",
        choices=["base", "refine"],
        default="refine",
        help="UFM模型版本（默认：refine）"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备（默认：cuda）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批量推理大小（默认：8）"
    )
    parser.add_argument(
        "--time_offset",
        type=float,
        default=1.0,
        help="时间偏移（秒），默认1.0秒"
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=1.0,
        help="Mask生成的最小阈值（默认：1.0）"
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=10.0,
        help="Mask生成的分位数（默认：10）"
    )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default=5.0,
        help="Mask生成的噪声阈值（默认：5.0）"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过已存在的输出文件"
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="保存可视化结果"
    )
    parser.add_argument(
        "--max_vis_frames",
        type=int,
        default=10,
        help="每个episode最多可视化的帧数（默认：10）"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=None,
        help="指定要处理的数据集名称（默认：处理所有temp_*数据集）"
    )
    parser.add_argument(
        "--inference_width",
        type=int,
        default=520,
        help="推理时的图像宽度（默认：520）"
    )
    parser.add_argument(
        "--inference_height",
        type=int,
        default=460,
        help="推理时的图像高度（默认：460）"
    )
    parser.add_argument(
        "--use_roi",
        action="store_true",
        help="使用ROI区域计算光流（仅在ROI内计算，其他区域默认为无运动）"
    )
    parser.add_argument(
        "--roi_x1",
        type=int,
        default=50,
        help="ROI左上角X坐标（默认：50）"
    )
    parser.add_argument(
        "--roi_y1",
        type=int,
        default=22,
        help="ROI左上角Y坐标（默认：22）"
    )
    parser.add_argument(
        "--roi_x2",
        type=int,
        default=540,
        help="ROI右下角X坐标（默认：540）"
    )
    parser.add_argument(
        "--roi_y2",
        type=int,
        default=425,
        help="ROI右下角Y坐标（默认：425）"
    )
    
    args = parser.parse_args()
    
    # 转换路径
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    if not data_root.exists():
        print(f"Error: Data root does not exist: {data_root}")
        return
    
    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 打印配置
    print("="*80)
    print("LeRobot v3.0 Optical Flow Annotation")
    print("="*80)
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Time offset: {args.time_offset}s")
    print(f"Inference size: {args.inference_width}x{args.inference_height}")
    if args.use_roi:
        print(f"ROI enabled: ({args.roi_x1}, {args.roi_y1}) to ({args.roi_x2}, {args.roi_y2})")
    print(f"Mask parameters: min_threshold={args.min_threshold}, top_percentile={args.top_percentile}, noise_threshold={args.noise_threshold}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Save visualizations: {args.save_vis}")
    print("="*80)
    
    # 加载模型
    model = load_ufm_model(args.model, args.device)
    
    # 扫描数据集
    if args.datasets:
        datasets = [data_root / name for name in args.datasets if (data_root / name).exists()]
    else:
        datasets = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("temp_")])
    
    if not datasets:
        print("No datasets found!")
        return
    
    print(f"\nFound {len(datasets)} datasets to process")
    
    # 处理每个数据集
    all_stats = []
    for dataset_path in datasets:
        stats = process_dataset(dataset_path, output_root, model, args)
        if stats:
            all_stats.append(stats)
    
    # 保存全局统计
    summary_path = output_root / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_datasets': len(all_stats),
            'datasets': all_stats,
            'parameters': vars(args)
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Summary saved to: {summary_path}")
    print("="*80)


if __name__ == "__main__":
    main()
