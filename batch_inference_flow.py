"""
批量推理脚本：对视频序列的每一帧计算flow。
遍历指定路径下的所有视频序列目录，计算相邻帧之间的光流。

Usage:
    python batch_inference_flow.py --data_root /path/to/data --output_root /path/to/output
    python batch_inference_flow.py --data_root /path/to/data --model refine --save_format npy
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

import cv2
import flow_vis
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "UniCeption"))
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence
from uniflowmatch.utils.viz import warp_image_with_flow

# 导入mask生成函数
from generate_mask import (
    generate_mask_magnitude_threshold,
    encode_rle
)


def load_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_correspondences(model, source_images, target_images, device="cuda"):
    """
    批量预测对应关系。
    
    Args:
        model: UFM模型
        source_images: list of (H, W, 3) numpy数组，或单个(H, W, 3)数组
        target_images: list of (H, W, 3) numpy数组，或单个(H, W, 3)数组
        device: 计算设备
    
    Returns:
        如果是批量输入，返回(flow_outputs, covisibilities)的列表
        如果是单个输入，返回单个(flow_output, covisibility)
    """
    is_batch = isinstance(source_images, list)
    
    if not is_batch:
        # 单个输入，转换为列表以便统一处理
        source_images = [source_images]
        target_images = [target_images]
    
    # 确定目标尺寸（使用第一个图像的尺寸）
    target_h, target_w = source_images[0].shape[:2]
    
    # 调整所有图像到相同尺寸并转换为tensor
    source_tensors = []
    target_tensors = []
    
    for src_img, tgt_img in zip(source_images, target_images):
        # Resize到目标尺寸
        if src_img.shape[:2] != (target_h, target_w):
            src_img = cv2.resize(src_img, (target_w, target_h))
        if tgt_img.shape[:2] != (target_h, target_w):
            tgt_img = cv2.resize(tgt_img, (target_w, target_h))
        
        # 转换为tensor (H, W, 3) -> (3, H, W)
        src_tensor = torch.from_numpy(src_img).permute(2, 0, 1).to(device)
        tgt_tensor = torch.from_numpy(tgt_img).permute(2, 0, 1).to(device)
        
        source_tensors.append(src_tensor)
        target_tensors.append(tgt_tensor)
    
    # 拼接成batch (B, 3, H, W)
    source_batch = torch.stack(source_tensors)
    target_batch = torch.stack(target_tensors)
    
    # 批量推理
    with torch.no_grad():
        result = model.predict_correspondences_batched(
            source_image=source_batch,
            target_image=target_batch,
        )
        
        # 提取结果
        flow_outputs = result.flow.flow_output.cpu().numpy()  # (B, 2, H, W)
        covisibilities = result.covisibility.mask.cpu().numpy()  # (B, H, W)
    
    # 返回结果
    if is_batch:
        # 返回列表
        return [(flow_outputs[i], covisibilities[i]) for i in range(len(source_images))]
    else:
        # 返回单个结果
        return flow_outputs[0], covisibilities[0]


def get_sorted_image_files(video_dir):
    """获取视频目录下所有图像文件，按帧序号排序。"""
    video_path = Path(video_dir)
    image_files = sorted(video_path.glob("*.png")) + sorted(video_path.glob("*.jpg")) + sorted(video_path.glob("*.jpeg"))
    
    # 按文件名排序（确保帧顺序正确）
    image_files = sorted(image_files, key=lambda x: int(x.stem))
    
    return image_files


def resize_to_same_height(img1, img2):
    """调整两个图像到相同高度，保持宽高比。"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if h1 == h2:
        return img1, img2
    
    target_h = min(h1, h2)
    img1_resized = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
    img2_resized = cv2.resize(img2, (int(w2 * target_h / h2), target_h))
    
    return img1_resized, img2_resized


def save_visualizations(source_image, target_image, flow_output, covisibility, output_dir, frame_idx):
    """
    保存可视化结果：将所有可视化合并为一张图。
    上排：flow可视化 + warped图像（水平拼接）
    下排：covisibility连续值 + covisibility二值化（水平拼接）
    整体：上下垂直拼接
    
    Args:
        source_image: 源图像 (H, W, 3) RGB格式
        target_image: 目标图像 (H, W, 3) RGB格式
        flow_output: flow输出 (2, H, W)
        covisibility: covisibility mask (H, W)
        output_dir: 输出目录
        frame_idx: 帧索引
    """
    output_path = Path(output_dir)
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成flow可视化
    flow_vis_img = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
    
    # 生成warped图像
    warped_image = warp_image_with_flow(source_image, None, target_image, flow_output.transpose(1, 2, 0))
    warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(warped_image)
    warped_image = np.clip(warped_image, 0, 255).astype(np.uint8)
    
    # 将flow_vis和warped_image水平拼接（上排）
    flow_vis_img, warped_image = resize_to_same_height(flow_vis_img, warped_image)
    top_row = np.hstack([flow_vis_img, warped_image])
    
    # 生成covisibility可视化
    covis_binary = (covisibility > 0.5).astype(np.uint8) * 255
    # 转为RGB格式（灰度图 -> RGB）
    covis_binary_rgb = np.stack([covis_binary, covis_binary, covis_binary], axis=2)
    
    covis_continuous = (covisibility * 255).astype(np.uint8)
    # 转为RGB格式（灰度图 -> RGB）
    covis_continuous_rgb = np.stack([covis_continuous, covis_continuous, covis_continuous], axis=2)
    
    # 将covisibility连续值和二值化水平拼接（下排）
    covis_continuous_rgb, covis_binary_rgb = resize_to_same_height(covis_continuous_rgb, covis_binary_rgb)
    bottom_row = np.hstack([covis_continuous_rgb, covis_binary_rgb])
    
    # 将上下两排垂直拼接，确保宽度一致
    h_top, w_top = top_row.shape[:2]
    h_bottom, w_bottom = bottom_row.shape[:2]
    
    if w_top != w_bottom:
        # 调整到相同宽度
        target_w = min(w_top, w_bottom)
        top_row = cv2.resize(top_row, (target_w, int(h_top * target_w / w_top)))
        bottom_row = cv2.resize(bottom_row, (target_w, int(h_bottom * target_w / w_bottom)))
    
    # 垂直拼接
    combined_vis = np.vstack([top_row, bottom_row])
    
    # 保存合并后的可视化图像
    combined_path = vis_dir / f"visualization_{frame_idx:04d}.png"
    cv2.imwrite(str(combined_path), cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))


def process_video_sequence(model, video_dir, output_dir, device="cuda", save_vis=True, 
                          batch_size=8, skip_existing=False, fps=30, direct_mask=False,
                          min_threshold=1.0, top_percentile=10, noise_threshold=5.0):
    """
    处理一个视频序列，计算帧之间的flow。
    
    Args:
        model: 加载的UFM模型
        video_dir: 视频序列目录路径
        output_dir: 输出目录路径
        device: 计算设备
        save_vis: 是否保存可视化结果
        batch_size: 批量推理的大小
        skip_existing: 如果输出文件已存在，是否跳过
        fps: 视频的实际帧率（Hz），用于下采样到5Hz
        direct_mask: 是否直接生成mask而不保存光流
        min_threshold: mask生成的最小阈值
        top_percentile: mask生成的分位数
        noise_threshold: mask生成的噪声阈值
    """
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    
    if direct_mask:
        # 直接生成mask模式：不创建子目录，直接在output_path的父目录保存
        # 使用序列名称作为文件名的一部分
        sequence_name = video_path.name
        masks_file = output_path.parent / f"masks_rle_{int(sequence_name):06d}.json"
        # 确保masks_file的目录存在
        masks_file.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and masks_file.exists():
            print(f"  Skipping {video_path.name} (mask file exists)")
            return
    else:
        # 保存光流模式：创建data目录
        data_dir = output_path / "data"  # 统一的数据目录
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = get_sorted_image_files(video_dir)
    
    if len(image_files) < 2:
        print(f"Warning: {video_dir} has less than 2 frames, skipping...")
        return
    
    print(f"Processing {len(image_files)} frames in {video_dir} (batch_size={batch_size}, fps={fps})")
    
    # 计算帧间隔：如果帧率 > 5Hz，下采样到5Hz
    target_fps = 5.0
    if fps > target_fps:
        frame_interval = int(fps)
        print(f"  Frame rate {fps}Hz > {target_fps}Hz, downsampling with interval={frame_interval}")
    else:
        frame_interval = int(fps)
        print(f"  Frame rate {fps}Hz <= {target_fps}Hz, using consecutive frames")
    
    # 准备所有帧对
    frame_pairs = []
    for i in range(len(image_files)):
        source_file = image_files[i]
        target_idx = i + frame_interval
        
        # 确保target_idx不超出范围
        if target_idx >= len(image_files):
            break
        
        target_file = image_files[target_idx]
        frame_idx = int(source_file.stem)
        
        if direct_mask:
            # 直接生成mask模式：不需要检查npz文件
            vis_path = None  # mask模式下不保存可视化
        else:
            # 保存光流模式
            npz_path = data_dir / f"data_{frame_idx:04d}.npz"
            vis_path = output_path / "visualizations" / f"visualization_{frame_idx:04d}.png" if save_vis else None
            
            # 检查文件是否存在
            if skip_existing:
                if npz_path.exists() and (vis_path is None or vis_path.exists()):
                    continue
        
        if direct_mask:
            frame_pairs.append({
                'source_file': source_file,
                'target_file': target_file,
                'frame_idx': frame_idx,
                'vis_path': None
            })
        else:
            frame_pairs.append({
                'source_file': source_file,
                'target_file': target_file,
                'frame_idx': frame_idx,
                'npz_path': npz_path,
                'vis_path': vis_path
            })
    
    if not frame_pairs:
        print(f"  All frames already processed, skipping...")
        return
    
    # 收集所有mask（direct_mask模式）
    all_masks = {}
    
    # 批量处理
    num_batches = (len(frame_pairs) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc=f"Processing {video_path.name}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frame_pairs))
        batch_pairs = frame_pairs[start_idx:end_idx]
        
        try:
            # 批量加载图像
            source_images = []
            target_images = []
            original_sizes = []  # 保存原始图像尺寸
            
            for pair in batch_pairs:
                source_img = load_image(pair['source_file'])
                target_img = load_image(pair['target_file'])
                
                # 保存原始尺寸
                original_sizes.append({
                    'source_size': source_img.shape[:2],
                    'target_size': target_img.shape[:2]
                })
                
                source_images.append(source_img)
                target_images.append(target_img)
            
            # 批量推理（内部会将图像resize到相同尺寸）
            results = predict_correspondences(model, source_images, target_images, device)
            
            # 获取推理时的目标尺寸（predict_correspondences会将所有图像resize到第一个图像的尺寸）
            inference_h, inference_w = source_images[0].shape[:2]
            
            # 保存结果
            batch_masks = {}  # 用于收集mask（direct_mask模式）
            
            for i, pair in enumerate(batch_pairs):
                flow_output, covisibility = results[i]
                original_size = original_sizes[i]
                
                # 如果图像被resize了，需要将flow和covisibility也resize回原始尺寸
                orig_h, orig_w = original_size['source_size']
                
                if (inference_h, inference_w) != (orig_h, orig_w):
                    # Resize flow (2, H, W) 和 covisibility (H, W) 回原始尺寸
                    # Flow需要调整scale，因为图像尺寸改变了
                    # flow[0]是x方向，flow[1]是y方向
                    flow_x = cv2.resize(
                        flow_output[0],
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR
                    ) * (orig_w / inference_w)  # x方向scale调整
                    
                    flow_y = cv2.resize(
                        flow_output[1],
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR
                    ) * (orig_h / inference_h)  # y方向scale调整
                    
                    flow_output = np.stack([flow_x, flow_y], axis=0)
                    
                    # 对于covisibility，只需要简单resize
                    covisibility = cv2.resize(
                        covisibility,
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                if direct_mask:
                    # 直接生成mask模式：生成mask并收集
                    mask, _ = generate_mask_magnitude_threshold(
                        flow_output,
                        min_threshold=min_threshold,
                        top_percentile=top_percentile,
                        noise_threshold=noise_threshold
                    )
                    batch_masks[pair['frame_idx']] = mask
                else:
                    # 保存光流模式：保存数据文件
                    np.savez(pair['npz_path'], flow=flow_output, covisibility=covisibility)
                    
                    # 保存可视化结果（使用原始尺寸的图像）
                    if save_vis:
                        save_visualizations(
                            source_images[i],  # 原始尺寸的图像
                            target_images[i],  # 原始尺寸的图像
                            flow_output,  # 已resize回原始尺寸
                            covisibility,  # 已resize回原始尺寸
                            output_path, 
                            pair['frame_idx']
                        )
            
            # 如果是direct_mask模式，将batch的mask添加到总字典中
            if direct_mask:
                all_masks.update(batch_masks)
        
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # 如果批量处理失败，尝试逐个处理
            for pair in batch_pairs:
                try:
                    source_img = load_image(pair['source_file'])
                    target_img = load_image(pair['target_file'])
                    
                    flow_output, covisibility = predict_correspondences(
                        model, source_img, target_img, device
                    )
                    
                    if direct_mask:
                        # 直接生成mask模式
                        mask, _ = generate_mask_magnitude_threshold(
                            flow_output,
                            min_threshold=min_threshold,
                            top_percentile=top_percentile,
                            noise_threshold=noise_threshold
                        )
                        all_masks[pair['frame_idx']] = mask
                    else:
                        # 保存光流模式
                        np.savez(pair['npz_path'], flow=flow_output, covisibility=covisibility)
                        
                        if save_vis:
                            save_visualizations(
                                source_img, target_img, flow_output, covisibility,
                                output_path, pair['frame_idx']
                            )
                except Exception as e2:
                    print(f"Error processing {pair['source_file']} -> {pair['target_file']}: {e2}")
                    continue
    
    # 如果是direct_mask模式，保存所有mask为RLE格式
    if direct_mask and all_masks:
        rle_data = {
            'frames': {},
            'metadata': {
                'num_frames': len(all_masks),
                'frame_indices': sorted(all_masks.keys()),
                'image_size': list(all_masks[list(all_masks.keys())[0]].shape)
            }
        }
        
        for frame_idx, mask in all_masks.items():
            rle = encode_rle(mask)
            rle_data['frames'][str(frame_idx)] = rle
        
        # 保存为JSON文件
        with open(masks_file, 'w') as f:
            json.dump(rle_data, f)
        
        # 计算压缩比
        original_size = sum(mask.nbytes for mask in all_masks.values())
        compressed_size = masks_file.stat().st_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        print(f"  Saved compressed masks: {masks_file}")
        print(f"  Compression ratio: {compression_ratio:.2f}x (original: {original_size/1024/1024:.2f}MB, compressed: {compressed_size/1024/1024:.2f}MB)")


def process_data_directory(model, data_root, output_root, device="cuda", save_vis=True,
                           batch_size=8, skip_existing=False, fps=30, direct_mask=False,
                           min_threshold=1.0, top_percentile=10, noise_threshold=5.0):
    """
    处理数据根目录下的所有视频序列。
    
    Args:
        model: 加载的UFM模型
        data_root: 数据根目录路径
        output_root: 输出根目录路径
        device: 计算设备
        save_vis: 是否保存可视化结果
        batch_size: 批量推理的大小
        skip_existing: 如果输出文件已存在，是否跳过
        fps: 视频的实际帧率（Hz），用于下采样到5Hz
        direct_mask: 是否直接生成mask而不保存光流
        min_threshold: mask生成的最小阈值
        top_percentile: mask生成的分位数
        noise_threshold: mask生成的噪声阈值
    """
    data_path = Path(data_root)
    output_path = Path(output_root)
    
    # 遍历所有image_X目录
    image_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("image") and "wrist" not in d.name])
    
    if not image_dirs:
        print(f"No 'image_*' directories found in {data_root}")
        return
    
    print(f"Found {len(image_dirs)} image directories")
    
    for image_dir in image_dirs:
        print(f"\nProcessing directory: {image_dir.name}")
        
        # 遍历该目录下的所有视频序列
        video_sequences = sorted([d for d in image_dir.iterdir() if d.is_dir()])
        
        if not video_sequences:
            print(f"  No video sequences found in {image_dir}")
            continue
        
        print(f"  Found {len(video_sequences)} video sequences")
        
        for video_seq in video_sequences[:]:
            # 创建对应的输出目录
            relative_path = video_seq.relative_to(data_path)
            output_seq_dir = output_path / relative_path
            
            try:
                process_video_sequence(
                    model, video_seq, output_seq_dir, device, save_vis,
                    batch_size=batch_size, skip_existing=skip_existing, fps=fps,
                    direct_mask=direct_mask,
                    min_threshold=min_threshold,
                    top_percentile=top_percentile,
                    noise_threshold=noise_threshold
                )
            except Exception as e:
                print(f"Error processing {video_seq}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="批量推理视频序列的flow")
    parser.add_argument(
        "--data_root", "-d",
        required=True,
        help="数据根目录路径（包含image_X目录的路径）"
    )
    parser.add_argument(
        "--output_root", "-o",
        required=True,
        help="输出根目录路径"
    )
    parser.add_argument(
        "--model",
        choices=["base", "refine"],
        default="base",
        help="模型版本"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="计算设备"
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="不保存可视化结果（默认保存）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批量推理的大小（默认8）"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果输出文件已存在则跳过（默认强制覆盖）"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="视频的实际帧率（Hz），用于下采样到5Hz。如果帧率>5Hz，会自动下采样（默认30Hz）"
    )
    parser.add_argument(
        "--direct_mask",
        action="store_true",
        help="直接生成mask而不保存光流文件（默认保存光流）"
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=1.0,
        help="mask生成的最小模长阈值（像素），用于direct_mask模式（默认：1.0）"
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=10.0,
        help="mask生成的分位数（0-100），用于direct_mask模式（默认：10）"
    )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default=5.0,
        help="mask生成的噪声阈值（像素），用于direct_mask模式（默认：5.0）"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading UFM {args.model} model...")
    if args.model == "refine":
        model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine")
    else:
        model = UniFlowMatchConfidence.from_pretrained("/mnt/raid0/data/UFM-Base")
    
    model.eval()
    model.to(args.device)
    print(f"Model loaded successfully! Using device: {args.device}")
    
    # 处理数据
    print(f"\nProcessing data from: {args.data_root}")
    print(f"Output will be saved to: {args.output_root}")
    if args.direct_mask:
        print(f"Mode: Direct mask generation (RLE compressed)")
        print(f"Mask parameters: min_threshold={args.min_threshold}, top_percentile={args.top_percentile}, noise_threshold={args.noise_threshold}")
    else:
        print(f"Mode: Flow saving")
        print(f"Data format: npz (flow + covisibility in one file)")
    print(f"Batch size: {args.batch_size}")
    print(f"FPS: {args.fps}Hz (will downsample to 5Hz if > 5Hz)")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Save visualizations: {not args.no_vis}\n")
    
    process_data_directory(
        model, args.data_root, args.output_root, args.device,
        save_vis=not args.no_vis,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        fps=args.fps,
        direct_mask=args.direct_mask,
        min_threshold=args.min_threshold,
        top_percentile=args.top_percentile,
        noise_threshold=args.noise_threshold
    )
    
    print("\n所有处理完成！")


if __name__ == "__main__":
    main()

