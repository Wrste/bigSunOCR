"""
BigSunOCR 工具函数
BigSunOCR Utility Functions

本文件包含一些常用的辅助函数，如图像处理、模型评估等
This file contains some common utility functions, such as image processing, model evaluation, etc.
"""

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import logging

# 配置日志记录器
# Configure logger
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# 图像处理函数
# Image processing functions

def load_image(image_path, target_size=(512, 128), normalize=True):
    """
    加载并预处理图像
    Load and preprocess image
    
    参数:
    - image_path: 图像路径
    - target_size: 目标大小，默认为(512, 128)
    - normalize: 是否归一化，默认为True
    
    返回:
    - img_tensor: 预处理后的图像张量
    - original_img: 原始图像（调整大小后）
    """
    try:
        # 读取图像
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像 | Cannot read image: {image_path}")
        
        # 转换为RGB
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 保存原始图像（调整大小后）
        # Save original image (after resizing)
        original_img = cv2.resize(img, target_size)
        
        # 归一化
        # Normalize
        if normalize:
            img = original_img.astype(np.float32) / 255.0
        else:
            img = original_img.astype(np.float32)
        
        # 转换为PyTorch张量并添加批次维度
        # Convert to PyTorch tensor and add batch dimension
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, original_img
    
    except Exception as e:
        logger.error(f"图像处理错误 | Image processing error: {e}")
        return None, None

def batch_process_images(image_paths, target_size=(512, 128), batch_size=16):
    """
    批量处理图像
    Batch process images
    
    参数:
    - image_paths: 图像路径列表
    - target_size: 目标大小，默认为(512, 128)
    - batch_size: 批次大小，默认为16
    
    返回:
    - batches: 批次列表，每个批次是一个张量
    """
    batches = []
    current_batch = []
    
    for path in tqdm(image_paths, desc="处理图像 | Processing images"):
        img_tensor, _ = load_image(path, target_size)
        if img_tensor is not None:
            current_batch.append(img_tensor)
            
            if len(current_batch) == batch_size:
                batches.append(torch.cat(current_batch, dim=0))
                current_batch = []
    
    # 处理剩余的图像
    # Process remaining images
    if current_batch:
        batches.append(torch.cat(current_batch, dim=0))
    
    return batches

# 词汇表处理函数
# Vocabulary processing functions

def load_vocab_dict(vocab_file_path):
    """
    加载词汇表字典
    Load vocabulary dictionary
    
    参数:
    - vocab_file_path: 词汇表文件路径
    
    返回:
    - vocab_dict: 词汇表字典，键为索引，值为字符
    """
    vocab_dict = {}
    try:
        with open(vocab_file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    index = int(parts[1])
                    vocab_dict[index] = word
        return vocab_dict
    except Exception as e:
        logger.error(f"加载词汇表错误 | Load vocabulary error: {e}")
        return {}

def decode_predictions(output, vocab_dict, remove_duplicates=True):
    """
    解码模型输出的预测结果
    Decode model output predictions
    
    参数:
    - output: 模型输出
    - vocab_dict: 词汇表字典
    - remove_duplicates: 是否移除重复的连续索引，默认为True
    
    返回:
    - decoded_text: 解码后的文本
    """
    # 使用贪婪解码（选择每个时间步概率最高的字符）
    # Use greedy decoding (select the character with the highest probability at each time step)
    _, max_indices = torch.max(output, 2)
    
    # 将张量转换为列表
    # Convert tensor to list
    indices = max_indices.cpu().numpy().tolist()[0]
    
    if remove_duplicates:
        # 移除重复的连续索引
        # Remove consecutive duplicates
        prev_idx = -1
        decoded_seq = []
        for idx in indices:
            if idx != prev_idx and idx != 0:  # 0通常是填充符号 | 0 is usually a padding symbol
                decoded_seq.append(idx)
            prev_idx = idx
    else:
        # 只移除填充符号
        # Only remove padding symbols
        decoded_seq = [idx for idx in indices if idx != 0]
    
    # 将索引转换为字符
    # Convert indices to characters
    decoded_text = [vocab_dict.get(idx, "") for idx in decoded_seq]
    
    return "".join(decoded_text)

# 模型评估函数
# Model evaluation functions

def calculate_accuracy(reference, hypothesis):
    """
    计算准确度
    Calculate accuracy
    
    参数:
    - reference: 参考文本
    - hypothesis: 预测文本
    
    返回:
    - accuracy: 准确度
    """
    # 计算字符级别的准确度
    # Calculate character-level accuracy
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    # 计算编辑距离
    # Calculate edit distance
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    edit_distance = dp[m][n]
    
    # 计算准确度
    # Calculate accuracy
    if m == 0:
        return 0.0
    
    accuracy = 1.0 - edit_distance / max(m, n)
    return accuracy

def evaluate_model(model, test_loader, vocab_dict, device):
    """
    评估模型
    Evaluate model
    
    参数:
    - model: 模型
    - test_loader: 测试数据加载器
    - vocab_dict: 词汇表字典
    - device: 设备
    
    返回:
    - avg_accuracy: 平均准确度
    - avg_time: 平均推理时间
    """
    model.eval()
    total_accuracy = 0.0
    total_time = 0.0
    count = 0
    
    with torch.no_grad():
        for batch, labels in tqdm(test_loader, desc="评估模型 | Evaluating model"):
            batch = batch.to(device)
            
            # 计时
            # Timing
            start_time = time.time()
            output = model(batch)
            end_time = time.time()
            
            inference_time = end_time - start_time
            total_time += inference_time
            
            # 解码预测结果
            # Decode prediction results
            for i in range(len(output)):
                pred_text = decode_predictions(output[i].unsqueeze(0), vocab_dict)
                ref_text = labels[i]
                
                accuracy = calculate_accuracy(ref_text, pred_text)
                total_accuracy += accuracy
                count += 1
    
    avg_accuracy = total_accuracy / count if count > 0 else 0.0
    avg_time = total_time / count if count > 0 else 0.0
    
    return avg_accuracy, avg_time

# 可视化函数
# Visualization functions

def visualize_prediction(image, prediction, reference=None, save_path=None):
    """
    可视化预测结果
    Visualize prediction result
    
    参数:
    - image: 图像
    - prediction: 预测文本
    - reference: 参考文本，默认为None
    - save_path: 保存路径，默认为None
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    
    if reference:
        title = f"预测 | Prediction: {prediction}\n参考 | Reference: {reference}"
    else:
        title = f"预测 | Prediction: {prediction}"
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存至 | Result saved to: {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    """
    绘制训练历史
    Plot training history
    
    参数:
    - history: 训练历史，包含'loss'和'accuracy'键
    - save_path: 保存路径，默认为None
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('模型损失 | Model Loss')
    plt.xlabel('轮次 | Epoch')
    plt.ylabel('损失 | Loss')
    
    # 绘制准确度
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('模型准确度 | Model Accuracy')
    plt.xlabel('轮次 | Epoch')
    plt.ylabel('准确度 | Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"训练历史已保存至 | Training history saved to: {save_path}")
    
    plt.show()

# 其他辅助函数
# Other utility functions

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    Ensure directory exists, create if not
    
    参数:
    - directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录 | Created directory: {directory}")

def get_available_device():
    """
    获取可用设备
    Get available device
    
    返回:
    - device: 设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
        print("Using CPU")
    
    return device 