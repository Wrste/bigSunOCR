"""
BigSunOCR 使用示例
BigSunOCR Usage Example

本示例展示了如何使用BigSunOCR模型进行数学公式识别
This example demonstrates how to use the BigSunOCR model for mathematical formula recognition
"""

import sys
import os

# 添加项目根目录到系统路径
# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import model

# 设置设备
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备 | Using device: {device}")

def load_vocab_dict(vocab_file_path):
    """
    加载词汇表字典
    Load vocabulary dictionary
    """
    vocab_dict = {}
    with open(vocab_file_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                index = int(parts[1])
                vocab_dict[index] = word
    return vocab_dict

def preprocess_image(image_path, target_size=(512, 128)):
    """
    预处理图像
    Preprocess image
    """
    # 读取图像
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像 | Cannot read image: {image_path}")
    
    # 转换为RGB
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小
    # Resize
    img = cv2.resize(img, target_size)
    
    # 归一化
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # 转换为PyTorch张量并添加批次维度
    # Convert to PyTorch tensor and add batch dimension
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, img

def decode_prediction(output, vocab_dict):
    """
    解码模型输出
    Decode model output
    """
    # 使用贪婪解码
    # Use greedy decoding
    _, max_indices = torch.max(output, 2)
    
    # 将张量转换为列表
    # Convert tensor to list
    indices = max_indices.cpu().numpy().tolist()[0]
    
    # 移除重复的连续索引
    # Remove consecutive duplicates
    prev_idx = -1
    decoded_seq = []
    for idx in indices:
        if idx != prev_idx and idx != 0:  # 0通常是填充符号 | 0 is usually a padding symbol
            decoded_seq.append(idx)
        prev_idx = idx
    
    # 将索引转换为字符
    # Convert indices to characters
    decoded_text = [vocab_dict.get(idx, "") for idx in decoded_seq]
    
    return "".join(decoded_text)

def main():
    """
    主函数
    Main function
    """
    # 模型和词汇表路径
    # Model and vocabulary paths
    model_path = "../model_data/latex_OCR_model.pth"
    vocab_path = "../Data-for-LaTeX_OCR/vocabDict/vocab_hotel_dict.txt"
    
    # 示例图像路径
    # Example image path
    image_path = "../examples/example_formula.jpg"
    
    # 检查文件是否存在
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"模型文件不存在 | Model file does not exist: {model_path}")
        print("请确保您已下载模型并放置在正确的位置")
        print("Please make sure you have downloaded the model and placed it in the correct location")
        return
    
    if not os.path.exists(vocab_path):
        print(f"词汇表文件不存在 | Vocabulary file does not exist: {vocab_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"示例图像不存在 | Example image does not exist: {image_path}")
        print("请提供一个有效的图像文件")
        print("Please provide a valid image file")
        return
    
    try:
        # 加载词汇表
        # Load vocabulary
        vocab_dict = load_vocab_dict(vocab_path)
        
        # 预处理图像
        # Preprocess image
        img_tensor, original_img = preprocess_image(image_path)
        
        # 加载模型
        # Load model
        print("正在加载模型... | Loading model...")
        ocr_model = model.Decode(device, 769)
        ocr_model.load_state_dict(torch.load(model_path, map_location=device))
        ocr_model.to(device)
        ocr_model.eval()
        print("模型加载成功 | Model loaded successfully")
        
        # 进行预测
        # Make prediction
        print("正在进行预测... | Making prediction...")
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            output = ocr_model(img_tensor)
        
        # 解码预测结果
        # Decode prediction result
        prediction = decode_prediction(output, vocab_dict)
        
        # 显示结果
        # Display result
        print("\n" + "="*50)
        print("预测结果 | Prediction Result:")
        print("="*50)
        print(f"LaTeX公式 | LaTeX Formula: {prediction}")
        print("="*50 + "\n")
        
        # 可视化结果
        # Visualize result
        plt.figure(figsize=(12, 6))
        plt.imshow(original_img)
        plt.title(f"预测结果 | Prediction: {prediction}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"发生错误 | An error occurred: {e}")

if __name__ == "__main__":
    main() 