import argparse
import cv2
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from model import model
import logging

# 配置日志记录器
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"Using device: {device}")

# 加载词汇表
vocab_file = "./Data-for-LaTeX_OCR/vocabDict/vocab_hotel_dict.txt"

def load_vocab_hotel_dict():
    """加载词汇表字典"""
    vocab_dict = {}
    with open(vocab_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                word = parts[0]
                index = int(parts[1])
                vocab_dict[index] = word
    return vocab_dict

def decode_predictions(out):
    """解码模型输出的预测结果"""
    vocab_dict = load_vocab_hotel_dict()
    # 使用贪婪解码（选择每个时间步概率最高的字符）
    _, max_indices = torch.max(out, 2)
    
    # 将张量转换为列表
    indices = max_indices.cpu().numpy().tolist()[0]
    
    # 移除重复的连续索引
    prev_idx = -1
    decoded_seq = []
    for idx in indices:
        if idx != prev_idx and idx != 0:  # 0通常是填充符号
            decoded_seq.append(idx)
        prev_idx = idx
    
    # 将索引转换为字符
    decoded_text = [vocab_dict.get(idx, "") for idx in decoded_seq]
    
    return "".join(decoded_text)

def load_image(path):
    """加载并预处理图像"""
    try:
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像: {path}")
        
        # 转换为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小为模型输入尺寸
        img = cv2.resize(img, (512, 128))
        
        # 归一化
        img = img.astype(np.float32) / 255.0
        
        # 转换为PyTorch张量并添加批次维度
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor
    
    except Exception as e:
        print(f"图像处理错误: {e}")
        print(f"Image processing error: {e}")
        return None

def predict(image_path, model_path="./model_data/latex_OCR_model.pth"):
    """使用模型预测图像中的LaTeX公式"""
    try:
        # 加载图像
        img_tensor = load_image(image_path)
        if img_tensor is None:
            return None
        
        # 加载模型
        print("正在加载模型...")
        print("Loading model...")
        
        # 创建模型实例
        ocr_model = model.Decode(device, 769)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            ocr_model.load_state_dict(torch.load(model_path, map_location=device))
            ocr_model.to(device)
            ocr_model.eval()
            print("模型加载成功")
            print("Model loaded successfully")
        else:
            print(f"模型文件不存在: {model_path}")
            print(f"Model file does not exist: {model_path}")
            return None
        
        # 预测
        print("正在进行预测...")
        print("Making prediction...")
        
        start_time = time.time()
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            output = ocr_model(img_tensor)
        
        # 解码预测结果
        prediction = decode_predictions(output)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        return {
            "prediction": prediction,
            "inference_time": inference_time
        }
    
    except Exception as e:
        print(f"预测过程中出错: {e}")
        print(f"Error during prediction: {e}")
        logging.error(f"Error during prediction: {e}")
        return None

def visualize_result(image_path, prediction):
    """可视化预测结果"""
    try:
        # 读取原始图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 显示图像
        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.title(f"预测结果 | Prediction: {prediction}")
        plt.axis('off')
        
        # 保存结果
        result_path = f"result_{os.path.basename(image_path)}"
        plt.savefig(result_path)
        print(f"结果已保存至: {result_path}")
        print(f"Result saved to: {result_path}")
        
        # 显示结果
        plt.show()
    
    except Exception as e:
        print(f"可视化结果时出错: {e}")
        print(f"Error visualizing result: {e}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LaTeX公式OCR识别 | LaTeX Formula OCR Recognition")
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径 | Input image path")
    parser.add_argument("--model_path", type=str, default="./model_data/latex_OCR_model.pth", 
                        help="模型路径 | Model path")
    parser.add_argument("--visualize", action="store_true", 
                        help="是否可视化结果 | Whether to visualize the result")
    
    args = parser.parse_args()
    
    # 检查图像路径
    if not os.path.exists(args.image_path):
        print(f"图像文件不存在: {args.image_path}")
        print(f"Image file does not exist: {args.image_path}")
        return
    
    # 进行预测
    result = predict(args.image_path, args.model_path)
    
    if result:
        print("\n" + "="*50)
        print("预测结果 | Prediction Result:")
        print("="*50)
        print(f"LaTeX公式 | LaTeX Formula: {result['prediction']}")
        print(f"推理时间 | Inference Time: {result['inference_time']:.4f} 秒 | seconds")
        print("="*50 + "\n")
        
        # 记录到日志
        logging.info(f"Image: {args.image_path}, Prediction: {result['prediction']}, Time: {result['inference_time']:.4f}s")
        
        # 可视化结果
        if args.visualize:
            visualize_result(args.image_path, result['prediction'])
    
if __name__ == "__main__":
    main() 