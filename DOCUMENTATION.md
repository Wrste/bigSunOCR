# BigSunOCR 项目文档 | Project Documentation

## 目录 | Table of Contents

- [项目简介 | Project Introduction](#项目简介--project-introduction)
- [技术架构 | Technical Architecture](#技术架构--technical-architecture)
- [安装指南 | Installation Guide](#安装指南--installation-guide)
- [使用方法 | Usage](#使用方法--usage)
- [API参考 | API Reference](#api参考--api-reference)
- [模型训练 | Model Training](#模型训练--model-training)
- [性能评估 | Performance Evaluation](#性能评估--performance-evaluation)
- [常见问题 | FAQ](#常见问题--faq)
- [贡献指南 | Contribution Guidelines](#贡献指南--contribution-guidelines)
- [许可证 | License](#许可证--license)

## 项目简介 | Project Introduction

BigSunOCR是一个专注于数学公式识别的光学字符识别（OCR）系统，由成都微珑汇科技有限公司（WIHEX.INC）的高级人工智能算法工程师开发。该项目旨在满足教育和研究领域对于低成本、高效率识别手写数学公式和复杂印刷公式的需求。

BigSunOCR is an Optical Character Recognition (OCR) system focused on mathematical formula recognition, developed by a senior artificial intelligence algorithm engineer from Chengdu Weilonghui Technology Co., Ltd. (WIHEX. INC). The project aims to meet the needs of the education and research fields for low-cost, efficient recognition of handwritten mathematical formulas and complex printed formulas.

### 主要特点 | Key Features

- **高精度识别**：针对手写和印刷数学公式进行了优化，识别准确率高
- **支持LaTeX**：直接输出LaTeX格式，方便在学术论文和教学材料中使用
- **轻量级模型**：模型体积小，推理速度快，适合在普通硬件上运行
- **易于集成**：提供简单的API和命令行工具，方便集成到其他系统中

- **High-accuracy Recognition**: Optimized for handwritten and printed mathematical formulas with high recognition accuracy
- **LaTeX Support**: Direct output in LaTeX format, convenient for use in academic papers and teaching materials
- **Lightweight Model**: Small model size, fast inference speed, suitable for running on ordinary hardware
- **Easy Integration**: Provides simple API and command-line tools for easy integration into other systems

## 技术架构 | Technical Architecture

BigSunOCR基于深度学习技术，主要参考了CRNN（Convolutional Recurrent Neural Network）的技术特点，并针对LaTeX公式的长序列特性进行了改进。

BigSunOCR is based on deep learning technology, mainly referring to the technical characteristics of CRNN (Convolutional Recurrent Neural Network), and has been improved for the long sequence characteristics of LaTeX formulas.

### 模型架构 | Model Architecture

模型主要由以下几个部分组成：

1. **特征提取网络**：使用ResNet18作为骨干网络，提取图像特征
2. **序列建模**：使用双向LSTM处理特征序列
3. **解码器**：使用CTC（Connectionist Temporal Classification）损失函数进行训练和解码

The model mainly consists of the following parts:

1. **Feature Extraction Network**: Uses ResNet18 as the backbone network to extract image features
2. **Sequence Modeling**: Uses bidirectional LSTM to process feature sequences
3. **Decoder**: Uses CTC (Connectionist Temporal Classification) loss function for training and decoding

### 技术创新点 | Technical Innovations

1. **改进的CNN输出张量形状**：通过改变CNN层输出的数据张量形状，并重新排列张量的维度顺序，实现了对图片LaTeX公式的长序列支持
2. **图像位置编码**：引入位置编码，增强模型对公式结构的理解
3. **SEBlock（Squeeze-and-Excitation Block）**：增强特征表示能力
4. **残差连接**：解决深层网络训练困难的问题

1. **Improved CNN Output Tensor Shape**: By changing the shape of the data tensor output by the CNN layer and rearranging the dimension order of the tensor, support for long sequences of image LaTeX formulas is achieved
2. **Image Position Encoding**: Introduces position encoding to enhance the model's understanding of formula structure
3. **SEBlock (Squeeze-and-Excitation Block)**: Enhances feature representation capability
4. **Residual Connections**: Solves the problem of training difficulties in deep networks

## 安装指南 | Installation Guide

### 环境要求 | Environment Requirements

- Python >= 3.8
- PyTorch == 2.4.0
- OpenCV == 4.5.5
- 其他依赖见requirements.txt

### 安装步骤 | Installation Steps

1. 克隆仓库 | Clone the repository
```bash
git clone https://github.com/yourusername/bigSunOCR.git
cd bigSunOCR
```

2. 安装依赖 | Install dependencies
```bash
pip install -r requirements.txt
```

3. 下载预训练模型 | Download pre-trained model
```bash
# 下载模型并放入model_data文件夹
# Download the model and put it in the model_data folder
# 模型下载链接 | Model download link: https://jidugs.wlhex.com/latex_OCR_model.pth
```

## 使用方法 | Usage

### 命令行使用 | Command Line Usage

使用预训练模型进行预测：

Using pre-trained model for prediction:

```bash
python predict.py --image_path path/to/your/image.jpg --visualize
```

参数说明 | Parameter description:
- `--image_path`: 输入图像路径 | Input image path
- `--model_path`: 模型路径，默认为`./model_data/latex_OCR_model.pth` | Model path, default is `./model_data/latex_OCR_model.pth`
- `--visualize`: 是否可视化结果 | Whether to visualize the result

### Python API使用 | Python API Usage

```python
from predict import predict

# 预测单个图像
# Predict a single image
result = predict("path/to/your/image.jpg")
print(f"LaTeX公式 | LaTeX Formula: {result['prediction']}")
print(f"推理时间 | Inference Time: {result['inference_time']:.4f} 秒 | seconds")
```

### 批量处理 | Batch Processing

```python
import os
from predict import predict
from utils import ensure_dir

# 输入和输出目录
# Input and output directories
input_dir = "path/to/input/images"
output_dir = "path/to/output/results"
ensure_dir(output_dir)

# 批量处理图像
# Batch process images
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        result = predict(image_path)
        
        if result:
            # 保存结果到文本文件
            # Save result to text file
            with open(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt"), "w") as f:
                f.write(result['prediction'])
            
            print(f"处理完成 | Processed: {filename}")
```

## API参考 | API Reference

### 主要模块 | Main Modules

#### `predict.py`

- `predict(image_path, model_path="./model_data/latex_OCR_model.pth")`: 使用模型预测图像中的LaTeX公式 | Use the model to predict LaTeX formulas in images
- `visualize_result(image_path, prediction)`: 可视化预测结果 | Visualize prediction results

#### `utils.py`

- `load_image(image_path, target_size=(512, 128), normalize=True)`: 加载并预处理图像 | Load and preprocess images
- `load_vocab_dict(vocab_file_path)`: 加载词汇表字典 | Load vocabulary dictionary
- `decode_predictions(output, vocab_dict, remove_duplicates=True)`: 解码模型输出的预测结果 | Decode the prediction results output by the model
- `calculate_accuracy(reference, hypothesis)`: 计算准确度 | Calculate accuracy
- `evaluate_model(model, test_loader, vocab_dict, device)`: 评估模型 | Evaluate model
- `visualize_prediction(image, prediction, reference=None, save_path=None)`: 可视化预测结果 | Visualize prediction results
- `plot_training_history(history, save_path=None)`: 绘制训练历史 | Plot training history
- `ensure_dir(directory)`: 确保目录存在 | Ensure directory exists
- `get_available_device()`: 获取可用设备 | Get available device

#### `model/model.py`

- `class SEBlock(nn.Module)`: Squeeze-and-Excitation Block
- `class ResidualBlock(nn.Module)`: 残差模块 | Residual module
- `class ResNet18(nn.Module)`: ResNet18骨干网络 | ResNet18 backbone network
- `class Image_CNN(nn.Module)`: 图像CNN特征提取器 | Image CNN feature extractor
- `class BidirectionalLSTM(nn.Module)`: 双向LSTM | Bidirectional LSTM
- `class Decode(nn.Module)`: 解码器 | Decoder

## 模型训练 | Model Training

### 数据准备 | Data Preparation

训练数据集包含1200个手写数学公式图片文件，测试集包含70个手写数学公式图片文件，验证集包含68个手写数学公式图片文件。

The training dataset contains 1200 handwritten mathematical formula image files, the test set contains 70 handwritten mathematical formula image files, and the validation set contains 68 handwritten mathematical formula image files.

### 训练过程 | Training Process

使用`trainModel.py`脚本进行训练：

Use the `trainModel.py` script for training:

```bash
python trainModel.py
```

训练参数可以在脚本中调整，包括学习率、批次大小、训练轮次等。

Training parameters can be adjusted in the script, including learning rate, batch size, training epochs, etc.

### 训练技巧 | Training Tips

1. **数据增强**：通过旋转、缩放、添加噪声等方式增加训练数据的多样性
2. **学习率调度**：使用余弦退火学习率调度器，提高训练稳定性
3. **梯度裁剪**：防止梯度爆炸问题
4. **早停**：当验证集性能不再提升时停止训练，防止过拟合

1. **Data Augmentation**: Increase the diversity of training data through rotation, scaling, adding noise, etc.
2. **Learning Rate Scheduling**: Use cosine annealing learning rate scheduler to improve training stability
3. **Gradient Clipping**: Prevent gradient explosion problems
4. **Early Stopping**: Stop training when validation set performance no longer improves to prevent overfitting

## 性能评估 | Performance Evaluation

### 训练效果 | Training Results

- 总损失 | Total Loss: 1.1550
- 平均损失 | Average Loss: 0.004812415804311362
- 平均准确度 | Average Accuracy: 0.98

### 测试效果 | Testing Results (CPU)

- 平均准确度 | Average Accuracy: 0.7133518634119548
- 平均推理时间 | Average Inference Time: 0.0876441 秒 | seconds

### 验证效果 | Validation Results (CPU)

- 平均准确度 | Average Accuracy: 0.778784725587209
- 平均推理时间 | Average Inference Time: 0.0857531323529412 秒 | seconds

### 测试环境 | Testing Environment

- CPU型号 | CPU Model: i7 14700HK
- 内存 | RAM: 32GB

## 常见问题 | FAQ

### 1. 模型无法正确识别复杂公式怎么办？

对于非常复杂的公式，可以尝试以下方法：
- 提高图像分辨率和质量
- 确保公式在图像中居中且清晰
- 对于手写公式，尽量保持书写工整

### 1. What if the model cannot correctly recognize complex formulas?

For very complex formulas, you can try the following methods:
- Improve image resolution and quality
- Ensure that the formula is centered and clear in the image
- For handwritten formulas, try to keep the writing neat

### 2. 如何提高识别准确率？

- 使用更高质量的图像
- 对图像进行预处理，如二值化、去噪等
- 针对特定领域的公式进行微调训练

### 2. How to improve recognition accuracy?

- Use higher quality images
- Preprocess images, such as binarization, denoising, etc.
- Fine-tune training for formulas in specific fields

### 3. 模型支持哪些数学符号？

模型支持大多数常见的数学符号，包括：
- 基本运算符（+, -, ×, ÷, =, etc.）
- 希腊字母（α, β, γ, etc.）
- 积分、求和、极限符号
- 分数、根号、上下标
- 矩阵表示

### 3. What mathematical symbols does the model support?

The model supports most common mathematical symbols, including:
- Basic operators (+, -, ×, ÷, =, etc.)
- Greek letters (α, β, γ, etc.)
- Integral, summation, limit symbols
- Fractions, square roots, superscripts and subscripts
- Matrix representation

## 贡献指南 | Contribution Guidelines

我们欢迎各种形式的贡献，包括但不限于：
- 报告问题和提出建议
- 提交代码改进
- 改进文档
- 分享使用经验

如果您想贡献代码，请遵循以下步骤：
1. Fork仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

We welcome contributions of all kinds, including but not limited to:
- Reporting issues and suggesting improvements
- Submitting code improvements
- Improving documentation
- Sharing usage experiences

If you want to contribute code, please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 许可证 | License

Apache License Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright (c) 2024 XingChengFu (bigSun), WLHEX INC.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. 