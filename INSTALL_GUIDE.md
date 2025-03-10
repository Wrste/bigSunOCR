# BigSunOCR 安装与使用指南 | Installation and Usage Guide

## 目录 | Table of Contents

- [环境要求 | Environment Requirements](#环境要求--environment-requirements)
- [安装步骤 | Installation Steps](#安装步骤--installation-steps)
- [快速开始 | Quick Start](#快速开始--quick-start)
- [常见问题 | Troubleshooting](#常见问题--troubleshooting)

## 环境要求 | Environment Requirements

BigSunOCR 需要以下环境：

- Python 3.8 或更高版本
- CUDA 支持（可选，用于 GPU 加速）
- 至少 4GB 内存（推荐 8GB 或更多）

BigSunOCR requires the following environment:

- Python 3.8 or higher
- CUDA support (optional, for GPU acceleration)
- At least 4GB of RAM (8GB or more recommended)

## 安装步骤 | Installation Steps

### 1. 克隆仓库 | Clone the Repository

```bash
git clone https://github.com/yourusername/bigSunOCR.git
cd bigSunOCR
```

### 2. 创建虚拟环境（推荐）| Create Virtual Environment (Recommended)

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖 | Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 下载预训练模型 | Download Pre-trained Model

从以下链接下载预训练模型，并将其放入 `model_data` 文件夹：

Download the pre-trained model from the following link and place it in the `model_data` folder:

```
https://jidugs.wlhex.com/latex_OCR_model.pth
```

或者使用以下命令下载（需要 curl）：

Or use the following command to download (requires curl):

```bash
# Windows
curl -o model_data/latex_OCR_model.pth https://jidugs.wlhex.com/latex_OCR_model.pth

# Linux/macOS
curl -o model_data/latex_OCR_model.pth https://jidugs.wlhex.com/latex_OCR_model.pth
```

### 5. 验证安装 | Verify Installation

运行以下命令验证安装是否成功：

Run the following command to verify that the installation was successful:

```bash
python -c "from model import model; print('安装成功！ | Installation successful!')"
```

## 快速开始 | Quick Start

### 使用命令行进行预测 | Using Command Line for Prediction

```bash
python predict.py --image_path path/to/your/image.jpg --visualize
```

参数说明 | Parameter description:
- `--image_path`: 输入图像路径 | Input image path
- `--model_path`: 模型路径，默认为 `./model_data/latex_OCR_model.pth` | Model path, default is `./model_data/latex_OCR_model.pth`
- `--visualize`: 是否可视化结果 | Whether to visualize the result

### 在 Python 代码中使用 | Using in Python Code

```python
from predict import predict

# 预测单个图像
# Predict a single image
result = predict("path/to/your/image.jpg")

if result:
    print(f"LaTeX公式 | LaTeX Formula: {result['prediction']}")
    print(f"推理时间 | Inference Time: {result['inference_time']:.4f} 秒 | seconds")
```

### 批量处理图像 | Batch Processing Images

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

### 运行示例 | Run Example

```bash
python examples/example.py
```

## 常见问题 | Troubleshooting

### 1. 找不到模型文件 | Model File Not Found

确保您已经下载了预训练模型并将其放在正确的位置：

Make sure you have downloaded the pre-trained model and placed it in the correct location:

```
model_data/latex_OCR_model.pth
```

### 2. CUDA 相关错误 | CUDA Related Errors

如果您遇到 CUDA 相关错误，可以尝试在 CPU 上运行：

If you encounter CUDA related errors, you can try running on CPU:

```python
# 在 utils.py 中修改
# Modify in utils.py
def get_available_device():
    return torch.device("cpu")
```

### 3. 依赖项安装失败 | Dependency Installation Failed

如果某些依赖项安装失败，可以尝试单独安装：

If some dependencies fail to install, you can try installing them separately:

```bash
pip install torch==2.4.0 torchvision==0.15.0
pip install opencv-python==4.5.5.64
pip install -r requirements.txt
```

### 4. 图像预处理问题 | Image Preprocessing Issues

如果识别结果不理想，可以尝试调整图像预处理参数：

If the recognition results are not ideal, you can try adjusting the image preprocessing parameters:

```python
# 在 utils.py 中的 load_image 函数中调整
# Adjust in the load_image function in utils.py
img = cv2.resize(img, (640, 160))  # 尝试更高的分辨率 | Try higher resolution
```

### 5. 内存不足 | Out of Memory

如果遇到内存不足的问题，可以尝试减小批次大小：

If you encounter out of memory issues, you can try reducing the batch size:

```python
# 在批处理时调整批次大小
# Adjust batch size during batch processing
batch_process_images(image_paths, batch_size=4)  # 默认为16 | Default is 16
```

如有其他问题，请参考详细文档或联系项目维护者。

For other issues, please refer to the detailed documentation or contact the project maintainer. 