"""
BigSunOCR Web演示
BigSunOCR Web Demo

一个简单的Web界面，用于演示BigSunOCR的功能
A simple web interface to demonstrate the functionality of BigSunOCR
"""

import os
import time
import base64
from io import BytesIO
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 导入项目模块
# Import project modules
from predict import predict
from utils import ensure_dir

app = Flask(__name__)

# 确保上传和结果目录存在
# Ensure upload and result directories exist
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ensure_dir(UPLOAD_FOLDER)
ensure_dir(RESULT_FOLDER)

@app.route('/')
def index():
    """
    渲染主页
    Render the home page
    """
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    """
    处理上传的图像并返回识别结果
    Process the uploaded image and return recognition results
    """
    if 'file' not in request.files:
        return jsonify({'error': '没有文件 | No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件 | No selected file'})
    
    if file:
        # 保存上传的文件
        # Save the uploaded file
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 进行预测
        # Make prediction
        result = predict(filepath)
        
        if not result:
            return jsonify({'error': '预测失败 | Prediction failed'})
        
        # 生成可视化结果
        # Generate visualization result
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.title(f"预测结果 | Prediction: {result['prediction']}")
        plt.axis('off')
        
        result_path = os.path.join(RESULT_FOLDER, f"result_{filename}.png")
        plt.savefig(result_path)
        plt.close()
        
        # 将图像转换为base64以便在前端显示
        # Convert image to base64 for display in frontend
        with open(result_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'inference_time': f"{result['inference_time']:.4f}",
            'image': img_data
        })
    
    return jsonify({'error': '处理失败 | Processing failed'})

# 创建HTML模板目录和文件
# Create HTML template directory and file
def create_template():
    """
    创建HTML模板
    Create HTML template
    """
    template_dir = 'templates'
    ensure_dir(template_dir)
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>BigSunOCR - 数学公式识别 | Mathematical Formula Recognition</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .upload-form {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .file-input {
            margin: 10px 0;
        }
        .submit-btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            display: none;
        }
        .result img {
            max-width: 100%;
            margin-top: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .error {
            color: red;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>
    <h1>BigSunOCR - 数学公式识别 | Mathematical Formula Recognition</h1>
    
    <div class="container">
        <p>上传一张包含数学公式的图像，系统将自动识别并转换为LaTeX格式。</p>
        <p>Upload an image containing mathematical formulas, and the system will automatically recognize and convert it to LaTeX format.</p>
        
        <form class="upload-form" id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" id="file" class="file-input" accept="image/*">
            <button type="submit" class="submit-btn">识别 | Recognize</button>
        </form>
        
        <div class="loading" id="loading">
            <p>正在处理，请稍候... | Processing, please wait...</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result" id="result">
            <h3>识别结果 | Recognition Result:</h3>
            <p><strong>LaTeX公式 | LaTeX Formula:</strong> <span id="prediction"></span></p>
            <p><strong>推理时间 | Inference Time:</strong> <span id="inference-time"></span> 秒 | seconds</p>
            <h3>可视化 | Visualization:</h3>
            <img id="result-image" src="" alt="识别结果 | Recognition Result">
        </div>
    </div>
    
    <script>
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('file');
            if (!fileInput.files.length) {
                document.getElementById('error').textContent = '请选择一个文件 | Please select a file';
                document.getElementById('error').style.display = 'block';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            fetch('/recognize', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    return;
                }
                
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('inference-time').textContent = data.inference_time;
                document.getElementById('result-image').src = 'data:image/png;base64,' + data.image;
                document.getElementById('result').style.display = 'block';
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = '处理请求时出错 | Error processing request: ' + error;
                document.getElementById('error').style.display = 'block';
            });
        });
    </script>
</body>
</html>
    """
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    # 创建模板
    # Create template
    create_template()
    
    # 启动Flask应用
    # Start Flask application
    print("启动Web演示，请访问 http://127.0.0.1:5000")
    print("Starting Web demo, please visit http://127.0.0.1:5000")
    app.run(debug=True) 