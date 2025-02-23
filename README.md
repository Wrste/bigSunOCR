#### Project Name

**bigSunOCR**

#### Project Description

This project is developed by a senior artificial intelligence algorithm engineer from Chengdu Weilonghui Technology Co., Ltd. (WIHEX. INC). The main goal of the project is to meet the company's requirements for low-cost training and inference related OCR tasks.

This project aims to achieve good results in handwritten mathematical formulas, printed formulas, complex formula samples, or comprehensive optical character recognition tasks.

**Instructions and references related to this project:**

This project mainly refers to the relevant technical characteristics of CRNN and improves the problem that CRNN does not support the LaTeX formula for long sequences. By changing the shape of the data tensor produced by the CNN layer and rearranging the dimension order of the tensor, the LaTeX formula for images can be implemented in a long-term manner. The project does not use attention mechanism, but instead fuses the features of the images into a ResNet18 network. After passing through two bidirectional LSTMs and the log_stoftmax activation function, the CTC loss function is used to calculate the loss and update the network gradient. Some methods are also used to optimize the network, such as image position encoding and SEBlock, which may not be added, but the effect may not be very good. You can try it yourself

**Related environment**

python>=3.8 

pytorch=2.4.0

opencv-python



**Project leader:**

Nickname: Big Sun

Email: 775106129@qq.com

Company email: fxc@wlhex.com
If you need to conduct academic research, you can also send me an email to obtain technical details

#### **train**

The training set contains 1200 handwritten mathematical formula image files, the testing set contains 70 handwritten mathematical formula image files, and the validation set contains 68 handwritten mathematical formula image files

**The final effect of model training**

Total Loss: 1.1550
Average Loss: 0.004812415804311362

Average accuracy: 0.98

#### **Testing and Verification**

**test on cpu **

Average accuracy: 0.7133518634119548
Average inference time: 0.0876441 Seconds

**val on cpu **

Average accuracy: 0.778784725587209
Average inference time: 0.0857531323529412 Seconds

**device config**

CPU model:i7 14700HK，RAM:32GB



#### 项目名称

**bigSunOCR**

#### 项目描述

这个项目由成都微珑汇科技有限公司（WIHEX.INC）的高级人工智能算法工程师开发。项目的主要目标是满足公司对于低成本训练和推理的相关OCR任务需要。

该项目旨在对手写数学公式、印刷体公式以及复杂公式样本或综合性光学字符识别任务上达到良好的效果。

**该项目相关说明与参考：**

该项目主要是的参考了CRNN的相关技术特点，并改进了CRNN对长序列的latex公式不支持的问题，通过改变CNN层出来的数据张量的形状，并重新排列张量的维度顺序，来实现对图片latex公式的长序列支持，项目没有采用注意力机制，而是通过将图片进行特征融合送入一个ResNet18网络，再经过两个双向lstm经过log_softmax激活函数后，再使用CTC损失函数计算损失并更新网络梯度，也采用了一些方法优化该网络例如:图片位置编码，SEBlock等也可以不加但是也许效果不太好，可以自己动手试试.

**相关环境**

python>=3.8 

pytorch=2.4.0

opencv-python



**项目负责人:**

昵称:太阳大(big sun)

邮箱:775106129@qq.com

公司邮箱:fxc@wlhex.com
需要做学术研究也可以发邮件给我，获得技术细节

#### 训练

训练集有1200个手写数学公式图片文件，测试集70个手写数学公式图片文件，验证集：68个手写数学公式图片。

**模型训练的最终效果**

总损失：1.1550
平均损失：0.004812415804311362

平均准确度：0.98

**测试和验证**

**cpu运行测试数据集**

平均准确度：0.7133518634119548
平均推理时间：0.0876441秒

**cpu上运行验证数据集**

平均准确度：0.778784725587209
平均推理时间：0.0857531323529412秒

**设备配置**

CPU型号：i7 14700HK，内存：32GB



#### license

Apache License                           

Version 2.0, January 2004                        

http://www.apache.org/licenses/ 

Copyright (c) 2024 XingChengFu (bigSun), WLHEX INC.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at     http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.