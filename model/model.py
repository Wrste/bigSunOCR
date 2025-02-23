import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3])  # Global Average Pooling
        y = self.fc1(y).relu()
        y = self.fc2(y).sigmoid()
        y = y.view(batch_size, channels, 1, 1)
        return x * y


# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, outchannels):
        super(ResidualBlock, self).__init__()
        self.channel_equal_flag = True
        if in_channels == outchannels:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=3, padding=1, stride=1)
        else:
            ## 对恒等映射分支的变换，当通道数发生变换时，分辨率变为原来的二分之一
            self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels, kernel_size=1, stride=2)
            self.bn1x1 = nn.BatchNorm2d(num_features=outchannels)
            self.channel_equal_flag = False

            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=outchannels,kernel_size=3,padding=1, stride=2)

        self.bn1 = nn.BatchNorm2d(num_features=outchannels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=outchannels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.channel_equal_flag == True:
            pass
        else:
            identity = self.conv1x1(identity)
            identity = self.bn1x1(identity)
            identity = self.relu(identity)

        out = identity + x
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,kernel_size=7,stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.conv2_1 = ResidualBlock(in_channels=64, outchannels=64)
        self.conv2_2 = ResidualBlock(in_channels=64, outchannels=64)

        # conv3_x
        self.conv3_1 = ResidualBlock(in_channels=64, outchannels=128)
        self.conv3_2 = ResidualBlock(in_channels=128, outchannels=128)

        # conv4_x
        self.conv4_1 = ResidualBlock(in_channels=128, outchannels=256)
        self.conv4_2 = ResidualBlock(in_channels=256, outchannels=256)

        # conv5_x
        self.conv5_1 = ResidualBlock(in_channels=256, outchannels=512)
        self.conv5_2 = ResidualBlock(in_channels=512, outchannels=512)




    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2_x
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        # conv3_x
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        # conv4_x
        x = self.conv4_1(x)
        x = self.conv4_2(x)

        # conv5_x
        x = self.conv5_1(x)
        x = self.conv5_2(x)


        return x


class Image_CNN(nn.Module):
    def __init__(self, device):
        self.device = device
        super(Image_CNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.res_net18 = ResNet18()
        self.se_block = SEBlock(512)  # SE Block for attention
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(x.shape)
        x = self.res_net18(x)
        # print(x.shape)

        # x = self.conv1(x)
        #
        # x = self.conv2(x)
        #
        # x = self.conv3(x)
        #
        # x = self.relu(self.conv4(x))
        #
        # x = self.dropout(x)
        #
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # print('conv7', x.shape)

        # 位置编码
        x = x.permute(0, 2, 3, 1)
        x = self.add_timing_signal_nd(x)
        x = x.permute(0, 3, 1, 2)

        x = self.se_block(x)  # 应用 SE Block

        return x  # 返回特征图





    def add_timing_signal_nd(self, x, min_timescale=1.0, max_timescale=1.0e4):

        # 将这个输入的图片形状放入数组
        static_shape = list(x.shape)  # [2, 512, 50, 120]
        """
        因此，num_dims = len(static_shape) - 2 在这个例子中等于 2，表示剩余的维度数量是图像的高度和宽度。
        在位置编码的上下文中，这些维度用于计算和应用位置编码，以帮助模型处理序列数据（如文本或图像块）中的位置信息。
        """
        num_dims = len(static_shape) - 2  # 2
        # 在代码中，channels = x.shape[-1] 这行代码的作用是获取张量 x 的最后一个维度的大小，通常用于确定张量中特征或通道的数量。
        channels = x.shape[-1]  # 512
        # 使用通道数 除以 (2*2)
        """
        num_timescales = channels // (num_dims * 2) 的计算是为了确定生成的正弦和余弦函数的数量，
        用于表示不同的时间尺度。这些时间尺度将用于生成位置编码，这里有128个，也就是要生成128个需要嵌入的数据
        """
        num_timescales = channels // (num_dims * 2)  # 512 // (2*2) = 128

        """
        log_timescale_increment 的计算目的是根据 max_timescale 和 min_timescale 的设置，以及用于生成位置编码的时间尺度数量 num_timescales，计算出一个适当的对数增量。
        这个增量将用于在不同时间尺度上生成正弦和余弦函数，以生成丰富和多样化的位置编码，帮助模型理解输入数据中的位置信息。
        """
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (float(num_timescales) - 1))

        """
        然后，根据指定的 min_timescale 和 log_timescale_increment，生成一组逆时间尺度
        逆时间尺度实际上是一个用来缩放和调整位置编码中频率的系数数组，而不是直接表示时间的逆。这个术语可能会让人误解，应该理解为在位置编码中用于调整频率的系数。
        """
        inv_timescales = min_timescale * torch.exp(
            torch.FloatTensor([i for i in range(num_timescales)]) * -log_timescale_increment)  # len == 128

        # 在图片的宽高中填充位置编码
        for dim in range(num_dims):  # dim == 0; 1
            length = x.shape[dim + 1]  # 要跳过前两个维度
            position = torch.arange(length).float()  # len == 50
            scaled_time = torch.reshape(position, (-1, 1)) * torch.reshape(inv_timescales, (1, -1))
            # [50,1] x [1,128] = [50,128]
            signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1).to(self.device)  # [50, 256]
            prepad = dim * 2 * num_timescales  # 0; 256
            postpad = channels - (dim + 1) * 2 * num_timescales  # 512-(1;2)*2*128 = 256; 0
            signal = F.pad(signal, (prepad, postpad, 0, 0))  # [50, 512]
            for _ in range(1 + dim):  # 1; 2
                signal = signal.unsqueeze(0)
            for _ in range(num_dims - 1 - dim):  # 1, 0
                signal = signal.unsqueeze(-2)
            x += signal  # [1, 14, 1, 512]; [1, 1, 14, 512]
        return x.clone().detach().requires_grad_(True)

# 预测序列长度
class prediction_sequence_length(nn.Module):
    def __init__(self, device, output_size):
        super().__init__()
        self.output_size = output_size
        self.fc = None  # 延迟初始化
        self.device = device

    def forward(self, x):
        if self.fc is None:
            # 自动根据第一次输入的维度初始化全连接层
            input_size = x.numel() // x.shape[0]  # 保留batch维度计算
            self.fc = nn.Linear(input_size, self.output_size)
            self.fc = self.fc.to(self.device)

        # 展平保留batch维度 (batch_size, features)
        x_flat = x.view(x.size(0), -1)
        x_out = self.fc(x_flat)
        # 直接取最大值索引，节省softmax计算
        return torch.argmax(x_out,dim=1)
# 可视化特征图
def plot_feature_maps(feature_maps, cols=8):
    for layer_name, feature_map in feature_maps.items():
        num_feature_maps = feature_map.size(1)
        rows = (num_feature_maps + cols - 1) // cols  # 计算行数
        fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))  # 动态调整图像大小

        for i in range(num_feature_maps):
            ax = axes[i // cols, i % cols]  # 计算行列索引
            ax.imshow(feature_map[0, i].cpu().detach().numpy(), cmap='viridis')
            ax.axis('off')

        # 隐藏多余的子图
        for j in range(num_feature_maps, rows * cols):
            axes[j // cols, j % cols].axis('off')

        plt.title(layer_name)
        plt.show()

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Assuming bidirectional=True

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out 的形状为 (batch_size, seq_len, hidden_size * 2) if bidirectional=True
        output = self.fc(lstm_out)
        # output 的形状为 (batch_size, seq_len, output_size)
        return output

class BidirectionalBERT(nn.Module):
    def __init__(self, input_size, output_size):
        super(BidirectionalBERT, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # 双向处理
        out_forward = self.fc1(x)
        out_backward = self.fc1(torch.flip(x, [1]))
        combined = out_forward + torch.flip(out_backward, [1])
        output = self.fc2(combined)
        return output


class Decode(nn.Module):
    def __init__(self, device, num_classes):
        super(Decode, self).__init__()
        self.device = device
        self.cnn = Image_CNN(device).to(device)




        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 512, 512),
            BidirectionalLSTM(512, 512,num_classes)
        ).to(device)

    def forward(self, image):
        x = self.cnn(image)
        # print(x.shape)
        # prediction_seq_length=prediction_sequence_length(self.device,1000)
        # seq_length=prediction_seq_length(x)
        # print(seq_length)
        # 有了预测的序列长度就可以拿这个序列长度去生成序列

        # 计算调整矩阵数值
        x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
        # 调整矩阵形状
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        x = torch.log_softmax(x, dim=2)
        return x
