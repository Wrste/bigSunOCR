# 新建模型进行简单预测
import hashlib
import math
import os
import subprocess
import random

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import datetime
import logging
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing
import matplotlib.pyplot as plt
import copy
import cv2
from model import model
import os
# device = torch.device("cpu" if torch.cuda.is_available() else "mps")

# device = torch.device("cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoding_file = "./Data-for-LaTeX_OCR/hand/formulas/vocab_hotel_encoding.txt"  # 编码后的样本文件路径
# 统一使用大字典
vocab_file = "./Data-for-LaTeX_OCR/vocabDict/vocab_hotel_dict.txt"  # 编码后的样本文件路径


# 训练图片加载
def load_images_train_paths():
    image_data = []
    with open("./Data-for-LaTeX_OCR/hand/matching/train.matching.txt", "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            image_index = line.split(" ")[0]
            # 记得修改这，这是真实读取得
            image_data.append(f"./Data-for-LaTeX_OCR/hand/images/images_train/{image_index}")
    return image_data


def load_images_train_paths_and_image_data():
    image_data = []
    with open("./Data-for-LaTeX_OCR/hand/matching/train.matching.txt", "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            image_index = line.split(" ")[0]
            # 记得修改这，这是真实读取得
            image_data_cv = cv2.imread(f"./Data-for-LaTeX_OCR/hand/images/images_train/{image_index}")

            rotated_image = cv2.rotate(image_data_cv, cv.ROTATE_180)
            # 确保两张图片的高度相同
            if image_data_cv.shape[0] != rotated_image.shape[0]:
                rotated_image = cv.resize(rotated_image, (image_data_cv.shape[1], image_data_cv.shape[0]))
            # 上下合并图片
            combined_image = np.vstack((image_data_cv, rotated_image))

            # height, width, channels = image_data_cv.shape
            image_data_cv = cv.resize(combined_image, (600, 600))
            image_data_cv = torch.tensor(image_data_cv).float()
            image_data.append(image_data_cv.permute(2, 0, 1))

    return image_data


# 加载词典
def load_vocab_fill():
    vocabs = []
    seq_length = []

    with open(encoding_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            words = line.strip().split(",")
            int_words = [int(word) for word in words]
            vocabs.append(torch.tensor(int_words))
            seq_length.append(len(int_words))

    padded_vocabs = pad_sequence(vocabs, batch_first=True, padding_value=0)
    input_length = padded_vocabs.ne(0).sum(dim=1).tolist()  # 计算填充后的长度

    return padded_vocabs, input_length, seq_length


def clip_gradient(optimizer, grad_clip):
    """
    梯度裁剪用于避免梯度爆炸
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# 划分训练集

class trainDataset(Dataset):
    def __init__(self, image, seq, input_seq_length, seq_length):
        self.image = image
        self.seq = seq
        self.seq_length = seq_length
        self.input_seq_length = input_seq_length

    def __len__(self):
        return len(self.image)

    def __getitem__(self, i):
        return self.image[i], self.seq[i], self.input_seq_length[i], self.seq_length[i]


def train(model, optimizer, scheduler, data_loader, epochs):
    global out, loss
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    min_loss = float('inf')  # 初始化为一个很大的值
    min_accuracy = 0  # 初始化为一个很大的值
    accumulation_steps = 4  # 梯度累积的步数

    for epoch in range(epochs):
        epoch_loss = 0.0
        forecast_latex = []  # 可以存储一些必要的输出
        accuracy = 0
        acc_sum = 0

        for i, (image, y, seq_length, y_lengths) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()

            # 将数据传输到GPU并确保内存节约
            image = image.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_lengths = y_lengths.to(device, non_blocking=True)

            # 模型预测输出
            out = model(image)  # 模型输出
            input_lengths = torch.tensor([out.size(0)] * y.size(0), dtype=torch.int).to(device)  # 输入长度（批次大小）

            # 计算CTC损失
            loss = ctc_loss(log_probs=out, targets=y, target_lengths=y_lengths, input_lengths=input_lengths)
            epoch_loss += loss.item()

            # 梯度累积
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            if (i + 1) % accumulation_steps == 0:
                # 每积累一定步数，更新模型
                optimizer.step()
                optimizer.zero_grad()

            # 打印和记录每个批次的损失值，避免过多的数据存储
            if i % 100 == 0:  # 每 100 个批次记录一次
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

            # 内存管理：清理不再使用的变量
            del out, y_lengths

            # 清理GPU缓存
            torch.cuda.empty_cache()

        # Epoch结束后更新学习率调度器（如果有的话）
        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {epoch_loss:.4f}")

        # 记录最小损失值和准确度（如果需要保存模型）
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            min_accuracy = accuracy  # 或者根据你的标准来计算准确度

        # 清理剩余的内存
        del image, y, loss
        torch.cuda.empty_cache()
        # scheduler.step()

        avg_loss = epoch_loss / len(data_loader)
        # print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")

        # print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
        # accuracy = accuracy / len(dataloader)
        # logging.info(f"Epoch {epoch + 1}/{epochs}")
        # if accuracy > min_accuracy:
        #     min_accuracy = accuracy
        #     # print(f"Epoch {epoch + 1}/{epochs}, Max Average accuracy: {accuracy}")
        #     # print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
        #     logging.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
        #     logging.info(f"Epoch {epoch + 1}/{epochs}, Max Average accuracy: {accuracy}")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss,
        #         'learning_rate': optimizer.param_groups[0]['lr']
        #     }, './model_data/latex_OCR_model.pth')

        # 只有当当前轮次的平均损失小于之前保存模型时的最小损失时才保存模型
        # test(model)
        if avg_loss < min_loss:
            min_loss = avg_loss
            # ok = True  # 初始化为 True，假设所有序列都可以渲染
            #
            # for data in forecast_latex:
            #     predictions = decode_predictions(data)
            #     if not can_compile_latex_expression(predictions):
            #         ok = False
            #         break  # 不需要再判断了
            # if ok:
            #     print("所有序列都可以渲染")
            #     # 在训练结束后进行解码
            #     torch.save(model.state_dict(), './model_data/latex_OCR_model.pth')
            #     break

            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
            # print(f"Epoch {epoch + 1}/{epochs}, Average accuracy: {accuracy}")
            logging.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss}")
            print(f"lr: {optimizer.param_groups[0]['lr']}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learning_rate': optimizer.param_groups[0]['lr']
            }, './model_data/latex_OCR_model.pth')


if __name__ == '__main__':

    # 加载图片
    image = load_images_train_paths_and_image_data()
    # 加载词典
    seq, input_seq_length, seq_length = load_vocab_fill()
    # 加载数据集
    dataset = trainDataset(image, seq, input_seq_length, seq_length)
    data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, pin_memory=True, num_workers=4)
    model = model.Decode(device, 769).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    checkpoint = torch.load('./model_data/latex_OCR_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train(model, optimizer, None, data_loader, 10000)
