import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import model
import logging
import datetime
# 配置日志记录器
logging.basicConfig(
    filename='app.log',  # 指定日志文件名
    filemode='a',  # 指定写入模式为追加模式
    format='%(asctime)s - %(message)s',  # 日志格式
    level=logging.INFO  # 日志级别
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
encoding_file = "./Data-for-LaTeX_OCR/hand/formulas/test_vocab_hotel_encoding.txt"  # 编码后的样本文件路径
# 统一使用大字典
vocab_file = "./Data-for-LaTeX_OCR/vocabDict/vocab_hotel_dict.txt"  # 编码后的样本文件路径


def calculate_accuracy(reference, hypothesis):
    ref_words = predictions(reference)  # 目标
    hyp_words = decode_predictions(hypothesis)  # 预测的
    # 计算准确匹配的单词数量
    correct_count = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref == hyp)

    # 总的单词数量
    total_words = len(ref_words)

    # 计算准确度
    accuracy = correct_count / total_words if total_words > 0 else 0.0

    return accuracy


def load_vocab():
    vocabs = []
    seq_length = []
    with open(encoding_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            words = line.strip().split(",")
            int_words = [int(word) for word in words]  # Convert each word to int
            vocabs.append(int_words)  # Append list of ints to vocabs
            seq_length.append(len(int_words))
    return vocabs


def load_vocab_hotel_dict():
    # 读取 字典文件 r 是读取文件模型 encoding 读取文件的编码 并且将对象指定到f中
    with open(vocab_file, "r", encoding="utf-8-sig") as f:
        # f.readlines() 实际意义是 按照行读取文件，并且把每一行放入一个list中
        lines = f.readlines()
        # 在这段代码中，new_dict = eval(lines[0]) 这行代码的作用是将文件的第一行文本（lines[0]）当作Python代码来执行，并返回执行结果。
        new_dict = eval(lines[0])
        return new_dict


# symbol_dict = {v: k for k, v in dict.items()}


#
def decode_predictions(out):
    # 使用贪婪解码（选择每个时间步概率最高的字符）
    _, preds = torch.max(out, dim=2)
    # print(preds)
    preds = remove_consecutive_duplicates(preds)
    preds = remove_zeros(preds)
    # print(preds)
    decoded_preds = []
    symbol_dict = load_vocab_hotel_dict()
    symbol_dict = {v: k for k, v in symbol_dict.items()}
    for pred in preds:
        decoded_preds.append(symbol_dict[pred.item()])
    decoded_string = ''.join(decoded_preds)
    # print(decoded_string)
    return decoded_string


# 不需要解码得预测
def predictions(preds):
    # print(preds)
    decoded_preds = []
    symbol_dict = load_vocab_hotel_dict()
    symbol_dict = {v: k for k, v in symbol_dict.items()}
    for pred in preds:
        decoded_preds.append(symbol_dict[pred.item()])
    decoded_string = ''.join(decoded_preds)
    # print(decoded_string)
    return decoded_string


# 加载测试数据集文件路径
def load_images_test_paths():
    image_data = []
    with open("./Data-for-LaTeX_OCR/hand/matching/test.matching.txt", "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            image_index = line.split(" ")[0]
            # 记得修改这，这是真实读取得
            image_data.append(f"./Data-for-LaTeX_OCR/hand/images/images_test/{image_index}")
    return image_data


# 验证集
def load_images_val_paths():
    image_data = []
    with open("./Data-for-LaTeX_OCR/hand/matching/val.matching.txt", "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            image_index = line.split(" ")[0]
            # 记得修改这，这是真实读取得
            image_data.append(f"./Data-for-LaTeX_OCR/hand/images/images_val/{image_index}")
    return image_data


# 训练集
def load_images_train_paths():
    image_data = []
    with open("./Data-for-LaTeX_OCR/hand/matching/train.matching.txt", "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            image_index = line.split(" ")[0]
            # 记得修改这，这是真实读取得
            image_data.append(f"./Data-for-LaTeX_OCR/hand/images/images_train/{image_index}")
    return image_data


def remove_zeros(tensor):
    # 找到非零元素的索引
    non_zero_indices = (tensor != 0).nonzero().squeeze(1)

    # 使用索引获取非零元素的张量
    cleaned_tensor = tensor[non_zero_indices]

    return cleaned_tensor


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


def load_image(path):
    image_data_cv = cv.imread(path)
    # image_data_cv = cv.resize(image_data_cv, (128, 128))
    rotated_image = cv.rotate(image_data_cv, cv.ROTATE_180)
    # 确保两张图片的高度相同
    if image_data_cv.shape[0] != rotated_image.shape[0]:
        rotated_image = cv.resize(rotated_image, (image_data_cv.shape[1], image_data_cv.shape[0]))
    # 上下合并图片
    combined_image = np.vstack((image_data_cv, rotated_image))
    image_data_cv = cv.resize(combined_image, (600, 600))
    image_data_cv = torch.tensor(image_data_cv).float()

    return image_data_cv.permute(2, 0, 1).unsqueeze(0)


def remove_consecutive_duplicates(tensor):
    if tensor.numel() < 2:
        return tensor
    cleaned_tensor = [tensor[0].item()]  # 初始化清理后的张量，将第一个元素添加到列表中
    for i in range(1, tensor.numel()):
        if tensor[i] != tensor[i - 1]:  # 如果当前元素不等于前一个元素，则添加到清理后的张量中
            cleaned_tensor.append(tensor[i].item())

    return torch.tensor(cleaned_tensor, dtype=tensor.dtype)


def sequence_to_string(sequence, symbol_dict):
    return ''.join([symbol_dict[idx] for idx in sequence])


def calculate_inference_speed(model, image_paths, device):
    total_time = 0
    inference_speed_list = []
    for image_path in image_paths:
        image_w, image_h = load_image(image_path)
        image_w = image_w.to(device)
        image_h = image_h.to(device)
        time_start = datetime.datetime.now()
        model.eval()  # 切换到评估模式
        with torch.no_grad():
            out = model(image_w, image_h)  # 再次前向传播获取预测
        time_end = datetime.datetime.now()
        total_time += (time_end - time_start).total_seconds()
        inference_speed_list.append((time_end - time_start).total_seconds())
    average_time = total_time / len(image_paths)
    print(f"Average Inference Time: {average_time:.4f} seconds per image")
    return inference_speed_list








def test(model):
    image_paths = load_images_test_paths()
    labels = load_vocab()
    total_accuracy = 0  # 用于累加准确度
    total_time = 0  # 用于累加推理时间
    index = 0
    model.eval()  # Switch to evaluation mode

    for image_path in image_paths:
        image_w = load_image(image_path)
        image_w = image_w.to(device)
        time_start = datetime.datetime.now()

        with torch.no_grad():
            out = model(image_w)  # Forward pass for prediction
            out_text = decode_predictions(out)
            print(out_text)
            time_end = datetime.datetime.now()
            out_test = out.cpu().detach()
            accuracy = calculate_accuracy(torch.tensor(labels[index]), out_test)
            total_accuracy += accuracy  # 累加准确度
            index += 1
            elapsed_time = (time_end - time_start).total_seconds()
            total_time += elapsed_time  # 累加推理时间
            print(f'Time elapsed: {elapsed_time} Seconds')
            print(f'accuracy: {accuracy}')

    average_accuracy = total_accuracy / len(image_paths)  # 计算平均准确度
    average_time = total_time / len(image_paths)  # 计算平均推理时间
    print(f"Average accuracy: {average_accuracy}")
    print(f"Average inference time: {average_time} Seconds")
    return average_accuracy, average_time


def test_single_image(model):
    image_w, image_h = load_image("./Data-for-LaTeX_OCR/hand/images/images_test/0.png")
    image_w = image_w.to(device)
    image_h = image_h.to(device)
    # print("rotated_img_h", image_w.shape)
    # print("rotated_img_h", image_h.shape)
    # 在训练结束后进行解码
    time_start = datetime.datetime.now()
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        out = model(image_w, image_h)  # 再次前向传播获取预测
        # out_lengths = torch.tensor([out.size(0)])  # 假设所有输出长度相同
        out = decode_predictions(out)
        print(out)
        time_end = datetime.datetime.now()
        print('Time elapsed:: %s Seconds' % (time_end - time_start))


def split_into_single_sample(data):
    # data 的形状为 (1216, 10, 769)
    num_samples = data.shape[1]
    # 使用 np.split 根据第二维度进行拆分
    # 结果是一个列表，每个元素的形状为 (1216, 1, 769)
    batches = np.split(data, num_samples, axis=1)
    # 确保每一份的形状是 (1216, 1, 769)
    return [batch.reshape(batch.shape[0], 1, batch.shape[2]) for batch in batches]


def batch_calculate_accuracy(labels, forecasts, ignore_index=0):
    accuracy = 0
    avg_accuracy = 0

    list_forecasts = split_into_single_sample(forecasts)
    for index in range(len(list_forecasts)):
        label = labels[index]
        mask = label != 0
        filtered_labels = label[mask]
        # print(filtered_labels)
        accuracy += calculate_accuracy(filtered_labels, list_forecasts[index])
    avg_accuracy = accuracy / len(labels)
    return avg_accuracy


if __name__ == '__main__':
    model = model.Decode(device, 769)
    checkpoint = torch.load('./model_data/latex_OCR_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model)
