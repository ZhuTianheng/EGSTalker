from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import numpy as np
import torch
import shutil

# 加载 Wav2Vec2 处理器和 HuBERT 模型
print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/hubert") # 从 Hugging Face 预训练模型库加载 Wav2Vec2 处理器
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/hubert") # 从 Hugging Face 预训练模型库加载 HuBERT 模型

# 定义一个从16kHz的wav文件中提取HuBERT特征的函数
def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)  # 使用 soundfile 读取 wav 文件
    hubert = get_hubert_from_16k_speech(speech_16k)  # 调用处理语音的函数
    return hubert  # 返回 HuBERT 特征

# 无梯度下的装饰器，加速模型推理
@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model
    hubert_model = hubert_model.to(device)  # 将模型移到指定的设备（GPU 或 CPU）
    if speech.ndim == 2:
        speech = speech[:, 0]  # 如果是双通道音频，取第一个通道，变成单通道
    # 使用 Wav2Vec2 处理器处理语音数据，得到输入值 (input_values)
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]，T为音频样本长度
    input_values_all = input_values_all.to(device)  # 将输入数据移到指定设备上
    
    # 由于长序列音频会导致内存不足，需要将其分割为多个片段进行处理
    # HuBERT 模型中使用的卷积神经网络的步幅为 [5,2,2,2,2,2]，总步幅为 320
    # 核大小为 [10,3,3,3,3,2,2]，所以卷积操作相当于一个核大小为 400，步幅为 320 的一维卷积
    # 计算输出时间步数公式：T = floor((t - k) / s)
     # 打印输入音频的长度
    input_length = input_values_all.shape[1]
    print(f"输入的音频长度 (input_length): {input_length}")
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel-stride)) // stride + 1

    print(f"分割的片段长度 (clip_length): {clip_length}")
    print(f"分割的次数 (num_iter): {num_iter}")
    print(f"卷积核大小 (kernel): {kernel}")
    print(f"步幅大小 (stride): {stride}")
    print(f"重叠长度 (kernel - stride): {kernel - stride}")
    print(f"期望的时间步数 (expected_T): {expected_T}")


    res_lst = []  # 用于存储每个分割片段的结果
    # 迭代处理每个片段
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]  # 获取音频片段
        hidden_states = hubert_model.forward(input_values).last_hidden_state  # 获取 HuBERT 模型的隐藏状态 [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])  # 将隐藏状态添加到结果列表中

    # 处理剩余的最后一段音频
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    if input_values.shape[1] >= kernel:  # 如果最后一段的长度大于等于卷积核的大小
        hidden_states = hubert_model(input_values).last_hidden_state  # 获取 HuBERT 模型的隐藏状态
        res_lst.append(hidden_states[0])

    # 拼接所有的结果片段，形成最终的特征向量
    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    # 确保拼接后的特征维度和期望的时间步数相符
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))  # 如果长度不足，使用填充操作
    else:
        ret = ret[:expected_T]  # 如果长度超过，截断至期望的时间步数

    return ret  # 返回最终的特征向量

# 定义一个使张量的第一个维度为偶数的函数
def make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:  # 如果第一个维度是奇数
        size[0] -= 1
        return tensor[:size[0]]  # 截断至偶数长度
    return tensor

import soundfile as sf
import numpy as np
import torch
from argparse import ArgumentParser
import librosa

# 解析命令行参数
parser = ArgumentParser()
parser.add_argument('--wav', type=str, help='')  # 添加wav文件路径的参数
args = parser.parse_args()

wav_name = args.wav  # 获取传入的wav文件路径

# 读取音频文件
speech, sr = sf.read(wav_name)  # 使用 soundfile 读取 wav 文件
speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)  # 使用 librosa 进行重采样，将采样率变为 16kHz
print("SR: {} to {}".format(sr, 16000))  # 打印采样率的变换情况

# 提取 HuBERT 隐藏特征
hubert_hidden = get_hubert_from_16k_speech(speech_16k)
hubert_hidden = make_even_first_dim(hubert_hidden).reshape(-1, 2, 1024)  # 调整特征张量的维度，确保第一个维度是偶数，并重塑形状
hubert_hidden = hubert_hidden.view(7999, 16, 128)  # 将形状直接调整为 [7999, 16, 128]
np.save(wav_name.replace('.wav', '.npy'), hubert_hidden.detach().numpy())  # 将特征保存为 .npy 文件
shutil.copy(wav_name.replace('.wav', '.npy'), wav_name.replace('.wav', '_ds.npy'))
print(hubert_hidden.detach().numpy().shape)  # 打印保存的特征的形状
