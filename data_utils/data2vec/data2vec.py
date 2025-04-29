import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Data2VecAudioModel
import torch.nn.functional as F
import shutil

def extract_data2vec_features(audio_path, chunk_size=7999):
    # 加载音频
    print("Loading audio file...")
    audio, sample_rate = sf.read(audio_path)
    print("Audio loading completed.")

    # 加载 Data2Vec 处理器和模型
    print("Loading Data2Vec processor and model...")
    processor = Wav2Vec2Processor.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/data2vec-audio-large-960h")
    model = Data2VecAudioModel.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/data2vec-audio-large-960h")
    print("Model loading completed.")

    #print(model)

    # 预处理音频
    print("Preprocessing audio...")
    inputs_list = []
    with tqdm(total=len(audio), desc="Preprocessing Audio", unit="samples") as pbar:
        for start in range(0, len(audio), chunk_size):
            end = start + chunk_size
            chunk = audio[start:end]
            inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            inputs_list.append(inputs.input_values)
            pbar.update(len(chunk))
    print(f"inputs-1 shape: {inputs_list[-1].shape}")

    # 对所有的分块进行统一长度处理
    for i in range(len(inputs_list)):
        if inputs_list[i].size(0) < chunk_size:
            padding_size = chunk_size - inputs_list[i].size(0)
            # 调整维度以便在第一个维度填充
            inputs_list[i] = inputs_list[i].unsqueeze(0)  # 在最前面添加一个新维度
            # 填充第一个维度
            inputs_list[i] = F.pad(inputs_list[i], (0, 0, padding_size, 0), "constant", 0)
            inputs_list[i] = inputs_list[i].squeeze(0)  # 移除第一个维度

    print(f"inputs-1 shape: {inputs_list[-1].shape}")
    # 合并处理后的输入
    input_values = torch.cat(inputs_list, dim=1)  # 按时间维度（batch_size 的维度）合并所有分块的输入

    # 将合并后的输入封装为与模型兼容的格式
    inputs = {'input_values': input_values}
    print(inputs['input_values'].shape)

    # 获取音频特征
    print("Extracting features from audio...")
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取最后一层隐藏状态
    features = outputs.last_hidden_state.numpy()

    print(f"Extracted features shape: {features.shape}")
    return features

# 使用函数提取特征
audio_path = "/home/zhutianheng/projects/GaussianTalker/datasets/Obama_data2vec/Obama01/aud.wav"  # 替换为你的 .wav 文件路径
features = extract_data2vec_features(audio_path)

# 保存特征为 .npy 文件
np.save(audio_path.replace(".wav", ".npy"), features)
shutil.copy(audio_path.replace('.wav', '.npy'), audio_path.replace('.wav', '_ds.npy'))
