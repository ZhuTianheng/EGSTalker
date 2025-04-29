import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Data2VecAudioForCTC
import shutil

# 1. 加载 Data2Vec 2.0 模型和特征提取器
model_name = "/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/data2vec-audio-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Data2VecAudioForCTC.from_pretrained(model_name)

# 2. 读取音频文件并处理成模型输入格式
def load_audio(file_path, sampling_rate=16000):
    audio, sr = sf.read(file_path, always_2d=False)
    if sr != sampling_rate:
        raise ValueError(f"Sampling rate of the file ({sr}) does not match the expected rate ({sampling_rate})")
    return torch.tensor(audio[:, 0] if audio.ndim == 2 else audio, dtype=torch.float32)

# 指定你的音频文件路径
audio_file_path = "/home/zhutianheng/projects/GaussianTalker/datasets/Obama_data2vec/Obama/aud.wav"

# 使用 tqdm 显示加载音频进度
audio_input = load_audio(audio_file_path)
with tqdm(total=len(audio_input), desc="Processing audio input", unit="samples") as pbar:
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    pbar.update(len(audio_input))

# 将输入封装为与模型兼容的格式
input_values = inputs.input_values
inputs = {'input_values': input_values}

# 4. 使用 Data2Vec 模型提取音频特征
with torch.no_grad():
    logits = model(**inputs).logits
'''
# 调整特征形状以匹配 DeepSpeech 提取的特征形状
sequence_length = logits.shape[1] // 16 * 16  # 调整序列长度为 16 的倍数
logits = logits[:, :sequence_length, :]
logits = logits.view(-1, 16, logits.shape[-1])
print("Logits01 Shape:", logits.shape)
linear_layer = torch.nn.Linear(logits.shape[0], 7999)
logits = linear_layer(logits.view(logits.shape[0], -1)).view(7999, 16, logits.shape[-1])
'''
# 调整特征形状以匹配 DeepSpeech 提取的特征形状
sequence_length = logits.shape[1] // 16 * 16  # 调整序列长度为 16 的倍数
logits = logits[:, :sequence_length, :]
logits = logits.view(-1, 16, logits.shape[-1])
print("Logits01 Shape:", logits.shape)
# 使用插值将窗口数量调整为 7999
logits = logits.permute(1, 2, 0)  # 将批次维度移动到最后，调整为 (C, L, N)
print("Logits02 Shape:", logits.shape)
logits = torch.nn.functional.interpolate(logits, size=7999, mode='linear', align_corners=False)  # 对批次维度进行插值
print("Logits cha Shape:", logits.shape)
logits = logits.permute(2, 0, 1).contiguous()  # 恢复为原始维度顺序 (N, C, L)
print("Logits03 Shape:", logits.shape)
logits = logits.contiguous().view(7999, 16, -1)
print("Logits04 Shape:", logits.shape)



# 5. 输出特征形状和部分特征数据
print("Logits Shape:", logits.shape)

# 保存音频特征
np.save(audio_file_path.replace(".wav", ".npy"), logits.numpy())
print(f"[INFO] Extracted features saved with shape: {logits.shape}")
shutil.copy(audio_file_path.replace('.wav', '.npy'), audio_file_path.replace('.wav', '_ds.npy'))
print(f'[INFO] ===== extracted audio labels =====')
