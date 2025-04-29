import torch
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def extract_audio_features(file_path, model_name="facebook/wav2vec2-base-960h"):
    # 加载模型和处理器
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    
    # 读取音频文件，使用librosa将音频转化为浮点数数组
    audio, sr = librosa.load(file_path, sr=16000)  # 将采样率设置为 16kHz
    
    # 处理音频输入，以符合Wav2Vec 2.0模型的输入要求
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    
    # 提取特征（不需要计算梯度）
    with torch.no_grad():
        outputs = model(inputs.input_values)
        hidden_states = outputs.last_hidden_state
    
    # 返回特征张量，维度为 (batch_size, sequence_length, feature_dim)
    return hidden_states

# 示例用法
if __name__ == "__main__":
    audio_path = "your_audio_file.wav"  # 替换为你自己的音频文件路径
    features = extract_audio_features(audio_path)
    print("音频特征的形状：", features.shape)
