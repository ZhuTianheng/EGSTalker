import numpy as np
import librosa
from transformers import Wav2Vec2Processor, WavLMModel
import torch
import torch.nn.functional as F

def preprocess_audio(file_path, target_sr=16000):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=target_sr)
    return audio

def extract_wavlm_features(audio, processor, model):
    # Process audio to get input values
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Extract features using WavLM model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state
    features = outputs.last_hidden_state.squeeze().cpu().numpy()
    
    # Ensure features are 3D
    if features.ndim < 3:
        features = np.expand_dims(features, axis=0)
    elif features.ndim > 3:
        raise ValueError(f"Unexpected number of dimensions in features: {features.ndim}")
    
    return features

def reshape_features(features):
    # 将numpy数组转换为torch tensor
    features_tensor = torch.from_numpy(features)
    
    # 调整维度顺序为 [batch, channel, length]
    features_tensor = features_tensor.permute(0, 2, 1)
    
    # 首先将1024降为29
    #features_29 = F.adaptive_avg_pool1d(features_tensor, 29)
    features_29 = features_tensor
    # 然后将2扩展为16
    # 将2个通道扩展为16个通道
    B, C, L = features_29.shape
    features_expanded = features_29.unsqueeze(1)  # [B, 1, C, L]
    features_16 = F.interpolate(
        features_expanded, 
        size=(16, L),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # [B, 16, L]
    
    # 转回numpy
    return features_16.numpy()

def save_features_to_npy(features, file_name):
    num_windows = 7999
    window_size = 16  # 修改为新的window_size
    feature_length = features.shape[1]
    print(f"Feature shape before padding: {features.shape}")
    
    # Calculate the number of windows we can create
    available_windows = feature_length - window_size + 1
    
    # If available windows are less than required, pad the features
    if available_windows < num_windows:
        padding_needed = num_windows - available_windows
        pad_each_side = padding_needed // 2
        
        # Generate Gaussian random padding
        np.random.seed(0)  # For reproducibility
        pad_values_start = np.random.normal(loc=0.0, scale=1.0, size=(features.shape[0], pad_each_side, features.shape[2]))
        pad_values_end = np.random.normal(loc=0.0, scale=1.0, size=(features.shape[0], pad_each_side, features.shape[2]))
        
        # Pad the features symmetrically
        features = np.concatenate((pad_values_start, features, pad_values_end), axis=1)
    
    # Recalculate feature length after padding
    feature_length = features.shape[1]
    
    # Initialize an array to hold all feature windows
    all_features = np.zeros((num_windows, window_size, 1024))  # 修改为新的目标维度
    
    # Create sliding windows
    for i in range(num_windows):
        all_features[i] = features[:, i:i+window_size, :]
    
    print(f"Features shape after sliding window and padding: {all_features.shape}")
    
    # Save features to .npy file
    np.save(file_name, all_features)

def main(audio_file_path, output_file_name):
    # Load Wav2Vec2 processor and WavLM model
    processor = Wav2Vec2Processor.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/wav2vec2-large-xlsr-53-esperanto")
    model = WavLMModel.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/wavlm")
    
    # Preprocess audio
    audio = preprocess_audio(audio_file_path)
    
    # Extract features
    features = extract_wavlm_features(audio, processor, model)
    
    # Reshape features to target dimensions
    features = reshape_features(features)
    
    # Save features to .npy file
    save_features_to_npy(features, output_file_name)

if __name__ == "__main__":
    audio_file_path = "/home/zhutianheng/projects/GaussianTalker/datasets/Obama_data2vec/Obama_hm/aud.wav"
    output_file_name = "/home/zhutianheng/projects/GaussianTalker/datasets/Obama_data2vec/Obama/aud_op.npy"

    main(audio_file_path, output_file_name) 