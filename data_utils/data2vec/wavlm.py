from transformers import Wav2Vec2Processor, WavLMModel
import soundfile as sf
import numpy as np
import torch

print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/wav2vec2-large-xlsr-53-esperanto")
print("Loading the WavLM Model...")
wavlm_model = WavLMModel.from_pretrained("/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/wavlm")


def get_wavlm_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    wavlm = get_wavlm_from_16k_speech(speech_16k)
    return wavlm

@torch.no_grad()
def get_wavlm_from_16k_speech(speech, device="cuda:0"):
    global wavlm_model
    wavlm_model = wavlm_model.to(device)
    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
    input_values_all = input_values_all.to(device)
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = wavlm_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    if input_values.shape[1] >= kernel:
        hidden_states = wavlm_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret

def make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] += 1
        return torch.cat((tensor, tensor[-1:]), dim=0)
    return tensor

import soundfile as sf
import numpy as np
import torch
from argparse import ArgumentParser
import librosa

parser = ArgumentParser()
parser.add_argument('--wav', type=str, help='')
args = parser.parse_args()

wav_name = args.wav

speech, sr = sf.read(wav_name)
speech_16k = librosa.resample(speech, orig_sr=sr, target_sr=16000)
print("SR: {} to {}".format(sr, 16000))

wavlm_hidden = get_wavlm_from_16k_speech(speech_16k)
wavlm_hidden = make_even_first_dim(wavlm_hidden).reshape(-1, 2, 1024)
np.save(wav_name.replace('.wav', '_wavlm.npy'), wavlm_hidden.detach().numpy())
print(wavlm_hidden.detach().numpy().shape) 