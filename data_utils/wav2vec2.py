import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2BertForCTC, AutoProcessor

import pyaudio
import soundfile as sf
import resampy

from queue import Queue
from threading import Thread, Event

# 线程函数，从输入流中读取音频帧并将其放入队列中
def _read_frame(stream, exit_event, queue, chunk):
    while True:
        if exit_event.is_set():  # 如果退出事件被触发，停止线程
            print(f'[INFO] read frame thread ends')
            break
        frame = stream.read(chunk, exception_on_overflow=False)  # 从流中读取一个音频块
        frame = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767  # 将帧数据归一化到 [-1, 1]
        queue.put(frame)  # 将归一化后的帧放入队列中

# 线程函数，从队列中取出音频帧并播放到输出流中
def _play_frame(stream, exit_event, queue, chunk):
    while True:
        if exit_event.is_set():  # 如果退出事件被触发，停止线程
            print(f'[INFO] play frame thread ends')
            break
        frame = queue.get()  # 从队列中获取下一个帧
        frame = (frame * 32767).astype(np.int16).tobytes()  # 将帧转换回 int16 格式以便播放
        stream.write(frame, chunk)  # 将帧写入输出流

# 自动语音识别 (ASR) 类定义
class ASR:
    def __init__(self, opt):
        self.opt = opt  # 存储配置选项
        self.play = opt.asr_play  # 处理过程中是否播放音频
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有 GPU 可用，则使用 GPU，否则使用 CPU
        self.fps = opt.fps  # 每秒的帧数（每帧 20 毫秒）
        self.sample_rate = 16000  # 音频采样率
        self.chunk = self.sample_rate // self.fps  # 每个块的采样数（例如，每 20 毫秒 320 个采样）
        self.mode = 'live' if opt.asr_wav == '' else 'file'  # 确定输入是实时的还是来自文件

        # 从模型中动态获取实际音频维度
        dummy_audio = torch.randn(1, self.chunk*5, device=self.device)  # 假设输入为一帧音频
        dummy_inputs = AutoProcessor.from_pretrained(opt.asr_model)(dummy_audio.cpu().numpy(), sampling_rate=self.sample_rate, return_tensors="pt")
        dummy_inputs = dummy_inputs['input_features'].to(self.device)  # 确保输入与模型期望的形状一致
        with torch.no_grad():
            dummy_output = Wav2Vec2BertForCTC.from_pretrained(opt.asr_model).to(self.device)(dummy_inputs).logits
        self.audio_dim = dummy_output.shape[-1]

        # ASR 处理的上下文和跨步大小设置
        self.context_size = opt.m  # 以帧数表示的上下文大小
        self.stride_left_size = opt.l  # 左跨步的帧数
        self.stride_right_size = opt.r  # 右跨步的帧数
        self.text = '[START]\n'  # 初始化输出文本为 '[START]'
        self.terminated = False  # ASR 过程是否已终止
        self.frames = []  # 用于存储处理过程中的音频帧的列表

        # 如果左跨步大于零，则用零填充左侧帧
        if self.stride_left_size > 0:
            self.frames.extend([np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)

        # 初始化线程事件和 PyAudio 实例
        self.exit_event = Event()
        self.audio_instance = pyaudio.PyAudio()

        # 为音频录制创建输入流（实时模式或文件模式）
        if self.mode == 'file':
            self.file_stream = self.create_file_stream()  # 从文件加载音频
        else:
            # 使用 PyAudio 创建实时输入流
            self.input_stream = self.audio_instance.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, output=False, frames_per_buffer=self.chunk)
            self.queue = Queue()  # 用于存储音频帧的队列
            self.process_read_frame = Thread(target=_read_frame, args=(self.input_stream, self.exit_event, self.queue, self.chunk))  # 读取音频帧的线程

        # 如果启用播放功能，设置输出流
        if self.play:
            self.output_stream = self.audio_instance.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=False, output=True, frames_per_buffer=self.chunk)
            self.output_queue = Queue()  # 用于存储待播放帧的队列
            self.process_play_frame = Thread(target=_play_frame, args=(self.output_stream, self.exit_event, self.output_queue, self.chunk))  # 播放音频帧的线程

        # 当前音频流的索引
        self.idx = 0

        # 从预训练模型中加载 ASR 模型和处理器
        print(f'[INFO] loading ASR model {self.opt.asr_model}...')
        self.processor = AutoProcessor.from_pretrained(opt.asr_model)  # 处理输入数据的处理器
        self.model = Wav2Vec2BertForCTC.from_pretrained(opt.asr_model).to(self.device)  # 用于语音转文本的 ASR 模型

        # 如果需要，初始化用于保存提取特征的变量
        if self.opt.asr_save_feats:
            self.all_feats = []

        # 用于存储提取特征的缓冲区（循环缓冲区以提高存储效率）
        self.feat_buffer_size = 4  # 要保留的特征缓冲区数量
        self.feat_buffer_idx = 0  # 当前特征缓冲区的索引
        self.feat_queue = torch.zeros(self.feat_buffer_size * self.context_size, self.audio_dim, dtype=torch.float32, device=self.device)

        # 初始化用于处理特征提取窗口的前后指针
        self.front = self.feat_buffer_size * self.context_size - 8  # 缓冲区开始处的假填充
        self.tail = 8  # 初始尾部位置
        # 注意力特征，初始化为零填充
        self.att_feats = [torch.zeros(self.audio_dim, 16, dtype=torch.float32, device=self.device)] * 4

        # 所需的预热步骤数量（用于实时模式下模型预热）
        self.warm_up_steps = self.context_size + self.stride_right_size + 8 + 2 * 3

        # 用于跟踪监听和播放线程状态的标志
        self.listening = False
        self.playing = False

    # 开始监听音频流的函数
    def listen(self):
        if self.mode == 'live' and not self.listening:
            print(f'[INFO] starting read frame thread...')
            self.process_read_frame.start()  # 启动读取音频帧的线程
            self.listening = True
        if self.play and not self.playing:
            print(f'[INFO] starting play frame thread...')
            self.process_play_frame.start()  # 启动播放音频帧的线程
            self.playing = True

    # 停止 ASR 过程的函数
    def stop(self):
        self.exit_event.set()  # 触发退出事件以停止所有线程

        # 如果启用了播放功能，停止并关闭输出流
        if self.play:
            self.output_stream.stop_stream()
            self.output_stream.close()
            if self.playing:
                self.process_play_frame.join()  # 等待播放线程结束
                self.playing = False

        # 在实时模式下停止并关闭输入流
        if self.mode == 'live':
            self.input_stream.stop_stream()
            self.input_stream.close()
            if self.listening:
                self.process_read_frame.join()  # 等待读取线程结束
                self.listening = False

    # 上下文管理器进入函数
    def __enter__(self):
        return self

    # 上下文管理器退出函数
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()  # 退出上下文时停止 ASR 过程
        if self.mode == 'live':
            self.text += '\n[END]'  # 在最终文本中附加 '[END]'
            print(self.text)  # 打印最终识别文本

    # 提取用于进一步处理的下一个特征窗口
    def get_next_feat(self):
        while len(self.att_feats) < 8:
            # 从队列中提取一个特征窗口（使用前后指针处理环绕）
            if self.front < self.tail:
                feat = self.feat_queue[self.front:self.tail]
            else:
                feat = torch.cat([self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

            self.front = (self.front + 2) % self.feat_queue.shape[0]
            self.tail = (self.tail + 2) % self.feat_queue.shape[0]

            self.att_feats.append(feat.permute(1, 0))  # 维度置换后将特征附加到列表中
        att_feat = torch.stack(self.att_feats, dim=0)  # 堆叠特征以创建张量
        self.att_feats = self.att_feats[1:]  # 从列表中移除最旧的特征
        return att_feat

    # 运行 ASR 过程的一个步骤（捕获音频并执行语音识别）
    def run_step(self):
        if self.terminated:
            return

        # 从流或队列中获取一个音频帧
        frame = self.get_audio_frame()
        if frame is None:
            # 如果没有更多的帧可用，将过程标记为已终止
            self.terminated = True
        else:
            self.frames.append(frame)  # 将帧添加到帧列表中
            if self.play:
                self.output_queue.put(frame)  # 如果启用播放，将帧放入输出队列
            # 如果上下文帧不足，跳过运行网络
            if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
                return

        # 连接帧以创建模型的输入张量
        inputs = np.concatenate(self.frames)
        # 丢弃旧帧以节省内存
        if not self.terminated:
            self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        # 使用 ASR 模型执行语音到文本的转换
        logits, labels, text = self.frame_to_text(inputs)
        feats = logits  # 使用 logits 作为特征以获得更好的唇同步效果

        # 如果需要，保存特征
        if self.opt.asr_save_feats:
            self.all_feats.append(feats)

        # 在循环缓冲区中记录特征（用于实时应用）
        if not self.terminated:
            start = self.feat_buffer_idx * self.context_size
            end = start + feats.shape[0]
            # 确保特征维度匹配
            if feats.shape[-1] != self.audio_dim:
                feats = feats[:, :self.audio_dim] if feats.shape[-1] > self.audio_dim else F.pad(feats, (0, self.audio_dim - feats.shape[-1]))
            self.feat_queue[start:end] = feats
            self.feat_buffer_idx = (self.feat_buffer_idx + 1) % self.feat_buffer_size

        # 将识别的文本附加到输出文本
        if text != '':
            self.text = self.text + ' ' + text

        # 当终止时，最终确定文本并在需要时保存特征
        if self.terminated:
            self.text += '\n[END]'
            print(self.text)
            if self.opt.asr_save_feats:
                print(f'[INFO] save all feats for training purpose... ')
                feats = torch.cat(self.all_feats, dim=0)  # 连接所有特征
                window_size = 16
                padding = window_size // 2
                feats = feats.view(-1, self.audio_dim).permute(1, 0).contiguous()  # 重塑特征
                feats = feats.view(1, self.audio_dim, -1, 1)  # 准备展开
                unfold_feats = F.unfold(feats, kernel_size=(window_size, 1), padding=(padding, 0), stride=(2, 1))
                unfold_feats = unfold_feats.view(self.audio_dim, window_size, -1).permute(2, 1, 0).contiguous()
                output_path = self.opt.asr_wav.replace('.wav', '_eo.npy') if 'esperanto' in self.opt.asr_model else self.opt.asr_wav.replace('.wav', '.npy')
                np.save(output_path, unfold_feats.cpu().numpy())
                print(f"[INFO] saved logits to {output_path}")

    # 从输入文件创建音频流
    def create_file_stream(self):
        stream, sample_rate = sf.read(self.opt.asr_wav)  # 从文件读取音频数据
        stream = stream.astype(np.float32)  # 将音频转换为 float32 格式

        # 如果音频有多个通道，仅使用第一个通道
        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        # 如果采样率与预期不符，重新采样音频
        if sample_rate != self.sample_rate:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        print(f'[INFO] loaded audio stream {self.opt.asr_wav}: {stream.shape}')
        return stream

    # 从流或队列中获取下一个音频帧
    def get_audio_frame(self):
        if self.mode == 'file':
            # 在文件模式下，从文件中读取下一个音频块
            if self.idx < self.file_stream.shape[0]:
                frame = self.file_stream[self.idx: self.idx + self.chunk]
                self.idx = self.idx + self.chunk
                return frame
            else:
                return None
        else:
            # 在实时模式下，从队列中获取下一个帧
            frame = self.queue.get()
            self.idx = self.idx + self.chunk
            return frame

    # 使用 ASR 模型将音频帧转换为文本
    def frame_to_text(self, frame):
        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            result = self.model(inputs['input_features'].to(self.device))  # 从 ASR 模型获取 logits
            logits = result.logits  # [1, N - 1, audio_dim]

        # 截取左右跨步
        left = max(0, self.stride_left_size)
        right = min(logits.shape[1], logits.shape[1] - self.stride_right_size + 1)
        if self.terminated:
            right = logits.shape[1]

        logits = logits[:, left:right]
        predicted_ids = torch.argmax(logits, dim=-1)  # 获取预测的标记索引
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()  # 解码为文本
        return logits[0], predicted_ids[0], transcription

    # 主函数，在终止前循环运行 ASR
    def run(self):
        self.listen()  # 开始监听音频流
        while not self.terminated:
            self.run_step()  # 处理音频并转换为文本

    # 清空音频队列以减少潜在的延迟
    def clear_queue(self):
        print(f'[INFO] clear queue')
        if self.mode == 'live':
            self.queue.queue.clear()
        if self.play:
            self.output_queue.queue.clear()

    # 实时 ASR 的预热函数，用于预热模型并稳定延迟
    def warm_up(self):
        self.listen()
        print(f'[INFO] warm up ASR live model, expected latency = {self.warm_up_steps / self.fps:.6f}s')
        t = time.time()
        for _ in range(self.warm_up_steps):
            self.run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time() - t
        print(f'[INFO] warm-up done, actual latency = {t:.6f}s')
        self.clear_queue()

# 主函数，处理命令行参数并启动 ASR
if __name__ == '__main__':
    import argparse

    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, default='')
    parser.add_argument('--play', action='store_true', help="play out the audio")
    parser.add_argument('--model', type=str, default='/home/zhutianheng/projects/GaussianTalker/data_utils/data2vec/models/wav2vec2-bert-CV16-en')
    parser.add_argument('--save_feats', action='store_true')
    parser.add_argument('--fps', type=int, default=50)  # 每秒的音频帧数
    parser.add_argument('-l', type=int, default=10)  # 左跨步大小
    parser.add_argument('-m', type=int, default=50)  # 上下文大小
    parser.add_argument('-r', type=int, default=10)  # 右跨步大小

    opt = parser.parse_args()

    # 设置 ASR 的其他选项
    opt.asr_wav = opt.wav
    opt.asr_play = opt.play
    opt.asr_model = opt.model
    opt.asr_save_feats = opt.save_feats

    # 检查是否使用了 DeepSpeech，如果是则抛出错误
    if 'deepspeech' in opt.asr_model:
        raise ValueError("DeepSpeech features should not use this code to extract...")

    # 使用提供的选项运行 ASR
    with ASR(opt) as asr:
        asr.run()
