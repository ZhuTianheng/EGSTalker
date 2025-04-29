import os
import cv2
import sys
import lpips
import numpy as np
from matplotlib import pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
import time
#from pytorch_fid import fid_score

class LMDMeter:
    def __init__(self, backend='dlib', region='mouth'):
        self.backend = backend
        self.region = region  # mouth 或 face

        if self.backend == 'dlib':
            import dlib

            # 手动加载 dlib 的人脸关键点检测模型
            self.predictor_path = './shape_predictor_68_face_landmarks.dat'
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError('请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载 dlib 模型文件')

            # 初始化人脸检测器和关键点预测器
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

        else:
            import face_alignment
            # 初始化 face_alignment 的人脸关键点检测器
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.V = 0
        self.N = 0
    
    def get_landmarks(self, img):
        # 从图像中提取关键点
        if self.backend == 'dlib':
            dets = self.detector(img, 1)
            for det in dets:
                shape = self.predictor(img, det)
                lms = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    lms[i, 0] = shape.part(i).x
                    lms[i, 1] = shape.part(i).y
                break
        else:
            lms = self.predictor.get_landmarks(img)[-1]
        
        lms = lms.astype(np.float32)
        return lms

    def clear(self):
        # 清除累积的数值
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        # 将输入转换为 numpy 数组，并将值缩放到 [0, 255] 范围
        outputs = []
        for inp in inputs:
            inp = inp.detach().cpu().numpy()
            inp = (inp * 255).astype(np.uint8)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        # 使用预测图像和真实图像更新 LMD 指标
        preds, truths = self.prepare_inputs(preds[0], truths[0])
        lms_pred = self.get_landmarks(preds)
        lms_truth = self.get_landmarks(truths)

        if self.region == 'mouth':
            # 仅使用嘴部的关键点（索引 48 到 67）
            lms_pred = lms_pred[48:68]
            lms_truth = lms_truth[48:68]

        # 通过减去均值来归一化关键点
        lms_pred = lms_pred - lms_pred.mean(0)
        lms_truth = lms_truth - lms_truth.mean(0)
        
        # 计算预测关键点和真实关键点之间的平均距离
        dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)
        
        self.V += dist
        self.N += 1
    
    def measure(self):
        # 返回平均 LMD 值
        return self.V / self.N

    def report(self):
        return f'LMD ({self.backend}) = {self.measure():.6f}' 


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        # 清除累积的数值
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        # 将输入转换为 numpy 数组
        outputs = []
        for inp in inputs:
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # 使用预测图像和真实图像更新 PSNR 指标
        preds, truths = self.prepare_inputs(preds, truths)
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        # 返回平均 PSNR 值
        return self.V / self.N

    def report(self):
        return f'PSNR = {self.measure():.6f}'


class LPIPSMeter:
    def __init__(self, net='alex', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        # 初始化 LPIPS 模型
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        # 清除累积的数值
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        # 将输入转换为 LPIPS 需要的格式
        outputs = []
        for inp in inputs:
            inp = inp.permute(0, 3, 1, 2).contiguous()
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        # 使用预测图像和真实图像更新 LPIPS 指标
        preds, truths = self.prepare_inputs(preds, truths)
        v = self.fn(truths, preds, normalize=True).mean().item()
        self.V += v
        self.N += 1
    
    def measure(self):
        # 返回平均 LPIPS 值
        return self.V / self.N

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'


class SSIMMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        # 清除累积的数值
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        # 将输入转换为 numpy 数组
        outputs = []
        for inp in inputs:
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        # 使用预测图像和真实图像更新 SSIM 指标
        preds, truths = self.prepare_inputs(preds, truths)
        height, width = preds[0].shape[1:3]
        win_size = min(7, height, width)  # 确保窗口大小不超过图像尺寸
        ssim_value = ssim(preds[0].transpose(1, 2, 0), truths[0].transpose(1, 2, 0),
                          channel_axis=-1, win_size=win_size, data_range=1.0)
        
        self.V += ssim_value
        self.N += 1

    def measure(self):
        # 返回平均 SSIM 值
        return self.V / self.N

    def report(self):
        return f'SSIM = {self.measure():.6f}'


if __name__ == "__main__":
    # 初始化所有指标
    lmd_meter = LMDMeter(backend='fan')
    psnr_meter = PSNRMeter()
    lpips_meter = LPIPSMeter()
    ssim_meter = SSIMMeter()
    #fid_meter = FIDMeter()

    # 清除所有指标
    lmd_meter.clear()
    psnr_meter.clear()
    lpips_meter.clear()
    ssim_meter.clear()
    #fid_meter.clear()

    # 从命令行参数获取视频路径
    vid_path_1 = sys.argv[1]
    vid_path_2 = sys.argv[2]

    # 打开视频捕获
    capture_1 = cv2.VideoCapture(vid_path_1)
    capture_2 = cv2.VideoCapture(vid_path_2)

    counter = 0
    start_time = time.time()

    try:
        while True:
            # 从两个视频中读取帧
            ret_1, frame_1 = capture_1.read()
            ret_2, frame_2 = capture_2.read()
            #frame_1 = cv2.resize(frame_1, (450, 450))
            #frame_2 = cv2.resize(frame_2, (450, 450))

            if not (ret_1 and ret_2):
                break
            
            # 将帧转换为张量并归一化到 [0, 1] 范围
            inp_1 = torch.FloatTensor(frame_1[..., ::-1] / 255.0)[None, ...].cuda()
            inp_2 = torch.FloatTensor(frame_2[..., ::-1] / 255.0)[None, ...].cuda()
            
            # 使用当前帧更新每个指标
            lmd_meter.update(inp_1, inp_2)
            psnr_meter.update(inp_1, inp_2)
            lpips_meter.update(inp_1, inp_2)
            ssim_meter.update(inp_1, inp_2)

            counter += 1
            if counter % 100 == 0:
                print(f'已处理 {counter} 帧')
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 释放视频捕获
        capture_1.release()
        capture_2.release()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"总处理时间: {elapsed_time:.2f} 秒")

    # 打印最终的指标结果
    print(lmd_meter.report())
    print(psnr_meter.report())
    print(lpips_meter.report())
    print(ssim_meter.report())
