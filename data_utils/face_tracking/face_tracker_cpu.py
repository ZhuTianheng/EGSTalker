import os

# 指定 CUDA 库的路径，确保程序可以找到正确版本的 CUDA
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH'

import sys
import cv2
import argparse
from pathlib import Path
import torch
import numpy as np
from data_loader import load_dir
from facemodel import Face_3DMM
from util import *
from render_3dmm import Render_3DMM
import copy

dir_path = os.path.dirname(os.path.realpath(__file__))

def set_requires_grad(tensor_list):
    for tensor in list(tensor_list):
        tensor.requires_grad = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path", type=str, default="obama/ori_imgs", help="idname of target person"
)
parser.add_argument("--img_h", type=int, default=512, help="image height")
parser.add_argument("--img_w", type=int, default=512, help="image width")
parser.add_argument("--frame_num", type=int, default=11000, help="image number")
args = parser.parse_args()

start_id = 0
end_id = args.frame_num

lms, img_paths = load_dir(args.path, start_id, end_id)
num_frames = lms.shape[0]
h, w = args.img_h, args.img_w
cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float)
id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
model_3dmm = Face_3DMM(
    os.path.join(dir_path, "3DMM"), id_dim, exp_dim, tex_dim, point_num
)

# only use one image per 40 to do fit the focal length
sel_ids = np.arange(0, num_frames, 40)
sel_num = sel_ids.shape[0]
arg_focal = 1600
arg_landis = 1e5

print(f'[INFO] fitting focal length...')

# fit the focal length
for focal in range(600, 1500, 100):
    id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
    euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
    trans = lms.new_zeros((sel_num, 3), requires_grad=True)
    trans.data[:, 2] -= 7
    focal_length = lms.new_zeros(1, requires_grad=False)
    focal_length.data += focal
    set_requires_grad([id_para, exp_para, euler_angle, trans])

    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=0.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=0.1)

    for iter in range(2000):
        id_para_batch = id_para.expand(sel_num, -1)
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
        )
        proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms[sel_ids].detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_frame.step()

    for iter in range(2500):
        id_para_batch = id_para.expand(sel_num, -1)
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
        )
        proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms[sel_ids].detach())
        loss_regid = torch.mean(id_para * id_para)
        loss_regexp = torch.mean(exp_para * exp_para)
        loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
        optimizer_idexp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_idexp.step()
        optimizer_frame.step()

        if iter % 1500 == 0 and iter >= 1500:
            for param_group in optimizer_idexp.param_groups:
                param_group["lr"] *= 0.2
            for param_group in optimizer_frame.param_groups:
                param_group["lr"] *= 0.2

    print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

    if loss_lan.item() < arg_landis:
        arg_landis = loss_lan.item()
        arg_focal = focal

print("[INFO] find best focal:", arg_focal)

print(f'[INFO] coarse fitting...')

# for all frames, do a coarse fitting
id_para = lms.new_zeros((1, id_dim), requires_grad=True)
exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
tex_para = lms.new_zeros(
    (1, tex_dim), requires_grad=True
)  # not optimized in this block
euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
trans = lms.new_zeros((num_frames, 3), requires_grad=True)
light_para = lms.new_zeros((num_frames, 27), requires_grad=True)

point_cloud = lms.new_zeros((num_frames, point_num, 3), requires_grad=True)

trans.data[:, 2] -= 7
focal_length = lms.new_zeros(1, requires_grad=True)
focal_length.data += arg_focal

set_requires_grad([id_para, exp_para, tex_para, euler_angle, trans, light_para])

optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=0.1)
optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

for iter in range(1500):
    id_para_batch = id_para.expand(num_frames, -1)
    geometry = model_3dmm.get_3dlandmarks(
        id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
    )
    proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
    loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms.detach())
    loss = loss_lan
    optimizer_frame.zero_grad()
    loss.backward()
    optimizer_frame.step()
    if iter == 1000:
        for param_group in optimizer_frame.param_groups:
            param_group["lr"] = 0.1

for param_group in optimizer_frame.param_groups:
    param_group["lr"] = 0.1

for iter in range(2000):
    id_para_batch = id_para.expand(num_frames, -1)
    geometry = model_3dmm.get_3dlandmarks(
        id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
    )
    proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
    loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms.detach())
    loss_regid = torch.mean(id_para * id_para)
    loss_regexp = torch.mean(exp_para * exp_para)
    loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
    optimizer_idexp.zero_grad()
    optimizer_frame.zero_grad()
    loss.backward()
    optimizer_idexp.step()
    optimizer_frame.step()
    if iter % 1000 == 0 and iter >= 1000:
        for param_group in optimizer_idexp.param_groups:
            param_group["lr"] *= 0.2
        for param_group in optimizer_frame.param_groups:
            param_group["lr"] *= 0.2

print(loss_lan.item(), torch.mean(trans[:, 2]).item())

print(f'[INFO] fitting light...')

batch_size = 32

device_default = torch.device("cpu")
device_render = torch.device("cpu")
renderer = Render_3DMM(arg_focal, h, w, batch_size, device_render)

sel_ids = np.arange(0, num_frames, int(num_frames / batch_size))[:batch_size]
imgs = []
for sel_id in sel_ids:
    imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
imgs = np.stack(imgs)
sel_imgs = torch.as_tensor(imgs)
sel_lms = lms[sel_ids]
sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
set_requires_grad([sel_light])

optimizer_tl = torch.optim.Adam([tex_para, sel_light], lr=0.1)
optimizer_id_frame = torch.optim.Adam([euler_angle, trans, exp_para, id_para], lr=0.01)

for iter in range(71):
    sel_exp_para, sel_euler, sel_trans = (
        exp_para[sel_ids],
        euler_angle[sel_ids],
        trans[sel_ids],
    )
    sel_id_para = id_para.expand(batch_size, -1)
    geometry = model_3dmm.get_3dlandmarks(
        sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy
    )
    proj_geo = forward_transform(geometry, sel_euler, sel_trans, focal_length, cxy)

    loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
    loss_regid = torch.mean(id_para * id_para)
    loss_regexp = torch.mean(sel_exp_para * sel_exp_para)

    sel_tex_para = tex_para.expand(batch_size, -1)
    sel_texture = model_3dmm.forward_tex(sel_tex_para)
    geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
    rott_geo = forward_rott(geometry, sel_euler, sel_trans)
    render_imgs = renderer(
        rott_geo.to(device_render),
        sel_texture.to(device_render),
        sel_light.to(device_render),
    )
    render_imgs = render_imgs.to(device_default)

    mask = (render_imgs[:, :, :, 3]).detach() > 0.0
    render_proj = sel_imgs.clone()
    render_proj[mask] = render_imgs[mask][..., :3].byte()
    loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)

    if iter > 50:
        loss = loss_col + loss_lan * 0.05 + loss_regid * 1.0 + loss_regexp * 0.8
    else:
        loss = loss_col + loss_lan * 3 + loss_regid * 2.0 + loss_regexp * 1.0
