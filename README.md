# EGSTalker
Real-Time Audio-Driven Talking Head Generation with Efficient Gaussian Deformation

---

## 🧠 Introduction

**EGSTalker** is a real-time audio-driven talking head generation framework based on 3D Gaussian deformation.  
We propose an efficient spatial-audio attention (ESAA) mechanism and Kolmogorov-Arnold Network (KAN) based deformation decoder to achieve high-fidelity and synchronized talking head synthesis with significant inference speed improvements.

This repository contains the official implementation of the paper:

> **EGSTalker: Real-Time Audio-Driven Talking Head Generation with Efficient Gaussian Deformation**  
> [Author] Tianheng Zhu, Yinfeng Yu*, Liejun Wang, Fuchun Sun, Wendong Zheng  
> (*Corresponding author)

## 🧩 Framework

The overall framework of EGSTalker is illustrated as follows:
<p align="center">
  <img src="https://raw.githubusercontent.com/ZhuTianheng/EGSTalker/main/docs/framework.png" width="70%">
</p>

The Efficient Spatial-Audio Attention (ESAA) module structure:
<p align="center">
  <img src="https://raw.githubusercontent.com/ZhuTianheng/EGSTalker/main/docs/esaa.png" width="70%">
</p>

## 📜 Paper

> [Coming soon]

## 📽️ Demo

### Audio-Driven Talking Head Synthesis
👉 [Download the demo video (EGSTalker.mp4)](https://github.com/ZhuTianheng/EGSTalker/tree/main/result-video)

## 📦 Installation

We recommend using **Conda** to set up the environment. The following commands will create and activate the `egstalker` environment using the provided `egstalker.yml` file.

```bash
git clone https://github.com/ZhuTianheng/EGSTalker.git
cd EGSTalker
git submodule update --init --recursive

# Create and activate conda environment
conda env create -f egstalker.yml
conda activate egstalker
```

Install optional dependencies (if not already included):

```bash
pip install -e submodules/custom-gaussian-rasterization
pip install -e submodules/simple-knn
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
## 📅 Download Dataset

We use talking portrait videos (3–5 minutes) from:

- [AD-NeRF](https://github.com/YudongGuo/AD-NeRF)
- [GeneFace](https://github.com/yerfor/GeneFace)
- [HDTF dataset](https://github.com/MRzzm/HDTF)

Example download:

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true -O data/obama/obama.mp4
```
## 🧾 Data Preparation

### 1. Face Parsing

```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth
```

### 2. 3D Morphable Model

Download Basel Face Model 2009 from [here](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details), and place `01_MorphableModel.mat` in:

```
data_utils/face_tracking/3DMM/
```

Then run:

```bash
cd data_utils/face_tracking
python convert_BFM.py
cd ../../
python data_utils/process.py ${YOUR_DATASET_DIR}/${DATASET_NAME}/${DATASET_NAME}.mp4 
```


## 🛠️ Usage

To train the model:

```bash
python train.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
                --model_path ${YOUR_MODEL_DIR} \
                --configs arguments/args.py
```
Rendering：
```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
                 --model_path ${YOUR_MODEL_DIR} \
                 --configs configs/egstalker_default.py \
                 --iteration 10000 \
                 --batch 16
```
Inference with Custom Audio:

Place `<custom_aud>.wav` and `<custom_aud>.npy` in `${YOUR_DATASET_DIR}/${DATASET_NAME}` and run:

```bash
python render.py -s ${YOUR_DATASET_DIR}/${DATASET_NAME} \
                 --model_path ${YOUR_MODEL_DIR} \
                 --configs configs/egstalker_default.py \
                 --iteration 10000 \
                 --batch 128 \
                 --custom_aud <custom_aud>.npy \
                 --custom_wav <custom_aud>.wav \
                 --skip_train \
                 --skip_test
```

---

