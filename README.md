# Deep-Learning-CT-Signature-for-Predicting-Occult-Liver-Metastases

## 0. Introduction

This repository accompanies our research on using deep learning to develop a CT-based signature for predicting occult liver metastases. The method is designed to assist in identifying patients at high risk of metastasis, potentially enabling earlier interventions and improved treatment planning.

![Model Overview](images/our_framework.jpg)

## 1. System Requirements🔧

- OS: Linux
- Python: >= 3.8
- Framework: Pytorch
- Recommended: CUDA-compatible GPU (CUDA >= 11.0)
- Dependencies: see **Sect.2**

## 2. Installation Guide📅

### Environment install
Clone this repository and navigate to the root directory of the project.

```bash
git clone https://github.com/GFizzz/Deep-Learning-CT-Signature-for-Predicting-Occult-Liver-Metastases.git

```
#### Install causal-conv1d

```bash
cd causal-conv1d

python setup.py install
```

#### Install mamba

```bash
cd mamba

python setup.py install
```

#### Install monai 

```bash
pip install monai
```

## 3. Demo

### Data collection

Due to privacy issues, you need to collect multi-phase CECT images (arterial phase, venous phase, delay phase) by yourself.

We will put the preprocessed images demo in the data directory "data/"

### Preprocessing
可以参考nnunet的预处理流程（https://github.com/MIC-DKFZ/nnUNet），对三期CECT图像分别经过预处理得到3d-fullres的npy文件，并将三期图像的mask乘上其肿瘤mask得到肿瘤区域的npy文件
After pre-processing, organize the data structure in this format:

- **data/arteria/demo1.npy**

- **data/venous/demo1.npy**

- **data/delayed/demo1.npy**

### Training 

When the pre-processing and data preparing process is done, we can train our model.

We mainly use the pre-processde data from last step: 

- **phase1_dir = "data/arteria/"**

- **phase2_dir = "data/venous/"**

- **phase3_dir = "data/venous/"**

```bash 
python train.py
```

The training logs and checkpoints are saved in:
**logdir = f"./logs/segmamba"**




### Inference 

When we have trained our models, we can inference all the data in testing set.

```bash 
python inference.py
```

When this process is done, the prediction cases will be put in this path:
**save_path = "./prediction_results/segmamba"**


---
## Acknowledgement
Many thanks for these repos for their great contribution!

https://github.com/ge-xing/SegMamba 

https://github.com/MIC-DKFZ/nnUNet


Thank you for your support and patience!


