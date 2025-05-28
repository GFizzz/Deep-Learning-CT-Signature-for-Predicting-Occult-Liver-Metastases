# Deep-Learning-CT-Signature-for-Predicting-Occult-Liver-Metastases

## Introduction

This repository accompanies our research on using deep learning to develop a CT-based signature for predicting occult liver metastases. The method is designed to assist in identifying patients at high risk of metastasis, potentially enabling earlier interventions and improved treatment planning.

![Model Overview](images/our_framework.jpg)

## 1. System Requirements🔧

- OS: Linux / macOS / Windows 10+
- Python: >= 3.8
- Recommended: CUDA-compatible GPU (CUDA >= 11.0)
- Dependencies:
  - numpy >= 1.21.0  
  - torch >= 1.10.0  
  - torchvision >= 0.11.0  
  - scikit-learn >= 1.0.0  
  - matplotlib >= 3.4.0  
  - (Other packages specified in `requirements.txt`)

Install all dependencies with:
```bash
pip install -r requirements.txt
```
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



---
## Acknowledgement
Many thanks for these repos for their great contribution!

https://github.com/ge-xing/SegMamba 

https://github.com/MIC-DKFZ/nnUNet


Thank you for your support and patience!


