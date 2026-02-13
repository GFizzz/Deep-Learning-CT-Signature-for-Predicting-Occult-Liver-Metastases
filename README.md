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

## 3. Instructions for Use & Demo

### Data collection

Due to privacy issues, you need to collect multi-phase CECT images (non-contrast phase, arterial phase, venous phase) by yourself.

We will put the preprocessed images demo in the data directory "data/"

#### Labels File Format

Patient labels are stored in an Excel file located in the `data/` directory.

- The Excel file (e.g., `labels.xlsx`) contains two columns:
  - **First column**: Patient ID (must match the folder names in `data/`)
  - **Second column**: Label (`0` or `1`), where:
    - `0` = Negative for occult liver metastases
    - `1` = Positive for occult liver metastases

### Preprocessing Instructions

Refering to the preprocessing pipeline of **nnU-Net** (https://github.com/MIC-DKFZ/nnUNet).  
Each phase of the contrast-enhanced CT (CECT) images is individually preprocessed to generate 3D full-resolution `.npy` files (`3d_fullres`).

After preprocessing, for each phase, the corresponding CT image is element-wise multiplied by its tumor mask and liver mask to extract the tumor region and liver region. This results in one `.npy` file per phase representing the tumor area and liver area only.
Then, organize the files as follows:
```text
data/

├── PancreasTumor/

│ ├── noncontrast/

│ │ ├── patient01.npy

│ │ ├── patient02.npy

│ │ ├── ...

│ ├── venous/

│ │ ├── patient01.npy

│ │ ├── patient02.npy

│ │ ├── ...

│ ├── arteria/

│ │ ├── patient01.npy

│ │ ├── patient02.npy

│ │ ├── ...

├── Liver/

│ ├── ...
```
### Training 

When the pre-processing and data preparing process is done, we can train our model.

We mainly use the pre-processed data from last step: 

- **phase1_dir = "data/PancreasTumor/noncontrast/"**

- **phase2_dir = "data/PancreasTumor/arteria/"**

- **phase3_dir = "data/PancreasTumor/venous/"**

- **liver_phase1_dir = "data/Liver/noncontrast/"**

- **liver_phase2_dir = "data/Liver/arteria/"**

- **liver_phase3_dir = "data/Liver/venous/"**

- **label_file = "data/label.xlsx"**

```bash 
python train.py
```

The training results and models are saved in: **"logs/model/"**


### Pretrained Weights

We provide pretrained model weights for reproducibility and evaluation.

You can download the pretrained checkpoints from the following link:

🔗 https://drive.google.com/drive/folders/1J9L-5HTu3gXzFG5o3Bs-VoF66-6rph0h

### Inference 

When we have trained our models, we can inference all the data in testing set.

```bash 
python inference.py
```

When this process is done, the prediction results will be put in **"test_results/"** as an `.xlsx` files.

### Demo: Inference patients with no label 

```bash 
python inference_nolabel.py
```

## 4. License

This project is covered under the **Apache 2.0 License**.


## Acknowledgement
Many thanks for these repos for their great contribution!

https://github.com/ge-xing/SegMamba 

https://github.com/MIC-DKFZ/nnUNet


Thank you for your support and patience!


