# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, RandGaussianNoise, RandBiasField, RandFlip,
)
class PancreasDataset(Dataset):
    def __init__(self, phase1_dir, phase2_dir, phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file=None):
        self.phase1_dir = phase1_dir
        self.phase2_dir = phase2_dir
        self.phase3_dir = phase3_dir
        self.liver_phase1_dir = liver_phase1_dir
        self.liver_phase2_dir = liver_phase2_dir
        self.liver_phase3_dir = liver_phase3_dir

        # Load all patient names from the given Excel file
        self.patient_names = self._load_patient_names(label_file)
        self.augmentation_transform1 = Compose([
            RandGaussianNoise(prob=0.7),
            RandBiasField(prob=0.7),
            RandFlip(prob=0.7, spatial_axis=0),
            # RandMotion(prob=0.2)
        ])

    def _load_patient_names(self, label_file):
        """从Excel文件中加载病人名"""
        patient_names = []
        df = pd.read_excel(label_file)
        for idx in range(len(df)):
            patient_name = df.iloc[idx, 0]  # 假设病人名在第一列
            patient_names.append(patient_name)
        return patient_names

    def _resize_liver(self, array):
        """Resize liver array to [64, 192, 192]."""
        depth, height, width = array.shape

        # First, resize each slice (height x width) to 192x192
        resized_slices = []
        for i in range(depth):
            resized_slice = cv2.resize(array[i, :, :], (192, 192), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(resized_slice)

        # Stack resized slices along the depth dimension
        resized_array = np.stack(resized_slices, axis=0)

        # Now, resize depth to 64 using zoom (depth interpolation)
        depth_scale = 64 / depth
        resized_array = zoom(resized_array, (depth_scale, 1, 1), order=1)  # order=1 means linear interpolation

        return resized_array
    def _pad_to_32(self, array):
        """将第一维度小于32的数组填充到32"""
        depth, height, width = array.shape
        if depth < 32:
            padded_array = np.zeros((32, height, width), dtype=array.dtype)
            padded_array[:depth, :, :] = array
            return padded_array
        return array

    def __len__(self):
        return len(self.patient_names)

    def __getitem__(self, idx):
        patient_name = self.patient_names[idx]
        
        # Load the three phase npy files
        phase1_path = os.path.join(self.phase1_dir, f"{patient_name}.npy")
        phase2_path = os.path.join(self.phase2_dir, f"{patient_name}.npy")
        phase3_path = os.path.join(self.phase3_dir, f"{patient_name}.npy")
        
        phase1_array = np.load(phase1_path)
        phase2_array = np.load(phase2_path)
        phase3_array = np.load(phase3_path)

        # Load the three phase npy files for liver
        liver_phase1_path = os.path.join(self.liver_phase1_dir, f"{patient_name}.npy")
        liver_phase2_path = os.path.join(self.liver_phase2_dir, f"{patient_name}.npy")
        liver_phase3_path = os.path.join(self.liver_phase3_dir, f"{patient_name}.npy")

        liver_phase1_array = np.load(liver_phase1_path)
        liver_phase2_array = np.load(liver_phase2_path)
        liver_phase3_array = np.load(liver_phase3_path)

        # Resize liver arrays to [64, 192, 192]
        liver_phase1_array = self._resize_liver(liver_phase1_array)
        liver_phase2_array = self._resize_liver(liver_phase2_array)
        liver_phase3_array = self._resize_liver(liver_phase3_array)

        # Pad arrays to have a first dimension of 32
        phase1_array = self._pad_to_32(phase1_array)
        phase2_array = self._pad_to_32(phase2_array)
        phase3_array = self._pad_to_32(phase3_array)

        # Convert pancreas arrays to PyTorch tensors and expand dims
        phase1_tensor = torch.tensor(phase1_array, dtype=torch.float32).unsqueeze(0)
        phase2_tensor = torch.tensor(phase2_array, dtype=torch.float32).unsqueeze(0)
        phase3_tensor = torch.tensor(phase3_array, dtype=torch.float32).unsqueeze(0)

        # Convert liver arrays to PyTorch tensors and expand dims
        liver_phase1_tensor = torch.tensor(liver_phase1_array, dtype=torch.float32).unsqueeze(0)
        liver_phase2_tensor = torch.tensor(liver_phase2_array, dtype=torch.float32).unsqueeze(0)
        liver_phase3_tensor = torch.tensor(liver_phase3_array, dtype=torch.float32).unsqueeze(0)

        # Concatenate pancreas tensors along the channel dimension
        pancreas_combined_tensor = torch.cat((phase1_tensor, phase2_tensor, phase3_tensor), dim=0)

        # Concatenate liver tensors along the channel dimension
        liver_combined_tensor = torch.cat((liver_phase1_tensor, liver_phase2_tensor, liver_phase3_tensor), dim=0)

        if self.transform:
            pancreas_combined_tensor = self.transform(pancreas_combined_tensor)
            liver_combined_tensor = self.transform(liver_combined_tensor)

        return pancreas_combined_tensor, liver_combined_tensor, patient_name



