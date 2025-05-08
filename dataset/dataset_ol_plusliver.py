import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import zoom

class PancreasDataset(Dataset):
    def __init__(self, phase1_dir, phase2_dir, phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file, num_class0=500, transform=None):
        self.phase1_dir = phase1_dir
        self.phase2_dir = phase2_dir
        self.phase3_dir = phase3_dir
        self.liver_phase1_dir = liver_phase1_dir
        self.liver_phase2_dir = liver_phase2_dir
        self.liver_phase3_dir = liver_phase3_dir

        self.transform = transform

        # Load all labels from the given Excel file
        self.labels_df = self._load_labels(label_file)
        
        # Filter out patients with missing files
        self.filtered_labels = self._filter_missing_files()

        self.class0_samples = [x for x in self.filtered_labels if x[1] == 0]
        self.class1_samples = [x for x in self.filtered_labels if x[1] == 1]

        if num_class0 is not None:
            self.class0_samples = self.class0_samples[:num_class0]

        self.filtered_labels = self.class0_samples + self.class1_samples

    def _load_labels(self, label_file):
        labels = []
        df = pd.read_excel(label_file)
        for idx in range(len(df)):
            # patient_name = df.iloc[idx, 1]
            # label = df.iloc[idx, 2]
            patient_name = df.iloc[idx, 0]
            label = df.iloc[idx, 1]
            labels.append((patient_name, label))
        return labels

    def _filter_missing_files(self):
        filtered_labels = []
        for patient_name, label in self.labels_df:
            phase1_path = os.path.join(self.phase1_dir, f"{patient_name}.npy")
            phase2_path = os.path.join(self.phase2_dir, f"{patient_name}.npy")
            phase3_path = os.path.join(self.phase3_dir, f"{patient_name}.npy")
            
            liver_phase1_path = os.path.join(self.liver_phase1_dir, f"{patient_name}.npy")
            liver_phase2_path = os.path.join(self.liver_phase2_dir, f"{patient_name}.npy")
            liver_phase3_path = os.path.join(self.liver_phase3_dir, f"{patient_name}.npy")

            if all(os.path.exists(p) for p in [phase1_path, phase2_path, phase3_path, liver_phase1_path, liver_phase2_path, liver_phase3_path]):
                filtered_labels.append((patient_name, label))
        return filtered_labels

    def _pad_to_32(self, array):
        depth, height, width = array.shape
        if depth < 32:
            # Create a new array with shape (32, height, width) and fill with zeros
            padded_array = np.zeros((32, height, width), dtype=array.dtype)
            # Copy the original array into the new array
            padded_array[:depth, :, :] = array
            return padded_array
        return array

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

    def __len__(self):
        return len(self.filtered_labels)
    
    def __getitem__(self, idx):
        # Get the patient name and label from the filtered list
        patient_name, label = self.filtered_labels[idx]
        
        # Load the three phase npy files for pancreas
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

        # Pad pancreas arrays to have a first dimension of 32
        phase1_array = self._pad_to_32(phase1_array)
        phase2_array = self._pad_to_32(phase2_array)
        phase3_array = self._pad_to_32(phase3_array)

        # Resize liver arrays to [64, 192, 192]
        liver_phase1_array = self._resize_liver(liver_phase1_array)
        liver_phase2_array = self._resize_liver(liver_phase2_array)
        liver_phase3_array = self._resize_liver(liver_phase3_array)

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

        # Create label tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pancreas_combined_tensor, liver_combined_tensor, label_tensor, patient_name

def test_pancreas_dataset():
    # Define the directories for the three phases
    pancreas_phase1_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/1max_tumor/output_boundingbox'
    pancreas_phase2_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/2max_tumor/output_boundingbox'
    pancreas_phase3_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/NC_tumor/output_boundingbox'
    
    liver_phase1_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/1max_liver/output_boundingbox'
    liver_phase2_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/2max_liver/output_boundingbox'
    liver_phase3_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/NC_liver/output_boundingbox'
    
    # Define the path to the label file
    label_file = '/data/gzf/tumor_transfer/Dataset/ncav_ol_labelsummary.xlsx'
    
    # Create an instance of the PancreasDataset
    dataset = PancreasDataset(pancreas_phase1_dir, pancreas_phase2_dir, pancreas_phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file, transform=None)
    
    # Test the dataset
    for i in range(len(dataset)):
        pancreas_tensor, liver_tensor, label, patient_name = dataset[i]
        print(f"Patient: {patient_name}, Pancreas Tensor Shape: {pancreas_tensor.shape}, Liver Tensor Shape: {liver_tensor.shape}, Label: {label}")

# test_pancreas_dataset()
