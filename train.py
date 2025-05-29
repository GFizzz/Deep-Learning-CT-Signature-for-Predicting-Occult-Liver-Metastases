import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset.dataset_ol_plusliver import *
from network.network_liver_metastases import *
import itertools
from monai.transforms import Compose, RandGaussianNoise, RandBiasField, RandFlip
# Define training and validation functions


class FocalLossBinary(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLossBinary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7
        preds = torch.sigmoid(preds)
        loss_1 = -self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = - (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)

def train(model, device, train_loader, optimizer, criterion,scheduler):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (pancreas_combined_tensor, liver_combined_tensor, label_tensor, patient_name) in enumerate(tqdm(train_loader)):
        pancreas_combined_tensor, liver_combined_tensor = pancreas_combined_tensor.to(device), liver_combined_tensor.to(device)
        label_tensor = label_tensor.to(device)
        optimizer.zero_grad()
        # outputs = model(liver.unsqueeze(1), tumor.unsqueeze(1))
        outputs = model(pancreas_combined_tensor, liver_combined_tensor)
        # print("outputs",outputs)
        # print("labels",labels)

        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label_tensor.size(0)
        correct += predicted.eq(label_tensor).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    return train_loss, train_accuracy

def validate(model, device, val_loader, criterion, best_accuracy, output_excel_path):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_names = []
    with torch.no_grad():
        for pancreas_combined_tensor, liver_combined_tensor, label_tensor, patient_name in tqdm(val_loader):
            pancreas_combined_tensor, liver_combined_tensor = pancreas_combined_tensor.to(device), liver_combined_tensor.to(device)
            label_tensor = label_tensor.to(device)
            outputs = model(pancreas_combined_tensor, liver_combined_tensor)
            loss = criterion(outputs, label_tensor)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += label_tensor.size(0)
            correct += predicted.eq(label_tensor).sum().item()
              
            all_labels.extend(label_tensor.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())  # 获取类别1的概率
            all_names.extend(patient_name)  # 保存病人名

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100. * correct / total
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Save results to Excel if current accuracy is the best
    if val_accuracy > best_accuracy:
        df = pd.DataFrame({
            'Patient Name': all_names,
            'Label': all_labels,
            'Probability': all_probabilities
        })
        df.to_excel(output_excel_path, index=False)
        best_accuracy = val_accuracy  # Update best accuracy
    
    return val_loss, val_accuracy, cm, all_labels, all_probabilities, best_accuracy

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100. * correct / total
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return val_loss, val_accuracy, cm

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    # plt.show()

# Main function for training and validation
def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Data paths and parameters
    # Define the directories for the three phases
    phase1_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/1max_tumor/output_boundingbox'
    phase2_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/2max_tumor/output_boundingbox'
    phase3_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/NC_tumor/output_boundingbox'

    liver_phase1_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/1max_liver/output_boundingbox'
    liver_phase2_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/2max_liver/output_boundingbox'
    liver_phase3_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/NC_liver/output_boundingbox'

    # Define the path to the label file
    label_file = "/data/gzf/tumor_transfer/Dataset/20250115_train.xlsx"
    # Create an instance of the PancreasDataset
    dataset = PancreasDataset(phase1_dir, phase2_dir, phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file, transform=None)
    batch_size = 1
    num_epochs = 100
    learning_rate = 0.0001

    train_transform = Compose([
        RandGaussianNoise(prob=0.3),
        RandBiasField(prob=0.3),
        RandFlip(prob=0.3, spatial_axis=0),
        # RandMotion(prob=0.2)
    ])

    # Create dataset and split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size])
    train_dataset = PancreasDataset(phase1_dir, phase2_dir, phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file, transform = train_transform)
    val_dataset = PancreasDataset(phase1_dir, phase2_dir, phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file,transform = None)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Initialize model
    model = POLMMamba(in_chans=3,
                    #  out_chans=out_channels,
                     depths=[2, 2, 2, 2],
                     feat_size=[48, 96, 192, 384]).to(device="cuda")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # class_weights = torch.tensor([1.0, 400.0/195.0], dtype=torch.float32).to(device)

    # class_weights = torch.tensor([1.0, 400.0/195.0], dtype=torch.float32)  # 根据类别分布调整权重
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # 创建 CosineAnnealingLR 调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
    # criterion = FocalLossBinary(alpha=0.3, gamma=1.5)
    # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_accuracy = 0.0
    best_confusion_matrix = None
    best_accuracy =0.0
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion,scheduler)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Validation
        # Validation
        val_loss, val_accuracy, cm, all_labels, all_probabilities, best_accuracy  = validate(model, device, val_loader, criterion,best_accuracy, 'result_tumor_transfer/20250306_liverOnly_best_validation_results.xlsx')
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save the model if validation accuracy has increased
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_confusion_matrix = cm
            torch.save(model.state_dict(), 'logs/model/result_tumor_transfer/liver_best_model_20250306.pth')

        # if epoch == 99:
        #     torch.save(model.state_dict(), 'final_model.pth')
        
        print(cm)
    
    # Plot and save confusion matrix of best validation accuracy
    plot_confusion_matrix(best_confusion_matrix, classes=['0', '1'], title='Confusion Matrix')

    print("Training finished!")

if __name__ == "__main__":
    main()
