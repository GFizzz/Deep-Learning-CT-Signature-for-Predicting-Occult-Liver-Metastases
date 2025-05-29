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
# Define training and validation functions

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

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def test(model, device, test_loader, criterion, output_excel_path):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_names = []
    
    with torch.no_grad():
        for pancreas_combined_tensor, liver_combined_tensor, label_tensor, name in tqdm(test_loader):
            pancreas_combined_tensor, liver_combined_tensor = pancreas_combined_tensor.to(device), liver_combined_tensor.to(device)
            labels = label_tensor.to(device)
            outputs = model(pancreas_combined_tensor, liver_combined_tensor)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())
            all_names.extend(name)

    test_loss = running_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Compute AUC
    auc = roc_auc_score(all_labels, all_probabilities)
    
    # Compute precision, recall, F1
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    # Save results to Excel
    df = pd.DataFrame({
        'Patient Name': all_names,
        'Label': all_labels,
        'Probability': all_probabilities
    })
    df.to_excel(output_excel_path, index=False)
    
    return test_loss, test_accuracy, auc, precision, recall, f1, cm


def main_test():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Define the directories for the three phases
    phase1_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/1max_tumor/output_boundingbox'
    phase2_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/2max_tumor/output_boundingbox'
    phase3_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/NC_tumor/output_boundingbox'

    liver_phase1_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/1max_liver/output_boundingbox'
    liver_phase2_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/2max_liver/output_boundingbox'
    liver_phase3_dir = '/data/gzf/tumor_transfer/Dataset/ol_2/NC_liver/output_boundingbox'
    # Define the path to the label file

    label_file = '/data/gzf/tumor_transfer/external_0303/external_label_0303.xlsx'
    
    # Create an instance of the PancreasDataset
    test_dataset = PancreasDataset(phase1_dir, phase2_dir, phase3_dir, liver_phase1_dir, liver_phase2_dir, liver_phase3_dir, label_file, transform=None)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Number of testing batches: {len(test_loader)}")
    
    # Initialize and load model
    model = POLMMamba(in_chans=3, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384]).to(device)
    # model.load_state_dict(torch.load('METALiv_best_model.pth'))
    # model.load_state_dict('result_tumor_transfer/best_model_20250210.pth')
    state_dict = torch.load('result_tumor_transfer/best_model_20250328_pancreasOnly.pth')
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Test the model
    test_loss, test_accuracy, auc, precision, recall, f1, cm = test(model, device, test_loader, criterion, 'result/testEX_results_0410_pancreasOnly.xlsx')
    
    # Print test results
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, classes=['0', '1'], title='Test Confusion Matrix')

    print("Testing finished!")


if __name__ == "__main__":
    main_test()
