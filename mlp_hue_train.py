import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.snack_dataset import SnackDataset
from utils.hue_histogram import hue_histogram
from utils.mlp import MLP, SnackDatasetTensor


DEVICE = "mps" # Mac GPU
BINS = 2400
N_PCA = 1600
BATCH_SIZE = 512
lr_rate = 0.01
num_epochs = 5000
num_class = 20

# Load the dataset
snacks = SnackDataset()

train_imgs = list(snacks.get_train_set())
val_imgs = list(snacks.get_validation_set())
test_imgs =list(snacks.get_test_set())

print(f"Train size: {len(train_imgs)} | Val size: {len(val_imgs)} | Test size: {len(test_imgs)}")

train_features = np.array([hue_histogram(img, bins=BINS) for img, _ in train_imgs])
train_labels = np.array([label for _, label in train_imgs]).reshape(-1, 1)
val_features = np.array([hue_histogram(img, bins=BINS) for img, _ in val_imgs])
val_labels = np.array([label for _, label in val_imgs]).reshape(-1, 1)
test_features = np.array([hue_histogram(img, bins=BINS) for img, _ in test_imgs])
test_labels = np.array([label for _, label in test_imgs]).reshape(-1, 1)

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Train a PCA model
pca = PCA(N_PCA)

# Convert data to tensors after applying PCA
train_features = pca.fit_transform(train_features)
val_features = pca.transform(val_features)
test_features = pca.transform(test_features)

# Create torch data loaders
train_dataset = SnackDatasetTensor(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_features = SnackDatasetTensor(val_features, val_labels)
val_loader = DataLoader(val_features, batch_size=1)
test_features = SnackDatasetTensor(test_features, test_labels)
test_loader = DataLoader(test_features, batch_size=1)

# Train the model
model = MLP(N_PCA, 128, 256, num_class)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

for epoch in range(num_epochs):
    model.train()
    for imgs, labels in train_loader:
        optimizer.zero_grad()
        imgs = imgs.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)
        
        # Forward pass
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step() 
    
    if epoch % 100 == 0:
        total_val = 0
        correct_val = 0
        model.eval()
        with torch.no_grad():
            for val_img, val_label in val_loader:
                val_img = val_img.to(DEVICE)
                val_label = val_label.to(DEVICE)
                output = model(val_img)
                output = torch.softmax(output, dim=1)
                output = torch.argmax(output, dim=1)
                output = output.cpu().detach()
                val_label = val_label.cpu().detach()[0]
                correct_val += (output == val_label).item()
                total_val += 1
        print(f"Epoch: {epoch} | Loss: {loss.item():.4f} | Validation accuracy: {correct_val/total_val:.4f}")