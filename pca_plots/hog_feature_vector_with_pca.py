# This sctipt plots the explained variance of the HOG feature vector with PCA components
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib import style 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils.create_hog import HOG
from utils.snack_dataset import SnackDataset


style.use('ggplot')

num_components = 1200

# Initialize the dataset
snacks = SnackDataset()

# Load the training set
train_imgs = list(snacks.get_train_set())
train_features = np.array([HOG(img).hog_image_rescaled.flatten() for img, _ in tqdm(train_imgs)])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_hue_features = scaler.fit_transform(train_features)

# Fit the PCA model
pca = PCA(n_components=num_components)
pca.fit(scaled_train_hue_features)

x_ticks = list(range(1, num_components + 1))

plt.figure(figsize=(12, 4))
plt.plot(x_ticks, np.cumsum(pca.explained_variance_ratio_), 
         label='type of feature', marker='')

plt.xlabel('Number of components')
plt.ylabel('Explained Variance')
plt.title('HOG Feature Vector: Explained Variance by PCA Components')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.xlim(1, num_components)
plt.ylim(0, 1)

plt.show()
