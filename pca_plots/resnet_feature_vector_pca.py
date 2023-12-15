# This script plots the explained variance of the Resnet feature vector with PCA components
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style 
from sklearn.decomposition import PCA

from utils.snack_dataset import SnackDataset
from utils.resnet_embedding import get_embeddings


style.use('ggplot')

n_components = 1500
DEVICE = 'mps' # Mac GPU, set to cuda if using cuda

snacks = SnackDataset()
train_imgs = list(snacks.get_train_set())
train_features = [img for img, _ in train_imgs]

# Get Hue Histogram features
print('Extracting Resnet feature vector...')
train_features = get_embeddings(train_features, gpu=DEVICE, data='train')

pca = PCA(n_components=n_components)
pca.fit(train_features)

x_ticks = list(range(1, n_components + 1))

plt.figure(figsize=(12, 4))
plt.plot(x_ticks, np.cumsum(pca.explained_variance_ratio_), 
         label='type of feature', marker='')

plt.xlabel('Number of components')
plt.ylabel('Explained Variance')
plt.title('Resnet Feature Vector: Explained Variance by PCA Components')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.xlim(1, n_components)
plt.ylim(0, 1)

plt.show()
