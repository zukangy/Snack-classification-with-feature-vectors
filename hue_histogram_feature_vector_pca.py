import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils.snack_dataset import SnackDataset
from utils.hue_histogram import hue_histogram


style.use('ggplot')

n_bins = 1600
num_components = 150

snacks = SnackDataset()

train_imgs = list(snacks.get_train_set())

# Get Hue Histogram features
print('Extracting Hue Histogram features...')
train_hue_features = np.array([hue_histogram(img, bins=n_bins) for img, _ in train_imgs])

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_hue_features = scaler.fit_transform(train_hue_features)

pca = PCA(n_components=num_components)
pca.fit(scaled_train_hue_features)

x_ticks = list(range(1, num_components + 1))

plt.figure(figsize=(12, 4))
plt.plot(x_ticks, np.cumsum(pca.explained_variance_ratio_), 
         label='type of feature', marker='')

# Adding aesthetics to match the provided image
plt.xlabel('Number of components')
plt.ylabel('Explained Variance')
plt.title('Hue Histogram Feature Vector: Explained Variance by PCA Components')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Set the x-axis and y-axis limits if needed
plt.xlim(1, num_components)
plt.ylim(0, 1)

# Show the plot
plt.show()
