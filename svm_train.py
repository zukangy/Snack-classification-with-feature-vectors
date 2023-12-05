import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.snack_dataset import SnackDataset
from utils.hue_histogram import hue_histogram


BINS = 2000

# Load the dataset
snacks = SnackDataset()

train_imgs = list(snacks.get_train_set())
val_imgs = list(snacks.get_validation_set())
test_imgs =list(snacks.get_test_set())

train_imgs.extend(val_imgs)

print(f"Train size: {len(train_imgs)} | Test size: {len(test_imgs)}")

train_features = np.array([hue_histogram(img, bins=BINS) for img, _ in train_imgs])
train_labels = np.array([label for _, label in train_imgs])
val_features = np.array([hue_histogram(img, bins=BINS) for img, _ in val_imgs])
val_labels = np.array([label for _, label in val_imgs])
test_features = np.array([hue_histogram(img, bins=BINS) for img, _ in test_imgs])
test_labels = np.array([label for _, label in test_imgs])

# Create a pipeline with MinMaxScaler, PCA, and SVC
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('pca', PCA()),
    ('svm', SVC(random_state=42))
])

# Define the parameter grid to search
param_grid = {
    'pca__n_components': [500, 1000, 1500],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto'],
    'svm__kernel': ['rbf', 'poly', 'sigmoid'],
    'svm__decision_function_shape': ['ovo', 'ovr'],
}

# Create and fit the grid search
# By default, GridSearchCV uses stratified k-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', 
                           verbose=2)
grid_search.fit(train_features, train_labels)

# Best parameters and model
print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Make predictions and calculate accuracy
y_pred = best_model.predict(test_features)
print(f"Test Accuracy: {accuracy_score(test_labels, y_pred)}")

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(test_labels, y_pred)
print("Confusion Matrix:\n", conf_matrix)
