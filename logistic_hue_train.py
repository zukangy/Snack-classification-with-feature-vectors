# Train a logistic regression on the hue feature vectors

import pandas as pd
import numpy as np 
import yaml
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.snack_dataset import SnackDataset
from utils.hue_histogram import hue_histogram


BINS = 1600
# Set to 1 to perform grid search, 0 to use best params; 
# mush have done grid search first before 0
grid_search = 1
best_pipeline = None

# Load the dataset
snacks = SnackDataset()

train_imgs = list(snacks.get_train_set())
val_imgs = list(snacks.get_validation_set())
test_imgs =list(snacks.get_test_set())

print(f"Train size: {len(train_imgs)} | Val size: {len(val_imgs)} | Test size: {len(test_imgs)}")

# Get Hue Histogram features
print('Extracting Hue Histogram features...')
train_hue_features = np.array([hue_histogram(img, bins=BINS) for img, _ in tqdm(train_imgs)])
val_hue_features = np.array([hue_histogram(img, bins=BINS) for img, _ in tqdm(val_imgs)])
test_hue_features = np.array([hue_histogram(img, bins=BINS) for img, _ in tqdm(test_imgs)])

# Get labels
train_labels = np.array([label for _, label in train_imgs])
val_labels = np.array([label for _, label in val_imgs])
test_labels = np.array([label for _, label in test_imgs])

# Create a pipeline with MinMaxScaler, PCA, and logistic regression
pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('pca', PCA(n_components=500)),
        ('logit', LogisticRegression(random_state=1))
    ])

if grid_search:
    combined_features = np.vstack((train_hue_features, val_hue_features))
    combined_labels = np.concatenate((train_labels, val_labels))
    
    # Create the test_fold array with -1 for training instances and 0 for validation instances
    train_indices = np.full(len(train_hue_features), -1, dtype=int)  # -1 for training data
    val_indices = np.zeros(len(val_hue_features), dtype=int)         # 0 for validation data
    test_fold = np.concatenate((train_indices, val_indices))
    
    # Create the PredefinedSplit
    ps = PredefinedSplit(test_fold)
    
    # Define the parameter grid to search
    param_grid = {
        'logit__penalty': ['l1', 'l2'],
        'logit__C': [.5, 1.],
        'logit__solver': ['lbfgs', 'liblinear']
    }

    # Create and fit the grid search
    # By default, GridSearchCV uses stratified k-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring='accuracy', 
                               verbose=3)
    grid_search.fit(combined_features, combined_labels)

    # Best parameters and model
    print("Best parameters:", grid_search.best_params_)
    best_pipeline = grid_search.best_estimator_

    # Write to a YAML file
    with open('./best_hue_logit_params.yml', 'w') as file:
        yaml.dump(grid_search.best_params_, file, default_flow_style=False)
        
if best_pipeline is None:
    # Best params from grid search
    with open('./best_hue_logit_params.yml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)
            
    # Initialize the pipeline with best params
    pca = PCA(n_components=500)
    logit = LogisticRegression(random_state=1, **best_params)
    
    best_pipeline = Pipeline([
                ('scaler', MinMaxScaler(feature_range=(0, 1))),
                ('pca', pca),
                ('logit', logit)
            ])
    
    best_pipeline.fit(train_hue_features, train_labels)
    
    
train_pred = best_pipeline.predict(train_hue_features)
train_accuracy = accuracy_score(train_labels, train_pred)

# Calculate accuracy and AUC for the validation set
val_pred = best_pipeline.predict(val_hue_features)
val_accuracy = accuracy_score(val_labels, val_pred)

# Calculate accuracy and AUC for the test set
test_pred = best_pipeline.predict(test_hue_features)
test_accuracy = accuracy_score(test_labels, test_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
    
# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(test_labels, test_pred)

# Fetch label names from the dataset
snack_names = [snacks.label_mapping(label) for label in range(20)]

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Convert to DataFrame for Seaborn
df_conf_matrix_normalized = pd.DataFrame(conf_matrix_normalized, index=snack_names, columns=snack_names)

# Plot the normalized confusion matrix
plt.figure(figsize=(15, 12))  # Adjust the size as needed
sns.heatmap(df_conf_matrix_normalized, annot=True, fmt=".2%", cmap='Blues', cbar=False)
plt.title('Normalized Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()