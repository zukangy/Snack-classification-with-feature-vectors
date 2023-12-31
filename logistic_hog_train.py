# Train a logistic regression on the HOG feature vectors
import pandas as pd
import numpy as np 
from tqdm import tqdm 
import yaml
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
from utils.create_hog import HOG


# Set to 1 to perform grid search, 0 to use best params; 
# mush have done grid search first before 0
grid_search = 0
hog_n_pca = 1200
grid_search = 0
best_model = None

# Load the dataset
snacks = SnackDataset()

train_imgs = list(snacks.get_train_set())
val_imgs = list(snacks.get_validation_set())
test_imgs =list(snacks.get_test_set())

print(f"Train size: {len(train_imgs)} | Val size: {len(val_imgs)} | Test size: {len(test_imgs)}")

# Get HOG features
print('Extracting HOG features...')
train_hog_features = np.array([HOG(img).hog_image_rescaled.flatten() for img, _ in tqdm(train_imgs)])
val_hog_features = np.array([HOG(img).hog_image_rescaled.flatten() for img, _ in tqdm(val_imgs)])
test_hog_features = np.array([HOG(img).hog_image_rescaled.flatten() for img, _ in tqdm(test_imgs)])

# Get labels 
train_labels = np.array([label for _, label in train_imgs])
val_labels = np.array([label for _, label in val_imgs])
test_labels = np.array([label for _, label in test_imgs])

# HOG features pipeline
hog_pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('pca', PCA(n_components=hog_n_pca))])

hog_pipeline.fit(train_hog_features)

train_hog_features = hog_pipeline.transform(train_hog_features)
val_hog_features = hog_pipeline.transform(val_hog_features)
test_hog_features = hog_pipeline.transform(test_hog_features)

if grid_search:
    combined_features = np.vstack((train_hog_features, val_hog_features))
    combined_labels = np.concatenate((train_labels, val_labels))
    
    # Create the test_fold array with -1 for training instances and 0 for validation instances
    train_indices = np.full(len(train_hog_features), -1, dtype=int)  # -1 for training data
    val_indices = np.zeros(len(val_hog_features), dtype=int)         # 0 for validation data
    test_fold = np.concatenate((train_indices, val_indices))
    
    # Create the PredefinedSplit
    ps = PredefinedSplit(test_fold)
    # Create a pipeline with MinMaxScaler, PCA, and SVC
    pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('pca', PCA(n_components=1200)),
        ('logic', LogisticRegression(random_state=42))
    ])

    # Define the parameter grid to search
    param_grid = {
        'logic__penalty': ['l1', 'l2'],
        'logic__C': [.5, 1.],
        'logic__solver': ['lbfgs', 'liblinear', 'sag']
    }

    # Create and fit the grid search
    # By default, GridSearchCV uses stratified k-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring='accuracy', 
                               verbose=3)
    grid_search.fit(combined_features, combined_labels)

    # Best parameters and model
    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Write to a YAML file
    with open('./best_params/best_hog_logic_params.yml', 'w') as file:
        yaml.dump(grid_search.best_params_, file, default_flow_style=False)
        
if best_model is None: 
    # Best params from grid search
    with open('./best_params/best_hue_hog_logit_params.yml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)
            
    # Initialize the model with best params
    best_model = LogisticRegression(**best_params, random_state=1)
    best_model.fit(train_hog_features, train_labels)
    
# Calculate accuracy and AUC for the training set
train_pred = best_model.predict(train_hog_features)
train_accuracy = accuracy_score(train_labels, train_pred)

# Calculate accuracy and AUC for the validation set
val_pred = best_model.predict(val_hog_features)
val_accuracy = accuracy_score(val_labels, val_pred)

# Calculate accuracy and AUC for the test set
test_pred = best_model.predict(test_hog_features)
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

df_conf_matrix_normalized = pd.DataFrame(conf_matrix_normalized, index=snack_names, columns=snack_names)

# Plot the normalized confusion matrix
plt.figure(figsize=(15, 12)) 
sns.heatmap(df_conf_matrix_normalized, annot=True, cmap='Blues', cbar=False)
plt.title('Normalized Confusion Matrix on HOG logistic regression')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()