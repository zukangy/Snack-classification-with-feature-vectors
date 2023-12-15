# Train a logistic regression on the HOG feature vectors
import pandas as pd
import numpy as np 
from tqdm import tqdm 
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils.snack_dataset import SnackDataset
from utils.create_hog import HOG


# Set to 1 to perform grid search, 0 to use best params; 
# mush have done grid search first before 0
grid_search = 1

# Load the dataset
snacks = SnackDataset()

train_imgs = list(snacks.get_train_set())
val_imgs = list(snacks.get_validation_set())
test_imgs =list(snacks.get_test_set())

# Combine train and validation sets as grid search will use cross-validation
train_imgs.extend(val_imgs)
print(f"Train size: {len(train_imgs)} | Test size: {len(test_imgs)}")

train_features = np.array([HOG(img).hog_image_rescaled.flatten() for img, _ in tqdm(train_imgs)])
test_features = np.array([HOG(img).hog_image_rescaled.flatten() for img, _ in tqdm(test_imgs)])

# Get labels
train_labels = np.array([label for _, label in train_imgs])
test_labels = np.array([label for _, label in test_imgs])

if grid_search:
    # Create a pipeline with MinMaxScaler, PCA, and SVC
    pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('pca', PCA()),
        ('svm', SVC(random_state=42))
    ])

    # Define the parameter grid to search
    param_grid = {
        'pca__n_components': [500, 1000],
        'svm__C': [0.1, 1],
        'svm__kernel': ['rbf', 'poly'],
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

    # Write to a YAML file
    with open('./best_hog_svm_params.yml', 'w') as file:
        yaml.dump(grid_search.best_params_, file, default_flow_style=False)
        
else:
    # Best params from grid search
    with open('./best_hog_svm_params.yml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)
    
    pca_params = {}
    model_params = {}
    for k, v in best_params.items():
        if k.startswith('pca'):
            pca_params[k.split('__')[1]] = v
        else:
            model_params[k.split('__')[1]] = v
            
    # Initialize the pipeline with best params
    pca = PCA(**pca_params)
    svm = SVC(**model_params, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', MinMaxScaler(feature_range=(0, 1))),
        ('pca', pca),
        ('svm', svm)
    ])
    
    # Fit the pipeline
    pipeline.fit(train_features, train_labels)
    
    # Make predictions and calculate accuracy
    y_pred = pipeline.predict(test_features)
    print(f"Test Accuracy: {accuracy_score(test_labels, y_pred)}")
    
    # Calculate and print the confusion matrix
    conf_matrix = confusion_matrix(test_labels, y_pred)
    
    # Fetch label names from the dataset
    snack_names = [snacks.label_mapping(label) for label in range(20)]
    
    df_conf_matrix = pd.DataFrame(conf_matrix, snack_names, snack_names)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(df_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('SVM on HOG Feature Vec Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()