# Train a svm model with resnet_50 embeddings
import pandas as pd
import numpy as np 
import yaml
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable

from utils.snack_dataset import SnackDataset


# Set to 1 to perform grid search, 0 to use best params; 
# mush have done grid search first before 0
grid_search = 0
best_pipeline = None
n_components = 1500

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()  # Set the model to evaluation mode

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

# embeddings function
def get_embedding(img):
    img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # Resnet 50 avg pool layer expects 2048 x 1 x 1
    embedding = torch.zeros(2048)

    def copy_data(m, i, o):
        embedding.copy_(o.data.squeeze())

    h = layer.register_forward_hook(copy_data)
    model(img)
    h.remove()

    return embedding

# Load the dataset
snacks = load_dataset("Matthijs/snacks")
train_snacks = snacks["train"]
val_snacks = snacks["validation"]
test_snacks = snacks["test"]

# # Get ResNet embeddings
train_embeddings = []
for i in tqdm(range(len(train_snacks))):
    embedding = get_embedding(train_snacks[i]["image"])
    # flatten and convert to numpy
    train_embeddings.append(embedding.detach().numpy().flatten())
    
val_embeddings = []
for i in tqdm(range(len(val_snacks))):
    embedding = get_embedding(val_snacks[i]['image'])
    # flatten and convert to numpy
    val_embeddings.append(embedding.detach().numpy().flatten())
    
test_embeddings = []
for i in tqdm(range(len(test_snacks))):
    embedding = get_embedding(test_snacks[i]['image'])
    # flatten and convert to numpy
    test_embeddings.append(embedding.detach().numpy().flatten())
    
train_embeddings = np.array(train_embeddings)
val_embeddings = np.array(val_embeddings)
test_embeddings = np.array(test_embeddings)

# Get labels
train_labels = np.array([img['label'] for img in train_snacks])
val_labels = np.array([img['label'] for img in val_snacks])
test_labels = np.array([img['label'] for img in test_snacks])

# Create a pipeline with MinMaxScaler, PCA, and SVC
pipeline = Pipeline([
    ('pca', PCA(n_components=n_components)),
    ('svm', SVC())
])

if grid_search:
    combined_features = np.vstack((train_embeddings, val_embeddings))
    combined_labels = np.concatenate((train_labels, val_labels))
    
    # Create the test_fold array with -1 for training instances and 0 for validation instances
    train_indices = np.full(len(train_embeddings), -1, dtype=int)  # -1 for training data
    val_indices = np.zeros(len(val_embeddings), dtype=int)         # 0 for validation data
    test_fold = np.concatenate((train_indices, val_indices))
    
    # Create the PredefinedSplit
    ps = PredefinedSplit(test_fold)
    
    # Define the parameter grid to search
    param_grid = {
        'svm__C': [0.1, 1],
        'svm__kernel': ['linear', 'poly'],
        'svm__decision_function_shape': ['ovo', 'ovr'],
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
    with open('./best_params/best_resnet_svm_params.yml', 'w') as file:
        yaml.dump(grid_search.best_params_, file, default_flow_style=False)
        
if best_pipeline is None:
    # Best params from grid search
    with open('./best_params/best_resnet_svm_params.yml') as file:
        best_params = yaml.load(file, Loader=yaml.FullLoader)
            
    # Initialize the pipeline with best params
    pca = PCA(n_components=n_components)
    svm = SVC(**best_params)
    
    best_pipeline = Pipeline([
                ('pca', pca),
                ('svm', svm)
            ])
    
    # Fit the pipeline
    best_pipeline.fit(train_embeddings, train_labels)
    
# Calculate accuracy and AUC for the training set
train_pred = best_pipeline.predict(train_embeddings)
train_accuracy = accuracy_score(train_labels, train_pred)

# Calculate accuracy and AUC for the validation set
val_pred = best_pipeline.predict(val_embeddings)
val_accuracy = accuracy_score(val_labels, val_pred)

# Calculate accuracy and AUC for the test set
test_pred = best_pipeline.predict(test_embeddings)
test_accuracy = accuracy_score(test_labels, test_pred)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
    
# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(test_labels, test_pred)

# Fetch label names from the dataset
snacks = SnackDataset()
snack_names = [snacks.label_mapping(label) for label in range(20)]

# Normalize the confusion matrix by row (i.e by the number of samples in each class)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

df_conf_matrix_normalized = pd.DataFrame(conf_matrix_normalized, index=snack_names, columns=snack_names)

# Plot the normalized confusion matrix
plt.figure(figsize=(15, 12)) 
sns.heatmap(df_conf_matrix_normalized, annot=True, cmap='Blues', cbar=False)
plt.title('Normalized Confusion Matrix on Resnet SVM')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()