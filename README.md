# Snack-classification-with-feature-vectors

This repository is dedicated to the final project for UC Berkeley MIDS W281 Computer Vision. 

This README.md file is for the ease of navigation to our demonstration scripts. 

## Overview
In this project, we aimed to experiment with several feature vectors for the snack image data and compare their efficiency against the ResNet embedding model. The feature vectors we had experimented with were HSV histogram and HOG, as well as a combined feature vector of these two.  

## Module Structure
```
project_root/
|-- best_params/                 # Stores the optimal hyper-parameters for each model
|-- pca_plots/                   # PCA plot for each feature vector
|-- utils/                       # utility functions for feature vector generation as well as dataset initialization
|-- weekly_reports/              # Materials for each weekly progress report in Jupyter notebook.
|-- final_vf.ipynb               # Experients with image augmentation
|-- logistic_hog_hue_train.py    # Logistic regression model on HOG and hue histogram feature vectors combined
|-- logistic_hog_train.py        # Logistic regression model on the HOG feature vector
|-- logistic_hue_train.py        # Logistic regression model on the hue histogram feature vector
|-- logit_resnet_train.py        # Logistic regression model on the ResNet-50 feature vector
|-- svm_hog_hue_train.py         # SVM model on HOG and hue histogram feature vectors combined
|-- svm_hog_train.py             # SVM model on HOG feature vector
|-- svm_hue_train.py             # SVM model on hue histogram feature vector
|-- svm_resnet_train.py          # SVM model on the ResNet-50 feature vector
|-- tsne_plot.ipynb              # t-SNE plots for all three feature vectors
```
