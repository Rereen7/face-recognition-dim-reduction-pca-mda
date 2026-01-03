# Face Classification with PCA and MDA (MATLAB)
Face classification in MATLAB using PCA/MDA for dimensionality reduction and classical classifiers including SVM, k-NN, and Bayes.


This repository implements face image classification using classical machine learning
methods combined with dimensionality reduction. The project is written entirely in
**MATLAB** and supports multiple datasets, classifiers, and experimental settings.

All experiments are configured and executed through a single script: `main.m`.

---

## Datasets

The following datasets are supported (provided as `.mat` files):

- **data.mat**
  - Face images with multiple expressions per subject
  - Tasks:
    - Task 1: Subject classification
    - Task 2: Expression classification (neutral vs. expression)

- **pose.mat**
  - Face images under different poses
  - Task: Subject classification only

- **illumination.mat**
  - Face images under different illumination conditions
  - Task: Subject classification only

---

## How to Run

1. Open **MATLAB**.
2. Place the following files in the same directory:
   - `main.m`
   - Dataset files (`data.mat`, `pose.mat`, `illumination.mat`)
   - All required helper functions (PCA, MDA, SVM, AdaBoost utilities)
3. Open `main.m`.
4. Set the configuration parameters in **Step 0**.
5. Run:
   ```matlab
   main

   
## Configuration

All experiments are configured in **Step 0** of `main.m`.  
Copy and modify the parameters below to reproduce or customize experiments.

```matlab
%% Dataset and Task
dataset_name = 'data';    % Options: 'data', 'pose', 'illumination'
task = 1;                 % 1 = subject classification
                           % 2 = expression classification (data.mat only)

%% Classifier Selection
classifier = 'svm';       % Options:
                           % 'bayes'        - Gaussian Bayes
                           % 'knn'          - k-Nearest Neighbors
                           % 'svm'          - Kernel SVM
                           % 'boosted_svm'  - AdaBoosted linear SVM

%% Dimensionality Reduction
proj_mode = 'pca';        % Options: 'pca' or 'mda'

% Projection dimensions for each dataset:
% [data, pose, illumination]
m_pca_vals = [20, 10, 20];
m_mda_vals = [1, 10, 20];

%% k-NN Parameters
k = 1;                    % Number of nearest neighbors

%% SVM Parameters
kernel_type = 'rbf';      % Options: 'rbf' or 'poly'

%% Boosted SVM Parameters
T = 10;                   % Number of AdaBoost rounds

%% Train/Test Split
train_ratio_task_1 = 0.80;   % Subject classification
train_ratio_task_2 = 0.70;   % Expression classification
