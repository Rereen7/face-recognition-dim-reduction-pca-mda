# face-recognition-dim-reduction-pca-mda
Face classification in MATLAB using PCA/MDA for dimensionality reduction and classical classifiers including SVM, k-NN, and Bayes.
# Face Classification with PCA and MDA (MATLAB)

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
