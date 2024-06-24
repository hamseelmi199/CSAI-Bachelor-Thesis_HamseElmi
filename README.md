
# CSAI Bachelor Thesis

**Author:** Hamse Elmi (2023232)  
**Supervisor:** Silvy Collin  
**Second Reader:** Lisa Lepp

## Overview

This repository contains the code for the Bachelor thesis titled "Ethical Deployment of Machine Learning Technologies for Predicting Mental States from EEG Data." The study aims to leverage machine learning algorithms to predict human mental states using electroencephalography (EEG) data while ensuring ethical AI practices.

## Objectives

- **Predict Mental States:** Utilize machine learning models to analyze EEG data and predict mental states.
- **Ensure Responsible AI:** Incorporate transparency, fairness, safety, and privacy in the deployment of machine learning models.
- **Model Evaluation:** Compare the performance of various machine learning models including Random Forest, Support Vector Machines (SVM), Gradient Boosting Machines, and Neural Networks.
- **Interpretability:** Use the SHERPA approach for electrode selection through SHAP to ensure model interpretability and transparency.

## Dataset

The EEG dataset used in this study is the test-retest resting and cognitive state EEG dataset collected by Wang, Duan, Dong, Ding, and Lei (2022).

## Methodology

1. **Data Preprocessing:** Clean and preprocess the EEG data for analysis.
2. **Model Training:** Train multiple machine learning models (Random Forest, SVM, Gradient Boosting Machines, Neural Networks) on the preprocessed data.
3. **Model Evaluation:** Evaluate the models based on their accuracy in predicting mental states.
4. **Interpretability:** Implement the SHERPA approach with SHAP for electrode selection to enhance model interpretability.


## Folder Structure
#Main Code
- 'code_final.ipynb' The main script in iPynb format that includes data preprocessing, model training, evaluation, and interpretability analysis.
- `code_final.py`: The main script that includes data preprocessing, model training, evaluation, and interpretability analysis.
#Model Results
- Includes model results for different splitting of positive and negative and balancing methods.
#Script for files
- 'filtered_file.csv' Includes the dataset labels for splitting data into positive and negative classes.
- 'Script for files.ipynb' The main iPynb script for labeling data from dataset and sorting them into the correct folders.
## Requirements

- Python 3.x
- Libraries:
  ```
  glob  # For file path expansion
  os  # For interacting with the operating system
  numpy  # For numerical computing
  pandas  # For data manipulation and analysis
  matplotlib  # For plotting
  mne  # For processing EEG data
  scikit-learn  # For various machine learning models and utilities
  scipy  # For statistical functions
  seaborn  # For visualization
  shap  # For explainability
  ```


## Acknowledgments

This research was conducted under the guidance of Silvy Collin and reviewed by Lisa Lepp. The dataset used in this study was provided by Wang, Duan, Dong, Ding, and Lei (2022a).

## Contact

For any questions or further information, please contact Hamse Elmi at h.elmi@tilburguniversity.edu/hamse-elmi@hotmail.com


