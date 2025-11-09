# Car Evaluation Classification: Decision Tree vs. XGBoost

This repository contains a project for a Data Scientist, showcasing an end-to-end machine learning workflow to classify car acceptability using the UCI Car Evaluation dataset. The project compares two models—Decision Tree and XGBoost—focusing on exploratory data analysis (EDA), preprocessing, model training, evaluation, and deriving actionable business insights.

## Case Study Overview

### Business Problem
In the automotive industry, evaluating a car's overall acceptability is crucial for manufacturers, dealers, and consumers. Factors like purchase price, maintenance cost, safety features, and passenger capacity influence whether a car is deemed "unacceptable," "acceptable," "good," or "very good." However, manual evaluation can be subjective and inefficient, especially with large inventories or market analyses.

**Objective:** Build and compare machine learning classifiers to predict a car's acceptability class based on its attributes. This model can assist automotive companies in prioritizing designs, optimizing pricing strategies, and recommending vehicles to customers. Specifically:
- Identify key features driving acceptability.
- Provide interpretable insights for business decisions, such as emphasizing safety in marketing.

This case study is solved in the Jupyter notebook [`Car_Evaluation_Classification_using_Decision_Tree_&_XGBoost.ipynb`](Car_Evaluation_Classification_using_Decision_Tree_&_XGBoost.ipynb), which demonstrates practical data science skills including data handling, modeling, and visualization.

## Dataset
- **Source:** UCI Machine Learning Repository - [Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation).
- **Description:** Derived from a hierarchical decision model, the dataset evaluates cars based on six categorical attributes:
  - `buying`: Buying price (vhigh, high, med, low).
  - `maint`: Maintenance cost (vhigh, high, med, low).
  - `doors`: Number of doors (2, 3, 4, 5more).
  - `persons`: Passenger capacity (2, 4, more).
  - `lug_boot`: Luggage boot size (small, med, big).
  - `safety`: Safety rating (low, med, high).
- **Target Variable:** `class` (unacc, acc, good, vgood) – Multi-class classification.
- **Size:** 1,728 samples, all categorical features (no missing values).

The notebook includes code to download and load the dataset directly.

## Methodology
The workflow follows a standard CRISP-DM process:
1. **Data Loading and EDA:** Inspect data distribution, check for imbalances, and visualize categorical features using bar plots.
2. **Preprocessing:** 
   - Label encoding for categorical variables.
   - Train-test split (80/20) with stratification to handle imbalance.
3. **Model Training and Hyperparameter Tuning:**
   - **Decision Tree Classifier:** 
   - **XGBoost Classifier:**
4. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and classification report.
5. **Visualization:** Feature importance plots, decision tree visualization, and confusion matrices.

Key libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn.

## Results and Insights
- **Model Performance:**
  - XGBoost: 99% accuracy on test set (outperforms Decision Tree due to ensemble boosting).
  - Decision Tree: 81% accuracy.

- **Feature Importance:**
  | Feature   | Importance (XGBoost) |
  |-----------|----------------------|
  | safety   | 0.358                |
  | persons  | 0.255                |
  | maint    | 0.142                |
  | buying   | 0.124                |
  | lug_boot | 0.087                |
  | doors    | 0.032                |

- **Business Insights:**
  - Safety and passenger capacity are the strongest predictors—recommend prioritizing these in car design and marketing.
  - High maintenance or buying costs significantly reduce acceptability; suggest cost optimization strategies.

Detailed results, including plots and metrics, are in the notebook.

