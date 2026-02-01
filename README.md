# Customer Churn Prediction: A Comparative Machine Learning Approach

## Project Overview
This repository contains a comprehensive machine learning system designed to predict customer attrition. Developed as a final-year engineering project, the study moves beyond basic classification by implementing a "Model Tournament" to evaluate and compare five distinct algorithmic architectures. The final system provides a deployment-ready dashboard for real-time churn risk assessment.

## The Model Tournament
To ensure the highest level of predictive reliability, five different algorithms were trained and validated. This comparative approach allowed for the analysis of different mathematical logics, ranging from linear boundaries to ensemble-based decision trees.

| Model | Algorithmic Logic | Performance Characteristics |
| :--- | :--- | :--- |
| **Random Forest** | Ensemble (Bagging) | **The Champion:** Exceptional stability and feature transparency. |
| **SVM** | Optimal Hyperplane | **Mathematical Peak:** Highest precision in high-dimensional space. |
| **KNN** | Instance-based | Effective at identifying local customer behavior clusters. |
| **Logistic Regression** | Linear Probabilistic | Provided the statistical baseline for the study. |
| **XGBoost** | Gradient Boosting | Optimized for minimizing residual errors in complex data. |

## Model Selection Strategy
While multiple models achieved high accuracy, the **Random Forest Classifier** was selected as the primary production model. This selection was based on the following engineering criteria:

* **Explainability:** Unlike "black-box" models, Random Forest allows for the calculation of Feature Importance, which is critical for business stakeholders to understand churn drivers.
* **Ensemble Robustness:** By aggregating 100+ decision trees, the model reduces variance and is significantly less prone to outliers than individual boundary-based models like SVM.
* **Deployment Efficiency:** The model maintains a small serialized footprint while providing fast inference times within the Streamlit interface.

## Data Engineering and Technical Challenges

### 1. Feature Engineering and Preprocessing
The raw dataset required significant transformation to be compatible with mathematical estimators. We implemented a dual-encoding strategy:

* **Label Encoding:** Applied to binary categorical features (e.g., Gender, TechSupport) to convert them into a machine-readable 0/1 format.
* **One-Hot Encoding:** Applied to multi-category features (e.g., Contract Type, Internet Service) to prevent the models from assuming a false numerical ranking.
* **Feature Scaling:** We utilized `StandardScaler` to normalize numerical ranges. This was strictly necessary for **Logistic Regression**, **KNN**, and **SVM**, as these models rely on distance and coordinate geometry; without scaling, high-value features like "Total Charges" would have mathematically overwhelmed "Tenure."

### 2. Addressing Class Imbalance (SMOTE Tuning)
The dataset was naturally imbalanced. To prevent the model from simply "guessing" the majority class:
* **Strategy:** We implemented SMOTE (Synthetic Minority Over-sampling Technique).
* **Optimization:** Initial 1:1 oversampling caused the models to overfit on synthetic noise. We optimized the `sampling_strategy` to **0.15–0.20**, providing a balanced learning signal that improved the F1-score without compromising real-world logic.

### 3. Solving the "100% Accuracy" Problem (Regularization)
When initial test runs yielded 100% accuracy, we identified this as "Overfitting." We applied specific regularization constraints to all five models to force them to learn general patterns:

| Model | Regularization Technique Applied |
| :--- | :--- |
| **Random Forest** | Restricted `max_depth` to 3 and tuned `min_samples_split`. |
| **SVM** | Adjusted the `C` parameter to create a "Soft Margin" for better generalization. |
| **XGBoost** | Utilized `reg_lambda` (L2 penalty) and limited tree depth. |
| **KNN** | Optimized `k-neighbors` to smooth decision boundaries against noise. |
| **Logistic Regression** | Implemented `L2` penalty to prevent coefficient explosion. |



The result is a model suite that maintains ~95% accuracy while remaining robust enough for production deployment.