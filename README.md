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

## System Architecture and Deployment

### 1. Project Directory Structure
To maintain professional engineering standards, the project is organized into modular directories. This ensures a clean separation between raw research, processed assets, and the production interface.

* **`/models`**: Contains serialized `.pkl` files for the trained Random Forest, SVM, and Scaler objects.
* **`/notebooks`**: Documented Jupyter Notebooks (v1, v2, v3) showing the evolution of the EDA and model training phases.
* **`/data`**: The source customer dataset used for training and validation.
* **`/dev_versions`**: Archived early-stage scripts and experimental dashboard iterations.
* **`app.py`**: The production Streamlit application script.
* **`requirements.txt`**: The comprehensive list of dependencies required to reconstruct the environment.

### 2. Local Installation and Usage
To run this project locally, ensure you have Python 3.9+ installed, then follow these steps:

**Step 1: Clone the Repository**

```bash
git clone [https://github.com/itsdakshjain/Customer-Churn-Prediction.git](https://github.com/itsdakshjain/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction
```

**Step 2: Create a Virtual Environment**

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Run the Application**

```bash
streamlit run app.py
```

### 3. Key Features
* **Real-time Risk Assessment:** Instant churn probability scoring based on live user input.
* **Smart UX (Auto-Jump):** Features an automated "scroll-to-result" anchor that glides the user directly to their analysis once the prediction is ready.
* **Dynamic Visualizations:** Dual-layered risk gauges and feature importance bar charts to explain the "Why" behind every prediction.
* **Intelligent UI:** Responsive design that adapts to mobile/desktop, featuring a "thinking" AI spinner and interactive success notifications.
* **Machine Learning Backend:** Modular integration of Random Forest and SVM architectures, allowing for model comparison and high-accuracy cross-validation.

### 4. Deployment
The application is optimized for cloud deployment via **Streamlit Cloud**. The production environment is configured to automatically synchronize with the `main` branch of this repository, ensuring that any updates to the models or UI are reflected in real-time on the live URL.

---

## License
This project is licensed under the **MIT License**—a permissive license that allows for personal and commercial use while providing a disclaimer of warranty. See the [LICENSE](LICENSE) file for the full text.

## Contact & Credits
**Developer:** Daksh Jain  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/itsdakshjain/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/itsdakshjain)
