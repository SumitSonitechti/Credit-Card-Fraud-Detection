# Credit Card Fraud Detection

This project develops a credit card fraud detection system leveraging machine learning models. It utilizes XGBoost as the primary model and a Support Vector Machine (SVM) as a baseline for comparison. A core focus is on effectively addressing the inherent class imbalance in fraud datasets through techniques like SMOTE (Synthetic Minority Over-sampling Technique).

## Project Overview

The implementation follows a structured analytical pipeline:

1.  **Data Loading**: The `creditcard.csv` dataset, containing anonymized transaction data, is loaded and initially inspected. *(Kaggle Dataset Link :- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)*
2.  **Data Preprocessing & Feature Engineering**: Raw data undergoes cleaning, including the removal of duplicate entries and handling of potential missing values. A new `Hour` feature is engineered from the `Time` column to capture temporal patterns, and the original `Time` column is subsequently dropped.
3.  **Model Setup & Data Preparation**: The dataset is split into training and testing sets using stratified sampling to preserve the original fraud rate. Features are standardized using `StandardScaler`. To mitigate the impact of class imbalance, SMOTE is applied exclusively to the training data, oversampling the minority (fraudulent) class.
4.  **Model Training**: An XGBoost classifier is trained on the full SMOTE-balanced training set. For practical comparison, an SVM (kernel='rbf') is trained on a smaller, stratified, and SMOTE-balanced subsample of the training data, acknowledging its higher computational cost on large datasets.
5.  **ROC-AUC Evaluation**: Model performance is quantitatively assessed using the Area Under the Receiver Operating Characteristic Curve (ROC-AUC), a robust metric for imbalanced classification tasks.
6.  **False Negative Analysis & Threshold Optimization**: A critical examination of model predictions, with a particular emphasis on False Negatives (missed fraud). This step involves analyzing confusion matrices, discussing the business implications of various error types, and exploring different prediction probability thresholds to optimize for fraud recall. The XGBoost model's feature importances are also analyzed, and ROC curves for both models are visualized.

## Key Findings

-   **High ROC-AUC Scores**: Both XGBoost and SVM models demonstrated excellent discriminatory power, achieving ROC-AUC scores exceeding 0.97, indicating strong capabilities in distinguishing fraudulent from legitimate transactions.
-   **Trade-off in Threshold Selection**: Analysis of varying prediction thresholds highlighted the crucial trade-off between minimizing False Negatives (capturing more fraud) and managing False Positives (reducing false alarms). Lowering the threshold significantly improved fraud detection recall but increased the number of legitimate transactions incorrectly flagged.
-   **Influential Features**: The feature importance analysis for XGBoost identified several key anonymized features (e.g., `V14`, `V4`, `V8`, `V12`) and engineered features (`Amount`, `Hour`) as highly predictive of fraudulent activity.

## Visualizations

Two critical plots are generated to aid in understanding model behavior:

-   `fraud_feature_importance_simple.png`: A horizontal bar chart illustrating the top 15 most important features as determined by the XGBoost model.
-   `fraud_roc_curve_simple.png`: A ROC curve plot comparing the performance of the XGBoost and SVM models.

## How to Run

1.  **Dependencies**: Ensure Python 3 and the necessary libraries are installed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, and `matplotlib`. These can be installed via pip:
    ```bash
    pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib
    ```
2.  **Data Acquisition**: Obtain the `creditcard.csv` dataset and place it in the same directory as this notebook or script.
   Kaggle Dataset Link :- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
4.  **Execution**: Execute the cells of the notebook sequentially to run the entire fraud detection pipeline.
