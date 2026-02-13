# ML Assignment 2

## Problem statement
Implement and evaluate multiple classification models on a single dataset, compare them using standard metrics, and prepare the project for deployment in a Streamlit app.

## Dataset description
- Dataset: Cardiovascular Disease Dataset (CSV file: Cardiovascular_Disease_Dataset.csv)
- Task: Binary classification (target column: target)
- Preprocessing: drop patientid, handle missing serumcholestrol (0 treated as missing), numeric imputation + scaling, categorical imputation + one-hot encoding.

## Models used
The following models are implemented on the same dataset:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9700 | 0.9983 | 0.9661 | 0.9828 | 0.9744 | 0.9384 |
| Decision Tree | 0.9750 | 0.9735 | 0.9744 | 0.9828 | 0.9785 | 0.9487 |
| KNN | 0.9550 | 0.9764 | 0.9652 | 0.9569 | 0.9610 | 0.9078 |
| Naive Bayes | 0.9600 | 0.9939 | 0.9500 | 0.9828 | 0.9661 | 0.9181 |
| Random Forest (Ensemble) | 0.9850 | 0.9995 | 0.9829 | 0.9914 | 0.9871 | 0.9692 |
| XGBoost (Ensemble) | 0.9850 | 0.9993 | 0.9913 | 0.9828 | 0.9870 | 0.9693 |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC; performs well with balanced classes and scaled features. |
| Decision Tree | Slightly higher accuracy than logistic regression but lower AUC; may overfit compared to ensembles. |
| KNN | Lowest accuracy among models; sensitive to feature scaling and neighborhood size. |
| Naive Bayes | Good recall and AUC; assumes feature independence which can limit precision. |
| Random Forest (Ensemble) | Best overall balance with high accuracy, AUC, and MCC; robust to feature noise. |
| XGBoost (Ensemble) | Matches Random Forest; strong precision and MCC with slightly different tradeoffs. |
