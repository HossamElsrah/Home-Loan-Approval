# Loan Status Prediction Project

This project aims to build a machine learning model that accurately predicts loan approval status based on applicant data. By experimenting with various classification models, the project highlights each model's performance in terms of accuracy, precision, and F1 score, providing insights into their suitability for predicting loan statuses.

## Table of Contents
- [Dataset](#dataset)
- [Objective](#objective)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation and Results](#evaluation-and-results)
- [Conclusion](#conclusion)

## Dataset
The dataset consists of applicant details like income, loan amount, credit history, and co-applicant information. Key features include:
- **Loan Amount**: Total loan amount requested.
- **Applicant Income** and **Coapplicant Income**: Applicant and co-applicant income details.
- **Credit History**: A binary indicator of previous credit history.
- **Education**: Applicant's education level.
- **Marital Status**: Whether the applicant is married or single.
- **Property Area**: The location type where the applicant resides.

Target Variable:
- **Loan_Status**: Indicates if the loan was approved (1) or not approved (0).

## Objective
The goal is to develop a model that can accurately predict loan status to assist financial institutions in making data-driven decisions regarding loan approvals.

## Data Preprocessing
The dataset required several preprocessing steps:
- **Missing Value Handling**: Missing values in key columns were imputed.
- **Outlier Removal**: To ensure model stability, outliers in income and loan amount were filtered based on defined thresholds.
- **Log Transformation**: Log transformations were applied to the `LoanAmount` column to address skewness.
- **Encoding Categorical Variables**: Columns such as education, marital status, and property area were encoded into numerical format to feed into the machine learning models.

## Exploratory Data Analysis (EDA)
EDA was performed to understand data distributions, correlations, and interactions between features. Visualizations like count plots, heatmaps, and histograms were generated to observe trends and patterns in loan approvals based on key features.

## Feature Engineering
Some engineered features include:
- **Income Bands**: Grouping income ranges for applicants and co-applicants to categorize loan eligibility.
- **Combined Income**: Summing applicant and co-applicant income as a combined indicator of financial capability.
  
These features were added to potentially enhance model accuracy.

## Modeling
The following models were trained, tuned, and evaluated for performance:
1. **Logistic Regression** - Baseline model for binary classification.
2. **Support Vector Classifier (SVC)** - A model suitable for capturing complex patterns, especially with non-linear kernels.
3. **K-Nearest Neighbors (KNN)** - A model that classifies based on the similarity of nearby data points.
4. **Random Forest Classifier (RFC)** - An ensemble learning method that builds multiple decision trees for better predictive power.

Each model underwent hyperparameter tuning using GridSearchCV, optimizing key parameters such as regularization in Logistic Regression, kernel type in SVC, neighbor count in KNN, and depth and estimator count in RFC.

## Evaluation and Results
Each model was evaluated based on its accuracy, F1 score, and precision. Below are the results:

- **Logistic Regression**:
  - **Accuracy**: 0.77
  - **F1 Score**: 0.77
  - **Precision**: 0.78
  - **Interpretation**: Logistic Regression provided a balanced baseline, performing reasonably well in accuracy and precision, making it useful when simplicity and interpretability are preferred.

- **Support Vector Classifier (SVC)**:
  - **Accuracy**: 0.85
  - **F1 Score**: 0.85
  - **Precision**: 0.80
  - **Interpretation**: SVC achieved the highest accuracy and F1 score, demonstrating its capacity to handle complex data patterns effectively. Its higher precision than Logistic Regression shows it effectively reduces false positives, making it ideal for minimizing the risk of loan default.

- **K-Nearest Neighbors (KNN)**:
  - **Accuracy**: 0.80
  - **F1 Score**: 0.80
  - **Precision**: 0.78
  - **Interpretation**: KNN performed moderately well with an accuracy of 0.80, achieving similar precision to Logistic Regression. KNN’s performance is sensitive to the number of neighbors and can work well with larger datasets where proximity-based prediction is effective.

- **Random Forest Classifier (RFC)**:
  - **Accuracy**: 0.77
  - **F1 Score**: 0.77
  - **Precision**: 0.82
  - **Interpretation**: While RFC shared the same accuracy as Logistic Regression, it achieved the highest precision at 0.82, indicating fewer false positives. RFC is beneficial when precision is prioritized, as it effectively captures feature importance across multiple decision trees.

## Conclusion
The SVC model emerged as the best-performing model for this project, balancing accuracy, precision, and F1 score at high levels. For scenarios where fewer false positives are crucial, the RFC and Logistic Regression models are suitable alternatives.
