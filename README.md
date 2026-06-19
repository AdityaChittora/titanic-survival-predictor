# Titanic: Machine Learning from Disaster

An end-to-end machine learning classification pipeline built to predict passenger survival rates on the Titanic. This project leverages modular engineering practices, class imbalance handling, and Scikit-Learn pipelines to ensure clean, reproducible data transformations and model inference.

## key Technical Highlights
* **Modular Preprocessing Pipelines:** Built a centralized `ColumnTransformer` using `make_column_transformer` to process numeric, binned continuous values, and binary values simultaneously, preventing data leakage during training splits.
* **Imbalanced Class Rectification:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** via `imblearn` pipelines to balance the training set, optimizing classification performance across minor classes.
* **Algorithm Evaluation:** Evaluated multiple estimators including **K-Nearest Neighbors (KNN Classifier)** and **Logistic Regression** under strict test constraints.
