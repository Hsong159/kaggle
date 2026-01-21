## Overview

This project explores the classic Titanic dataset to predict passenger survival
using interpretable machine learning models. The goal is not only to achieve
strong predictive performance, but also to understand which factors most
influenced survival outcomes.

## Dataset

- Source: Kaggle Titanic - Machine Learning from Disaster
- Target variable: `Survived`
- Number of samples: 891
- Task: Binary classification

## Feature Engineering

I performed domain-driven feature selection and preprocessing:

- Dropped identifiers and high-noise columns (`PassengerId`, `Ticket`, `Cabin`)
- Encoded categorical variables using one-hot encoding:
  - `Sex` → `Sex_male`
  - `Embarked` → `Embarked_Q`, `Embarked_S`
- Handled missing values using median imputation for `Age`
- Focused on interpretable, low-dimensional features

## Feature Selection

I applied L1-regularized logistic regression (Lasso) to perform embedded
feature selection. This allowed the model to automatically eliminate weak
predictors by shrinking their coefficients to zero.

As a result:
- Retained: `Sex_male`, `Pclass`, `Age`, `Fare`, `Embarked_S`
- Removed: `Embarked_Q`

This confirmed that passenger sex and socioeconomic status were the strongest
predictors of survival.

## Model

- Model: Logistic Regression
- Regularization: L1 (Lasso)
- Pipeline:
  - StandardScaler
  - LogisticRegression (liblinear solver)

A scikit-learn Pipeline was used to prevent data leakage and ensure consistent
preprocessing across training and validation data.

## Evaluation

- Train/validation split: 80/20 (stratified)
- Metric: Accuracy, ROC-AUC

The final model achieved approximately 80% validation accuracy, which is
consistent with strong baseline performance on this dataset.