# Multi-Class Classification on Imbalanced Data

**Kingston University · CI7521 Machine Learning & Deep Learning · Coursework 1**

Benchmarking eight classical ML classifiers on a 5-class imbalanced dataset (OpenML #4538, 9,873 samples × 32 features), with systematic hyperparameter tuning, scaler comparison, and resampling strategies to handle class imbalance.

## Problem

Multi-class classification over 5 classes (D / H / P / R / S) with heavy class imbalance. The goal: find the classifier + preprocessing pipeline that maximises **balanced accuracy** on the held-out test set, not just raw accuracy (which would be misleading on imbalanced data).

## Approach

**Classifiers compared**

| # | Model | Library |
|---|-------|---------|
| 1 | Linear Discriminant Analysis (LDA) | scikit-learn |
| 2 | Quadratic Discriminant Analysis (QDA) | scikit-learn |
| 3 | Decision Tree | scikit-learn |
| 4 | K-Nearest Neighbours (KNN) | scikit-learn |
| 5 | Logistic Regression | scikit-learn |
| 6 | Support Vector Machine (SVM) | scikit-learn |
| 7 | Random Forest | scikit-learn |
| 8 | Gaussian Naive Bayes | scikit-learn |

**Pipeline for each model**

1. **Preprocessing comparison** — Quantile Transformer, Power Transformer, Standard Scaler, normalisation, and no-scaling baseline
2. **Resampling comparison** — original (imbalanced) vs SMOTE oversampling vs random undersampling (via `imbalanced-learn`)
3. **Feature selection** — LinearSVC-based selection vs no feature selection
4. **Hyperparameter tuning** — two-stage: `RandomizedSearchCV` to find a promising region, then `GridSearchCV` to refine
5. **Evaluation** — balanced accuracy, macro/micro one-vs-rest ROC-AUC, per-class precision/recall/F1, confusion matrix

## Key findings

- **Quantile Transformer** gave the most consistent gains across models; Power Transformer showed overfitting signs on LDA
- **SMOTE oversampling** outperformed undersampling on every model — the minority classes had enough variance that synthetic samples helped; undersampling threw away too much signal
- **LinearSVC-based feature selection** improved generalisation on LDA but had no effect on Decision Tree — a useful reminder that feature selection interacts with the model's own implicit selection
- Increasing `min_samples_leaf` and `min_samples_split` for the Decision Tree improved balanced accuracy by reducing overfitting on the majority class

## Tech

```
Python · scikit-learn · imbalanced-learn · pandas · NumPy · Matplotlib · Seaborn
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook CI7521_CW1_2026_Group_19.ipynb
```

Dataset is fetched automatically via `sklearn.datasets.fetch_openml(data_id=4538)` — no manual download needed.

## Team

Group 19 · 4-person team · Individual contribution: hyperparameter tuning, scaler/resampling comparisons, model evaluation

## Context

Coursework for CI7521 Machine Learning & Deep Learning at Kingston University, taught by Dr Jad Abbass. Part of the MSc Artificial Intelligence programme.
