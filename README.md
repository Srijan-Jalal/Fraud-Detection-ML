# Credit Card Fraud Detection — Classical ML Baseline

Anomaly detection on the Kaggle Credit Card Fraud dataset
(284,807 transactions | 0.172% fraud prevalence)

## 📁 Project Structure
fraud-detection-ml/
│
├── data/                        ← Dataset goes here (see setup below)
│   └── creditcard.csv           ← NOT committed to GitHub
│
├── notebooks/
│   └── fraud_detection.ipynb    ← Main notebook (all steps)
│
├── outputs/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── results_summary.csv
│
├── .gitignore
├── requirements.txt
└── README.md

## Problem
Financial fraud detection is a classic imbalanced classification problem.
Standard ML models fail because fraud represents <0.2% of transactions.

## Approach
Compared three anomaly detection strategies:
- **Isolation Forest** — unsupervised, tree-based anomaly scoring
- **One-Class SVM** — trained on normal transactions only
- **XGBoost + SMOTE** — supervised with synthetic minority oversampling

## Key Results
| Model             | AUC-PR | AUC-ROC | F1 (Fraud) |
|-------------------|--------|---------|------------|
| Isolation Forest  | 0.1916 | 0.9536  | 0.3077     |
| XGBoost + SMOTE   | 0.8391 | 0.9783  | 0.5266     |
| One-Class SVM     | 0.3817 | 0.9405  | 0.2321     |

## Key Findings

- **XGBoost + SMOTE dominates on AUC-PR (0.8391)** — outperforming 
  Isolation Forest (0.1916) by 338% and One-Class SVM (0.3817) by 120%, 
  confirming that label-guided learning with oversampling is far superior 
  to unsupervised methods on severely imbalanced data.

- **AUC-ROC is misleading here** — all three models score above 0.94 on 
  AUC-ROC, yet F1 scores reveal a vastly different story (0.53 vs 0.23). 
  This confirms that AUC-PR and F1 are the correct evaluation metrics for 
  imbalanced fraud detection — not AUC-ROC.

- **Unsupervised methods fall short** — despite requiring no labels, 
  Isolation Forest (F1: 0.31) and One-Class SVM (F1: 0.23) struggle with 
  precision, producing too many false positives for practical deployment.

- **Motivation for hybrid approach** — XGBoost requires labelled fraud 
  examples which are scarce in real-world systems. This gap motivates 
  Project 2 and 3: a VAE-based semi-supervised detector that achieves 
  strong performance without relying on labelled fraud data.

## Setup
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection.ipynb

## Dataset
Kaggle Credit Card Fraud Detection:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Part of
MEXT 2026 research portfolio — building toward a hybrid
statistical + deep learning anomaly detection framework.
