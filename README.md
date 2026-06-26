# Loan Default Risk Predictor

A machine learning project that predicts whether a loan applicant is likely to default, based on their financial profile.

This project extends concepts from the **TATA GenAI Powered Data Analytics Job Simulation** (Forage) — exploratory data analysis, risk profiling, and delinquency prediction — into an independent, end-to-end machine learning workflow.

## What it does

Given an applicant's financial details (income, loan amount, credit score, debt-to-income ratio, employment history, etc.), the model predicts the probability that they will default on the loan.

## Dataset

8,000 simulated loan applicant records, built to reflect realistic relationships between financial attributes and default risk (similar in structure to public credit-risk datasets such as those found on Kaggle). Columns include:

- `age`, `annual_income`, `loan_amount`, `credit_score`
- `employment_years`, `existing_loans`, `debt_to_income_ratio`, `loan_term_months`
- `default` (target: 1 = defaulted, 0 = repaid)

## Approach

1. **Exploratory Data Analysis** — checked data quality, visualized default distribution, and examined how credit score and debt-to-income ratio relate to default risk
2. **Modeling** — trained and compared two models:
   - Logistic Regression (interpretable baseline)
   - Random Forest Classifier (captures non-linear patterns)
3. **Evaluation** — accuracy, ROC-AUC score, confusion matrix, and feature importance analysis

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~71% | 0.717 |
| Random Forest | ~72% | 0.715 |

**Top predictors of default:** Credit score (most important), followed by loan amount, annual income, and debt-to-income ratio.

## Tools used

Python, pandas, scikit-learn, matplotlib, seaborn — run in Jupyter Notebook / Google Colab.

## How to run

1. Open `Loan_Default_Risk_Predictor.ipynb` in Jupyter or Google Colab
2. Make sure `loan_data.csv` is in the same folder (or uploaded to Colab)
3. Run all cells in order

## Possible next steps

- Try gradient boosting models (XGBoost, LightGBM)
- Tune hyperparameters with cross-validation
- Validate against a real-world Kaggle credit risk dataset
