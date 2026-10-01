# Customer Churn Prediction (Telco) — Machine Learning

University of Lahore · Faculty of Information Technology · Course: Artificial Intelligence (Assignment 1)

## Project overview
A complete machine-learning pipeline that predicts whether a telecom customer will **leave (Churn = Yes)** or **stay (Churn = No)**, using the Telco Customer Churn dataset (7,043 customers, 21 columns).

## Pipeline
1. Data loading and understanding
2. Data preprocessing (blank `TotalCharges`, data types, duplicates, unnecessary columns, inconsistent labels)
3. Exploratory data analysis (churn distribution, churn vs. contract / tenure / charges / services)
4. Feature engineering and encoding (`NumServices`, `AvgMonthlySpend`, `TenureGroup`; binary + one-hot encoding)
5. Train/test split (80/20, stratified) and training of Logistic Regression, Decision Tree and Random Forest
6. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices)
7. Prediction on new customers with a reusable `predict_churn()` function

## Results (test set, 1,407 customers)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.723 | 0.487 | 0.789 | 0.602 | 0.834 |
| Decision Tree | 0.701 | 0.463 | 0.786 | 0.583 | 0.818 |
| **Random Forest** | **0.751** | **0.521** | 0.786 | **0.627** | **0.841** |

**Best model: Random Forest.** Main churn drivers: contract type, tenure, fiber-optic internet, monthly charges and payment method.

## Files
| File | Description |
|---|---|
| `Churn_Prediction_Notebook.ipynb` | Full notebook (code, explanations, outputs) |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset |
| `churn_model.joblib` | Saved final model (Random Forest) with scaler and column list |
| `requirements.txt` | Python dependencies |

## How to run
```bash
git clone <this-repository-url>
cd <repository-folder>
pip install -r requirements.txt
jupyter notebook Churn_Prediction_Notebook.ipynb
```
Then choose **Kernel → Restart & Run All**. The CSV must be in the same folder as the notebook.
(Google Colab: upload the notebook and the CSV, then Runtime → Run all.)

## Use the saved model
```python
import joblib
bundle = joblib.load("churn_model.joblib")
model, scaler, columns = bundle["model"], bundle["scaler"], bundle["columns"]
```
