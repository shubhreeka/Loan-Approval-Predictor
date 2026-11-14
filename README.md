# Loan Approval Predictor: Hybrid Rules + XGBoost

A web app that predicts whether a loan application is likely to be approved. It combines **bank-style policy rules**
(CIBIL cut-off, income-based loan limits) with an **XGBoost** model trained on applicant data, and returns a decision
with a confidence score.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-EB5E28)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)

---

## How a decision is made

```
Applicant form ──► Rule 1: CIBIL score < 650?                 ──► Reject (policy)
                   Rule 2: loan > income × {3, 4, 5}?          ──► Reject (policy)
                           (multiplier by income bracket:
                            < ₹5 L → 3×, < ₹20 L → 4×, else 5×)
                   Otherwise ──► XGBoost model ──► Approve / Reject + confidence %
```

The **hybrid design** mirrors real lending. Hard eligibility rules are explainable and auditable, and the ML model handles the
complex interactions between the remaining factors.

**Features used by the model (14):** annual income, loan amount, CIBIL score, car/house ownership, marital status,
gender, education, employment type, income-proof type, dependents, number of assets, gold holdings and existing loans.

## Results

On a held-out test set (20%, 200 applications):

| Metric | Score |
|---|---|
| ROC-AUC | **0.998** |
| Accuracy | 99.0% |
| F1 (rejected class) | 0.67 |

> The data is **imbalanced**: only 4.5% of applications are rejections, so a model that always says "approve" would already
> reach 95.5% accuracy. That is why ROC-AUC and the F1 on rejections are the meaningful numbers here. Next steps:
> class weighting (`scale_pos_weight`), stratified cross-validation and threshold tuning.

**Most important features:** existing loans, employment type, CIBIL score, loan amount.

## App flow
1. **Applicant details:** name, age and KYC documents (Aadhaar, PAN, ITR, Form 16). In this demo the documents are
   only checked for presence and **never stored**.
2. **Financial profile:** income, loan amount, CIBIL score, assets and so on.
3. **Result:** approved or rejected, with the model's approval probability or the rule that triggered the rejection.

## Run it

```bash
git clone https://github.com/shubhreeka/Loan-Approval-Predictor.git
cd Loan-Approval-Predictor
pip install -r requirements.txt
python train_xgboost.py     # optional: retrain -> xgb_model.pkl, label_encoders.pkl
python app.py               # http://127.0.0.1:5001
```

## Project structure

```
├── app.py                       # Flask app: rules + model inference
├── train_xgboost.py             # Training script (label encoding + XGBoost)
├── loan_approval_dataset.xlsx   # 1,000 applications, 14 features + TARGET
├── xgb_model.pkl, label_encoders.pkl
├── templates/                   # personal.html → index.html → result.html
└── static/style.css
```

## Author

**Shubhreeka Saxena**, B.Tech AI & ML, VIPS-TC (GGSIPU)
[LinkedIn](https://www.linkedin.com/in/shubhreeka-saxena/) · [GitHub](https://github.com/shubhreeka)
