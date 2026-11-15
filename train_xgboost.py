import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # relative paths below work from any terminal folder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle

df = pd.read_excel("loan_approval_dataset.xlsx")

if 'Number_of_Assets' not in df.columns:
    df['Number_of_Assets'] = 1  
if 'Gold_Worth' not in df.columns:
    df['Gold_Worth'] = 200000   
if 'Other_Loans' not in df.columns:
    df['Other_Loans'] = 'False' 

features = [
    'Annual_Income',
    'Loan_Amount',
    'CIBIL_Score',
    'Owns_Car',
    'Owns_House',
    'Married',
    'Gender',
    'Education_Level',
    'Employment_Type',
    'Income_Proof',
    'Dependents',
    'Number_of_Assets',
    'Gold_Worth',
    'Other_Loans'
]

X = df[features]
y = df['TARGET'] 

label_encoders = {}
for col in X.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model Trained.")
