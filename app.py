import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # relative paths below work from any terminal folder
from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

@app.route('/')
def personal():
    return render_template('personal.html')

@app.route('/loan', methods=['POST'])
def loan():
    name = request.form['name']
    age = request.form['age']

    documents = {
        'aadhaar': request.files['aadhaar'],
        'pan': request.files['pan'],
        'itr': request.files['itr'],
        'form16': request.files['form16']
    }
    # Privacy by design: KYC documents are only checked for presence in this demo,
    # never written to disk.
    missing = [key for key, file in documents.items() if not (file and file.filename)]
    if missing:
        return f"Missing documents: {', '.join(missing)}", 400

    return render_template('index.html', name=name, age=age)

@app.route('/result', methods=['POST'])
def result():
    try:
        # Extract form values
        annual_income = float(request.form['Annual_Income'])
        loan_amount = float(request.form['Loan_Amount'])
        cibil_score = float(request.form['CIBIL_Score'])

        # 1️⃣ CIBIL Score Rule
        if cibil_score < 650:
            return render_template('result.html',
                approved="Rejected: Your CIBIL score is too low for loan approval.",
                confidence="Rule-based rejection"
            )

        # 2️⃣ Progressive Loan-to-Income Ratio
        if annual_income < 500000:
            multiplier = 3
        elif annual_income < 2000000:
            multiplier = 4
        else:
            multiplier = 5

        max_loan_allowed = annual_income * multiplier

        if loan_amount > max_loan_allowed:
            return render_template('result.html',
                approved=f"Rejected: Requested loan ({loan_amount:,.0f}) exceeds allowed limit ({max_loan_allowed:,.0f}) based on your income bracket.",
                confidence="Rule-based rejection"
            )

        # 3️⃣ Build ML Input
        input_data = {
            'Annual_Income': annual_income,
            'Loan_Amount': loan_amount,
            'CIBIL_Score': cibil_score,
            'Owns_Car': request.form['Owns_Car'],
            'Owns_House': request.form['Owns_House'],
            'Married': request.form['Married'],
            'Gender': request.form['Gender'],
            'Education_Level': request.form['Education_Level'],
            'Employment_Type': request.form['Employment_Type'],
            'Income_Proof': request.form['Income_Proof'],
            'Dependents': int(request.form['Dependents']),
            'Number_of_Assets': int(request.form['Number_of_Assets']),
            'Gold_Worth': float(request.form['Gold_Worth']),
            'Other_Loans': request.form['Other_Loans']
        }

        # Apply Label Encoding where needed
        for col, val in input_data.items():
            if col in label_encoders:
                le = label_encoders[col]
                input_data[col] = le.transform([val])[0]

        # Convert to NumPy Array for Prediction
        input_array = np.array([list(input_data.values())])

        # 4️⃣ ML Prediction
        prediction = model.predict(input_array)[0]
        proba_default = model.predict_proba(input_array)[0][1]
        proba_approve = 1 - proba_default

        approval = (
            "Approved! Congratulations! Based on your inputs, your loan is likely to be approved."
            if prediction == 0
            else "Rejected, Sorry! Based on your inputs, your loan is unlikely to be approved."
        )

        confidence = f"{proba_approve * 100:.2f}%"

        return render_template('result.html', approved=approval, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5001)
