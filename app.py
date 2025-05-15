from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Load model artifacts
with open('churn_model.pkl', 'rb') as f:
    dt = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

categorical_cols = ['Gender']
numerical_cols = [
    'Age', 'Support Calls', 'Payment Delay',
    'Total Spend', 'Last Interaction'
]

def get_retention_recommendation(customer_data, prediction):
    if prediction == 0:
        return {
            "action": "No immediate action needed",
            "message": "Customer is not predicted to churn. Maintain current engagement.",
            "recommendations": []
        }

    recommendations = []

    if customer_data['Support Calls'] >= 8:
        if customer_data['Support Calls'] >= 15:
            recommendations.append({
                "feature": "VIP Support",
                "description": "Immediate escalation to senior support team with 24hr resolution SLA",
                "rationale": f"Extremely high support calls ({customer_data['Support Calls']}) indicate serious unresolved issues"
            })
        else:
            recommendations.append({
                "feature": "Dedicated Support",
                "description": "Assign a dedicated account manager for immediate issue resolution",
                "rationale": f"Multiple support calls ({customer_data['Support Calls']}) suggest recurring problems"
            })

    if customer_data['Payment Delay'] > 10:
        if customer_data['Payment Delay'] > 30:
            recommendations.append({
                "feature": "Payment Relief",
                "description": "Offer payment plan with first month free and reduced installments",
                "rationale": f"Severe payment delay ({customer_data['Payment Delay']} days) indicates financial distress"
            })
        else:
            recommendations.append({
                "feature": "Payment Flexibility",
                "description": "Waive late fees and extend due date by 2 weeks",
                "rationale": f"Payment delay ({customer_data['Payment Delay']} days) may indicate temporary cash flow issues"
            })

    if customer_data['Total Spend'] < 1000:
        recommendations.append({
            "feature": "Value Boost",
            "description": "Free upgrade to premium features for 60 days",
            "rationale": f"Mid-range spending (${customer_data['Total Spend']}) suggests opportunity to demonstrate value"
        })
    else:
        recommendations.append({
            "feature": "Elite Retention",
            "description": "Personalized account review with executive team and custom benefits package",
            "rationale": f"High-value customer (${customer_data['Total Spend']}) worth exceptional retention efforts"
        })

    if customer_data['Age'] <= 44:
        recommendations.append({
            "feature": "Next-Gen Engagement",
            "description": "Access to beta features and innovation community",
            "rationale": f"Younger customer (age {customer_data['Age']}) may value cutting-edge features"
        })

    if customer_data['Last Interaction'] > 20:
        recommendations.append({
            "feature": "Reactivation Campaign",
            "description": "Personalized 'We want you back' offer with time-sensitive benefits",
            "rationale": f"{customer_data['Last Interaction']} days since last interaction indicates disengagement"
        })

    prioritized_recommendations = sorted(
        recommendations,
        key=lambda x: 1 if "VIP" in x["feature"] else
                      2 if "Payment" in x["feature"] else 3
    )

    return {
        "action": "Immediate retention action required",
        "message": f"Customer matches {len(recommendations)} key churn indicators",
        "recommendations": prioritized_recommendations
    }


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/predict.html')
def serve_predict():
    return render_template('predict.html')


@app.route('/feedback.html')
def serve_feedback():
    return render_template('feedback.html')


@app.route('/upload.html')
def serve_upload():
    return render_template('upload.html')


# --- NEW: Handle single customer prediction (from form input) ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Extract data and typecast numeric fields properly
        customer_data = {}
        for col in categorical_cols:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400
            customer_data[col] = data[col]

        for col in numerical_cols:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400
            try:
                customer_data[col] = float(data[col])
            except:
                return jsonify({'error': f'Invalid value for {col}'}), 400

        # Encode categorical
        encoded_data = customer_data.copy()
        for col in categorical_cols:
            encoded_data[col] = label_encoders[col].transform([customer_data[col]])[0]

        # Scale numeric
        num_values = [customer_data[col] for col in numerical_cols]
        scaled_nums = scaler.transform([num_values])[0]

        for i, col in enumerate(numerical_cols):
            encoded_data[col] = scaled_nums[i]

        # Prepare DataFrame with feature order
        input_df = pd.DataFrame([encoded_data], columns=dt.feature_names_in_)

        # Predict
        pred = dt.predict(input_df)[0]
        proba = dt.predict_proba(input_df)[0][1]

        recs = get_retention_recommendation(customer_data, pred)

        risk_factors = []
        if customer_data['Support Calls'] >= 8:
            risk_factors.append(f"High support calls ({customer_data['Support Calls']})")
        if customer_data['Payment Delay'] > 10:
            risk_factors.append(f"Payment delay ({customer_data['Payment Delay']} days)")
        if customer_data['Last Interaction'] > 20:
            risk_factors.append(f"Recent inactivity ({customer_data['Last Interaction']} days)")

        result = {
            'prediction': 'Yes' if pred == 1 else 'No',
            'probability': f"{proba:.1%}",
            'risk_factors': risk_factors,
            'recommendations': recs,
            'customer_data': customer_data
        }

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in single prediction: {e}")
        return jsonify({'error': str(e)}), 500


# --- Existing file upload bulk prediction ---
@app.route('/upload', methods=['POST'])
def upload_predict():
    try:
        csv_file = request.files.get('csv_file')
        if not csv_file or csv_file.filename == '':
            return jsonify({'error': 'No file uploaded'}), 400

        df = pd.read_csv(csv_file)

        # Validate required columns
        required_cols = categorical_cols + numerical_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing required columns: {missing_cols}'}), 400

        # Preprocess data
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = label_encoders[col].transform(df_encoded[col])
        df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        df_encoded = df_encoded[dt.feature_names_in_]

        # Predict
        preds = dt.predict(df_encoded)
        probs = dt.predict_proba(df_encoded)[:, 1]

        results = []
        for i, row in df.iterrows():
            pred = preds[i]
            proba = probs[i]
            cust_data = row.to_dict()
            recs = get_retention_recommendation(cust_data, pred)
            risk_factors = []
            if cust_data['Support Calls'] >= 8:
                risk_factors.append(f"High support calls ({cust_data['Support Calls']})")
            if cust_data['Payment Delay'] > 10:
                risk_factors.append(f"Payment delay ({cust_data['Payment Delay']} days)")
            if cust_data['Last Interaction'] > 20:
                risk_factors.append(f"Recent inactivity ({cust_data['Last Interaction']} days)")

            results.append({
                'prediction': 'Yes' if pred == 1 else 'No',
                'probability': f"{proba:.1%}",
                'risk_factors': risk_factors,
                'recommendations': recs,
                'customer_data': cust_data
            })

        return jsonify({'results': results})

    except Exception as e:
        logging.error(f"Error in upload prediction: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
