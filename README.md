# 📉 Customer Churn Prediction

This project predicts whether a customer is likely to **churn** (i.e., leave a service or subscription) based on their demographic, support, and usage data. It uses a machine learning model served via Flask and features a modern, responsive frontend for input and visual output.

---

## 🚀 Features

- Predict churn with probability scores
- Identify key risk factors for churn
- Generate actionable recommendations
- Clean and modern UI with dark theme
- Real-time inference using Flask backend

---

## 🧠 Tech Stack

### Frontend
- HTML5, CSS3 (Dark Mode Styling)
- JavaScript (Vanilla JS)

### Backend
- Python (Flask)
- scikit-learn (for ML model)
- Pandas, NumPy (data processing)

---

## 🔍 Prediction Inputs

| Field                | Description                            |
|---------------------|----------------------------------------|
| Age                 | Customer's age                         |
| Gender              | Male / Female                          |
| Support Calls       | Number of support calls made           |
| Payment Delay       | Days since last payment was due        |
| Total Spend         | Total amount spent by the customer     |
| Last Interaction    | Days since last customer interaction   |

---

## 🧪 Output

- **Prediction:** `Yes` or `No` for churn
- **Probability Meter**: Visual meter of churn probability
- **Risk Factors**: Tags of top contributing reasons
- **Recommendations**: Customized tips to retain customer

---

## 🖥️ Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/PrathamSachan91/Customer_Churn_Prediction.git
cd customer-churn-prediction
churn_prediction\Scripts\activate
python app.py    

## Contributors
1. Harsh Kumar
2. Pratham Sachan