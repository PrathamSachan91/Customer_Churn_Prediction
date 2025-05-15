# 📉 Customer Churn Prediction

This project predicts whether a customer is likely to **churn** (i.e., leave a service or subscription) based on their demographic, support, and usage data. It uses a machine learning model served via Flask and features a modern, responsive frontend for input and visual output.

---

## 🚀 Features

- ✅ Predict churn with probability scores  
- 🔎 Identify key risk factors for churn  
- 💡 Generate actionable retention recommendations  
- 📁 Upload CSV file and download updated results  
- 🌙 Clean and modern UI with dark theme  
- ⚙️ Real-time inference using Flask backend  

---

## 🧠 Tech Stack

### Frontend
- HTML5, CSS3 (Dark Mode Styling)
- JavaScript (Vanilla JS)

### Backend
- Python (Flask)
- scikit-learn (ML model)
- Pandas, NumPy (data processing)

---

## 🔍 Prediction Inputs

| Field              | Description                                |
|-------------------|--------------------------------------------|
| Age               | Customer's age                             |
| Gender            | Male / Female                              |
| Support Calls     | Number of support calls made               |
| Payment Delay     | Days since last payment was due            |
| Total Spend       | Total amount spent by the customer         |
| Last Interaction  | Days since last customer interaction       |

---

## 🧪 Output

- **Prediction**: `Yes` or `No` for churn  
- **Probability Meter**: Visual indicator of churn probability  
- **Risk Factors**: Highlighted issues contributing to churn  
- **Recommendations**: Tailored retention actions  
- **📁 CSV Upload Support**: Upload a CSV and get back an annotated CSV with predictions, churn probabilities, risk tags, and recommendations

---

## 📂 CSV Upload Format

### Required Input Columns:
- `Age`  
- `Gender`  
- `Support Calls`  
- `Payment Delay`  
- `Total Spend`  
- `Last Interaction`

### Output CSV Includes:
- Original input columns  
- `Prediction` (`Yes` / `No`)  
- `Churn Probability` (e.g., `78.4%`)  
- `Risk Factors`  
- `Recommended Actions`  

---

## 🖥️ Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/PrathamSachan91/Customer_Churn_Prediction.git
cd customer-churn-prediction
churn_prediction\Scripts\activate
python app.py    
