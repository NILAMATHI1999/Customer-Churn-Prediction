# 📉 Customer Churn Prediction App  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)  
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)  

An **ML-powered web app for predicting customer churn** using **Streamlit** and **Random Forest**. The app allows users to input customer details and get real-time churn predictions with probability scores.

---

## 🔍 Problem Statement

Customer churn is when customers stop using a company’s service. Retaining customers is crucial for subscription-based businesses.  
The goal of this project is to:
- Predict **whether a customer will stay or churn** based on their demographics and service usage.
- Provide a **simple web interface** to make predictions accessible for business users.
- Learn how to build and deploy a **Machine Learning app**.

---

## 📊 Dataset Info

- **Dataset Name**: Telco Customer Churn  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Size**: 7043 rows × 21 columns  

### ✅ Key Preprocessing Steps
- Dropped `customerID` column.
- Converted `TotalCharges` to numeric and handled missing values.
- Applied **Label Encoding** for categorical features.
- Scaled numerical features using **StandardScaler**.

---

## ⚙️ Features Used for Prediction

| Feature Name       | Description                        |
|--------------------|----------------------------------|
| gender             | Gender of the customer            |
| SeniorCitizen      | Whether the customer is a senior  |
| Partner            | Has a partner                     |
| Dependents         | Has dependents                   |
| tenure             | Number of months with company     |
| Contract           | Contract type                     |
| InternetService    | Type of internet                  |
| PaymentMethod      | Payment method                   |
| MonthlyCharges     | Monthly billing amount            |
| TotalCharges       | Lifetime billing amount           |
| ...and other binary service features                |

---

## 🧠 Model Details

- **Algorithm**: RandomForestClassifier (scikit-learn)
- **Why Random Forest?**
  - Handles categorical and numerical data well
  - Good baseline accuracy and interpretability

### 🔬 Evaluation Metrics

| Metric     | Value  |
|------------|--------|
| Accuracy   | ~79%   |

**Confusion Matrix**:  
![Confusion Matrix](images/confusion_matrix.png)

---

## 🖥️ Streamlit Web App

The app provides:

- **Sidebar Inputs**: Gender, Contract, Internet Service, Monthly Charges, etc.
- **Outputs**:
  - ✅ Prediction (Churn / Not Churn)
  - 📈 Churn Probability (e.g., 67.2%)
  - 🧾 Recent Predictions Log (`prediction_logs.csv`)

---

## 🧰 Tools & Technologies Used

| Category           | Tools / Libraries         |
|--------------------|--------------------------|
| Language           | Python 3                 |
| ML Framework       | scikit-learn             |
| Web Framework      | Streamlit                |
| Data Handling      | pandas, numpy            |
| Visualization      | matplotlib, seaborn      |
| Model Persistence  | joblib                   |

---

## ▶️ How to Run This Project

```bash
# 1️⃣ Clone or Download the Repository
git clone https://github.com/yourusername/customer_churn_project.git
cd customer_churn_project

# 2️⃣ Create & Activate Virtual Environment (Recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit App
streamlit run app.py

# The app will open in your browser at:
# 👉 http://localhost:8501
