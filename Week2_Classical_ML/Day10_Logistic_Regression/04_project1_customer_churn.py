"""
PROJECT 1: CUSTOMER CHURN PREDICTION
=====================================
Predict which customers will leave (churn) vs stay

Business Context:
- Telecom company with 7,000+ customers
- Monthly subscription service
- Cost to acquire new customer: ₹5,000
- Cost to retain existing customer: ₹500
- Goal: Identify at-risk customers and prevent churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score)

print("="*70)
print("PROJECT 1: CUSTOMER CHURN PREDICTION")
print("="*70)

# BUSINESS PROBLEM

print("""
╔══════════════════════════════════════════════════════════════╗
║               BUSINESS PROBLEM: CUSTOMER CHURN               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Company: TeleCom India (Mobile Service Provider)           ║
║  Problem: 20% annual churn rate                             ║
║                                                              ║
║  Financial Impact:                                           ║
║  • Average customer lifetime value: ₹50,000                 ║
║  • Customer acquisition cost: ₹5,000                        ║
║  • Retention campaign cost: ₹500/customer                   ║
║                                                              ║
║  Goal: Build ML model to predict churn                       ║
║  Success Metric: Catch 80%+ of churners (high recall)       ║
║                                                              ║
║  Business Value:                                             ║
║  If we identify 1,000 churners and retain 70%:              ║
║  Saved revenue = 700 × ₹50,000 = ₹3.5 crore                ║
║  Campaign cost = 1,000 × ₹500 = ₹5 lakh                    ║
║  Net benefit = ₹3 crore annually!                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# GENERATE REALISTIC TELECAM DATA

print("\n" + "="*70)
print("STEP 1: DATA GENERATION")
print("="*70)

np.random.seed(42)
n_customers = 7000

# Customer Demographics
age = np.random.normal(40, 15, n_customers).clip(18, 80)
gender = np.random.choice(['Male', 'Female'], n_customers)
senior_citizen = (age >= 65).astype(int)

# Account Information
tenure_months = np.random.exponential(24, n_customers).clip(1, 72)  # Time with company
monthly_charges = np.random.normal(65, 25, n_customers).clip(20, 150)
total_charges = tenure_months * monthly_charges + np.random.normal(0, 200, n_customers)

# Services
phone_service = np.random.choice([0, 1], n_customers, p=[0.1, 0.9])
multiple_lines = np.where(phone_service == 1,
                          np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
                          0)
internet_service = np.random.choice([0, 1, 2], n_customers, p=[0.2, 0.4, 0.4])    #0=No, 1=DSL, 2=Fiber
online_security = np.where(internet_service > 0,
                           np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
                           0)
tech_support = np.where(internet_service > 0,
                        np.random.choice([0, 1], n_customers, p=[0.5, 0.5]),
                        0)

# Contract
contract_type = np.random.choice([0, 1, 2], n_customers, p=[0.55, 0.24, 0.21])  # 0=Month-to-Month, 1=One Year, 2=Two Year
paperless_billing = np.random.choice([0, 1], n_customers, p=[0.4, 0.6])
payment_method = np.random.choice([0, 1, 2, 3], n_customers)    # 0=Electronic, 1=Mailed check, 2=Bank Transfer, 3=Credit Card

# Calculate churn probability (complex realistic relationship)
churn_logit = (
    -3.5 +  # Base
    -0.05 * tenure_months +  # Longer tenure = less churn
    0.02 * monthly_charges +  # Higher charges = more churn
    0.8 * (contract_type == 0) +  # Month-to-month = more churn
    -0.5 * (contract_type == 2) +  # Two-year = less churn
    -0.4 * online_security +  # Security = less churn
    -0.3 * tech_support +  # Support = less churn
    0.5 * (internet_service == 2) +  # Fiber = more churn (higher price)
    0.3 * senior_citizen +  # Senior = more churn
    -0.2 * (payment_method == 3)  # Credit card = less churn
)

churn_probability = 1 / (1 + np.exp(-churn_logit))
churned = (churn_probability > np.random.uniform(0, 1, n_customers)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': [f'CUST{i:05d}' for i in range(1, n_customers + 1)],
    'Age': age.round(0).astype(int),
    'Gender': gender,
    'SeniorCitizen': senior_citizen,
    'Tenure_Months': tenure_months.round(0).astype(int),
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,  # 0=No, 1=DSL, 2=Fiber
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'Contract': contract_type,  # 0=Month-to-month, 1=One year, 2=Two year
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges.round(2),
    'TotalCharges': total_charges.round(2),
    'Churned': churned
})

print(f"✅ Generated dataset: {len(df)} customers")
print(f"\nFirst 10 rows:")
print(df.head(10))