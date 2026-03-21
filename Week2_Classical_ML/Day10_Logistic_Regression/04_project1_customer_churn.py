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

# EXPLORATORY DATA ANALYSIS

print("\n" + "="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"  Total customers: {len(df):,}")
print(f"  Features: {df.shape[1] - 2}")  # Exclude CustomerID and target

print(f"\n📈 Churn Statistics:")
churn_counts = df['Churned'].value_counts()
churn_rate = df['Churned'].mean()
print(f"    Stayed (0): {churn_counts[0]:,} ({(1-churn_rate)*100:.1f}%)")
print(f"    Churned (1): {churn_counts[1]:,} ({churn_rate*100:.1f}%)")
print(f"    Overall churn rate: {churn_rate:.1%}")

if churn_rate < 0.3:
    print(f"  ⚠️  Slightly imbalanced (but manageable)")

print(df.describe().round(2))

print(f"\n🔍 Churn Analysis by Key Features:")

# Tenure vs Churn
tenure_churn = df.groupby(pd.cut(df['Tenure_Months'], bins=[0, 12, 24, 36, 100]))['Churned'].mean()
for tenure_bin, rate in tenure_churn.items():
    print(f"    {tenure_bin}: {rate:.1%}")

# Contract vs Churn
contract_names = {0: 'Month-to-Month', 1: 'One year', 2: 'Two year'}
contract_churn = df.groupby('Contract')['Churned'].mean()
print(f"\nChurn Rate by Contract")
for contract, rate in contract_churn.items():
    print(f"    {contract_names[contract]}: {rate:.1%}")

# Monthly charges vs Churn
df['Charge_Tier'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
charge_churn = df.groupby('Charge_Tier')['Churned'].mean()
print(f"\nChurn Rate by Monthly Charges:")
for tier, rate in charge_churn.items():
    print(f"    {tier}: {rate:.1%}")

# FEATURE ENGINEERING

print("\n" + "="*70)
print("STEP 3: FEATURE ENGINEERING")
print("="*70)

# Create new features
df['ChargesPerMonth'] = df['TotalCharges'] / (df['Tenure_Months'] + 1)  # Avoid division by zero
df['LongTermCustomer'] = (df['Tenure_Months'] >= 24).astype(int)
df['HighValue'] = (df['MonthlyCharges'] >= 70).astype(int)
df['HasSupport'] = (df['OnlineSecurity'] == 1) | (df['TechSupport'] == 1).astype(int)

# One-hot encode gender
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)

print("✅ Created new features:")
print("  • ChargesPerMonth: Monthly spending rate")
print("  • LongTermCustomer: Tenure >= 24 months")
print("  • HighValue: Monthly charges >= ₹70")
print("  • HasSupport: Has any support service")
print("  • Gender_Male: Binary gender encoding")

# PREPARE FOR DATA MODELING

print("\n" + "="*70)
print("STEP 4: DATA PREPARATION")
print("="*70)

# Select features
feature_cols = [
    'Age', 'SeniorCitizen', 'Gender_Male', 'Tenure_Months',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'TechSupport', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
    'TotalCharges', 'ChargesPerMonth', 'LongTermCustomer',
    'HighValue', 'HasSupport'
]

X = df[feature_cols].values
y = df['Churned'].values

print(f"Feature selected: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} customers")
print(f"Test set:  {len(X_test)} customers")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features standardized (mean=0, std=1)")

# MODEL TRAINING

print("\n" + "="*70)
print("STEP 5: MODEL TRAINING")
print("="*70)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Logistic Regression model trained!")

# Feature inportance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n📊 Top 10 Most Important Features:")
print(f"{'Feature':>25} {'Coefficient':>15} {'Impact':>15}")
print("-"*60)
for idx, row in feature_importance.head(10).iterrows():
    impact = 'Increase Churn' if row['Coefficient'] > 0 else 'Reduce Churn'
    print(f"{row['Feature']:>25} {row['Coefficient']:>15.4f} {impact:>15}")

# MODEL EVALUATION

print("\n" + "="*70)
print("STEP 6: MODEL EVALUATION")
print("="*70)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"{'─'*50}")
print(f"Accuracy:  {accuracy:.1%}")
print(f"Precision: {precision:.1%}")
print(f"Recall:    {recall:.1%} ← KEY METRIC (catch churners)")
print(f"F1-Score:  {f1:.3f}")
print(f"ROC-AUC:   {roc_auc:.3f}")

print(f"\n📋 Confusion Matrix:")
print(f"                PREDICTED")
print(f"              ┌─────────┬─────────┐")
print(f"              │ Stay (0)│Churn (1)│")
print(f"    ┌─────────┼─────────┼─────────┤")
print(f"  A │ Stay    │  {tn:>5}  │  {fp:>5}  │")
print(f"  C │  (0)    │  (TN)   │  (FP)   │")
print(f"  T ├─────────┼─────────┼─────────┤")
print(f"  U │ Churn   │  {fn:>5}  │  {tp:>5}  │")
print(f"  A │  (1)    │  (FN)   │  (TP)   │")
print(f"  L └─────────┴─────────┴─────────┘")

print(f"\nInterpretation:")
print(f"  ✅ Correctly identified {tp} churners (True Positives)")
print(f"  ✅ Correctly identified {tn} loyal customers (True Negatives)")
print(f"  ❌ Missed {fn} churners (False Negatives - COSTLY!)")
print(f"  ⚠️  {fp} false alarms (False Positives - wasted retention effort)")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall')
print(f"\n🔄 Cross-Validation (5-fold):")
print(f"  Recall scores: {cv_scores}")
print(f"  Mean recall: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# BUSINESS IMPACT ANALYSIS

print("\n" + "="*70)
print("STEP 7: BUSINESS IMPACT ANALYSIS")
print("="*70)

# Financial calculations
retention_cost = 500  # ₹500 per customer
customer_ltv = 50000  # ₹50,000 lifetime value
retention_success_rate = 0.7  # 70% of identified churners can be retained

# Scenario 1: Without ML model (no intervention)
total_churners = churn_counts[1]
lost_revenue_no_ml = total_churners * customer_ltv

print(f"\n💰 SCENARIO 1: Without ML Model")
print(f"{'─'*50}")
print(f"  Total churners: {total_churners:,}")
print(f"  Lost revenue: ₹{lost_revenue_no_ml/1e7:.2f} crore")
print(f"  Retention cost: ₹0 (no intervention)")
print(f"  Net loss: ₹{lost_revenue_no_ml/1e7:.2f} crore")

# Scenario 2: With ML model
identified_churners = tp  # True positives in test set
missed_churners = fn
false_alarms = fp

# Scale up to full customer base
scale_factor = len(df) / len(y_test)
identified_churners_full = int(identified_churners * scale_factor)
false_alarms_full = int(false_alarms * scale_factor)

retained_customers = int(identified_churners_full * retention_success_rate)
saved_revenue = retained_customers * customer_ltv
total_retention_cost = (identified_churners_full + false_alarms_full) * retention_cost
net_benefit = saved_revenue - total_retention_cost

print(f"\n💰 SCENARIO 2: With ML Model")
print(f"{'─'*50}")
print(f"  Identified churners: {identified_churners_full:,}")
print(f"  False alarms: {false_alarms_full:,}")
print(f"  Retention campaigns: {identified_churners_full + false_alarms_full:,}")
print(f"  Successfully retained: {retained_customers:,} (70% of identified)")
print(f"  Saved revenue: ₹{saved_revenue/1e7:.2f} crore")
print(f"  Campaign cost: ₹{total_retention_cost/1e5:.2f} lakh")
print(f"  Net benefit: ₹{net_benefit/1e7:.2f} crore")

print(f"\n🎉 ROI of ML Model:")
roi = (net_benefit / total_retention_cost) * 100
print(f"  Return on Investment: {roi:.0f}%")
print(f"  For every ₹1 spent on retention: ₹{roi/100:.2f} saved!")