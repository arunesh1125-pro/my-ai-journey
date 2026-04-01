"""
COMPREHENSIVE ML PROJECT: HR EMPLOYEE ATTRITION PREDICTION
===========================================================

Business Context:
- Tech company with 1,470 employees
- 16% annual attrition rate (industry avg: 13%)
- Cost to replace employee: ₹15 lakh (recruitment + training)
- Goal: Predict who will leave & take preventive action

This project demonstrates:
✅ Complete end-to-end ML pipeline
✅ Multiple algorithm comparison
✅ Hyperparameter tuning
✅ Feature engineering
✅ Business insights & recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)

print("="*80)
print("COMPREHENSIVE ML PROJECT: HR EMPLOYEE ATTRITION PREDICTION")
print("="*80)

# BUSINESS PROBLEM

print("""
╔════════════════════════════════════════════════════════════════════╗
║           BUSINESS PROBLEM: EMPLOYEE ATTRITION                     ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Company: TechCorp India (Software & IT Services)                  ║
║  Problem: 16% annual attrition (237 employees left last year)      ║
║                                                                    ║
║  Financial Impact:                                                 ║
║  • Replacement cost: ₹15 lakh per employee                        ║
║  • Annual loss: 237 × ₹15L = ₹35.55 crore                        ║
║  • Productivity loss during transition: ~₹10 crore                ║
║  • Total annual cost: ₹45+ crore                                  ║
║                                                                    ║
║  ML Solution Goals:                                                ║
║  1. Predict who will leave (6 months advance notice)               ║
║  2. Identify key drivers of attrition                              ║
║  3. Recommend targeted retention strategies                        ║
║                                                                    ║
║  Success Metrics:                                                  ║
║  • 80%+ recall (catch most at-risk employees)                     ║
║  • Interpretable insights for HR action                            ║
║  • Reduce attrition from 16% to 12% (50 employees saved)          ║
║  • Projected savings: 50 × ₹15L = ₹7.5 crore/year                ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

# STEP 1: DATA GENERATION

print("\n" + "="*80)
print("STEP 1: DATA GENERATION & LOADING")
print("="*80)

np.random.seed(42)
n_employees = 1470

# Demographics
age = np.random.normal(37, 9, n_employees).clip(22, 60)
gender = np.random.choice(['Male', 'Female'], n_employees, p=[0.6, 0.4])
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], 
                                  n_employees, p=[0.32, 0.46, 0.22])
distance_from_home = np.random.exponential(10, n_employees).clip(1, 50)

# Job characteristics
department = np.random.choice(['Sales', 'R&D', 'HR'], 
                             n_employees, p=[0.45, 0.45, 0.10])
job_role = np.random.choice(['Sales Executive', 'Research Scientist', 
                             'Software Engineer', 'Manager', 'HR'], 
                            n_employees)
job_level = np.random.choice([1, 2, 3, 4, 5], n_employees, 
                             p=[0.35, 0.30, 0.20, 0.10, 0.05])
years_at_company = np.random.exponential(7, n_employees).clip(0, 40)
years_in_role = (years_at_company * np.random.uniform(0.4, 0.9, n_employees)).clip(0, 30)
years_with_manager = (years_in_role * np.random.uniform(0.5, 1.0, n_employees)).clip(0, 20)

# Compensation & benefits
monthly_income = (
    30000 + 
    20000 * job_level + 
    5000 * years_at_company +
    np.random.normal(0, 15000, n_employees)
).clip(25000, 200000)

percent_salary_hike = np.random.normal(15, 5, n_employees).clip(5, 30)
stock_options = np.random.choice([0, 1, 2, 3], n_employees, p=[0.6, 0.25, 0.10, 0.05])

# Work-life balance & satisfaction
work_life_balance = np.random.choice([1, 2, 3, 4], n_employees, p=[0.08, 0.25, 0.40, 0.27])
job_satisfaction = np.random.choice([1, 2, 3, 4], n_employees, p=[0.10, 0.23, 0.37, 0.30])
environment_satisfaction = np.random.choice([1, 2, 3, 4], n_employees, 
                                           p=[0.10, 0.22, 0.40, 0.28])
relationship_satisfaction = np.random.choice([1, 2, 3, 4], n_employees,
                                            p=[0.08, 0.20, 0.42, 0.30])

# Performance & development
performance_rating = np.random.choice([1, 2, 3, 4], n_employees, p=[0.05, 0.15, 0.40, 0.40])
training_times_last_year = np.random.poisson(3, n_employees).clip(0, 6)
years_since_promotion = np.random.exponential(2, n_employees).clip(0, 15)

# Work conditions
overtime = np.random.choice([0, 1], n_employees, p=[0.72, 0.28])
business_travel = np.random.choice([0, 1, 2], n_employees, p=[0.71, 0.19, 0.10])
num_companies_worked = np.random.poisson(2.5, n_employees).clip(0, 9)

# Calculate attrition probability (realistic complex relationships)
attrition_logit = (
    -0.04 * (age - 35) +  # Younger more likely to leave
    -0.00004 * monthly_income +  # Lower salary = higher attrition
    -0.35 * job_satisfaction +  # Low satisfaction = leaves
    -0.30 * work_life_balance +  # Poor balance = leaves
    0.80 * overtime +  # Overtime increases attrition
    0.30 * business_travel +  # Travel increases attrition
    -0.20 * stock_options +  # Stock options retain
    0.25 * distance_from_home / 10 +  # Far commute = more likely to leave
    -0.06 * years_at_company +  # Longer tenure = less likely to leave
    0.12 * years_since_promotion +  # No promotion = frustration
    0.08 * num_companies_worked +  # Job hoppers
    -0.03 * percent_salary_hike +  # Low hike = leaves
    -0.20 * performance_rating +  # High performers don't leave (usually)
    -0.15 * environment_satisfaction +  # Bad environment = leaves
    0.50 * (marital_status == 'Single')  # Single = more mobile
)
# Convert to probability
attrition_probability = 1 / (1 + np.exp(-attrition_logit))
threshold = np.percentile(attrition_probability, 84)
attrition = (attrition_probability > threshold).astype(int)
# attrition = np.random.binomial(1, attrition_probability)

# Create DataFrame
df = pd.DataFrame({
    'EmployeeID': [f'EMP{i:04d}' for i in range(1, n_employees + 1)],
    'Age': age.round(0).astype(int),
    'Gender': gender,
    'MaritalStatus': marital_status,
    'DistanceFromHome': distance_from_home.round(0).astype(int),
    'Department': department,
    'JobRole': job_role,
    'JobLevel': job_level,
    'YearsAtCompany': years_at_company.round(1),
    'YearsInCurrentRole': years_in_role.round(1),
    'YearsWithCurrManager': years_with_manager.round(1),
    'MonthlyIncome': monthly_income.round(0).astype(int),
    'PercentSalaryHike': percent_salary_hike.round(1),
    'StockOptionLevel': stock_options,
    'WorkLifeBalance': work_life_balance,
    'JobSatisfaction': job_satisfaction,
    'EnvironmentSatisfaction': environment_satisfaction,
    'RelationshipSatisfaction': relationship_satisfaction,
    'PerformanceRating': performance_rating,
    'TrainingTimesLastYear': training_times_last_year,
    'YearsSinceLastPromotion': years_since_promotion.round(1),
    'OverTime': overtime,
    'BusinessTravel': business_travel,
    'NumCompaniesWorked': num_companies_worked,
    'Attrition': attrition
})

print(f"✅ Generated dataset: {len(df)} employees")
print(f"\nFirst 10 employees:")
print(df.head(10))

# STEP 2: EXPLORATORY DATA ANALYSIS

print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

attrition_rate = df['Attrition'].mean()
attrition_count = df['Attrition'].sum()

print(f"\n📊 Attrition Overview:")
print(f"  Total employees: {len(df)}")
print(f"  Employees left: {attrition_count} ({attrition_rate:.1%})")
print(f"  Employees stayed: {len(df) - attrition_count} ({(1-attrition_rate):.1%})")

left = df[df['Attrition'] == 1]
stayed = df[df['Attrition'] == 0]

print(f"\n📈 Key Metrics Comparison:")
print(f"\n{'Metric':<25} {'Stayed':<15} {'Left':<15} {'Difference':<15}")
print("-" * 70)
print(f"{'Avg Age':<25} {stayed['Age'].mean():>14.1f}  {left['Age'].mean():>14.1f}  {stayed['Age'].mean() - left['Age'].mean():>14.1f}")
print(f"{'Avg Monthly Income':<25} ₹{stayed['MonthlyIncome'].mean():>12,.0f}  ₹{left['MonthlyIncome'].mean():>12,.0f}  ₹{stayed['MonthlyIncome'].mean() - left['MonthlyIncome'].mean():>12,.0f}")
print(f"{'Avg Years at Company':<25} {stayed['YearsAtCompany'].mean():>14.1f}  {left['YearsAtCompany'].mean():>14.1f}  {stayed['YearsAtCompany'].mean() - left['YearsAtCompany'].mean():>14.1f}")
print(f"{'Avg Job Satisfaction':<25} {stayed['JobSatisfaction'].mean():>14.2f}  {left['JobSatisfaction'].mean():>14.2f}  {stayed['JobSatisfaction'].mean() - left['JobSatisfaction'].mean():>14.2f}")
print(f"{'Work-Life Balance':<25} {stayed['WorkLifeBalance'].mean():>14.2f}  {left['WorkLifeBalance'].mean():>14.2f}  {stayed['WorkLifeBalance'].mean() - left['WorkLifeBalance'].mean():>14.2f}")
print(f"{'% Working Overtime':<25} {stayed['OverTime'].mean()*100:>13.1f}%  {left['OverTime'].mean()*100:>13.1f}%  {(stayed['OverTime'].mean() - left['OverTime'].mean())*100:>13.1f}%")

# Department analysis
print(f"\n📊 Attrition by Department:")
dept_attrition = df.groupby('Department')['Attrition'].agg(['sum', 'count', 'mean'])
dept_attrition.columns = ['Left', 'Total', 'Attrition Rate']
dept_attrition['Attrition Rate'] = dept_attrition['Attrition Rate'] * 100
print(dept_attrition.to_string())

print("\n" + "="*80)

# FEATURE ENGINEERING

print("STEP 3: FEATURE ENGINEERING")
print("="*80)

# Encode categorical variables
df['Gender_Male'] = (df['Gender'] == 'Male').astype(int)
df['MaritalStatus_Single'] = (df['MaritalStatus'] == 'Single').astype(int)
df['MaritalStatus_Married'] = (df['MaritalStatus'] == 'Married').astype(int)
df['Dept_Sales'] = (df['Department'] == 'Sales').astype(int)
df['Dept_RD'] = (df['Department'] == 'R&D').astype(int)

# Create derived features
df['YearsPerCompany'] = df['YearsAtCompany'] / (df['NumCompaniesWorked'] + 1)
df['PromotionGap'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
df['TenureRatio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['IncomePerYear'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
df['SatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                           df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4

print("✅ Created derived features:")
print("  • YearsPerCompany: Average tenure per company")
print("  • PromotionGap: Years since last promotion")
print("  • TenureRatio: Stability in current role")
print("  • IncomePerYear: Income growth rate")
print("  • SatisfactionScore: Overall satisfaction (1-4)")

# Select features for modeling
feature_cols = [
    'Age', 'Gender_Male', 'MaritalStatus_Single', 'MaritalStatus_Married',
    'DistanceFromHome', 'Dept_Sales', 'Dept_RD', 'JobLevel',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager',
    'MonthlyIncome', 'PercentSalaryHike', 'StockOptionLevel',
    'WorkLifeBalance', 'JobSatisfaction', 'EnvironmentSatisfaction',
    'RelationshipSatisfaction', 'PerformanceRating', 
    'TrainingTimesLastYear', 'YearsSinceLastPromotion',
    'OverTime', 'BusinessTravel', 'NumCompaniesWorked',
    'YearsPerCompany', 'PromotionGap', 'TenureRatio', 
    'IncomePerYear', 'SatisfactionScore'
]

X = df[feature_cols].values
y = df['Attrition'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData prepared:")
print(f"  Training set: {len(X_train)} employees")
print(f"  Test set: {len(X_test)} employees")
print(f"  Features: {len(feature_cols)}")

# STEP 4: MODEL TRAINING & COMPARISON

print("\n" + "="*80)
print("STEP 4: MODEL TRAINING & COMPARISON")
print("="*80)

print("\n🔧 Training 4 models...")

# Model 1: Logistic Regression (Baseline)
print("\n1️⃣  Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Model 2: Decision Tree
print("2️⃣  Decision Tree...")
dt_model = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train, y_train)

# Model 3: Random Forest (with tuning)
print("3️⃣  Random Forest (with GridSearch)...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [20, 50],
    'min_samples_leaf': [10, 20]
}
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring='f1', n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
print(f"   Best params: {rf_grid.best_params_}")

# Model 4: XGBoost (with tuning)
print("4️⃣  XGBoost (with GridSearch)...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_base = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_base, xgb_params, cv=3, scoring='f1', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
print(f"   Best params: {xgb_grid.best_params_}")

print("\n✅ All models trained!")

# STEP 5: MODEL EVALUATION & COMPARISON

print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION & COMPARISON")
print("="*80)

# Store models and names
models = {
    'Logistic Regression': (lr_model, X_test_scaled),
    'Decision Tree': (dt_model, X_test),
    'Random Forest': (rf_model, X_test),
    'XGBoost': (xgb_model, X_test)
}

results = []

for name, (model, X_test_data) in models.items():
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    })

results_df = pd.DataFrame(results)

print(f"\n📊 MODEL COMPARISON:")
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

print(f"\n🏆 WINNER:")
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_f1 = results_df['F1-Score'].max()
print(f"  {best_model_name} (F1-Score: {best_f1:.3f})")

# Use best model for detailed analysis
best_model_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_info = models[best_model_name]
best_model = best_model_info[0]
best_X_test = best_model_info[1]

# Detailed evaluation of best model
y_pred_best = best_model.predict(best_X_test)
y_pred_proba_best = best_model.predict_proba(best_X_test)[:, 1]
cm_best = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm_best.ravel()


print(f"\n📋 DETAILED RESULTS ({best_model_name}):")
print(f"                PREDICTED")
print(f"              ┌──────────┬──────────┐")
print(f"              │  Stay    │  Leave   │")
print(f"    ┌─────────┼──────────┼──────────┤")
print(f"  A │ Stay    │   {tn:>4}   │   {fp:>4}   │")
print(f"  C │  (0)    │   (TN)   │   (FP)   │")
print(f"  T ├─────────┼──────────┼──────────┤")
print(f"  U │ Leave   │   {fn:>4}   │   {tp:>4}   │")
print(f"  A │  (1)    │   (FN)   │   (TP)   │")
print(f"  L └─────────┴──────────┴──────────┘")

print(f"\nBusiness Interpretation:")
print(f"  ✅ Correctly identified {tp} at-risk employees (can intervene!)")
print(f"  ✅ Correctly identified {tn} stable employees")
print(f"  ❌ Missed {fn} at-risk employees (false sense of security)")
print(f"  ⚠️  Flagged {fp} stable employees (wasted retention effort)")

# STEP 6: FEATURE IMPORTANCE

print("\n" + "="*80)
print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance for best model
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
else:
    # For Logistic Regression, use absolute coefficient
    importance = np.abs(best_model.coef_[0])

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(f"\n📊 Top 15 Attrition Drivers:")
print(importance_df.head(15).to_string(index=False))

print(f"\n💡 Key Insights:")
top_3_features = importance_df.head(3)['Feature'].tolist()
print(f"  Top 3 factors driving attrition: ")
for i, feat in enumerate(top_3_features, 1):
    print(f"    {i}. {feat}")

# STEP 7: BUSINESS IMPACT ANALYSIS

print("\n" + "="*80)
print("STEP 7: BUSINESS IMPACT ANALYSIS")
print("="*80)

# Current state
total_employees = len(df)
current_attrition = attrition_count
replacement_cost = 1500000  # ₹15 lakh per employee
current_annual_loss = current_attrition * replacement_cost

# With ML intervention
recall_achieved = results_df.loc[results_df['Model'] == best_model_name, 'Recall'].values[0]
identified_at_risk = int(current_attrition * recall_achieved)
retention_success_rate = 0.6  # 60% of identified can be retained
employees_saved = int(identified_at_risk * retention_success_rate)

retention_cost_per_employee = 150000  # ₹1.5 lakh (bonus/promotion/transfer)
total_retention_cost = identified_at_risk * retention_cost_per_employee

prevented_loss = employees_saved * replacement_cost
net_benefit = prevented_loss - total_retention_cost

print(f"\n💰 FINANCIAL ANALYSIS:")
print("-" * 70)
print(f"\nCURRENT STATE (Without ML):")
print(f"  Employees leaving annually: {current_attrition}")
print(f"  Replacement cost per employee: ₹{replacement_cost/1e5:.1f} lakh")
print(f"  Annual attrition cost: ₹{current_annual_loss/1e7:.2f} crore")

print(f"\nWITH ML SYSTEM:")
print(f"  At-risk employees identified: {identified_at_risk} ({recall_achieved:.1%} recall)")
print(f"  Retention campaigns deployed: {identified_at_risk}")
print(f"  Expected retention success: 60%")
print(f"  Employees saved: {employees_saved}")
print(f"  Prevented turnover cost: ₹{prevented_loss/1e7:.2f} crore")
print(f"  Retention campaign cost: ₹{total_retention_cost/1e7:.2f} crore")
print(f"  NET ANNUAL BENEFIT: ₹{net_benefit/1e7:.2f} crore")

roi = (net_benefit / total_retention_cost) * 100
print(f"\n🎉 ROI of ML System: {roi:.0f}%")
print(f"   For every ₹1 spent on retention: ₹{roi/100:.2f} saved!")

# STEP 8: VISUALIZATIONS

print("\n" + "="*80)
print("STEP 8: CREATING COMPREHENSIVE DASHBOARD")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)

# Plot 1: Attrition Rate
ax1 = fig.add_subplot(gs[0, 0])
attrition_data = df['Attrition'].value_counts()
colors_attrition = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax1.pie(attrition_data.values, 
                                     labels=['Stayed', 'Left'],
                                     autopct='%1.1f%%', colors=colors_attrition,
                                     startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
ax1.set_title('Overall Attrition Rate', fontweight='bold', fontsize=12)

# Plot 2: Attrition by Department
ax2 = fig.add_subplot(gs[0, 1])
dept_pivot = df.groupby('Department')['Attrition'].mean()
bars = ax2.bar(dept_pivot.index, dept_pivot.values * 100, 
              color=['#3498db', '#e74c3c', '#f39c12'], 
              edgecolor='black', linewidth=2)
ax2.set_ylabel('Attrition Rate (%)', fontweight='bold', fontsize=10)
ax2.set_title('Attrition by Department', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)

# Plot 3: Income Distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist([stayed['MonthlyIncome']/1000, left['MonthlyIncome']/1000], 
         bins=25, label=['Stayed', 'Left'],
         color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax3.set_xlabel('Monthly Income (₹ Thousands)', fontweight='bold', fontsize=10)
ax3.set_ylabel('Frequency', fontweight='bold', fontsize=10)
ax3.set_title('Income Distribution', fontweight='bold', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Overtime Impact
ax4 = fig.add_subplot(gs[0, 3])
overtime_attrition = df.groupby('OverTime')['Attrition'].mean() * 100
bars = ax4.bar(['No Overtime', 'Overtime'], overtime_attrition.values,
              color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=2)
ax4.set_ylabel('Attrition Rate (%)', fontweight='bold', fontsize=10)
ax4.set_title('Impact of Overtime', fontweight='bold', fontsize=12)
ax4.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=10)

# Plot 5: Model Comparison
ax5 = fig.add_subplot(gs[1, :2])
x = np.arange(len(results_df))
width = 0.15
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_metrics = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
    offset = width * (i - 1.5)
    bars = ax5.bar(x + offset, results_df[metric], width, 
                   label=metric, color=color, edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Score', fontweight='bold', fontsize=11)
ax5.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
ax5.set_xticks(x)
ax5.set_xticklabels(results_df['Model'], fontsize=10)
ax5.legend(fontsize=10)
ax5.set_ylim(0, 1)
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Feature Importance
ax6 = fig.add_subplot(gs[1, 2:])
top_15 = importance_df.head(15)
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_15)))
bars = ax6.barh(range(len(top_15)), top_15['Importance'],
               color=colors_imp, edgecolor='black', linewidth=1.5)
ax6.set_yticks(range(len(top_15)))
ax6.set_yticklabels(top_15['Feature'], fontsize=9)
ax6.set_xlabel('Importance', fontweight='bold', fontsize=11)
ax6.set_title(f'Top 15 Attrition Drivers ({best_model_name})', 
             fontweight='bold', fontsize=13)
ax6.grid(axis='x', alpha=0.3)

# Plot 7: ROC Curves
ax7 = fig.add_subplot(gs[2, :2])
for name, (model, X_test_data) in models.items():
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax7.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')

ax7.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax7.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax7.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax7.set_title('ROC Curves - All Models', fontweight='bold', fontsize=13)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 8: Confusion Matrix (Best Model)
ax8 = fig.add_subplot(gs[2, 2:])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
            xticklabels=['Stay', 'Leave'],
            yticklabels=['Stay', 'Leave'],
            ax=ax8, annot_kws={'size': 13, 'weight': 'bold'})
ax8.set_ylabel('Actual', fontweight='bold', fontsize=11)
ax8.set_xlabel('Predicted', fontweight='bold', fontsize=11)
ax8.set_title(f'Confusion Matrix ({best_model_name})', 
             fontweight='bold', fontsize=13)

# Plot 9: Business Impact Summary
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')

impact_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                         BUSINESS IMPACT SUMMARY                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  CURRENT STATE (Without ML):                                                         ║
║  • Annual attrition: {current_attrition} employees ({attrition_rate:.1%})                                         ║
║  • Replacement cost: ₹{current_annual_loss/1e7:.2f} crore                                                       ║
║                                                                                      ║
║  WITH ML INTERVENTION:                                                               ║
║  • Best Model: {best_model_name:65} ║
║  • At-risk identified: {identified_at_risk} employees ({recall_achieved:.1%} recall)                                    ║
║  • Employees saved: {employees_saved} (60% retention success rate)                                 ║
║  • Prevented cost: ₹{prevented_loss/1e7:.2f} crore                                                        ║
║  • Retention campaigns: ₹{total_retention_cost/1e7:.2f} crore                                                     ║
║  • NET ANNUAL BENEFIT: ₹{net_benefit/1e7:.2f} crore                                                    ║
║                                                                                      ║
║  ROI: {roi:.0f}% (₹{roi/100:.1f} saved per ₹1 spent)                                                     ║
║                                                                                      ║
║  TOP 3 ATTRITION DRIVERS:                                                            ║
║  1. {top_3_features[0]:80} ║
║  2. {top_3_features[1]:80} ║
║  3. {top_3_features[2]:80} ║
║                                                                                      ║
║  RECOMMENDED ACTIONS:                                                                ║
║  • Deploy model to score all employees monthly                                       ║
║  • Targeted retention for top 20% risk scores                                        ║
║  • Focus on improving satisfaction scores & work-life balance                        ║
║  • Review compensation for high-risk/high-performers                                 ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

ax9.text(0.5, 0.5, impact_text, transform=ax9.transAxes,
         fontsize=9, verticalalignment='center', horizontalalignment='center',
         family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('HR ANALYTICS: EMPLOYEE ATTRITION PREDICTION - COMPLETE ML PIPELINE',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('02_hr_analytics_complete.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_hr_analytics_complete.png")

# ============================================
# STEP 9: DEPLOYMENT RECOMMENDATIONS
# ============================================

print("\n" + "="*80)
print("STEP 9: DEPLOYMENT RECOMMENDATIONS")
print("="*80)

recommendations = f"""
🚀 DEPLOYMENT STRATEGY:

1. MONTHLY RISK SCORING
   → Score all employees at month-end
   → Risk categories:
     • High Risk (>70% probability): Immediate intervention
     • Medium Risk (40-70%): Watch closely
     • Low Risk (<40%): Standard engagement
   
2. TARGETED INTERVENTIONS BY RISK LEVEL
   
   HIGH RISK (Top 10%):
   • One-on-one meeting with department head
   • Salary review & possible adjustment
   • Career development plan
   • Flexible work arrangements
   • Stock option grants
   
   MEDIUM RISK (Next 20%):
   • Skip-level meetings with senior management
   • Training & development opportunities
   • Project rotation options
   • Team building activities
   
3. PROACTIVE ACTIONS (Based on Top Drivers)
   
   {top_3_features[0]}:
   → Implement quarterly satisfaction surveys
   → Address issues within 2 weeks
   
   {top_3_features[1]}:
   → Review compensation annually
   → Market benchmarking for critical roles
   
   {top_3_features[2]}:
   → Flexible work hours
   → Remote work options
   → Overtime limits & monitoring

4. MONITORING & IMPROVEMENT
   
   Weekly:
   • Track intervention success rates
   • Monitor attrition by department/role
   
   Monthly:
   • Retrain model with new attrition data
   • Update risk scores
   • Generate executive dashboard
   
   Quarterly:
   • ROI analysis
   • Model performance audit
   • A/B test new retention strategies

5. SUCCESS METRICS
   
   Primary:
   • Reduce attrition from {attrition_rate:.0%} to 12% (1-year goal)
   • Save {employees_saved}+ employees annually
   • ROI > 400%
   
   Secondary:
   • Improve avg satisfaction scores
   • Reduce turnover in critical roles
   • Increase internal promotion rate

EXPECTED OUTCOMES:
✅ {employees_saved} employees retained annually
✅ ₹{net_benefit/1e7:.2f} crore saved per year
✅ Improved employee morale
✅ Reduced recruitment costs
✅ Preserved institutional knowledge
✅ {roi:.0f}% ROI on retention programs
"""

print(recommendations)

# Save comprehensive report
with open('hr_analytics_deployment_report.txt', 'w', encoding='utf-8') as f:
    f.write("HR EMPLOYEE ATTRITION PREDICTION - DEPLOYMENT REPORT\n")
    f.write("="*80 + "\n\n")
    f.write("MODEL COMPARISON:\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBEST MODEL: {best_model_name}\n")
    if hasattr(best_model, 'get_params'):
        f.write(f"Hyperparameters: {best_model.get_params()}\n\n")
    f.write("BUSINESS IMPACT:\n")
    f.write(f"  Net Annual Benefit: ₹{net_benefit/1e7:.2f} crore\n")
    f.write(f"  Employees Saved: {employees_saved}\n")
    f.write(f"  ROI: {roi:.0f}%\n\n")
    f.write("TOP 10 ATTRITION DRIVERS:\n")
    for idx, row in importance_df.head(10).iterrows():
        f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

print("\n✅ Saved deployment report: hr_analytics_deployment_report.txt")

print("\n" + "="*80)
print("COMPREHENSIVE PROJECT COMPLETE!")
print("="*80)
print(f"\n🎉 Summary:")
print(f"  • Trained 4 models (Logistic Reg, Decision Tree, Random Forest, XGBoost)")
print(f"  • Best Model: {best_model_name}")
print(f"  • Business Impact: ₹{net_benefit/1e7:.2f} crore saved annually")
print(f"  • ROI: {roi:.0f}%")
print(f"  • Production-ready deployment plan created")