"""
BIG PROJECT: LOAN APPROVAL DECISION SYSTEM
==========================================
Complete end-to-end ML project with Decision Trees

Business Context:
- Bank processing 10,000+ loan applications monthly
- Manual review takes 3-5 days
- Goal: Automate 70% of decisions, flag 30% for review
- Reduce processing time from 3 days to 10 minutes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


print("="*80)
print("PROJECT: INTELLIGENT LOAN APPROVAL SYSTEM")
print("="*80)

# BUSINESS PROBLEM

print("""
╔════════════════════════════════════════════════════════════════════╗
║             BUSINESS PROBLEM: LOAN APPROVAL                        ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Bank: IndiaFirst Bank                                             ║
║  Problem: Manual loan approval is slow and inconsistent            ║
║                                                                    ║
║  Current Process:                                                  ║
║  • 10,000 applications/month                                       ║
║  • Manual review: 3-5 days per application                         ║
║  • Inconsistent decisions (human bias)                             ║
║  • High operational cost (₹500/application)                        ║
║                                                                    ║
║  Solution: ML-Powered Decision System                              ║
║  • Instant decision for clear-cut cases (70%)                      ║
║  • Flag borderline cases for human review (30%)                    ║
║  • Reduce processing time: 3 days → 10 minutes                     ║
║  • Consistent, unbiased decisions                                  ║
║                                                                    ║
║  Success Metrics:                                                  ║
║  • 85%+ accuracy on test set                                       ║
║  • High recall (don't miss good borrowers)                         ║
║  • Interpretable (explain rejections to customers)                 ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

# DATA GENERATION

print("\n" + "="*80)
print("STEP 1: DATA GENERATION")
print("="*80)

np.random.seed(42)
n_applications = 5000

# Applicant information
age = np.random.normal(35, 10, n_applications).clip(21, 65)
annual_income = np.random.lognormal(11.5, 0.6, n_applications).clip(200000, 5000000)  # ₹2L to ₹50L
employment_years = np.random.exponential(5, n_applications).clip(0, 40)
credit_score = np.random.normal(700, 100, n_applications).clip(300, 900)

# Loan details
loan_amount = np.random.uniform(100000, 5000000, n_applications)  # ₹1L to ₹50L
loan_term = np.random.choice([12, 24, 36, 48, 60, 84, 120], n_applications)  # months

# Financial ratios
debt_to_income = np.random.uniform(0, 0.8, n_applications)
existing_loans = np.random.poisson(1, n_applications).clip(0, 5)
property_value = loan_amount * np.random.uniform(1.5, 3, n_applications)

# Demographics
education = np.random.choice([0, 1, 2], n_applications, p=[0.3, 0.5, 0.2])  # 0=HS, 1=Bachelor, 2=Master+
marital_status = np.random.choice([0, 1], n_applications, p=[0.4, 0.6])  # 0=Single, 1=Married
dependents = np.random.poisson(1, n_applications).clip(0, 5)

# Calculate approval probability (realistic business logic)
approval_logit = (
    -5.0 +
    0.03 * (age - 30) +  # Age matters (sweet spot 30-50)
    0.000002 * annual_income +  # Higher income = better
    0.05 * employment_years +  # Stable employment
    0.008 * (credit_score - 600) +  # Credit score critical
    -4 * debt_to_income +  # High DTI = risky
    -0.3 * existing_loans +  # Too many loans = risky
    -0.0000005 * loan_amount +  # Large loans = higher risk
    0.5 * education +  # Education helps
    0.3 * marital_status +  # Married = stable
    -0.2 * dependents  # More dependents = expenses
)

approval_probability = 1 / (1 + np.exp(-approval_logit))
loan_approved = (approval_probability > np.random.uniform(0, 1, n_applications)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'ApplicantID': [f'APP{i:05d}' for i in range(1, n_applications + 1)],
    'Age': age.round(0).astype(int),
    'AnnualIncome': annual_income.round(0).astype(int),
    'EmploymentYears': employment_years.round(1),
    'CreditScore': credit_score.round(0).astype(int),
    'LoanAmount': loan_amount.round(0).astype(int),
    'LoanTerm': loan_term,
    'DebtToIncome': debt_to_income.round(3),
    'ExistingLoans': existing_loans,
    'PropertyValue': property_value.round(0).astype(int),
    'Education': education,  # 0=HS, 1=Bachelor, 2=Master+
    'MaritalStatus': marital_status,  # 0=Single, 1=Married
    'Dependents': dependents,
    'LoanApproved': loan_approved  # 0=Rejected, 1=Approved
})

print(f"✅ Generated {len(df)} loan applications")
print(f"\nFirst 10 applications:")
print(df.head(10))

# EXPLORATORY DATA ANALYSIS

print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

approval_rate = df['LoanApproved'].mean()
print(f"\n📊 Overall Approval Rate: {approval_rate:.1%}")

approved = df[df['LoanApproved'] == 1]
rejected = df[df['LoanApproved'] == 0]

print(f"\nApproved: {len(approved)} ({len(approved)/len(df)*100:.1f}%)")
print(f"Rejected: {len(rejected)} ({len(rejected)/len(df)*100:.1f}%)")

print(f"\n📈 Key Metrics by Approval Status:")
print(f"\n{'Metric':<20} {'Approved':<15} {'Rejected':<15}")
print("-" * 50)
print(f"{'Avg Income':<20} ₹{approved['AnnualIncome'].mean():>12,.0f}  ₹{rejected['AnnualIncome'].mean():>12,.0f}")
print(f"{'Avg Credit Score':<20} {approved['CreditScore'].mean():>14.0f}  {rejected['CreditScore'].mean():>14.0f}")
print(f"{'Avg Age':<20} {approved['Age'].mean():>14.1f}  {rejected['Age'].mean():>14.1f}")
print(f"{'Avg DTI':<20} {approved['DebtToIncome'].mean():>14.2f}  {rejected['DebtToIncome'].mean():>14.2f}")
print(f"{'Avg Loan Amount':<20} ₹{approved['LoanAmount'].mean():>12,.0f}  ₹{rejected['LoanAmount'].mean():>12,.0f}")

# DATA PREPARATION

print("\n" + "="*80)
print("STEP 3: DATA PREPARATION")
print("="*80)

# Create derived features
df['IncomeToLoan'] = df['AnnualIncome'] / df['LoanAmount']
df['PropertyToLoan'] = df['PropertyValue'] / df['LoanAmount']
df['MonthlyIncome'] = df['AnnualIncome'] / 12
df['EstimatedEMI'] = df['LoanAmount'] / df['LoanTerm'] * 1.1  # Rough EMI estimate

print("✅ Created derived features:")
print("  • IncomeToLoan: Can applicant afford this loan?")
print("  • PropertyToLoan: Collateral ratio")
print("  • MonthlyIncome: For EMI calculations")
print("  • EstimatedEMI: Rough monthly payment")

# Select features
feature_cols = [
    'Age', 'AnnualIncome', 'EmploymentYears', 'CreditScore',
    'LoanAmount', 'LoanTerm', 'DebtToIncome', 'ExistingLoans',
    'PropertyValue', 'Education', 'MaritalStatus', 'Dependents',
    'IncomeToLoan', 'PropertyToLoan', 'MonthlyIncome', 'EstimatedEMI'
]

X = df[feature_cols].values
y = df['LoanApproved'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} applications")
print(f"Test set:  {len(X_test)} applications")

# MODEL TRAINING & HYPERPARAMETER TUNING

print("\n" + "="*80)
print("STEP 4: MODEL TRAINING & HYPERPARAMETER TUNING")
print("="*80)

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [20, 50, 100],
    'min_samples_leaf': [10, 20, 30],
    'criterion': ['gini', 'entropy']
}

print("🔍 Testing combinations of hyperparameters...")
print(f"Total combinations: {len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['criterion'])}")

# Grid Search with Cross-Validation
tree_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    tree_model,
    param_grid,
    cv=5,
    scoring='f1',  # Balance precision and recall
    n_jobs=-1, # This parameter tells the function to use all available CPU cores to run the different parameter combinations in parallel, significantly speeding up the search process.
    verbose=1 # : This controls how much information is printed to the console during execution. A value of 1 provides basic progress updates. 
)

grid_search.fit(X_train, y_train)

print("\n✅ Best hyperparameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV F1-Score: {grid_search.best_score_:.3f}")

# Use best model
best_tree = grid_search.best_estimator_

# MODEL EVALUATION

print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION")
print("="*80)

# Predictions
y_pred_train = best_tree.predict(X_train)
y_pred_test = best_tree.predict(X_test)

# Metrics
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"\n🎯 MODEL PERFORMANCE:")
print("-" * 50)
print(f"Train Accuracy: {train_acc:.1%}")
print(f"Test Accuracy:  {test_acc:.1%}")
print(f"Precision:      {precision:.1%} (of predicted approvals, % correct)")
print(f"Recall:         {recall:.1%} (of actual good borrowers, % caught)")
print(f"F1-Score:       {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"                PREDICTED")
print(f"              ┌──────────┬──────────┐")
print(f"              │ Reject   │ Approve  │")
print(f"    ┌─────────┼──────────┼──────────┤")
print(f"  A │ Reject  │   {tn:>4}   │   {fp:>4}   │")
print(f"  C │  (0)    │   (TN)   │   (FP)   │")
print(f"  T ├─────────┼──────────┼──────────┤")
print(f"  U │ Approve │   {fn:>4}   │   {tp:>4}   │")
print(f"  A │  (1)    │   (FN)   │   (TP)   │")
print(f"  L └─────────┴──────────┴──────────┘")

print(f"\nBusiness Interpretation:")
print(f"  ✅ Correctly approved: {tp} good borrowers (True Positives)")
print(f"  ✅ Correctly rejected: {tn} risky borrowers (True Negatives)")
print(f"  ❌ Wrongly approved: {fp} risky borrowers (False Positives - LOSS RISK!)")
print(f"  ❌ Wrongly rejected: {fn} good borrowers (False Negatives - lost customers)")

# FEATURE IMPORTANCE

print("\n" + "="*80)
print("STEP 6: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': best_tree.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n📊 Top 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

print("\n💡 Business Insights:")
top_feature = importance_df.iloc[0]['Feature']
print(f"  • {top_feature} is the most critical factor in loan decisions")
print(f"  • Focus data quality efforts on top 5 features")
print(f"  • Least important features can be dropped to simplify model")

# BUSINESS IMPACT ANALYSIS

print("\n" + "="*80)
print("STEP 7: BUSINESS IMPACT ANALYSIS")
print("="*80)

# Current manual process
monthly_applications = 10000
manual_cost_per_app = 500  # ₹500 per manual review
manual_time_days = 3
current_monthly_cost = monthly_applications * manual_cost_per_app

# With ML system
auto_approval_rate = 0.7  # 70% automated
manual_review_rate = 0.3  # 30% flagged for review

automated_apps = int(monthly_applications * auto_approval_rate)
manual_apps = int(monthly_applications * manual_review_rate)

ml_cost_per_app = 10  # ₹10 for ML prediction
automated_cost = automated_apps * ml_cost_per_app
manual_cost = manual_apps * manual_cost_per_app
total_ml_cost = automated_cost + manual_cost

monthly_savings = current_monthly_cost - total_ml_cost
annual_savings = monthly_savings * 12

print(f"\n💰 COST ANALYSIS:")
print("-" * 50)
print(f"Current Manual Process:")
print(f"  Applications/month: {monthly_applications:,}")
print(f"  Cost per application: ₹{manual_cost_per_app}")
print(f"  Monthly cost: ₹{current_monthly_cost/1e5:.2f} lakh")
print(f"  Processing time: {manual_time_days} days")

print(f"\nWith ML System:")
print(f"  Automated (70%): {automated_apps:,} applications")
print(f"  Manual review (30%): {manual_apps:,} applications")
print(f"  ML cost: ₹{automated_cost/1e5:.2f} lakh")
print(f"  Manual cost: ₹{manual_cost/1e5:.2f} lakh")
print(f"  Total monthly cost: ₹{total_ml_cost/1e5:.2f} lakh")
print(f"  Processing time: 10 minutes (automated)")

print(f"\n🎉 SAVINGS:")
print(f"  Monthly savings: ₹{monthly_savings/1e5:.2f} lakh")
print(f"  Annual savings: ₹{annual_savings/1e7:.2f} crore")

roi = (annual_savings / (total_ml_cost * 12)) * 100
print(f"  ROI: {roi:.0f}%")

# VISUALIZATIONS

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

# Plot 1: Approval Rate
ax1 = fig.add_subplot(gs[0, 0])
approval_data = df['LoanApproved'].value_counts()
colors_approval = ['#e74c3c', '#2ecc71']
wedges, texts, autotexts = ax1.pie(approval_data.values, 
                                     labels=['Rejected', 'Approved'],
                                     autopct='%1.1f%%', colors=colors_approval,
                                     startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(13)
ax1.set_title('Overall Approval Rate', fontweight='bold', fontsize=13)

# Plot 2: Credit Score Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist([rejected['CreditScore'], approved['CreditScore']], 
         bins=30, label=['Rejected', 'Approved'],
         color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Credit Score', fontweight='bold', fontsize=11)
ax2.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax2.set_title('Credit Score Distribution', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Income vs Loan Amount
ax3 = fig.add_subplot(gs[0, 2])
scatter = ax3.scatter(df['AnnualIncome']/1e5, df['LoanAmount']/1e5,
                     c=df['LoanApproved'], cmap='RdYlGn',
                     s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Annual Income (₹ Lakhs)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Loan Amount (₹ Lakhs)', fontweight='bold', fontsize=11)
ax3.set_title('Income vs Loan Amount (by Approval)', fontweight='bold', fontsize=13)
plt.colorbar(scatter, ax=ax3, label='Approved')
ax3.grid(True, alpha=0.3)

# Plot 4: Feature Importance
ax4 = fig.add_subplot(gs[1, :])
top_10 = importance_df.head(10)
colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
bars = ax4.barh(range(len(top_10)), top_10['Importance'],
               color=colors_imp, edgecolor='black', linewidth=1.5)
ax4.set_yticks(range(len(top_10)))
ax4.set_yticklabels(top_10['Feature'], fontsize=10)
ax4.set_xlabel('Importance', fontweight='bold', fontsize=11)
ax4.set_title('Top 10 Most Important Features for Loan Approval', 
             fontweight='bold', fontsize=14)
ax4.grid(axis='x', alpha=0.3)

for i, (feature, imp) in enumerate(zip(top_10['Feature'], top_10['Importance'])):
    ax4.text(imp + 0.005, i, f'{imp:.3f}', va='center', 
            fontweight='bold', fontsize=9)

# Plot 5: Confusion Matrix
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
            xticklabels=['Reject', 'Approve'],
            yticklabels=['Reject', 'Approve'],
            ax=ax5, annot_kws={'size': 14, 'weight': 'bold'})
ax5.set_ylabel('Actual', fontweight='bold', fontsize=11)
ax5.set_xlabel('Predicted', fontweight='bold', fontsize=11)
ax5.set_title('Confusion Matrix', fontweight='bold', fontsize=13)

# Plot 6: Model Metrics
ax6 = fig.add_subplot(gs[2, 1])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [test_acc, precision, recall, f1]
colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
bars = ax6.bar(metrics, values, color=colors_metrics, 
              edgecolor='black', linewidth=2)
ax6.set_ylim(0, 1)
ax6.set_ylabel('Score', fontweight='bold', fontsize=11)
ax6.set_title('Model Performance Metrics', fontweight='bold', fontsize=13)
ax6.grid(axis='y', alpha=0.3)

for i, v in enumerate(values):
    ax6.text(i, v + 0.02, f'{v:.3f}', ha='center', 
            fontweight='bold', fontsize=10)

# Plot 7: Decision Tree Visualization
ax7 = fig.add_subplot(gs[2, 2])
plot_tree(best_tree, filled=True, 
         feature_names=feature_cols,
         class_names=['Reject', 'Approve'],
         ax=ax7, fontsize=7, max_depth=3)  # Show first 3 levels only
ax7.set_title('Decision Tree (First 3 Levels)', fontweight='bold', fontsize=13)

plt.suptitle('LOAN APPROVAL SYSTEM - COMPLETE ANALYSIS',
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig('03_loan_approval_system.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 03_loan_approval_system.png")

# DEPLOYMENT RECOMMENDATIONS

print("\n" + "="*80)
print("STEP 8: DEPLOYMENT RECOMMENDATIONS")
print("="*80)

recommendations = f"""
🚀 DEPLOYMENT PLAN:

1. AUTOMATED APPROVAL (70% of applications)
   → Confidence threshold: > 0.7 probability
   → Instant approval via mobile app/web
   → Email notification within 10 minutes
   
2. HUMAN REVIEW QUEUE (30% of applications)
   → Borderline cases (0.3 - 0.7 probability)
   → Flagged with model's reasoning
   → Loan officer reviews within 24 hours
   
3. AUTOMATIC REJECTION (0% - fully automated now)
   → Send rejection with clear reasons
   → Offer financial counseling
   → Allow re-application after 6 months

FEATURE MONITORING:
Weekly:
  • Track approval rates by feature segments
  • Monitor default rates on approved loans
  • Flag data quality issues

Monthly:
  • Retrain model with new approved/rejected loans
  • A/B test: ML vs human decisions
  • Update feature importance rankings

EXPLAIN DECISIONS TO CUSTOMERS:
Top rejection reasons:
  1. {importance_df.iloc[0]['Feature']}: Below threshold
  2. {importance_df.iloc[1]['Feature']}: Insufficient
  3. {importance_df.iloc[2]['Feature']}: High risk indicator

EXPECTED OUTCOMES:
✅ Process 70% applications instantly (7,000/month)
✅ Reduce processing time: 3 days → 10 minutes
✅ Save ₹{annual_savings/1e7:.2f} crore annually
✅ Consistent, unbiased decisions
✅ Better customer experience
✅ {roi:.0f}% ROI
"""

print(recommendations)

# Save model details
with open('loan_approval_report.txt', 'w', encoding='utf-8') as f:
    f.write("LOAN APPROVAL SYSTEM - DEPLOYMENT REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: Decision Tree Classifier\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
    f.write(f"Performance:\n")
    f.write(f"  Test Accuracy: {test_acc:.1%}\n")
    f.write(f"  Precision: {precision:.1%}\n")
    f.write(f"  Recall: {recall:.1%}\n")
    f.write(f"  F1-Score: {f1:.3f}\n\n")
    f.write(f"Business Impact:\n")
    f.write(f"  Annual Savings: ₹{annual_savings/1e7:.2f} crore\n")
    f.write(f"  ROI: {roi:.0f}%\n")
    f.write(f"  Processing Time: 3 days → 10 minutes\n\n")
    f.write("Top 5 Features:\n")
    for idx, row in importance_df.head(5).iterrows():
        f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

print("\n✅ Saved deployment report: loan_approval_report.txt")

print("\n" + "="*80)
print("PROJECT COMPLETE: LOAN APPROVAL SYSTEM")
print("="*80)