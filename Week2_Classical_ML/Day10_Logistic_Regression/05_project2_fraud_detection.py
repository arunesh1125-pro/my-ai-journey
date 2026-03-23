"""
PROJECT 2: CREDIT CARD FRAUD DETECTION
=======================================
Detect fraudulent transactions in highly imbalanced dataset

Business Context:
- Credit card company processing millions of transactions daily
- Fraud rate: ~0.17% (highly imbalanced!)
- Average fraud amount: ₹25,000
- Average legitimate transaction: ₹5,000
- Goal: Catch fraud while minimizing false alarms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print("="*70)
print("PROJECT 2: CREDIT CARD FRAUD DETECTION")
print("="*70)

# BUSINESS PROBLEM

print("""
╔══════════════════════════════════════════════════════════════╗
║            BUSINESS PROBLEM: FRAUD DETECTION                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Company: PaySecure India (Payment Processing)              ║
║  Problem: 0.17% fraud rate (172 frauds per 100,000 txns)    ║
║                                                              ║
║  Challenge: EXTREME CLASS IMBALANCE                          ║
║  • Legitimate transactions: 99.83%                           ║
║  • Fraudulent transactions: 0.17%                            ║
║                                                              ║
║  Financial Impact per Fraud:                                 ║
║  • Average fraud amount: ₹25,000                            ║
║  • Detection cost: ₹50 (automated review)                   ║
║  • Manual review cost: ₹500 (if flagged)                    ║
║  • Customer inconvenience: High (false positives)            ║
║                                                              ║
║  Success Criteria:                                           ║
║  • Catch 85%+ of fraud (High Recall)                        ║
║  • Keep false alarms low (Precision > 10%)                  ║
║  • Balance customer experience with fraud prevention         ║
║                                                              ║
║  Why Imbalance Matters:                                      ║
║  A naive model predicting "all legitimate" achieves          ║
║  99.83% accuracy but catches ZERO fraud!                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# GENERATE REALISTIC FRAUD DATA

print("\n" + "="*70)
print("STEP 1: DATA GENERATION (Imbalanced Dataset)")
print("="*70)

np.random.seed(42)

# Generate mostly legitimate transactions
n_legitimate = 100000
n_fraud = 172  # 0.17% fraud rate

print(f"Generating dataset:")
print(f"  Legitimate transactions: {n_legitimate:,}")
print(f"  Fraudulent transactions: {n_fraud}")
print(f"  Imbalance ratio: {n_legitimate/n_fraud:.1f}:1")

# Legitimate transactions
legitimate_time = np.random.uniform(0, 172800, n_legitimate)  # 48 hours in seconds
legitimate_amount = np.random.lognormal(8.5, 0.8, n_legitimate)  # ₹5,000 average
legitimate_v1 = np.random.normal(0, 1.5, n_legitimate)
legitimate_v2 = np.random.normal(0, 1.5, n_legitimate)
legitimate_v3 = np.random.normal(0, 1.5, n_legitimate)
legitimate_v4 = np.random.normal(0, 1.5, n_legitimate)
legitimate_v5 = np.random.normal(0, 1.5, n_legitimate)
legitimate_v6 = np.random.normal(0, 1.5, n_legitimate)
legitimate_v7 = np.random.normal(0, 1.5, n_legitimate)

# Fraudulent transactions (different patterns)
fraud_time = np.random.uniform(0, 172800, n_fraud)
fraud_amount = np.random.lognormal(10.1, 0.6, n_fraud)  # ₹25,000 average (higher!)
fraud_v1 = np.random.normal(-3, 2, n_fraud)  # Different distribution
fraud_v2 = np.random.normal(2.5, 2, n_fraud)
fraud_v3 = np.random.normal(-2, 2, n_fraud)
fraud_v4 = np.random.normal(3, 2, n_fraud)
fraud_v5 = np.random.normal(-1.5, 2, n_fraud)
fraud_v6 = np.random.normal(2, 2, n_fraud)
fraud_v7 = np.random.normal(-2.5, 2, n_fraud)

# Combine
time = np.concatenate([legitimate_time, fraud_time])
amount = np.concatenate([legitimate_amount, fraud_amount])
v1 = np.concatenate([legitimate_v1, fraud_v1])
v2 = np.concatenate([legitimate_v2, fraud_v2])
v3 = np.concatenate([legitimate_v3, fraud_v3])
v4 = np.concatenate([legitimate_v4, fraud_v4])
v5 = np.concatenate([legitimate_v5, fraud_v5])
v6 = np.concatenate([legitimate_v6, fraud_v6])
v7 = np.concatenate([legitimate_v7, fraud_v7])
fraud_label = np.concatenate([np.zeros(n_legitimate), np.ones(n_fraud)])

# Shuffle
shuffle_idx = np.random.permutation(len(time))
time = time[shuffle_idx]
amount = amount[shuffle_idx]
v1, v2, v3, v4, v5, v6, v7 = v1[shuffle_idx], v2[shuffle_idx], v3[shuffle_idx], v4[shuffle_idx], v5[shuffle_idx], v6[shuffle_idx], v7[shuffle_idx]
fraud_label = fraud_label[shuffle_idx]

# Create DataFrame
df = pd.DataFrame({
    'Time': time,
    'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5, 'V6': v6, 'V7': v7,
    'Amount': amount,
    'Class': fraud_label.astype(int)
})

print(f"\n✅ Generated dataset: {len(df):,} transactions")
print(f"\nFirst 10 rows:")
print(df.head(10))

# EXPLORATORY DATA ANALYSIS

print("\n" + "="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"  Total transactions: {len(df):,}")
print(f"  Features: {df.shape[1] - 1}")

print(f"\n⚠️  CLASS IMBALANCE ANALYSIS:")
class_dist = df['Class'].value_counts()
fraud_rate = df['Class'].mean()
print(f"  Legitimate (0): {class_dist[0]:,} ({(1-fraud_rate)*100:.2f}%)")
print(f"  Fraud (1):      {class_dist[1]:,} ({fraud_rate*100:.2f}%)")
print(f"  Imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1")
print(f"\n  ⚠️  EXTREME IMBALANCE!")
print(f"     This is realistic for fraud detection")
print(f"     Standard ML approaches will fail!")

print(f"\n💰 Transaction Amount Analysis:")
legit_amounts = df[df['Class'] == 0]['Amount']
fraud_amounts = df[df['Class'] == 1]['Amount']
print(f"  Legitimate - Mean: ₹{legit_amounts.mean():,.2f}, Median: ₹{legit_amounts.median():,.2f}")
print(f"  Fraud      - Mean: ₹{fraud_amounts.mean():,.2f}, Median: ₹{fraud_amounts.median():,.2f}")
print(f"  → Fraudulent transactions are {fraud_amounts.mean()/legit_amounts.mean():.2f}x larger on average!")

print(f"\n🕐 Time Distribution:")
print(f"  Dataset covers: {df['Time'].max()/3600:.1f} hours")
print(f"  Legitimate avg time: {legit_amounts.mean():.1f}s")
print(f"  Fraud avg time: {fraud_amounts.mean():.1f}s")

# NAIVE APPROACH (BASELINE)

print("\n" + "="*70)
print("STEP 3: NAIVE BASELINE (Why Simple Accuracy Fails)")
print("="*70)

# Split data
X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {len(X_train):,} transactions")
print(f"Test set:  {len(X_test):,} transactions")
print(f"Test fraud rate: {y_test.mean()*100:.2f}%")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train naive model (no special handling of imbalance)
print(f"\n🔧 Training NAIVE Logistic Regression...")
naive_model = LogisticRegression(max_iter=1000, random_state=42)
naive_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_naive = naive_model.predict(X_test_scaled)
y_pred_proba_naive = naive_model.predict_proba(X_test_scaled)[:, 1]

accuracy_naive = accuracy_score(y_test, y_pred_naive)
precision_naive = precision_score(y_test, y_pred_naive, zero_division=0)
recall_naive = recall_score(y_test, y_pred_naive)
f1_naive = f1_score(y_test, y_pred_naive)

cm_naive = confusion_matrix(y_test, y_pred_naive)
tn_n, fp_n, fn_n, tp_n = cm_naive.ravel()

print(f"\n❌ NAIVE MODEL RESULTS:")
print(f"{'─'*50}")
print(f"  Accuracy:  {accuracy_naive:.2%}  ← Looks great! But...")
print(f"  Precision: {precision_naive:.2%}")
print(f"  Recall:    {recall_naive:.2%}  ← TERRIBLE! Missed most fraud!")
print(f"  F1-Score:  {f1_naive:.4f}")

print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn_n:,}")
print(f"    False Positives: {fp_n}")
print(f"    False Negatives: {fn_n} ← Missed this many frauds!")
print(f"    True Positives:  {tp_n}")

print(f"\n  💡 Why This Fails:")
print(f"     Model learned to predict 'legitimate' for everything!")
print(f"     It achieves high accuracy but catches almost NO fraud.")
print(f"     This is the imbalanced data problem!")

# APPROACH 1: CLASS WEIGHTS

print("\n" + "="*70)
print("STEP 4: APPROACH 1 - Class Weights")
print("="*70)

print("""
Strategy: Give more importance to fraud class during training
- Penalize fraud misclassification more heavily
- Forces model to pay attention to minority class
""")

# Calculate class weights
fraud_weight = len(y_train) / (2 * np.sum(y_train))
legit_weight = len(y_train) / (2 * np.sum(y_train == 0))

print(f"Class weights:")
print(f"  Legitimate (0): {legit_weight:.2f}")
print(f"  Fraud (1):      {fraud_weight:.2f}")
print(f"  → Fraud errors are {fraud_weight/legit_weight:.0f}x more costly!")

# Train with class weights
model_weighted = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',    # Automatically balances
    random_state = 42
)
model_weighted.fit(X_train_scaled, y_train)

# Evaluate
y_pred_weighted = model_weighted.predict(X_test_scaled)
y_pred_proba_weighted = model_weighted.predict_proba(X_test_scaled)[:, 1]

accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
precision_weighted = precision_score(y_test, y_pred_weighted)
recall_weighted = recall_score(y_test, y_pred_weighted)
f1_weighted = f1_score(y_test, y_pred_weighted)

cm_weighted = confusion_matrix(y_test, y_pred_weighted)
tn_w, fp_w, fn_w, tp_w = cm_weighted.ravel()

print(f"\n✅ CLASS WEIGHTS MODEL RESULTS:")
print(f"{'─'*50}")
print(f"  Accuracy:  {accuracy_weighted:.2%}")
print(f"  Precision: {precision_weighted:.2%}")
print(f"  Recall:    {recall_weighted:.2%}  ← Much better!")
print(f"  F1-Score:  {f1_weighted:.4f}")

print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn_w:,}")
print(f"    False Positives: {fp_w:,} ← More false alarms (trade-off)")
print(f"    False Negatives: {fn_w} ← Fewer missed frauds!")
print(f"    True Positives:  {tp_w} ← Caught more fraud!")

# APPROACH 2: SMOTE (Oversampling)

print("\n" + "="*70)
print("STEP 5: APPROACH 2 - SMOTE (Synthetic Oversampling)")
print("="*70)

print("""
Strategy: Create synthetic fraud examples
- SMOTE = Synthetic Minority Over-sampling Technique
- Generates new fraud samples by interpolating between existing ones
- Balances the dataset
""")

print(f"\nOriginal training set class distribution:")
print(f"  Before SMOTE: {Counter(y_train)}")

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"  After SMOTE:  {Counter(y_train_smote)}")
print(f"  → Dataset is now balanced!")

# Train on SMOTE data
model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

# Evaluate
y_pred_smote = model_smote.predict(X_test_scaled)
y_pred_proba_smote = model_smote.predict_proba(X_test_scaled)[:, 1]

accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)

cm_smote = confusion_matrix(y_test, y_pred_smote)
tn_s, fp_s, fn_s, tp_s = cm_smote.ravel()

print(f"\n✅ SMOTE MODEL RESULTS:")
print(f"{'─'*50}")
print(f"  Accuracy:  {accuracy_smote:.2%}")
print(f"  Precision: {precision_smote:.2%}")
print(f"  Recall:    {recall_smote:.2%}  ← Excellent!")
print(f"  F1-Score:  {f1_smote:.4f}")

print(f"\n  Confusion Matrix:")
print(f"    True Negatives:  {tn_s:,}")
print(f"    False Positives: {fp_s:,}")
print(f"    False Negatives: {fn_s}")
print(f"    True Positives:  {tp_s}")

# THRESHOLD TUNING

print("\n" + "="*70)
print("STEP 6: APPROACH 3 - Threshold Tuning")
print("="*70)

print("""
Strategy: Adjust decision threshold from default 0.5
- Lower threshold → Catch more fraud (higher recall)
- Higher threshold → Fewer false alarms (higher precision)
- Find optimal threshold for business needs
""")

# Test different thresholds
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_results = []
for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba_weighted >= thresh).astype(int)

    if y_pred_thresh.sum() > 0:
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

        threshold_results.append({
            'Threshold': thresh,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

threshold_df = pd.DataFrame(threshold_results)

print(f"\n📊 Threshold Analysis:")
print(threshold_df.to_string(index=False))

# Find optimal threshold (maximize F1)
optimal_idx = threshold_df['F1'].idxmax()
optimal_threshold = threshold_df.iloc[optimal_idx]['Threshold']
optimal_f1 = threshold_df.iloc[optimal_idx]['F1']


print(f"\n✅ Optimal Threshold: {optimal_threshold}")
print(f"   F1-Score: {optimal_f1:.4f}")

# Use optimal threshold
y_pred_optimal = (y_pred_proba_weighted >= optimal_threshold).astype(int)
precision_optimal = precision_score(y_test, y_pred_optimal)
recall_optimal = recall_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)

cm_optimal = confusion_matrix(y_test, y_pred_optimal)
tn_o, fp_o, fn_o, tp_o = cm_optimal.ravel()

print(f"\n✅ OPTIMIZED THRESHOLD MODEL:")
print(f"{'─'*50}")
print(f"  Precision: {precision_optimal:.2%}")
print(f"  Recall:    {recall_optimal:.2%}")
print(f"  F1-Score:  {f1_optimal:.4f}")

# MODEL COMPARISON

print("\n" + "="*70)
print("STEP 7: MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['Naive', 'Class Weights', 'SMOTE', 'Optimal Threshold'],
    'Accuracy': [accuracy_naive, accuracy_weighted, accuracy_smote, accuracy_score(y_test, y_pred_optimal)],
    'Precision': [precision_naive, precision_weighted, precision_smote, precision_optimal],
    'Recall': [recall_naive, recall_weighted, recall_smote, recall_optimal],
    'F1-Score': [f1_naive, f1_weighted, f1_smote, f1_optimal],
    'TP': [tp_n, tp_w, tp_s, tp_o],
    'FP': [fp_n, fp_w, fp_s, fp_o],
    'FN': [fn_n, fn_w, fn_s, fn_o]
})

print(f"\n📊 Complete Model Comparison:")
print(comparison_df.to_string(index=False))

print(f"\n💡 Key Insights:")
print(f"  • Naive model: High accuracy but useless (low recall)")
print(f"  • Class Weights: Simple, effective improvement")
print(f"  • SMOTE: Best recall, catches most fraud")
print(f"  • Threshold Tuning: Fine-tune precision/recall trade-off")

# BUSINESS IMPACT ANALYSIS

print("\n" + "="*70)
print("STEP 8: BUSINESS IMPACT ANALYSIS")
print("="*70)

# Use SMOTE model (best recall)
avg_fraud_amount = fraud_amounts.mean()
review_cost = 500  # Manual review per flagged transaction
detection_cost = 50  # Automated detection cost

# Scale to annual volume (test set × 365 days)
annual_factor = 365 * (len(df) / len(y_test))

# Scenario 1: No Model
total_fraud_annual = int(y_test.sum() * annual_factor)
fraud_loss_no_model = total_fraud_annual * avg_fraud_amount

print(f"\n💰 SCENARIO 1: No Fraud Detection")
print(f"{'─'*50}")
print(f"  Annual frauds: {total_fraud_annual:,}")
print(f"  Total fraud loss: ₹{fraud_loss_no_model/1e7:.2f} crore")
print(f"  Detection cost: ₹0")
print(f"  Net loss: ₹{fraud_loss_no_model/1e7:.2f} crore")

# Scenario 2: With SMOTE Model
detected_fraud_annual = int(tp_s * annual_factor)
missed_fraud_annual = int(fn_s * annual_factor)
false_alarms_annual = int(fp_s * annual_factor)

fraud_loss_with_model = missed_fraud_annual * avg_fraud_amount
total_detection_cost = (detected_fraud_annual + false_alarms_annual) * detection_cost
total_review_cost = false_alarms_annual * review_cost
total_cost = total_detection_cost + total_review_cost
net_benefit = fraud_loss_no_model - fraud_loss_with_model - total_cost

print(f"\n💰 SCENARIO 2: With Fraud Detection (SMOTE Model)")
print(f"{'─'*50}")
print(f"  Detected frauds: {detected_fraud_annual:,} ({recall_smote:.1%} catch rate)")
print(f"  Missed frauds: {missed_fraud_annual:,}")
print(f"  False alarms: {false_alarms_annual:,}")
print(f"  Prevented loss: ₹{detected_fraud_annual * avg_fraud_amount/1e7:.2f} crore")
print(f"  Remaining fraud loss: ₹{fraud_loss_with_model/1e7:.2f} crore")
print(f"  Detection cost: ₹{total_detection_cost/1e5:.2f} lakh")
print(f"  Review cost: ₹{total_review_cost/1e5:.2f} lakh")
print(f"  Total operational cost: ₹{total_cost/1e5:.2f} lakh")
print(f"  NET BENEFIT: ₹{net_benefit/1e7:.2f} crore")

roi = (net_benefit / total_cost) * 100
print(f"\n🎉 ROI of Fraud Detection System:")
print(f"  Return on Investment: {roi:.0f}%")
print(f"  For every ₹1 spent: ₹{roi/100:.1f} saved in fraud losses!")

# VISUALIZATION

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

# Plot 1: Class Imbalance
ax1 = fig.add_subplot(gs[0, 0])
class_counts = df['Class'].value_counts()
colors_class = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['Legitimate', 'Fraud'], class_counts.values, 
               color=colors_class, edgecolor='black', linewidth=2)
ax1.set_ylabel('Count', fontweight='bold', fontsize=11)
ax1.set_title('Extreme Class Imbalance', fontweight='bold', fontsize=13)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(class_counts.values):
    ax1.text(i, v, f'{v:,}\n({v/len(df)*100:.2f}%)', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: Amount Distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(legit_amounts, bins=50, alpha=0.6, label='Legitimate', 
         color='blue', edgecolor='black', density=True)
ax2.hist(fraud_amounts, bins=30, alpha=0.6, label='Fraud',
         color='red', edgecolor='black', density=True)
ax2.set_xlabel('Transaction Amount (₹)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Density', fontweight='bold', fontsize=11)
ax2.set_title('Transaction Amount Distribution', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 100000)
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparisoon - Recall
ax3 = fig.add_subplot(gs[0, 2])
models = comparison_df['Model']
recalls = comparison_df['Recall']
colors_recall = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
bars = ax3.barh(models, recalls, color=colors_recall, edgecolor='black', linewidth=2)
ax3.set_xlabel('Recall (Fraud Detection Rate)', fontweight='bold', fontsize=11)
ax3.set_title('Model Comparison: Recall', fontweight='bold', fontsize=13)
ax3.set_xlim(0, 1)
ax3.grid(axis='x', alpha=0.3)
for i, v in enumerate(recalls):
    ax3.text(v + 0.02, i, f'{v:.1%}', va='center', fontweight='bold', fontsize=10)

# Plot 4: COnfusion Matrices Comparison
ax4 = fig.add_subplot(gs[1, :])
cms = [cm_naive, cm_weighted, cm_smote, cm_optimal]
titles = ['Naive', 'Class Weights', 'SMOTE', 'Optimal Threshold']

for idx, (cm_data, title) in enumerate(zip(cms, titles)):
    ax = plt.subplot(2, 4, idx + 1)
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'],
                ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_ylabel('Actual', fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=12)

# Plot 5: Precision-Recall Trade-off
ax5 = fig.add_subplot(gs[2, 0])
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba_smote)
ax5.plot(recalls, precisions, linewidth=3, color='#9b59b6')
ax5.fill_between(recalls, precisions, alpha=0.3, color='#9b59b6')
ax5.set_xlabel('Recall', fontweight='bold', fontsize=11)
ax5.set_ylabel('Precision', fontweight='bold', fontsize=11)
ax5.set_title('Precision-Recall Curve (SMOTE)', fontweight='bold', fontsize=13)
ax5.grid(True, alpha=0.3)

# Plot 6: ROC Curve Comparison
ax6 = fig.add_subplot(gs[2, 1])
fpr_naive, tpr_naive, _ = roc_curve(y_test, y_pred_proba_naive)
fpr_weighted, tpr_weighted, _ = roc_curve(y_test, y_pred_proba_weighted)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred_proba_smote)

ax6.plot(fpr_naive, tpr_naive, linewidth=2, label=f'Naive (AUC={roc_auc_score(y_test, y_pred_proba_naive):.3f})', color='#e74c3c')
ax6.plot(fpr_weighted, tpr_weighted, linewidth=2, label=f'Weighted (AUC={roc_auc_score(y_test, y_pred_proba_weighted):.3f})', color='#f39c12')
ax6.plot(fpr_smote, tpr_smote, linewidth=2, label=f'SMOTE (AUC={roc_auc_score(y_test, y_pred_proba_smote):.3f})', color='#2ecc71')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax6.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax6.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
ax6.set_title('ROC Curves Comparison', fontweight='bold', fontsize=13)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# Plot 7: threshold Impact
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(threshold_df['Threshold'], threshold_df['Precision'], 
         'o-', linewidth=2, markersize=6, label='Precision', color='#3498db')
ax7.plot(threshold_df['Threshold'], threshold_df['Recall'],
         's-', linewidth=2, markersize=6, label='Recall', color='#e74c3c')
ax7.plot(threshold_df['Threshold'], threshold_df['F1'],
         '^-', linewidth=2, markersize=6, label='F1-Score', color='#2ecc71')
ax7.axvline(x=optimal_threshold, color='purple', linestyle='--', 
            linewidth=2, alpha=0.7, label=f'Optimal ({optimal_threshold})')
ax7.set_xlabel('Decision Threshold', fontweight='bold', fontsize=11)
ax7.set_ylabel('Score', fontweight='bold', fontsize=11)
ax7.set_title('Threshold Tuning Analysis', fontweight='bold', fontsize=13)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 8: Business Impact Summary
ax8 = fig.add_subplot(gs[3, :])
ax8.axis('off')

impact_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         BUSINESS IMPACT SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WITHOUT FRAUD DETECTION:                                                    ║
║  • Annual fraud transactions: {total_fraud_annual:,}                                    ║
║  • Total fraud loss: ₹{fraud_loss_no_model/1e7:.2f} crore                                          ║
║                                                                              ║
║  WITH FRAUD DETECTION (SMOTE Model):                                         ║
║  • Frauds detected: {detected_fraud_annual:,} ({recall_smote:.1%} catch rate)                        ║
║  • Prevented loss: ₹{detected_fraud_annual * avg_fraud_amount/1e7:.2f} crore                                        ║
║  • Missed frauds: {missed_fraud_annual:,}                                                   ║
║  • False alarms: {false_alarms_annual:,} (customer inconvenience)                         ║
║  • Operational cost: ₹{total_cost/1e5:.2f} lakh                                             ║
║  • NET BENEFIT: ₹{net_benefit/1e7:.2f} crore annually                                      ║
║                                                                              ║
║  ROI: {roi:.0f}% (₹{roi/100:.1f} saved per ₹1 spent)                                         ║
║                                                                              ║
║  KEY METRICS (SMOTE Model):                                                  ║
║  • Precision: {precision_smote:.1%} ({precision_smote*100:.0f}% of flagged transactions are fraud)        ║
║  • Recall: {recall_smote:.1%} (catching {recall_smote*100:.0f}% of all fraud)                           ║
║  • F1-Score: {f1_smote:.3f} (balanced performance)                                     ║
║                                                                              ║
║  RECOMMENDATION: Deploy SMOTE model with continuous monitoring               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

ax8.text(0.5, 0.5, impact_text, transform=ax8.transAxes,
         fontsize=10, verticalalignment='center', horizontalalignment='center',
         family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('CREDIT CARD FRAUD DETECTION - COMPLETE ANALYSIS', 
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('06_fraud_detection_complete.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 06_fraud_detection_complete.png")

# ACTIONABLE RECOMMENDATIONS

print("\n" + "="*70)
print("STEP 9: DEPLOYMENT RECOMMENDATIONS")
print("="*70)

recommendations = f"""
🎯 RECOMMENDED APPROACH: SMOTE + Threshold Tuning

DEPLOYMENT STRATEGY:

1. PRIMARY MODEL: SMOTE-based Logistic Regression
   → Best fraud detection rate ({recall_smote:.1%})
   → Acceptable false alarm rate
   → Real-time scoring capability

2. THRESHOLD CONFIGURATION:
   → Standard threshold: {optimal_threshold} (balanced)
   → High-value transactions (>₹50,000): Lower threshold (0.2)
     • Catch more fraud on expensive transactions
   → Low-value transactions (<₹5,000): Higher threshold (0.5)
     • Reduce false alarms on small purchases

3. AUTOMATED ACTIONS:
   → Score > 0.8: Block transaction immediately
   → Score 0.5-0.8: SMS verification required
   → Score 0.3-0.5: Flag for manual review within 24h
   → Score < 0.3: Allow transaction

4. MONITORING & MAINTENANCE:
   Weekly:
   • Track precision, recall, false alarm rate
   • Monitor fraud patterns (new attack vectors)
   • Adjust thresholds based on performance

   Monthly:
   • Retrain model with new fraud examples
   • Update SMOTE parameters
   • A/B test alternative approaches

   Quarterly:
   • Full system audit
   • Cost-benefit analysis
   • Explore advanced models (Random Forest, XGBoost)

5. CUSTOMER EXPERIENCE:
   → SMS: "Unusual transaction detected. Reply Y to confirm"
   → Reduce friction: 2-factor auth only when needed
   → False positive handling: Easy appeal process

6. CONTINUOUS IMPROVEMENT:
   → Collect feedback on false positives
   → Label new fraud patterns quickly
   → Adversarial testing (simulate attacks)

EXPECTED OUTCOMES:
✅ Prevent ₹{net_benefit/1e7:.2f} crore fraud losses annually
✅ Catch {recall_smote:.0%}+ of fraudulent transactions
✅ Maintain customer satisfaction (minimal false alarms)
✅ {roi:.0f}% ROI on fraud detection system
"""

print(recommendations.format(
    recall_smote=recall_smote,
    optimal_threshold=optimal_threshold,
    net_benefit=net_benefit,
    roi=roi
))

# Save detailed report
with open('fraud_detection_report.txt', 'w', encoding='utf-8') as f:
    f.write("CREDIT CARD FRAUD DETECTION - DETAILED REPORT\n")
    f.write("="*70 + "\n\n")
    f.write("Model Comparison:\n")
    f.write(comparison_df.to_string(index=False))
    f.write("\n\nBusiness Impact:\n")
    f.write(f"  Net Benefit: ₹{net_benefit/1e7:.2f} crore annually\n")
    f.write(f"  ROI: {roi:.0f}%\n")
    f.write(f"  Frauds Detected: {detected_fraud_annual:,}\n")
    f.write(f"  Prevented Loss: ₹{detected_fraud_annual * avg_fraud_amount/1e7:.2f} crore\n")

print("\n✅ Saved detailed report: fraud_detection_report.txt")

print("\n" + "="*70)
print("PROJECT 2 COMPLETE: CREDIT CARD FRAUD DETECTION")
print("="*70)