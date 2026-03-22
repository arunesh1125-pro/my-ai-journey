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