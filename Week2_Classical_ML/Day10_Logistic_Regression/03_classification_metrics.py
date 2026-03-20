"""
CLASSIFICATION EVALUATION METRICS
==================================
Beyond accuracy: Understanding precision, recall, F1, ROC-AUC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, roc_auc_score, precision_recall_curve)

print("="*70)
print("CLASSIFICATION METRICS: COMPREHENSIVE GUIDE")
print("="*70)

# WHY ACCURACY ISN'T ENOUGH

print("""
╔══════════════════════════════════════════════════════════════╗
║              WHY ACCURACY ISN'T ENOUGH                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Example: Fraud Detection                                    ║
║  Dataset: 10,000 transactions                                ║
║  Fraudulent: 100 (1%)                                        ║
║  Legitimate: 9,900 (99%)                                     ║
║                                                              ║
║  Naive Model: Always predict "Not Fraud"                     ║
║  Accuracy = 9,900/10,000 = 99%  😱                          ║
║                                                              ║
║  But this model is USELESS!                                  ║
║  It never catches ANY fraud!                                 ║
║                                                              ║
║  This is why we need better metrics:                         ║
║  • Precision: Of predicted fraud, how many were real?        ║
║  • Recall: Of actual fraud, how many did we catch?           ║
║  • F1 Score: Balance of precision and recall                 ║
║  • ROC-AUC: Overall discriminative ability                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# CONFUSION MATRIX

print("\n" + "="*70)
print("1. CONFUSION MATRIX")
print("="*70)

print("""
The foundation of all classification metrics!

                    PREDICTED
                ┌─────────┬─────────┐
                │ Class 0 │ Class 1 │
    ┌───────────┼─────────┼─────────┤
    │ Class 0   │   TN    │   FP    │  TN = True Negative  (✓ Correct)
A   │           │         │         │  FP = False Positive (✗ Type I Error)
C   ├───────────┼─────────┼─────────┤
T   │ Class 1   │   FN    │   TP    │  FN = False Negative (✗ Type II Error)
U   │           │         │         │  TP = True Positive  (✓ Correct)
A   └───────────┴─────────┴─────────┘
L

Reading the Matrix:
- Diagonal (TN + TP) = Correct predictions
- Off-diagonal (FP + FN) = Errors

Example: Medical Test
TN = Healthy patient, test says healthy ✓
FP = Healthy patient, test says sick ✗ (False alarm)
FN = Sick patient, test says healthy ✗ (Missed diagnosis - DANGEROUS!)
TP = Sick patient, test says sick ✓
""")

# GENERATE EXAMPLE DATA

print("\n" + "="*70)
print("EXAMPLE: LOAN DEFAULT PREDICTION")
print("="*70)

# Generate load data
np.random.seed(42)
n = 1000

# Features
income = np.random.normal(50000, 20000, n) # Actual income
credit_score = np.random.normal(650, 100, n)  # Creadit Score
debt_ratio = np.random.uniform(0, 0.8, n)   # Debt-to-income ratio
employment_years = np.random.uniform(0, 30, n)  # Years employed

# Target (will default: 1, will repay: 0)
# Higher risk if: low income, low credit score, high debt ratio, short employment
default_prob = (                         # z
    -0.00001 * income +
    -0.002 * credit_score +
    2.0 * debt_ratio +
    -0.05 * employment_years +
    2.0
)

default_prob = 1 / (1 + np.exp(-default_prob))  # Sigmoid σ(z)
will_default = (default_prob > np.random.uniform(0, 1, n)).astype(int)

# Create DataFrame
df_loan = pd.DataFrame({
    'Income': income,
    'Credit_Score': credit_score,
    'Debt_Ratio': debt_ratio,
    'Employment_Years': employment_years,
    'Will_Default': will_default
})

print(f"Dataset: {len(df_loan)} loan applications")
print(f"\nClass distribution: ")
class_dist = df_loan['Will_Default'].value_counts()
print(f"    Will Repay (0): {class_dist[0]} ({class_dist[0]/len(df_loan)*100:.1f}%)")
print(f"    Will Default (1): {class_dist[1]} ({class_dist[1]/len(df_loan)*100:.1f}%)")

if class_dist[0] / class_dist[1] > 2 or class_dist[1] / class_dist[0] > 2:
    print("  ⚠️  IMBALANCED DATASET - Accuracy alone will be misleading!")

# Prepare data
X = df_loan[['Income', 'Credit_Score', 'Employment_Years']].values
y = df_loan["Will_Default"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# CONFUSION MATRIX

print("\n" + "="*70)
print("CONFUSION MATRIX ANALYSIS")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n                PREDICTED")
print(f"              ┌──────────┬──────────┐")
print(f"              │ Repay (0)│Default(1)│")
print(f"    ┌─────────┼──────────┼──────────┤")
print(f"  A │ Repay   │   {tn:>4}   │   {fp:>4}   │")
print(f"  C │  (0)    │   (TN)   │   (FP)   │")
print(f"  T ├─────────┼──────────┼──────────┤")
print(f"  U │ Default │   {fn:>4}   │   {tp:>4}   │")
print(f"  A │  (1)    │   (FN)   │   (TP)   │")
print(f"  L └─────────┴──────────┴──────────┘")

print(f"\nInterpretation: ")
print(f"  TN = {tn}: Correctly predicted will repay")
print(f"  FP = {fp}: Wrongly predicted default (rejected good customers)")
print(f"  FN = {fn}: Wrongly predicted repay (RISKY! gave loan to defaulters)")
print(f"  TP = {tp}: Correctly predicted default")

# ALL METRICS

print("\n" + "="*70)
print("ALL CLASSIFICATION METRICS")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"\n1. ACCURACY")
print(f"   Formula: (TP + TN) / Total")
print(f"   Calculation: ({tp} + {tn}) / {len(y_test)} = {accuracy:.4f}")
print(f"   Interpretation: {accuracy:.1%} of all predictions were correct")
print(f"   ⚠️  Can be misleading with imbalanced data!")

print(f"\n2. PRECISION (Positive Predictive Value)")
print(f"   Formula: TP / (TP + FP)")
print(f"   Calculation: {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"   Question: Of all predicted defaults, {precision:.1%} were actually defaults")
print(f"   Use case: When False Positives are costly (rejecting good customers)")

print(f"\n3. RECALL (Sensitivity, True Positive Rate)")
print(f"   Formula: TP / (TP + FN)")
print(f"   Calculation: {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"   Question: Of all actual defaults, we caught {recall:.1%}")
print(f"   Use case: When False Negatives are costly (missing fraud/disease)")

print(f"\n4. F1 SCORE (Harmonic Mean of Precision & Recall)")
print(f"   Formula: 2 × (Precision × Recall) / (Precision + Recall)")
print(f"   Calculation: 2 × ({precision:.4f} × {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")
print(f"   Interpretation: Balanced metric, good when classes are imbalanced")

print(f"\n5. ROC-AUC (Area Under Receiver Operating Characteristic)")
print(f"   Score: {roc_auc:.4f}")
print(f"   Range: 0.5 (random) to 1.0 (perfect)")
print(f"   Interpretation: Overall ability to discriminate between classes")
print(f"   Grade: ", end="")
if roc_auc > 0.9:
    print("Excellent!")
elif roc_auc > 0.8:
    print("Good")
elif roc_auc > 0.7:
    print("Fair")
else:
    print("Needs improvement")

# PRECISION-RECALL TRADEOFF

print("\n" + "="*70)
print("PRECISION-RECALL TRADEOFF")
print("="*70)

print("""
The Dilemma:
- Increase threshold (0.5 -> 0.7):
      -> Higer precision (fewer false alarms)
      -> Lower recall (miss more positives)

- Decrease threshold (0.5 -> 0.3):
      -> Higher recall (catch more positivies)
      -> Lower precision (more false alarms)

Business Decision Examples:
    
Scenerio 1: Spam Filter
  Priority: Don't lose important emails (minimize FN)
  Action: Lower threshold → Maximize RECALL
  Trade-off: Accept some spam in inbox
      
Scenerio 2: Fraud Detedction
  Priority: Catch ALL fraud (minimize FN)
  Action: Lower threshold → Maximize RECALL
  Trade-off: More false alerts to investigate
      
Scenario 3: Medical Diagnosis (Cancer)
  Priority: Don't miss any cases (minimize FN)
  Action: Lower threshold → Maximize RECALL
  Trade-off: More follow-up tests (false positives)
      
Scenario 4: Marketing Campaign
  Priority: Only contact interested customers (minimize FP)
  Action: Raise threshold → Maximize PRECISION
  Trade-off: Miss some potential customers
""")

# VISUALIZATIONS

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: Confusion Matrix Heatmap
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Repay (0)', 'Default (1)'],
            yticklabels=['Repay (0)', 'Default (1)'],
            ax=ax1, annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_ylabel('Actual', fontweight='bold', fontsize=12)
ax1.set_xlabel('Predicted', fontweight='bold', fontsize=12)
ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=14)

# Add labels
ax1.text(0.5, 0.25, f'TN\n{tn}', ha='center', va='center',
         fontsize=10, color='darkblue', weight='bold')
ax1.text(1.5, 0.25, f'FP\n{fp}', ha='center', va='center',
         fontsize=10, color='darkred', weight='bold')
ax1.text(0.5, 1.25, f'FN\n{fn}', ha='center', va='center',
         fontsize=10, color='darkred', weight='bold')
ax1.text(1.5, 1.25, f'TP\n{tp}', ha='center', va='center',
         fontsize=10, color='darkblue', weight='bold')

# Plot 2: Metrics Comparison
ax2 = fig.add_subplot(gs[0, 1])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax2.barh(metrics, values, color=colors, edgecolor='black', linewidth=2)
ax2.set_xlim(0, 1)
ax2.set_xlabel('Score', fontweight='bold', fontsize=12)
ax2.set_title('All Metrics Comparison', fontweight='bold', fontsize=14)
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(values):
    ax2.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=11)

# Plot 3: ROC Curve
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, threshold_roc = roc_curve(y_test, y_pred_proba)
ax3.plot(fpr, tpr, linewidth=3, color='#2ecc71', label=f'ROC (AUC={roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
ax3.fill_between(fpr, tpr, alpha=0.3, color='#2ecc71')
ax3.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax3.set_ylabel('True Positive Rate (Recall)', fontweight='bold', fontsize=12)
ax3.set_title('ROC Curve', fontweight='bold', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Precision-Recall Curve
ax4 = fig.add_subplot(gs[1, 0])
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
ax4.plot(recalls, precisions, linewidth=3, color='#e74c3c')
ax4.set_xlabel('Recall', fontweight='bold', fontsize=12)
ax4.set_ylabel('Precision', fontweight='bold', fontsize=12)
ax4.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
ax4.grid(True, alpha=0.3)
ax4.fill_between(recalls, precisions, alpha=0.3, color='#e74c3c')

# Plot 5: Threshold Impact
ax5 = fig.add_subplot(gs[1, 1:])
thresholds_test = np.linspace(0, 1, 100)
precisions_list = []
recalls_list = []
f1_scores_list = []

for thresh in thresholds_test:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    if y_pred_thresh.sum() > 0: # Avoid division by zero
        p = precision_score(y_test, y_pred_thresh, zero_division=0)
        r = recall_score(y_test, y_pred_thresh, zero_division=0)
        f = f1_score(y_test, y_pred_thresh, zero_division=0)
    else:
        p, r, f = 0, 0, 0
    precisions_list.append(p)
    recalls_list.append(r)
    f1_scores_list.append(f1)

ax5.plot(thresholds_test, precisions_list, linewidth=2, label='Precision', color='#3498db')
ax5.plot(thresholds_test, recalls_list, linewidth=2, label='Recall', color='#e74c3c')
ax5.plot(thresholds_test, f1_scores_list, linewidth=2, label='F1-Score', color='#2ecc71')
ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Default (0.5)')
ax5.set_xlabel('Decision Threshold', fontweight='bold', fontsize=12)
ax5.set_ylabel('Score', fontweight='bold', fontsize=12)
ax5.set_title('Impact of Threshold on Metrics', fontweight='bold', fontsize=14)
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# Plot 6: Probablity Distribution
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(y_pred_proba[y_test==0], bins=30, alpha=0.6, label='Actual: Repay',
         color='blue', edgecolor='black')
ax6.hist(y_pred_proba[y_test==1], bins=30, alpha=0.6, label='Actual: Default',
         color='red', edgecolor='black')
ax6.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax6.set_xlabel('Predicted Probability', fontweight='bold', fontsize=12)
ax6.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax6.set_title('Probability Distribution by Class', fontweight='bold', fontsize=14)
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Classification Report as Tables
ax7 =fig.add_subplot(gs[2, 1:])
ax7.axis('off')

report = classification_report(y_test, y_pred, target_names=['Repay', 'Default'],
                               output_dict=True)
report_df = pd.DataFrame(report).T

# Create table
table_data = []
for idx, row in report_df.iterrows():
    if idx in ['Repay', 'Default', 'accuracy', 'macro avg', 'weighted avg']:
        if idx == 'accuracy':
            table_data.append([idx, '', '', f"{row['precision']:.3f}", ''])
        else:
            table_data.append([
                idx, 
                f"{row.get('precision', 0):.3f}",
                f"{row.get('recall', 0):.3f}",
                f"{row.get('f1-score', 0):.3f}",
                f"{int(row.get('support', 0))}" if 'support' in row else ''
            ])

table = ax7.table(cellText=table_data,
                  colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax7.set_title('Classification Report', fontweight='bold', fontsize=14, pad=20)
plt.suptitle('COMPREHENSIVE CLASSIFICATION METRICS DASHBOARD', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('04_classification_metrics_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 04_classification_metrics_dashboard.png")

print("\n" + "="*70)
print("CLASSIFICATION METRICS COMPLETE!")
print("="*70)