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

