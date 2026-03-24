"""
PROJECT 3: MEDICAL DIAGNOSIS CLASSIFIER
========================================
Predict disease presence from patient health metrics

Business Context:
- Healthcare provider screening for heart disease
- Early detection saves lives and reduces treatment costs
- High stakes: False Negatives (missed diagnoses) are DANGEROUS
- Goal: Maximize recall while maintaining reasonable precision
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
print("PROJECT 3: MEDICAL DIAGNOSIS CLASSIFIER")
print("="*70)

# MEDICAL PROBLEM STATEMENT

print("""
╔══════════════════════════════════════════════════════════════╗
║           MEDICAL PROBLEM: HEART DISEASE DETECTION           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Healthcare Provider: CardioHealth India                     ║
║  Problem: Early detection of heart disease risk              ║
║                                                              ║
║  Why This Matters:                                           ║
║  • Heart disease: #1 cause of death in India (28% deaths)   ║
║  • Early detection → 80% reduction in mortality              ║
║  • Average treatment cost: ₹5 lakh (late stage)             ║
║  • Prevention cost: ₹50,000 (early intervention)            ║
║                                                              ║
║  Clinical Stakes:                                            ║
║  ┌────────────────────────────────────────────────────────┐ ║
║  │ FALSE NEGATIVE (Type II Error):                       │ ║
║  │ → Patient has disease, model says healthy             │ ║
║  │ → Patient doesn't get treatment                       │ ║
║  │ → POTENTIALLY FATAL! ☠️                                │ ║
║  │ → This is the WORST outcome                           │ ║
║  │                                                        │ ║
║  │ FALSE POSITIVE (Type I Error):                        │ ║
║  │ → Healthy patient, model says disease                 │ ║
║  │ → Patient gets follow-up tests                        │ ║
║  │ → Some anxiety, but NOT dangerous                     │ ║
║  │ → Acceptable trade-off                                │ ║
║  └────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  Success Criteria:                                           ║
║  • Recall > 95% (catch nearly ALL diseases)                 ║
║  • Precision > 70% (minimize unnecessary tests)             ║
║  • Principle: "Better safe than sorry"                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# GENERATE REALISTIC MEDICAL DATA

print("\n" + "="*70)
print("STEP 1: PATIENT DATA GENERATION")
print("="*70)

np.random.seed(42)
n_patients = 1000

print(f"Generating medical records for {n_patients} patients...")

# Patient demographics
age = np.random.normal(54, 10, n_patients).clip(30, 80)
sex = np.random.choice([0, 1], n_patients, p=[0.32, 0.68])  # 0=Female, 1=Male

# Clinical measurements
resting_bp = np.random.normal(130, 20, n_patients).clip(90, 200)  # Blood pressure (mm Hg)
cholesterol = np.random.normal(240, 50, n_patients).clip(150, 400)  # mg/dl
fasting_bs = np.random.choice([0, 1], n_patients, p=[0.85, 0.15])  # Fasting blood sugar > 120 mg/dl
max_hr = np.random.normal(150, 25, n_patients).clip(70, 200)  # Maximum heart rate
exercise_angina = np.random.choice([0, 1], n_patients, p=[0.68, 0.32])  # Exercise induced angina
oldpeak = np.random.exponential(1, n_patients).clip(0, 6)  # ST depression
num_vessels = np.random.choice([0, 1, 2, 3], n_patients, p=[0.6, 0.2, 0.15, 0.05])  # Coronary arteries

# Calculate disease probability (complex realistic relationship)
disease_logit = (
    -8.0 +  # Base (healthy baseline)
    0.05 * age +  # Age increases risk
    0.8 * sex +  # Males higher risk
    0.01 * resting_bp +  # High BP increases risk
    0.005 * cholesterol +  # High cholesterol increases risk
    0.5 * fasting_bs +  # High blood sugar increases risk
    -0.015 * max_hr +  # Lower max HR = higher risk
    1.2 * exercise_angina +  # Exercise angina = strong indicator
    0.6 * oldpeak +  # ST depression indicates risk
    0.8 * num_vessels  # More affected vessels = higher risk
)

disease_probability = 1 / (1 + np.exp(-disease_logit))
has_disease = (disease_probability > np.random.uniform(0, 1, n_patients)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'PatientID': [f'P{i:04d}' for i in range(1, n_patients + 1)],
    'Age': age.round(0).astype(int),
    'Sex': sex,  # 0=Female, 1=Male
    'RestingBP': resting_bp.round(0).astype(int),
    'Cholesterol': cholesterol.round(0).astype(int),
    'FastingBS': fasting_bs,  # 0=Normal, 1=High
    'MaxHR': max_hr.round(0).astype(int),
    'ExerciseAngina': exercise_angina,  # 0=No, 1=Yes
    'Oldpeak': oldpeak.round(1),
    'NumVessels': num_vessels,
    'HeartDisease': has_disease  # 0=Healthy, 1=Disease
})

print(f"✅ Generated dataset: {len(df)} patient records")
print(f"\nFirst 10 patients:")
print(df.head(10))

# EXPLORATORY DATA ANALYSIS

print("\n" + "="*70)
print("STEP 2: CLINICAL DATA ANALYSIS")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"  Total patients: {len(df):,}")
print(f"  Clinical features: {df.shape[1] - 2}")  # Exclude PatientID and target

print(f"\n🏥 Disease Prevalence:")
disease_counts = df['HeartDisease'].value_counts()
disease_rate = df['HeartDisease'].mean()
print(f"  Healthy (0):  {disease_counts[0]:,} ({(1-disease_rate)*100:.1f}%)")
print(f"  Disease (1):  {disease_counts[1]:,} ({disease_rate*100:.1f}%)")
print(f"  Prevalence rate: {disease_rate:.1%}")

if 0.3 < disease_rate < 0.7:
    print(f"  ✅ Relatively balanced dataset")

print(f"\n📋 Clinical Statistics by Disease Status:")

# Age analysis
healthy_age = df[df['HeartDisease'] == 0]['Age'].mean()
disease_age = df[df['HeartDisease'] == 1]['Age'].mean()
print(f"\nAge:")
print(f"  Healthy patients:  {healthy_age:.1f} years")
print(f"  Disease patients:  {disease_age:.1f} years")
print(f"  → Disease patients are {disease_age - healthy_age:.1f} years older on average")

# Gender analysis
male_disease_rate = df[df['Sex'] == 1]['HeartDisease'].mean()
female_disease_rate = df[df['Sex'] == 0]['HeartDisease'].mean()
print(f"\nGender:")
print(f"  Male disease rate:   {male_disease_rate:.1%}")
print(f"  Female disease rate: {female_disease_rate:.1%}")
print(f"  → Males {male_disease_rate/female_disease_rate:.1f}x more likely to have disease")

# Blood pressure
healthy_bp = df[df['HeartDisease'] == 0]['RestingBP'].mean()
disease_bp = df[df['HeartDisease'] == 1]['RestingBP'].mean()
print(f"\nResting Blood Pressure:")
print(f"  Healthy:  {healthy_bp:.0f} mm Hg")
print(f"  Disease:  {disease_bp:.0f} mm Hg")

# Cholesterol
healthy_chol = df[df['HeartDisease'] == 0]['Cholesterol'].mean()
disease_chol = df[df['HeartDisease'] == 1]['Cholesterol'].mean()
print(f"\nCholesterol:")
print(f"  Healthy:  {healthy_chol:.0f} mg/dl")
print(f"  Disease:  {disease_chol:.0f} mg/dl")

# Exercise angina (strong indicator)
angina_disease_rate = df[df['ExerciseAngina'] == 1]['HeartDisease'].mean()
no_angina_disease_rate = df[df['ExerciseAngina'] == 0]['HeartDisease'].mean()
print(f"\nExercise Angina:")
print(f"  With angina:    {angina_disease_rate:.1%} disease rate")
print(f"  Without angina: {no_angina_disease_rate:.1%} disease rate")
print(f"  → Angina is a strong risk indicator!")

# DATA PREPARATION

print("\n" + "="*70)
print("STEP 3: DATA PREPARATION")
print("="*70)

# Select features
feature_cols = [
    'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS',
    'MaxHR', 'ExerciseAngina', 'Oldpeak', 'NumVessels'
]

X = df[feature_cols].values
y = df['HeartDisease'].values

print(f"Features selected: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} patients")
print(f"Test set:  {len(X_test)} patients")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features standardized")

# MODEL TRAINING

print("\n" + "="*70)
print("STEP 4: MODEL TRAINING")
print("="*70)

# Train standard model
print("Training standard Logistic Regression...")
model_standard = LogisticRegression(max_iter=1000, random_state=42)
model_standard.fit(X_train_scaled, y_train)

# Train with class weights (prefer catching disease)
print("Training weighted model (prioritize disease detection)...")
model_weighted = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 2},  # 2x penalty for missing disease
    random_state=42
)
model_weighted.fit(X_train_scaled, y_train)

print("✅ Models trained!")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient_Standard': model_standard.coef_[0],
    'Coefficient_Weighted': model_weighted.coef_[0],
    'Abs_Coefficient': np.abs(model_weighted.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n📊 Clinical Risk Factors (Feature Importance):")
print(f"{'Feature':>20} {'Coefficient':>15} {'Clinical Impact':>30}")
print("-"*70)
for idx, row in feature_importance.head(9).iterrows():
    impact = 'Increases Risk' if row['Coefficient_Weighted'] > 0 else 'Protective Factor'
    print(f"{row['Feature']:>20} {row['Coefficient_Weighted']:>15.4f} {impact:>30}")

# MODEL EVALUATION - STANDARD

print("\n" + "="*70)
print("STEP 5: MODEL EVALUATION - STANDARD MODEL")
print("="*70)

y_pred_std = model_standard.predict(X_test_scaled)
y_pred_proba_std = model_standard.predict_proba(X_test_scaled)[:, 1]

accuracy_std = accuracy_score(y_test, y_pred_std)
precision_std = precision_score(y_test, y_pred_std)
recall_std = recall_score(y_test, y_pred_std)
f1_std = f1_score(y_test, y_pred_std)
roc_auc_std = roc_auc_score(y_test, y_pred_proba_std)

cm_std = confusion_matrix(y_test, y_pred_std)
tn_std, fp_std, fn_std, tp_std = cm_std.ravel()

print(f"\n📊 STANDARD MODEL PERFORMANCE:")
print(f"{'─'*50}")
print(f"  Accuracy:  {accuracy_std:.1%}")
print(f"  Precision: {precision_std:.1%}")
print(f"  Recall:    {recall_std:.1%} ← KEY METRIC!")
print(f"  F1-Score:  {f1_std:.3f}")
print(f"  ROC-AUC:   {roc_auc_std:.3f}")

print(f"\n🏥 Clinical Interpretation:")
print(f"                PREDICTED")
print(f"              ┌──────────┬──────────┐")
print(f"              │ Healthy  │ Disease  │")
print(f"    ┌─────────┼──────────┼──────────┤")
print(f"  A │ Healthy │   {tn_std:>4}   │   {fp_std:>4}   │")
print(f"  C │  (0)    │   (TN)   │   (FP)   │")
print(f"  T ├─────────┼──────────┼──────────┤")
print(f"  U │ Disease │   {fn_std:>4}   │   {tp_std:>4}   │")
print(f"  A │  (1)    │   (FN)   │   (TP)   │")
print(f"  L └─────────┴──────────┴──────────┘")

print(f"\nClinical Outcomes:")
print(f"  ✅ Correctly diagnosed {tp_std} disease cases (True Positives)")
print(f"  ✅ Correctly cleared {tn_std} healthy patients (True Negatives)")
print(f"  ❌ MISSED {fn_std} disease cases (False Negatives) ☠️ DANGEROUS!")
print(f"  ⚠️  {fp_std} false alarms (False Positives - unnecessary follow-up)")

if fn_std > 0:
    print(f"\n  ⚠️  CRITICAL: {fn_std} patients with disease sent home!")
    print(f"     These patients may develop serious complications.")
    print(f"     False Negatives are UNACCEPTABLE in medical diagnosis!")

# MODEL EVALUATION - WEIGHTED

print("\n" + "="*70)
print("STEP 6: MODEL EVALUATION - WEIGHTED MODEL")
print("="*70)

y_pred_wt = model_weighted.predict(X_test_scaled)
y_pred_proba_wt = model_weighted.predict_proba(X_test_scaled)[:, 1]

accuracy_wt = accuracy_score(y_test, y_pred_wt)
precision_wt = precision_score(y_test, y_pred_wt)
recall_wt = recall_score(y_test, y_pred_wt)
f1_wt = f1_score(y_test, y_pred_wt)
roc_auc_wt = roc_auc_score(y_test, y_pred_proba_wt)

cm_wt = confusion_matrix(y_test, y_pred_wt)
tn_wt, fp_wt, fn_wt, tp_wt = cm_wt.ravel()

print(f"\n📊 WEIGHTED MODEL PERFORMANCE:")
print(f"{'─'*50}")
print(f"  Accuracy:  {accuracy_wt:.1%}")
print(f"  Precision: {precision_wt:.1%}")
print(f"  Recall:    {recall_wt:.1%} ← IMPROVED!")
print(f"  F1-Score:  {f1_wt:.3f}")
print(f"  ROC-AUC:   {roc_auc_wt:.3f}")

print(f"\n🏥 Clinical Interpretation:")
print(f"                PREDICTED")
print(f"              ┌──────────┬──────────┐")
print(f"              │ Healthy  │ Disease  │")
print(f"    ┌─────────┼──────────┼──────────┤")
print(f"  A │ Healthy │   {tn_wt:>4}   │   {fp_wt:>4}   │")
print(f"  C │  (0)    │   (TN)   │   (FP)   │")
print(f"  T ├─────────┼──────────┼──────────┤")
print(f"  U │ Disease │   {fn_wt:>4}   │   {tp_wt:>4}   │")
print(f"  A │  (1)    │   (FN)   │   (TP)   │")
print(f"  L └─────────┴──────────┴──────────┘")

print(f"\nImprovement over standard model:")
print(f"  False Negatives: {fn_std} → {fn_wt} (reduction of {fn_std - fn_wt})")
print(f"  Recall: {recall_std:.1%} → {recall_wt:.1%} (improvement of {(recall_wt - recall_std)*100:.1f} percentage points)")

if fn_wt < fn_std:
    print(f"  ✅ BETTER! Fewer missed diagnoses!")
else:
    print(f"  ⚠️  No improvement in missed diagnoses")

# OPTIMAL THRESHOLD FOR MEDICAL USE

print("\n" + "="*70)
print("STEP 7: OPTIMAL THRESHOLD TUNING FOR MEDICAL USE")
print("="*70)


print("""
Medical Principle: "First, do no harm"
→ Prioritize catching disease (high recall)
→ Accept more false alarms as trade-off
→ Better safe than sorry!
""")

# Test thresholds
thresholds_medical = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
threshold_results = []

for thresh in thresholds_medical:
    y_pred_thresh = (y_pred_proba_wt >= thresh).astype(int)
    
    if y_pred_thresh.sum() > 0:
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        cm_thresh = confusion_matrix(y_test, y_pred_thresh)
        tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
        
        threshold_results.append({
            'Threshold': thresh,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'FN': fn_t,
            'FP': fp_t
        })

threshold_df = pd.DataFrame(threshold_results)

print(f"\n📊 Threshold Analysis:")
print(threshold_df.to_string(index=False))

# Choose threshold that maximizes recall while keeping precision > 70%
medical_threshold_candidates = threshold_df[threshold_df['Precision'] >= 0.70]
if len(medical_threshold_candidates) > 0:
    optimal_medical_idx = medical_threshold_candidates['Recall'].idxmax()
    optimal_medical_threshold = medical_threshold_candidates.iloc[optimal_medical_idx]['Threshold']
    optimal_medical_recall = medical_threshold_candidates.iloc[optimal_medical_idx]['Recall']
    optimal_medical_fn = int(medical_threshold_candidates.iloc[optimal_medical_idx]['FN'])
else:
    # If no threshold meets precision requirement, maximize recall
    optimal_medical_idx = threshold_df['Recall'].idxmax()
    optimal_medical_threshold = threshold_df.iloc[optimal_medical_idx]['Threshold']
    optimal_medical_recall = threshold_df.iloc[optimal_medical_idx]['Recall']
    optimal_medical_fn = int(threshold_df.iloc[optimal_medical_idx]['FN'])

print(f"\n✅ OPTIMAL MEDICAL THRESHOLD: {optimal_medical_threshold}")
print(f"   Recall: {optimal_medical_recall:.1%}")
print(f"   False Negatives: {optimal_medical_fn}")

# Use optimal threshold
y_pred_optimal = (y_pred_proba_wt >= optimal_medical_threshold).astype(int)
precision_optimal = precision_score(y_test, y_pred_optimal)
recall_optimal = recall_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)

cm_optimal = confusion_matrix(y_test, y_pred_optimal)
tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()

print(f"\n📊 OPTIMIZED MODEL (Threshold={optimal_medical_threshold}):")
print(f"{'─'*50}")
print(f"  Precision: {precision_optimal:.1%}")
print(f"  Recall:    {recall_optimal:.1%} ← Maximized!")
print(f"  F1-Score:  {f1_optimal:.3f}")
print(f"  False Negatives: {fn_opt} (missed diagnoses)")
print(f"  False Positives: {fp_opt} (extra follow-ups)")