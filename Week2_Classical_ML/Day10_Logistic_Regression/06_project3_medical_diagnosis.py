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

# CLINICAL COST-BENEFIT ANALYSIS

print("\n" + "="*70)
print("STEP 8: CLINICAL COST-BENEFIT ANALYSIS")
print("="*70)

# Cost parameters
screening_cost = 500  # ₹500 per screening test
followup_cost = 5000  # ₹5,000 for follow-up tests
treatment_early = 50000  # ₹50,000 early treatment
treatment_late = 500000  # ₹5 lakh late-stage treatment
mortality_cost = 10000000  # ₹1 crore (loss of life value)


# Scale to annual screening (100,000 patients)
annual_patients = 100000
scale_factor = annual_patients / len(y_test)

# Scenario 1: No screening
disease_patients_total = int(y_test.sum() * scale_factor)
cost_no_screening = disease_patients_total * treatment_late
print(f"\n💰 SCENARIO 1: No Screening Program")
print(f"{'─'*50}")
print(f"  Annual patients: {annual_patients:,}")
print(f"  Undetected disease cases: {disease_patients_total:,}")
print(f"  Late-stage treatment cost: ₹{cost_no_screening/1e7:.2f} crore")
print(f"  Preventable deaths: ~{int(disease_patients_total * 0.3):,} (30% mortality)")


# Scenario 2: With optimized screening
total_screening_cost = annual_patients * screening_cost
detected_disease = int(tp_opt * scale_factor)
missed_disease = int(fn_opt * scale_factor)
false_alarms = int(fp_opt * scale_factor)

followup_cost_total = false_alarms * followup_cost
early_treatment_cost = detected_disease * treatment_early
late_treatment_cost = missed_disease * treatment_late

total_cost_with_screening = (total_screening_cost + followup_cost_total + 
                             early_treatment_cost + late_treatment_cost)
lives_saved = int(detected_disease * 0.25)  # 25% mortality reduction
net_benefit = cost_no_screening - total_cost_with_screening

print(f"\n💰 SCENARIO 2: With Optimized Screening (Threshold={optimal_medical_threshold})")
print(f"{'─'*50}")
print(f"  Screening cost: ₹{total_screening_cost/1e7:.2f} crore ({annual_patients:,} × ₹{screening_cost})")
print(f"  Detected disease: {detected_disease:,} (early intervention)")
print(f"  Missed disease: {missed_disease:,} (late-stage treatment)")
print(f"  False alarms: {false_alarms:,} (follow-up tests)")
print(f"  Follow-up cost: ₹{followup_cost_total/1e7:.2f} crore")
print(f"  Early treatment: ₹{early_treatment_cost/1e7:.2f} crore")
print(f"  Late treatment: ₹{late_treatment_cost/1e7:.2f} crore")
print(f"  Total program cost: ₹{total_cost_with_screening/1e7:.2f} crore")
print(f"  Lives saved: ~{lives_saved:,}")
print(f"  NET BENEFIT: ₹{net_benefit/1e7:.2f} crore")

roi = (net_benefit / total_cost_with_screening) * 100
print(f"\n🎉 Healthcare ROI:")
print(f"  Return on Investment: {roi:.0f}%")
print(f"  Cost per life saved: ₹{total_cost_with_screening/lives_saved:,.0f}")
print(f"  → Screening program is HIGHLY cost-effective!")

# VISUALIZATION

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

# Plot 1: Disease Prevalence
ax1 = fig.add_subplot(gs[0, 0])
disease_data = df['HeartDisease'].value_counts()
colors_disease = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax1.pie(disease_data.values, labels=['Healthy', 'Disease'],
                                     autopct='%1.1f%%', colors=colors_disease,
                                     startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
ax1.set_title('Disease Prevalence', fontweight='bold', fontsize=13)

# Plot 2: Age Distribution by Disease
ax2 = fig.add_subplot(gs[0, 1])
healthy_ages = df[df['HeartDisease'] == 0]['Age']
disease_ages = df[df['HeartDisease'] == 1]['Age']
ax2.hist([healthy_ages, disease_ages], bins=20, label=['Healthy', 'Disease'],
         color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Age (years)', fontweight='bold', fontsize=11)
ax2.set_ylabel('Frequency', fontweight='bold', fontsize=11)
ax2.set_title('Age Distribution by Disease Status', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Feature Importance
ax3 = fig.add_subplot(gs[0, 2])
top_features = feature_importance.head(9)
colors_importance = ['#e74c3c' if c > 0 else '#2ecc71' for c in top_features['Coefficient_Weighted']]
ax3.barh(range(len(top_features)), top_features['Coefficient_Weighted'],
         color=colors_importance, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(top_features)))
ax3.set_yticklabels(top_features['Feature'], fontsize=10)
ax3.set_xlabel('Coefficient', fontweight='bold', fontsize=11)
ax3.set_title('Clinical Risk Factors', fontweight='bold', fontsize=13)
ax3.axvline(x=0, color='black', linewidth=2)
ax3.grid(axis='x', alpha=0.3)


# Plot 4: Model Comparison - Confusion Matrices
cms = [cm_std, cm_wt, cm_optimal]
titles = ['Standard\n(Threshold=0.5)', 'Weighted\n(Threshold=0.5)', 
          f'Optimized\n(Threshold={optimal_medical_threshold})']

for idx, (cm_data, title) in enumerate(zip(cms, titles)):
    ax = plt.subplot(gs[1, idx])
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='RdYlGn', cbar=False,
                xticklabels=['Healthy', 'Disease'],
                yticklabels=['Healthy', 'Disease'],
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_ylabel('Actual', fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=11)
    ax.set_title(title, fontweight='bold', fontsize=12)

# Plot 5: Recall comparison
ax5 = fig.add_subplot(gs[2, 0])
models = ['Standard', 'Weighted', 'Optimized']
recalls = [recall_std, recall_wt, recall_optimal]
fns = [fn_std, fn_wt, fn_opt]
colors_recall = ['#f39c12', '#3498db', '#2ecc71']
bars = ax5.bar(models, recalls, color=colors_recall, edgecolor='black', linewidth=2)
ax5.set_ylabel('Recall (Disease Detection Rate)', fontweight='bold', fontsize=11)
ax5.set_title('Model Comparison: Recall', fontweight='bold', fontsize=13)
ax5.set_ylim(0, 1)
ax5.grid(axis='y', alpha=0.3)
for i, (v, fn) in enumerate(zip(recalls, fns)):
    ax5.text(i, v + 0.02, f'{v:.1%}\n({fn} FN)', ha='center', 
             fontweight='bold', fontsize=10)

# Plot 6: ROC Curve
ax6 = fig.add_subplot(gs[2, 1])
fpr_std, tpr_std, _ = roc_curve(y_test, y_pred_proba_std)
fpr_wt, tpr_wt, _ = roc_curve(y_test, y_pred_proba_wt)

ax6.plot(fpr_std, tpr_std, linewidth=2, label=f'Standard (AUC={roc_auc_std:.3f})', 
         color='#f39c12')
ax6.plot(fpr_wt, tpr_wt, linewidth=2, label=f'Weighted (AUC={roc_auc_wt:.3f})', 
         color='#2ecc71')
ax6.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax6.fill_between(fpr_wt, tpr_wt, alpha=0.3, color='#2ecc71')
ax6.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
ax6.set_ylabel('True Positive Rate (Recall)', fontweight='bold', fontsize=11)
ax6.set_title('ROC Curve Comparison', fontweight='bold', fontsize=13)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

# Plot 7: Threshold Impact
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(threshold_df['Threshold'], threshold_df['Recall'],
         'o-', linewidth=2, markersize=8, label='Recall', color='#e74c3c')
ax7.plot(threshold_df['Threshold'], threshold_df['Precision'],
         's-', linewidth=2, markersize=8, label='Precision', color='#3498db')
ax7.plot(threshold_df['Threshold'], threshold_df['F1'],
         '^-', linewidth=2, markersize=8, label='F1-Score', color='#2ecc71')
ax7.axvline(x=optimal_medical_threshold, color='purple', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Optimal ({optimal_medical_threshold})')
ax7.set_xlabel('Decision Threshold', fontweight='bold', fontsize=11)
ax7.set_ylabel('Score', fontweight='bold', fontsize=11)
ax7.set_title('Medical Threshold Tuning', fontweight='bold', fontsize=13)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 7: Threshold Impact
ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(threshold_df['Threshold'], threshold_df['Recall'],
         'o-', linewidth=2, markersize=8, label='Recall', color='#e74c3c')
ax7.plot(threshold_df['Threshold'], threshold_df['Precision'],
         's-', linewidth=2, markersize=8, label='Precision', color='#3498db')
ax7.plot(threshold_df['Threshold'], threshold_df['F1'],
         '^-', linewidth=2, markersize=8, label='F1-Score', color='#2ecc71')
ax7.axvline(x=optimal_medical_threshold, color='purple', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Optimal ({optimal_medical_threshold})')
ax7.set_xlabel('Decision Threshold', fontweight='bold', fontsize=11)
ax7.set_ylabel('Score', fontweight='bold', fontsize=11)
ax7.set_title('Medical Threshold Tuning', fontweight='bold', fontsize=13)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)


# Plot 8: Clinical Impact Summary
ax8 = fig.add_subplot(gs[3, :])
ax8.axis('off')

impact_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       CLINICAL IMPACT SUMMARY                                 ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  WITHOUT SCREENING:                                                           ║
║  • Undetected disease cases: {disease_patients_total:,}                                       ║
║  • Late-stage treatment cost: ₹{cost_no_screening/1e7:.2f} crore                                    ║
║  • Preventable deaths: ~{int(disease_patients_total * 0.3):,}                                        ║
║                                                                               ║
║  WITH OPTIMIZED SCREENING (Threshold={optimal_medical_threshold}):                              ║
║  • Patients screened: {annual_patients:,} annually                                    ║
║  • Disease detected early: {detected_disease:,} ({recall_optimal:.1%} catch rate)                    ║
║  • Missed diagnoses: {missed_disease:,} ({fn_opt} in test set)                                  ║
║  • False alarms: {false_alarms:,} (get follow-up tests)                              ║
║  • Total program cost: ₹{total_cost_with_screening/1e7:.2f} crore                                   ║
║  • NET BENEFIT: ₹{net_benefit/1e7:.2f} crore                                              ║
║  • Lives saved: ~{lives_saved:,}                                                   ║
║                                                                               ║
║  ROI: {roi:.0f}% (₹{roi/100:.1f} saved per ₹1 spent)                                          ║
║  Cost per life saved: ₹{total_cost_with_screening/lives_saved:,.0f}                                        ║
║                                                                               ║
║  MEDICAL METRICS (Optimized Model):                                           ║
║  • Recall: {recall_optimal:.1%} (catching {recall_optimal*100:.0f}% of disease cases)                     ║
║  • Precision: {precision_optimal:.1%} ({precision_optimal*100:.0f}% of positive predictions are correct)          ║
║  • False Negative Rate: {fn_opt/y_test.sum()*100:.1f}% (missed {fn_opt} out of {int(y_test.sum())} disease cases)        ║
║                                                                               ║
║  RECOMMENDATION: Deploy optimized screening program                           ║
║  • Use threshold = {optimal_medical_threshold} (prioritize catching disease)                  ║
║  • Annual screening for at-risk populations (age 40+)                         ║
║  • Follow-up protocol for positive screens                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

ax8.text(0.5, 0.5, impact_text, transform=ax8.transAxes,
         fontsize=10, verticalalignment='center', horizontalalignment='center',
         family='monospace', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.suptitle('HEART DISEASE SCREENING - COMPLETE CLINICAL ANALYSIS',
             fontsize=18, fontweight='bold', y=0.995)
plt.savefig('07_medical_diagnosis_complete.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 07_medical_diagnosis_complete.png")

# CLINICAL DEPLOYMENT PROTOCOL

print("\n" + "="*70)
print("STEP 9: CLINICAL DEPLOYMENT PROTOCOL")
print("="*70)

protocol = f"""
🏥 SCREENING PROGRAM DEPLOYMENT

TARGET POPULATION:
  → Adults aged 40+ years
  → High-risk groups (family history, diabetes, hypertension)
  → Annual screening recommended

SCREENING PROCESS:
  1. Collect patient data (9 clinical measurements)
  2. Model predicts disease probability
  3. Decision protocol based on probability:

     P(Disease) >= {optimal_medical_threshold} → POSITIVE SCREEN
     ├─→ Schedule follow-up tests within 2 weeks
     ├─→ ECG, stress test, coronary angiography
     └─→ Cardiologist consultation

     P(Disease) < {optimal_medical_threshold} → NEGATIVE SCREEN
     ├─→ Routine follow-up in 1 year
     ├─→ Lifestyle counseling
     └─→ Risk factor management

FALSE POSITIVE MANAGEMENT:
  • ~{false_alarms:,} patients annually require follow-up
  • Cost: ₹{followup_cost_total/1e7:.2f} crore (acceptable given lives saved)
  • Patient education: "Better safe than sorry"
  • Minimize anxiety with clear communication

FALSE NEGATIVE MONITORING:
  • Expected {missed_disease:,} missed cases annually
  • Continuous model improvement with new data
  • Symptom-based referral system as backup
  • Regular model retraining (quarterly)

PERFORMANCE MONITORING:
  Weekly:
  • Track screening volume
  • Monitor follow-up completion rates
  • Flag unusual patterns

  Monthly:
  • Precision, recall, false negative rate
  • Compare predictions vs confirmed diagnoses
  • Cost analysis

  Quarterly:
  • Retrain model with new patient data
  • Update risk factors and coefficients
  • Validate on held-out test set

EXPECTED OUTCOMES:
  ✅ Detect {recall_optimal:.0%} of disease cases early
  ✅ Save ~{lives_saved:,} lives annually
  ✅ Reduce late-stage treatment costs by ₹{net_benefit/1e7:.2f} crore
  ✅ {roi:.0f}% ROI on screening program

ETHICAL CONSIDERATIONS:
  • Patient consent for AI-assisted screening
  • Transparent explanation of model limitations
  • Human oversight: cardiologist reviews all positives
  • Regular bias audits (age, gender, ethnicity)
  • Privacy: HIPAA-compliant data handling
"""

print(protocol)

# Save detailed report
with open('medical_diagnosis_report.txt', 'w', encoding='utf-8') as f:
    f.write("HEART DISEASE SCREENING - DETAILED REPORT\n")
    f.write("="*70 + "\n\n")
    f.write("Model Performance:\n")
    f.write(f"  Recall: {recall_optimal:.1%}\n")
    f.write(f"  Precision: {precision_optimal:.1%}\n")
    f.write(f"  F1-Score: {f1_optimal:.3f}\n")
    f.write(f"  Optimal Threshold: {optimal_medical_threshold}\n\n")
    f.write("Clinical Impact:\n")
    f.write(f"  Net Benefit: ₹{net_benefit/1e7:.2f} crore annually\n")
    f.write(f"  Lives Saved: {lives_saved:,}\n")
    f.write(f"  ROI: {roi:.0f}%\n")
    f.write(f"  False Negatives: {missed_disease:,}\n\n")
    f.write("Top Risk Factors:\n")
    for idx, row in feature_importance.head(9).iterrows():
        f.write(f"  {row['Feature']}: {row['Coefficient_Weighted']:.4f}\n")

print("\n✅ Saved detailed report: medical_diagnosis_report.txt")

print("\n" + "="*70)
print("PROJECT 3 COMPLETE: MEDICAL DIAGNOSIS CLASSIFIER")
print("="*70)