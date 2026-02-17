import pandas as pd
import numpy as np

print("="*70)
print("PHASES 1 & 2: UNDERSTAND DATA + QUALITY ASSESSMENT")
print("="*70)

#Load dataset
df = pd.read_csv('titanic.csv')

 # PHASE 1: UNDERSTAND THE DATA

print("\n" + "="*70)
print("1.1 BASIC INFORMATION")
print("="*70)

print(f"\n📊 Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\n📋 Column Names:\n{df.columns.to_list()}")

print(f"\n🔤 Data Types:")
print(df.dtypes)

print(f"\n🎯 Target Variable: 'Survived'")
print(f"   0 = Did not survive ")
print(f"   1 = Survived ")
print()

 # BUSINESS CONTEXT

print("="*70)
print("1.2 BUSINESS CONTEXT (FEATURE DICTIONARY)")
print("="*70)

feature_dict = {
    'PassengerId': 'Unique ID for each paassenger',
    'Survived': 'TARGET: 0=No, 1=Yes',
    'Pclass': 'Ticket class: 1=1st, 2=2nd, 3=3rd',
    'Name': 'Full name including title',
    'Sex': 'Gender: male/female',
    'Age': 'Age in years (fractional is < 1)',
    'SibSp': 'Number of siblings/spouse aboard',
    'Parch': 'Number of Parents/children aboard',
    'Ticket': 'Ticket number',
    'Fare': 'Passenger fare in British pounds',
    'Cabin': 'Cabin number (mostly missing)',
    'Embarked': 'Port: C=Cherbourg, Q=Queenstowm, S=Southampton'
}

for col, desc in feature_dict.items():
    print(f"  {col:15s}: {desc}")
print()

 # STATICTICAL SUMMARY

print("="*70)
print("1.3 STATISTICAL SUMMARY")
print("="*70)

print("\nNumeric Columns: ")
print(df.describe().round(2).to_string())

print("\nCategorical Columns: ")
print(df.describe(include=['object']).to_string())

# TARGET VARIBALE ANALYSIS

print("\n" + "="*70)
print("1.4 TARGET VARIABLE: SURVIVAL ANALYSIS")
print("="*70)

survival_counts = df['Survived'].value_counts()
survival_pct = df['Survived'].value_counts(normalize=True) * 100

print(f"\nSurvival Distribution: ")
print(f"  Survived (1):    {survival_counts[1]:4d} passengers  ({survival_pct[1]:.1f}%)")
print(f"  Survived (0):    {survival_counts[0]:4d} passengers  ({survival_pct[0]:.1f}%)")

if abs(survival_pct[1] - 50) > 20:
    print("⚠️  CLASS IMBALANCE DETECTED! ")
    print("   When training ML model, consider: ")
    print("   - Oversampling minority class (SMOTE) ")
    print("   - Undersampling majority class")
    print("   - Using class_weight='balanced'")
    print("    - Using AUC-ROC instead of accuracy")

else:
    print("✅ Classes are reasonably balanced")
print()

 # PHASE 2: DATA QUALITY ASSESSMENT

print("="*70)
print("PHASE 2: DATA QUALITY ASSESSMENT")
print("="*70)

# 2.1 Missing Values
print("\n2.1 MISSING VALUES ANALYSIS")
print("-"*50)

missing = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percent': (df.isnull().sum() / len(df) *100).round(2).values,
    'Dtype': df.dtypes.values
})
missing = missing[missing['Missing_Count'] > 0].sort_values(
    'Missing_Count', ascending=False
)

print(missing.to_string(index=False))
print()

# Categorize columns by missing %
for _, row in missing.iterrows():
    pct = row['Missing_Percent']
    col = row['Column']
    if pct > 70:
        action = "❌ DROP column (too many missing)"
    elif pct > 30:
        action = "⚠️  Consider dropping OR careful imputation"
    elif pct > 0:
        action = "✅ Impute (median/mode/model-based)"
    print(f" {col}: {pct}% missing -> {action}")
print()

# 2.2. Duplicates
print("2.2 DUPLICATE DETECTION")
print("-"*50)
duplicates = df.duplicated().sum()
if duplicates > 0:
    print("Duplicate rows: ")
    print(df[df.duplicated(keep=False)])
else:
    print("✅ No duplicate rows found")
print()

# 2.3 Data Type Validation
print("2.3 DATA TYPE VALIDATION")
print("-"*50)

expected_types={
    'PassengerId': 'int64',
    'Survived': 'int64',
    'Pclass': 'int64',
    'Age': 'float64',
    'SibSp': 'int64',
    'Parch': 'int64',
    'Fare': 'float64'
}

for col, expected in expected_types.items():
    actual = str(df[col].dtype)
    status = "✅" if actual == expected else "⚠️ "
    print(f"  {col:15s}: expected={expected:10s} actual={actual} {status}")
print()

# 2.4 Invalid Values
print("2.4 INVALID VALUES CHECK: ")
print("-"*50)

# Age Validation
age_issues = df[df['Age'].notna() & ((df['Age'] < 0) | (df['Age'] > 120))]
print(f"age issues (< 0 or > 120): {len(age_issues)}")

# Fare validation
fare_issues = df[df['Fare'].notna() & (df['Fare'] < 0)]
print(f"Negative fares: {len(fare_issues)}")

# Pclass validation
pclass_issues = df[~df['Pclass'].isin([1, 2, 3])]
print(f"Invalid Pclass: {len(pclass_issues)}")

# Sex validation
sex_issues = df[~df['Sex'].isin(['male', 'female'])]
print(f"Invalid Sex Values: {len(sex_issues)}")

# Survived validation
survived_issues = df[~df['Survived'].isin([0, 1])]
print(f"Invalid Survived Values: {len(survived_issues)}")

print("✅ Phase 1 & 2 Complete!")