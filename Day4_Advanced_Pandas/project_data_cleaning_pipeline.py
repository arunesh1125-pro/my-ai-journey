"""
CAPSTONE PROJECT: Professional Data Cleaning Pipeline
Clean a completely messy dataset from raw to ML-ready!
This demonstrates your full data cleaning skills.
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime

print("="*70)
print("PROFESSIONAL DATA CLEANING PIPELINE")
print("="*70)

 # STEP 0: CREATE MESSY DATASET

print("\n📥 STEP 0: Loading Raw Data...")

np.random.seed(42)

# Simulate extremely messy real-world HR dataset
raw_data = {
    'emp_id': ['E001', 'E002', 'E003', 'E002', 'E004', 'E005',
               'E006', None, 'E008', 'E009', 'E010'],
    'employee_name': ['  Alice Johnson  ', 'BOB Smith', 'charlie brown',
                      'BOB Smith', 'DAVID LEE', 'eve Wilson ',
                      'Frank O Brien', 'Grace kim', '  Henry  Park  ',
                      None, 'Jack Chen'],
    'age': ['25', '30', '35', '30', '-5', '28',
            '150', '32', '29', '31', 'twenty-seven'],
    'department': ['IT', 'hr', 'IT', 'hr', 'Finance',
                   'HR', 'it', 'Finance', None, 'IT', 'FINANCE'],
    'salary': ['50,000', '60000', '75,000', '60000', '55000',
               '70,000', '65000', '-1000', '80,000', '72000', None],
    'join_date': ['2021-01-15', '15/02/2021', 'March 3, 2021',
                  '15/02/2021', '04-04-2021', '2021-05-10',
                  '06/2021/15', 'July 7 2021', '2021-08-20',
                  '2021-09-01', '2021-10-15'],
    'email': ['alice.j@company.com', 'BOB.smith@Company.COM',
              'charlie.b@company.com', 'BOB.smith@Company.COM',
              'not_an_email', 'eve.w@company.com',
              'frank.ob@company.com', 'grace.k@company.com',
              'henry.p@company.com', None, 'jack.c@company.com'],
    'phone': ['+91 98765 43210', '9876543211', '(987) 654-3212',
              '9876543211', '98765-43213', '91-9876543214',
              'not-a-phone', '9876543216', '9876543217',
              '9876543218', '9876543219'],
    'performance_score': [4.5, 3.8, 4.2, 3.8, None, 4.7,
                          3.5, 4.0, None, 3.9, 4.1],
    'city': ['Chennai', 'MUMBAI', ' Delhi ', 'MUMBAI', 'bangalore',
             'Kolkata', 'HYDERABAD', 'Pune ', ' Chennai ', 'Mumbai', 'Delhi']
}

df_raw = pd.DataFrame(raw_data)
print(f"Raw dataset: {df_raw.shape}")
print(df_raw)
print()

 # STEP 1: ASSESSMENT REPORT

print("\n" + "="*70)
print("📊 STEP 1: DATA QUALITY ASSESSMENT REPORT")
print("="*70)

def assess_data_quality(df):
    """Generate data quality report"""
    report = []

    for col in df.columns:
        col_data = df[col]
        report.append({
            'Column':col,
            'Type': str(col_data.dtype),
            'Total': len(col_data),
            'Missing': col_data.isnull().sum(),
            'Missing_%': round(col_data.isnull().sum() / len(col_data) * 100, 2),
            'Unique': col_data.nunique(),
            'Duplicates': len(col_data) - col_data.nunique()
        })
    return pd.DataFrame(report)

quality_report = assess_data_quality(df_raw)
print(quality_report.to_string(index=False))
print()

# Issues found
print("\n🚨 ISSUES IDENTIFIED:")
print("1. Duplicate rows (BOB Smith appears twice)")
print("2. Missing values in name, emp_id, email, salary")
print("3. Invalid age values (-5, 150, 'twenty-seven')")
print("4. Inconsistent department names (IT, it, hr, HR)")
print("5. Salary with commas and negative values")
print("6. Multiple date formats")
print("7. Invalid phone numbers")
print("8. Invalid email address")
print("9. City names with extra spaces and inconsistent case")
print()

# STEP 2: REMOVE DUPLICATES

print("="*70)
print("🔄 STEP 2: REMOVING DUPLICATES")
print("="*70)

df = df_raw.copy()
before = len(df)
df = df.drop_duplicates(subset=['emp_id', 'employee_name'], keep='first')
after = len(df)
df = df.reset_index(drop=True)
print(f"Removed {before - after} duplicate rows: {before} -> {after}")
print()

# STEP 3: CLEAN TEXT COLUMNS

print("="*70)
print("📝 STEP 3: CLEANING TEXT COLUMNS")
print("="*70)

# Clean names
df['employee_name'] = (
    df['employee_name']
    .str.strip()
    .str.title()
    .str.replace(r'\s+', ' ', regex=True)
)
print("Names cleaned ✅ ")

# Standardize department
df['department'] = (
    df['department']
    .str.strip()
    .str.title()
    .replace({'Hr':'HR', 'It':'IT'})
)
print("Departments standardized ✅")

# Standardize city
df['city'] = (
    df['city']
    .str.strip()
    .str.title()
)
print("Cities cleaned ✅")

# lowercase email
df['email'] = df['email'].str.strip().str.lower()
print("Emails standardized ✅ ")
print()

 # STEP 4: FIX NUMERIC COLUMNS

print("="*70)
print("🔢 STEP 4: FIXING NUMERIC COLUMNS")
print("="*70)

# Fix salary

df['salary'] = (
    df['salary']
    .str.replace(',', '', regex=False)
    .pipe(pd.to_numeric, errors='coerce')
)
# Mark negative salaries as invalid
df.loc[df['salary'] < 0, 'salary'] = np.nan
print("Salary cleaned ✅ ")

# Fix age
df['age'] = pd.to_numeric(df['age'], errors='coerce')
# Mark impossible ages as Invalid
df.loc[(df['age'] < 18) | (df['age'] > 100), 'age'] = np.nan
print(" Age cleaned ✅")

 # STEP 5: FIX DATES

print("="*70)
print("📅 STEP 5: STANDARDIZING DATES")
print("="*70)

def parse_date(date_str):
    """Try multiple date formats"""
    if pd.isna(date_str):
        return pd.NaT
    
    formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y',
        '%d-%m-%Y', '%m/%d/%Y', '%B %d %Y',
        '%m/%Y/%d'
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue

    try:
        return pd.to_datetime(date_str, format=fmt)
    except:
        return pd.NaT
    
df['join_date'] = df['join_date'].apply(parse_date)
print(" Dates standardized ✅")
print(df[['employee_name', 'join_date']])
print()

# STEP 6: FIX PHONE NUMBERS

print("="*70)
print("📱 STEP 6: CLEANING PHONE NUMBERS")
print("="*70)

def clean_phone(phone):
    """Standardize phone to 10 digits"""
    if pd.isna(phone):
         return np.nan
    digits = re.sub(r'\D', '', str(phone))
    if len(digits) == 12 and digits.startswith('91'):
        digits = digits[2:]
    if len(digits) == 10 and digits.isdigit():
        return digits
    return np.nan

df['phone'] = df['phone'].apply(clean_phone)
print(" Phone numbers cleaned ✅")
print()

# STEP 7: VALIDATE EMAIL

print("="*70)
print("📧 STEP 7: VALIDATING EMAILS")
print("="*70)

email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df['email_valid'] = df['email'].str.match(email_pattern, na=False)
df.loc[~df['email_valid'], 'email'] = np.nan
df = df.drop('email_valid', axis=1)
print(" Emails validated ✅")
print()

# STEP 8: HANDLE MISSING VALUES

print("="*70)
print("🔧 STEP 8: HANDLING MISSING VALUES")
print("="*70)

# Age: fill with median
df['age'] = df['age'].fillna(df['age'].median())
print(f"Age: filled {df['age'].isnull().sum()} missing with median")

# Salary: fill with departments median
df['salary'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)
print(f"Salary: filled with department median")

# Performance: fill with department mean
df['performance_score'] = df.groupby('department')['performance_score'].transform(
    lambda x: x.fillna(x.mean()).round(2)
)
print(f"Performance: filled with department mean")

# Employee ID: generate for missing
mask = df['emp_id'].isnull()
df.loc[mask, 'emp_id'] = [f'E{i:03d}' for i in range(900, 900 + mask.sum())]
print(f"Employee IDs: generated for missing")
print()

 # STEP 9: HANDLE OUTLIERS

print("="*70)
print("⚠️  STEP 9: HANDLING OUTLIERS")
print("="*70)

for col in ['salary', 'performance_score']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"{col}: clipped {len(outliers)} outliers to [{lower:.2f}, {upper:.2f}]")
    else:
        print(f"{col}: no outliers found ✅")
print()

#  STEP 10: FEATURE ENGINEERING

print("="*70)
print("⚙️  STEP 10: FEATURE ENGINEERING")
print("="*70)

# Experience in years
df['years_experience'] = (
    (pd.Timestamp.now() - df['join_date']).dt.days / 365.25
).round(1)

# Salary grade
df['salary_grade'] = pd.cut(
    df['salary'],
    bins=[0, 55000, 65000, 75000, float('inf')],
    labels=['Entry', 'Mid', 'Senior', 'Lead']   
)

# Performance category
df['performance_category'] = pd.cut(
    df['performance_score'],
    bins=[0, 3.5, 4.0, 4.5, 5.0],
    labels=['Needs Improvement', 'Good', 'Excellent', 'Outstanding']
)

print("New features created: ")
print("✅ years_experience")
print("✅ salary_grade")
print("✅ performance_category")
print()

 # FINAL REPORT

print("="*70)
print("📋 FINAL CLEANED DATASET")
print("="*70)

print(df.to_string())
print()

print("="*70)
print("📊 CLEANING SUMMARY REPORT")
print("="*70)

print(f"""
CLEANING PIPELINE SUMMARY
{"="*40}
Original Records:    {len(df_raw)}
Duplicates Removed:  {len(df_raw) - len(df)}
Final Records:       {len(df)}

DATA QUALITY IMPROVEMENTS:
{"="*40}
✅ Duplicate rows removed
✅ Text columns standardized
✅ Invalid numeric values fixed
✅ Date formats standardized
✅ Phone numbers validated
✅ Email addresses validated
✅ Missing values imputed
✅ Outliers handled
✅ New features engineered

COLUMN SUMMARY:
{"="*40}
""")
for col in df.columns:
    missing = df[col].isnull().sum()
    status = "✅" if missing == 0 else f"⚠️  {missing} missing"
    print(f"{col:25s} {status}")

# Save cleaned data
df.to_csv('cleaned_employee_data.csv', index=False)
print(f"\n✅ Saved cleaned data to 'cleaned_employee_data.csv'")

print("""
PIPELINE COMPLETE!
Raw messy data -> Professional ML-ready dataset
""")
