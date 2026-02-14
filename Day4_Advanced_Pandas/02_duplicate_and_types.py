import pandas as pd
import numpy as np

 # HANDLING DUPLICATES

print("="*70)
print("DETECTING AND REMOVING DUPLICATES")
print("="*70)

# Create dataset with duplicates
data = {
    'ID': [1,2,3,2,4,5,3,6],
    'Name': ['Alice','Bob','Charlie','Bob',
             'David','Eve','Charlie','Frank'],
    'Email': ['alice@email.com', 'bob@email.com',
              'charlie@email.com', 'bob@email.com',
              'david@email.com', 'eve@email.com',
              'charlie@email.com','frank@email.com'],
    'Score': [85,90,78,92,88,95,78,82]
}

df = pd.DataFrame(data)
print("Dataset with duplicates: ")
print(df)
print()

# Detect duplicates
print("Duplicate rows: ")
print(df.duplicated())
print()

print(f"Number of duplicated rows: {df.duplicated().sum()}")
print()

# Show actual duplicate rows
print("Actual duplicate rows: ")
print(df[df.duplicated(keep=False)])
print()

# Duplicate based on specific columns
print("Duplicated based on 'ID' only: ")
print(df.duplicated(subset=['ID']))
print()

# Remove duplicates
df_no_dup = df.drop_duplicates()
print(f"After drop_duplicates(): {len(df_no_dup)} rows")
print(df_no_dup)
print()

# Keep last occurence (no first)
df_keep_last = df.drop_duplicates(keep='last')
print("Keeping last occurence: ")
print(df_keep_last)
print()

 # FIXING DATA TYPES

print("="*70)
print("FIXING DATA TYPE INCONSISTENCIES")
print("="*70)

# Create dataset with wrong data types
messy_data = {
    'ID': ['001', '002', '003', '004', '005'],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': ['25', '30', '35.0', '28', 'thirty'],  # String ages!
    'Salary': ['50,000', '60,000', '75000', '55,000', '70000'],  # Commas!
    'Joined': ['2026-01-15', '2026-02-20', '01/03/2026',
               'March 4, 2026', '2026-05-10'],  # Multiple date formats!
    'Active': ['True', 'False', '1', 'yes', 'no']  # Mixed booleans!
}

df1 = pd.DataFrame(messy_data)
print("Messy dataset: ")
print(df1)
print()

print("Original data types: ")
print(df1.dtypes)
print()

 # FIX 1: Numeric columns

print("--- FIXING NUMERIC COLUMNS ---")

# Fix ID (string -> integer)

df1['ID'] = df1['ID'].astype(int)
print("ID fixed: ", df1['ID'].tolist())

# Fix AGE (Handle non-numeric values)
df1['Age_clean'] = pd.to_numeric(df1['Age'], errors='coerce')
print("Age fixed (non-numeric -> NaN): ", df1['Age_clean'].tolist())

# Fill NaN age with median
df1['Age_clean'] = df1['Age_clean'].fillna(df1['Age_clean'].median())
print("Age after filling NaN: ", df1['Age_clean'].tolist())

# Fix Salary (remove commas, convert to float)
df1['Salary_clean'] = df1['Salary'].str.replace(',', '').astype(float)
print("Salary fixed: ", df1['Salary_clean'].tolist())
print()

 # FIX 2: Date Columns

print("--- FIXING DATE COLUMNS ---")

# Multiple format handling
def parse_date_flexible(date_str):
    """Try multiple date formats"""
    formats = [
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%B %d, %Y',
        '%m/%d/%Y'
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    # Last resort: let pandas infer
    try:
        return pd.to_datetime(date_str, infer_datetime_format=True)
    except:
        return pd.NaT
    
df1['Joined_clean'] = df1['Joined'].apply(parse_date_flexible)
print("Date fixed: ")
print(df1[['Joined', 'Joined_clean']])
print()

 # FIX 3: Boolean Columns

print("--- FIXING BOOLEAN COLUMNS ---")

def standardize_boolean(value):
    """Convert various boolean representations"""
    true_values = ['true','1','yes','y','on']
    false_values = ['false', '0','no','n','off']

    val = str(value).lower().strip()

    if val in true_values:
        return True
    elif val in false_values:
        return False
    else:
        return np.nan

df1['Active_clean'] = df1['Active'].apply(standardize_boolean)
print("Boolean fixed: ")
print(df1[['Active','Active_clean']])
print()

 # FINAL CLEANED DATASET

print("="*70)
print("FINAL CLEANED DATASET")
print("="*70)

df_final = pd.DataFrame({
    'ID': df1['ID'],
    'Name': df1['Name'],
    'Age': df1['Age_clean'],
    'Salary': df1['Salary_clean'],
    'Joined': df1['Joined_clean'],
    'Active': df1['Active_clean']
})

print(df_final)
print()
print("Data types: ")
print(df_final.dtypes)