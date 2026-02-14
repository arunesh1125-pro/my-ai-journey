import pandas as pd
import numpy as np

print("="*70)
print("HANDLING MISSING DATA - THE PROFESSIONAL WAY")
print("="*70)

 # CREATE MESSY DATASET

 # Simulate real-world messy employee data
data = {
    'Employee_ID': [1,2,3,4,5,6,7,8,9,10],
    'Name': ['Alice','Bob',None,'David','Eve',
             'Frank','Grace','Henry',None,'Jack'],
    'Age': [25, np.nan, 35, 28, np.nan,
            32, 29, np.nan, 27, 33],
    'Department': ['IT', 'HR', 'IT', None, 'Finance',
                   'HR', None, 'IT', 'Finance', 'IT'],
    'Salary': [50000, 60000, np.nan, 55000, 70000,
               np.nan, 65000, 80000, 72000, np.nan],
    'Performance': [4.5, np.nan, 3.8, 4.2, 4.7,
                    np.nan, 4.0, 3.5, np.nan, 4.3],
    'Years_Experience': [3, 8, np.nan, 5, 10, 
                         np.nan, 7, 12, 6, np.nan]
}

df = pd.DataFrame(data)
print("Raw Dataset: ")
print(df)
print()

 # DETECTING MISSING VALUES

print("="*70)
print("STEP 1: DETECTING MISSING VALUES")
print("="*70)

# Check for nulls
print("isnull() - Which cells are null: ")
print(df.isnull())
print()

# Count nulls per column
print("Null count per column: ")
null_counts = df.isnull().sum()
print(null_counts)
print()

# Percentage of nulls
print("Null percentage per column: ")
null_perentage = (df.isnull().sum() / len(df)) * 100
print(null_perentage.round(2))
print()

# Visual summary
print("Missing data summary: ")
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2).values,
    'Data_Type': df.dtypes.values

})
print(missing_summary)
print()

 # STRATEGY 1: DROP MISSING VALUES

print("="*70)
print("STRATEGY 1: DROPPING MISSING VALUES")
print("="*70)

# Drop rows with ANY null
df_dropped_any = df.dropna()
print(f"Original: {len(df)} rows")
print(f"After dropna() (any): {len(df_dropped_any)} rows")
print(df_dropped_any)
print()

# Drop rows where ALL Values are null
df_dropped_all = df.dropna(how='all')
print(f"After dropna(how='all'): {len(df_dropped_all)} rows")
print(df_dropped_all)
print()

# Drop rows with null in SPECIFIC columns
df_dropped_name = df.dropna(subset='Name')
print(f"After dropping rows with null Name: {len(df_dropped_name)} rows")
print(df_dropped_name)
print()

# Drop columns with too many nulls (>50% missing)
df_drop_cols = df.dropna(axis=1, thresh=int(0.5 * len(df)))
print(f"After dropping column >50% null: {df_drop_cols.shape}")
print()

 # STRATEGY 2: FILL MISSING VALUES

print("="*70)
print("STRATEGY 2: FILLING MISSING VALUES")
print("="*70)

df_filled = df.copy()

# Fill with constant
df_filled['Name'] = df_filled['Name'].fillna('Unknown')
df_filled['Department'] = df_filled['Department'].fillna('Unassigned')
print("After filling Name and Department: ")
print(df_filled[['Name', 'Department']].head())
print()

# Fill numeric with MEAN
mean_age = df['Age'].mean()
df_filled['Age'] = df_filled['Age'].fillna(mean_age)
print(f"Mean Age: {mean_age:.2f}")
print("After filling Age with mean: ")
print(df_filled['Age'])
print()

# Fill numeric with MEDIAN (better for skewed data!)
median_salary = df['Salary'].median()
df_filled['Salary'] = df_filled['Salary'].fillna(median_salary)
print(f"Median Salary: {median_salary:.2f}")
print("After filling Salary with median: ")
print(df_filled['Salary'])
print()

# Fill with MODE (for categorical)
mode_dept = df['Department'].mode()[0]
df_filled['Department'] = df_filled['Department'].fillna(mode_dept)
print(f"Mode Department: {mode_dept}")
print()

# Group-based filling (POWERFUL!)
# Fill AGE based on department average
print("Department-wise average age: ")
dept_age_avg = df.groupby('Department')['Age'].transform('mean')
df['Age_filled'] = df['Age'].fillna(dept_age_avg)
print(df[['Name','Department','Age','Age_filled']])
print()

# Forward fill (use previous valid value)
df_ffill = df.copy()
df_ffill['Performance'] = df_ffill['Performance'].ffill()
print("After forward fill (Performance): ")
print(df_ffill['Performance'])
print()

# Backward fill
df_bfill = df.copy()
df_bfill['Performance'] = df_bfill['Performance'].bfill()
print("After backward fill (Performance): ")
print(df_bfill['Performance'])
print()

 # STRATEGY 3: INTERPOLATION

print("="*70)
print("STRATEGY 3: INTERPOLATION")
print("="*70)

# Create time-series like data
ts_data = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
print("Original:\n",ts_data)
print()

# Linear interpolation
interpolated = ts_data.interpolate(method='linear')
print('Linear interpolated: ', interpolated.values)
print()

# Polynomial interpolation of order=2
ply_interpolated = ts_data.interpolate(method='polynomial', order=2)
print('Polynomial interpolation of order 2: ', ply_interpolated.values)
print()

 # PROFESSIONAL MISSING DATA DECISION GUIDE

print("="*70)
print("WHEN TO USE WHICH STRATEGY")
print("="*70)

guide = """
MISSING DATA DECISION GUIDE:

1. DROP rows when:
   - Missing in key identifier columns (ID, Name)
   - Very few rows affected (<5% of data)
   - Missing at random (not systematic)

2. FILL with MEAN when:
   - Numeric data, roughly normal distribution
   - Not too many missing values (<20%)

3. FILL with MEDIAN when:
   - Numeric data, skewed distribution
   - Presence of outliers
   - Salary, price, age data

4. FILL with MODE when:
   - Categorical data
   - Department, city, category columns

5. GROUP-BASED FILL when:
   - Value depends on category
   - Example: Fill age based on department average

6. INTERPOLATION when:
   - Time-series data
   - Sequential measurements

7. KEEP as NaN when:
   - Missingness itself is informative
   - Create "is_missing" binary feature instead
"""
print(guide)