import pandas as pd
import numpy as np

"""
MERGING = Combining datasets (like SQL JOINs)
CRITICAL for ML: Often your data is in multiple tables/files!
"""

print("="*70)
print("MERGING AND JOINING DATASETS")
print("="*70)
 
 # CREATE SAMPLE DATASETS

# Employee table
employees = pd.DataFrame({
    'Employee_ID': [1, 2, 3, 4, 5, 6],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Department_ID': [10, 20, 10, 30, 20, 40],
    'Salary': [50000, 60000, 75000, 55000, 70000, 65000]
})

# Department table
departments = pd.DataFrame({
    'Department_ID': [10, 20, 30, 50],  # Note: 40 missing, 50 extra
    'Department_Name': ['Engineering', 'Marketing', 'Finance', 'Legal'],
    'Location': ['Chennai', 'Mumbai', 'Delhi', 'Bangalore']
})

# Performance table
performance = pd.DataFrame({
    'Employee_ID': [1, 2, 3, 4, 5, 7],  # Note: 6 missing, 7 extra
    'Year': [2024, 2024, 2024, 2024, 2024, 2024],
    'Rating': [4.5, 3.8, 4.2, 3.5, 4.7, 4.0]
})

print("Employees: ")
print(employees)
print()

print("Departments: ")
print(departments)
print()

print("Performance: ")
print(performance)
print()

 # INNER JOIN (Default) - Only matching records

print("="*70)
print("INNER JOIN - Only matching records in BOTH tables")
print("="*70)

inner_join = pd.merge(
    employees,
    departments,
    on='Department_ID',
    how='inner'
)
print("Employees INNER JOIN Departments: ")
print(inner_join)
print(f"\nRows: {len(inner_join)} (Frank excluded - Dept_ID 40 not in departments)")
print()

 # LEFT JOIN - All left records, matching right

print("="*70)
print("LEFT JOIN - All left table + matching right")
print("="*70)

left_join = pd.merge(
    employees,
    departments,
    on='Department_ID',
    how='left'
)
print("Employees LEFT JOIN Departments: ")
print(left_join)
print(f"\nRows: {len(left_join)} (Frank included with NaN department info)")
print()

 # RIGHT JOIN - All right record, matching left

print("="*70)
print("RIGHT JOIN - All right table + matching left")
print("="*70)

right_join = pd.merge(
    employees,
    departments,
    on='Department_ID',
    how='right'
)
print("Employees RIGHT JOIN Department")
print(right_join)
print(f"\nRows: {len(right_join)} (Legal dept included with NaN employee info)")
print()

 # OUTER JOIN -All records from both tables

print("="*70)
print("OUTER JOIN - All records from BOTH tables")
print("="*70)

outer_join = pd.merge(
    employees,
    departments,
    on='Department_ID',
    how='outer'
)
print("Employees OUTER JOIN Departments: ")
print(outer_join)
print()

 # JOING ON DIFFERENT COLUMNS

print("="*70)
print("JOIN with DIFFERENT COLUMN NAMES")
print("="*70)

# Rename column to simulate real-world scenerio
performance_renamed = performance.rename(columns={'Employee_ID': 'Emp_ID'})

# Use left_on and right_on
joined = pd.merge(
    employees,
    performance_renamed,
    left_on='Employee_ID',
    right_on='Emp_ID',
    how='left'
)
print("Employees with Performancer: ")
print(joined)
print()

 # MULTI-TABLE JOIN (COmmon in real projects!)

print("="*70)
print("MULTI-TABLE JOIN")
print("="*70)

# Step 1: Join employees with departments
step1 = pd.merge(employees, departments, on='Department_ID', how='left')

# Step 2: Join result with Performance
final = pd.merge(step1, performance, on='Employee_ID', how='left')

print("Complete Employee Report: ")
print(final)
print()

 # CONCAT (Stacking datasets)

print("="*70)
print("CONCATENATING DATASETS (Stacking)")
print("="*70)

# Stacking rows (new data coming in)
employees_2023 = pd.DataFrame({
    'Employee_ID': [7, 8],
    'Name': ['Grace','Henry'],
    'Department_ID': [10, 20],
    'Salary': [68000, 72000]
})

# Vertical concat (add rows)
all_employees = pd.concat([employees, employees_2023], ignore_index=True)
print("After concatenating new employees: ")
print(all_employees)
print()

# Horizontal concat (add columns)
extra_info = pd.DataFrame({
    'Email':[f"{name.lower()}@company.com" for name in employees['Name']],
    'Phone': [f"98{i:08d}" for i in range(len(employees))]
})

employees_full = pd.concat([employees, extra_info], axis=1)
print("After adding contact info: ")
print(employees_full)
print()

 # PRACTICAL EXERCISE

print("="*70)
print("PRACTICAL EXERCISE: Build Complete Employee Report")
print("="*70)

# Full analysis
full_report = pd.merge(employees, departments, on='Department_ID', how='left')
full_report = pd.merge(full_report, performance, on='Employee_ID', how='left')

# Add analysis colums
full_report['Salary_vs_Dept_Avg'] = full_report.groupby('Department_Name')['Salary'].transform(
    lambda x: x - x.mean()
)

full_report['Performance_category'] = pd.cut(
    full_report['Rating'],
    bins=[0,3.5,4.0,4.5,5.0],
    labels=['Needs Improvement', 'Good', 'Excellent', 'Outstanding'],
    right=True
)

print("Complete Employee Report with Analysis: ")
print(full_report)