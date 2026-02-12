import pandas as pd
import numpy as np

"""
DATAFRAME = 2D labeled data structure (like Excel sheet or SQL table)
Most important Pandas object for data analysis!
"""

 # CREATING DATAFRAMES

print("="*60)
print("CREATING DATAFRAMES")
print("="*60)

# From dictionary
data_dict = {
    'Name': ['Arunesh', 'Boopesh', 'Chandhru','Dhinesh', 'Evanesh'],
    'Age': [21, 20, 23, 24, 32],
    'City': ['Kallakurichi','Trichy','Salem','Madurai','Kovai'],
    'Salary': [50000,60000,75000,55000,70000]
}

df = pd.DataFrame(data_dict)
print("DataFrame from dictionary: ")
print(df)
print()

 # BASIC INFORMATION

print("="*60)
print("DATAFRAME INFO")
print("="*60)

print("Shape: ", df.shape) # (rows,columns)
print("Columns: ", df.columns.tolist())
print("Index: ", df.index.tolist())
print("Data types:\n", df.dtypes)
print()

# Quick look
print("First 3 rows (head): ")
print(df.head(3))
print()

print("Last 2 rows (tail): ")
print(df.tail(2))
print()

# Statistical summary
print("Statistical summary: ")
print(df.describe())
print()

# info (concise summary)
print("DataFrame Info: ")
df.info()
print()

 # SELECTING COLUMNS

print("="*60)
print("SELECTING COLUMNS")
print("="*60)

# Single column (return Series)
print("Name column (Series): ")
print(df['Name'])
print(f"Type: {type(df[['Name']])}")
print()

# Multiple columns
print("Name and Age: ")
print(df[['Name', 'Age']])
print()

 # SELECTING ROWS

print("="*60)
print("SELECTING ROWS")
print("="*60)

# By position (iloc)
print("First row (iloc): ")
print(df.iloc[0])
print()

print("First 3 rows (iloc): ")
print(df.iloc[0:3])
print()

# BY label (loc)
df_with_index = df.set_index('Name')
print("\nDataFrame with Name as index: ")
print(df_with_index)
print()

print("Arunesh's data (loc): ")
print(df_with_index.loc['Arunesh'])
print()

 # BOOLEAN INDEXING

print("="*60)
print("BOOLEAN INDEXING (Filtering)")
print("="*60)

# People older than 30
print("People older than 30: ")
print(df[df['Age']>30])
print()

# People from Madurai
print("People from Madurai: ")
print(df[df['City'] == 'Madurai'])
print()

# Multiple Conditions (AND)
print("People older than 28 AND salary > 60000")
print(df[(df['Age']>28) & (df['Salary'] > 60000)])
print()

# Multiple Conditions (OR)
print("People from Salem OR Kovai: ")
print(df[(df['City'] == 'Salem') | (df['City'] == 'Kovai')])
print()

 # ADDING/MODIFYING COLUMNS

print("="*60)
print("ADDING/MODIFYING COLUMNS")
print("="*60)

# Add new column
df['Experience'] = [3, 4, 6, 5, 8]
print("After adding Experience: ")
print(df)
print()

# Calculate from existing columns
df['Salary_per_Year_Exp'] = df['Salary'] / df['Experience']
print("After adding calculated column: ")
print(df)
print()

# Conditional column
df['Senior'] = df['Age'] > 30
print("After adding boolean column: ")
print(df)
print()

 # DELETING COLUMNS/ROWS

print("="*60)
print("DELETING COLUMNS/ROWS")
print("="*60)

# Drop column
df_copy = df.copy()
df_copy = df_copy.drop('Senior', axis=1) # axis=1 for columns
print("After dropping 'Senior' column: ")
print(df_copy)
print()

# Drop row
df_copy2 = df.copy()
df_copy2 = df_copy2.drop(0, axis=0) #axis=0 for rows
print("After dropping first row: ")
print(df_copy2)
print()

# Drop multiple rows
df_copy3 = df.copy()
df_copy3 = df_copy3.drop([0, 2, 4])
print("After dropping rows 0, 2, 4: ")
print(df_copy3)
print()

 # SORTING

print("="*60)
print("SORTING")
print("="*60)

# Sort by single column
print("Sorted by Age (ascending): ")
print(df.sort_values('Age'))
print()

print("Sort by Salary (descending): ")
print(df.sort_values('Salary', ascending=False))
print()

# Sort by multiple columns
print("Sort by City, then Age: ")
print(df.sort_values(['City', 'Age']))