import numpy as np
import pandas as pd

# Create a sample student dataset
np.random.seed(42)

data = {
    'Student_ID': range(1, 21),
    'Name': [f"Student_{i}" for i in range(1, 21)],
    'Math': np.random.randint(40, 100, 20),
    'Physics': np.random.randint(35, 100, 20),
    'Chemistry': np.random.randint(40, 100, 20),
    'English': np.random.randint(50, 100, 20),
    'Computer': np.random.randint(45, 100, 20),
    'City': np.random.choice(['Chennai','Mumbai','Delhi','Banglore'], 20)
}

df = pd.DataFrame(data)
print("Student Dataset: ")
print(df)
print()

# Task 1: Basic Info
print("Dataset Shape: ", df.shape)
print("Column names: ", df.columns.tolist())
print("\nData types: ")
print(df.dtypes)
print()

# Task 2: Add total and average columns
df['Total'] = df[['Math', 'Physics', 'Chemistry', 'English', 'Computer']].sum(axis=1)
df['Average'] = df['Total'] / 5
print("After adding Total and Average: ")
print(df[['Name', 'Total', 'Average']].head())
print()

# Task 3: Add grade column
def get_grade(avg):
    if avg >= 90: return 'A+'
    elif avg >=80: return 'A'
    elif avg >=70: return 'B'
    elif avg >= 60: return 'C'
    elif avg >= 50: return 'D'
    else: return 'F'

df['Grade'] = df['Average'].apply(get_grade)
print("After adding Grade: ")
print(df[['Name', 'Average', 'Grade']].head(10))
print()

# Task 4: Filter Students
print("Students with average > 80: ")
print(df[df['Average'] > 80][['Name', 'Average', 'Grade']])
print()

print("Students from Chennai: ")
print(df[df['City'] == 'Chennai'][['Name', 'City', 'Average']])
print()

print("Students who failed Math (< 40): ")
print(df[df['Math'] < 40][['Name', 'Math']])
print()

# Task 5: Sort operations
print("Top 5 students by average: ")
print(df.nlargest(5, 'Average')[['Name','Average','Grade']])
print()

print("Bottom 5 students by average: ")
print(df.nsmallest(5, 'Average')[['Name','Average','Grade']])
print()

# Task 6: Statistics
print("Subject-wise statistics: ")
subjects = ['Math','Physics','Chemistry','English','Computer']
for subject in subjects:
    print(f"\n{subject}: ")
    print(f"  Mean: {df[subject].mean():.2f}") 
    print(f"  Max: {df[subject].max()}")
    print(f"  Min: {df[subject].min()}")
    print(f"   Pass rate (>=40): {(df[subject] >= 40).sum() / len(df) * 100:.1f}%")