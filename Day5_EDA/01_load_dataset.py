"""
DATASET: Titanic Survival Dataset
==================================
One of the most famous datasets in ML history.
Goal: Predict who survived the Titanic disaster.

WHY TITANIC?
- Real historical data (April 1912)
- Mix of numeric and categorical features
- Clear target variable (Survived: 0 or 1)
- Perfect for learning EDA and ML
- Industry-standard benchmark dataset
"""

import pandas as pd
import numpy as np

print("="*70)
print("PHASE 1: LOADING AND UNDERSTANDING THE DATASET")
print("="*70)

 # CREATE TITANIC-LIKE DATASET

 # We'll create a realistic dataset mimicking Titanic structure
np.random.seed(42)

n = 891   # Actual Titanic passenger count

# Passenger class distribution (1st: 24%, 2nd: 21%, 3rd: 55%)
pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])

# Sex (male: 65%, female: 35%)
sex = np.random.choice(['male', 'female'], n, p=[0.65, 0.35])

# Age (some missing ~20%)
age = np.where(
    np.random.randn(n) < 0.2, # where function checks condition, is_True, is_False, 
    np.nan, 
    np.clip(np.random.normal(30, 14, n), 0.5, 80)  #(30, 14, n) - mean=30, std=14. clips them between 0.5 and 80 to ensure realistic ages
)

# Siblings/Spouses
sibsp = np.random.choice([0,1,2,3,4,5], n,
                          p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.005])

# Parents/Children
parch = np.random.choice([0,1,2,3,4,5], n, p=[0.76, 0.13, 0.08, 0.02, 0.005, 0.005])

# Fare (based on class)
fare = np.where(
    pclass==1,
    np.clip(np.random.normal(84, 78, n), 0, 512),
    np.where(
        pclass==2,
        np.clip(np.random.normal(20, 13, n), 0, 73),
        np.clip(np.random.normal(13, 11, n), 0, 69)
    )
)

# Some missing fares
fare = np.where(np.random.random(n) < 0.02, np.nan, fare)

# Embarked port (S: 72%, C: 19%, Q: 9%)
embarked = np.where( 
    np.random.random(n) < 0.02,
    np.nan,
    np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09])
)

# Cabin (many missing ~77%)
cabin_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
cabin = np.where(
    np.random.random(n) < 0.77,
    np.nan,
    [f"{np.random.choice(cabin_letters)}{np.random.randint(1, 150)}"
     for _ in range(n)]
)

# Survival (based on realistic factors)
# Women and children first! Higher class = better survival
base_survival = (
    (sex == 'female').astype(float)*0.4 +
    (pclass == 1).astype(float)*0.25 +
    (pclass == 2).astype(float)*0.10 +
    np.random.random(n)*0.25
)
survived = (base_survival > 0.45).astype(int)

# Create names
first_names_male = ['James', 'John', 'William', 'Thomas', 'Henry',
                    'Charles', 'George', 'Joseph', 'Robert', 'Edward']
first_names_female = ['Mary', 'Anna', 'Emma', 'Elizabeth', 'Margaret',
                      'Minnie', 'Ida', 'Bertha', 'Alice', 'Clara']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
              'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor',
              'Anderson', 'Thomas', 'Jackson', 'White', 'Harris']
names = []
for s in sex:
    if s == 'male':
        title = np.random.choice(['Mr.', 'Dr.', 'Rev.'], p=[0.9, 0.05, 0.05])
        first = np.random.choice(first_names_male)
    else:
        title = np.random.choice(['Miss', 'Mrs.', 'Dr.'], p=[0.55, 0.43, 0.02])
        first = np.random.choice(first_names_female)
    last = np.random.choice(last_names)
    names.append(f"{last}, {title} {first}")

# Ticket numbers
tickets = [f"{np.random.choice(['PC', 'CA', 'A/', 'SOTON', 'SC', ''])}"
          f"{np.random.randint(1000, 999999)}" for _ in range(n)]

# Create DataFrame
df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Survived': survived,
    'Pclass': pclass,
    'Name': names,
    'Sex': sex,
    'Age': age.round(1),
    'SibSp': sibsp,
    'Parch': parch,
    'Ticket': tickets,
    'Fare': fare.round(2),
    'Cabin': cabin,
    'Embarked': embarked
})

# Save dataset
df.to_csv('titanic.csv', index=False)

print("✅ Dataset created and saved as 'titanic.csv' ")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows: ")
print(df.head())
print(f"\nData types: ")
print(df.dtypes)