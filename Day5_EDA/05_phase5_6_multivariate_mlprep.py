import numpy as np
import pandas as pd

print("="*70)
print("PHASES 5 & 6: MULTIVARIATE ANALYSIS + ML PREPARATION")
print("="*70)

df = pd.read_csv('titanic.csv')

 # PHASE 5: MULTIVARIATE ANALYSIS

print("\n5.1 CORRELATION HEATMAP (TEXT VERSION)")
print("="*70)

numeric_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
corr = numeric_df.corr().round(2)

# ASCII heatmap
cols = corr.columns.tolist()
print(f"\n{'':12s}", end="")
for c in cols:
    print(f"{c[:6]:>8s}", end="")
print()

for i, row_name in enumerate(cols):
    print(f"{row_name[:12]:12s}", end="")
    for j, col_name in enumerate(cols):
        val = corr.loc[row_name, col_name]
        if val >=0.5:
            symbol = "████"
        elif val >=0.3:
            symbol = "▓▓▓░"
        elif val >=0.1:
            symbol = "▒▒░░"
        elif val >=-0.1:
            symbol = "░░░░"
        elif val >=-0.3:
            symbol = "▒▒░░"
        elif val >=-0.5:
            symbol = "▓▓▓░"
        else:
            symbol = "████"
        print(f"{val:+6.2f} ", end="")
    print()

print("\nLegend: ████=Strong, ▓▓=Moderate, ▒=Weak, ░=Negligible")
print()

# 5.2 PATTERN DISCOVERY

print("5.2 KEY PATTERN DISCOVERY")
print("="*70)

# Three-way analysis: Pclass × Sex × Survival
print("\nPclass × Sex × Survival (3-way analysis):")
print("-"*60)

three_way = df.groupby(['Pclass', 'Sex'])['Survived'].agg(
    ['count', 'sum', 'mean']
).round(3)

three_way.columns = ['Total', 'Survived', 'Rate']
print(three_way)
print()

# Family compostition analysis
df['Family_Size'] = df['SibSp'] + df["Parch"] + 1
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
df['Family_Type'] = pd.cut(
    df['Family_Size'],
    bins=[0,1,4,20],
    labels=['Alone', 'Small Family', 'Large Family']
)

print("\nFamily Type vs Survival: ")
print("-"*50)
family_survival = df.groupby('Family_Type', observed=True)['Survived'].agg(
    ['count', 'mean']
).round(3)

family_survival.columns = ['Count', 'Survival_Rate']
print(family_survival)
print()

# Extract title from name
df['Title'] = df['Name'].str.extract(r',\s([^\.]+)\.')
df['Title'] = df['Title'].str.strip()

# Group rare titles
common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
df['Title_grouped'] = df['Title'].apply(
    lambda x: x if x in common_titles else 'Rare'
)

print("\nTitle vs Survival: ")
print("-"*50)
title_survival = df.groupby('Title_grouped')['Survived'].agg(
    ['count', 'mean']
).sort_values('mean', ascending=False).round(3)
title_survival.columns = ['Count', 'Survival_Rate']
print(title_survival)
print()
print("💡 INSIGHT: Title captures both gender AND social status!")
print("💡 'Mrs' and 'Miss' titles = female = high survival")
print("💡 'Master' = young boy = some preferential treatment")
print()

 # PHASE 6: ML PREPARATION

print("="*70)
print("PHASE 6: CREATING ML-READY DATASET")
print("="*70)

df_ml = df.copy()

# Step 1: Extract useful features
print("\n6.1 FEATURE ENGINEERING: ")

# Title extraction
df_ml['Title'] = df_ml['Name'].str.extract(r',\s([^\.]+)\.')
df_ml['Title'] = df_ml['Name'].str.strip()
common_titles1 = ['Mr', 'Mrs', 'Miss', 'Master', 'Dr']
df_ml['Title'] = df_ml['Title'].apply(
    lambda x : x if x in common_titles1 else 'Rare'
)
print("   ✅ Title extracted from Name")

# Family features
df_ml['Family_Size'] = df_ml['SibSp'] + df_ml['Parch'] + 1
df_ml['Alone'] = (df_ml['Family_Size'] == 1).astype(int)
print("   ✅ Family_Size and Is_Alone created")

# Deck from Cabin
df_ml['Deck'] = df_ml['Cabin'].str[0].fillna('Unknown')
print("   ✅ Deck extracted from Cabin")

# Age groups
df_ml['Age_Group'] = pd.cut(
    df_ml['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
)
print("   ✅ Age_Group created")

# Fare categories
df_ml['Fare_Category'] = pd.qcut(
    df_ml['Fare'].fillna(df_ml['Fare'].median()),
    q=4,
    labels = ['low', 'Medium', 'High', 'Very High']
)
print("   ✅ Fare_Category created")
print()

# Handling missing values
print("6.2 IMPUTATION:")

# Age imputation by Title
title_age_map = df_ml.groupby('Title')['Age'].median()  # Calculate Median Age by Title
df_ml['Age'] = df_ml.apply(    # Iterate through each row of the DataFrame
    lambda row: title_age_map.get(row['Title'], df_ml['Age'].median)
    if pd.isna(row['Age']) else row['Age'],   # Checks if the 'Age' in the current row is missing (NaN). If the Age is not missing, it keeps the original Age.
    axis = 1
)
print("   ✅ Age imputed by Title median")

# Fare imputation
df_ml['Fare'] = df_ml['Fare'].fillna(df_ml.groupby('Pclass')['Fare'].transform('median'))
print("   ✅ Fare imputed by Pclass median")

# Embarked imputation
df_ml['Embarked'] = df_ml['Embarked'].fillna(df_ml['Embarked'].mode()[0])
print("   ✅ Embarked imputed with mode")
print()

# Step 3: Feature transformation
print("6.3 TRANSFORMATIONS: ")

# Log transform Fare (reduce skewness)
df_ml['Fare_Log'] = np.log1p(df_ml['Fare'])
print("   ✅ Log transform applied to Fare")

# Age normaliztion
df_ml['Age_normalized'] = (df_ml['Age'] - df_ml['Age'].mean()) / df_ml['Age'].std()
print("   ✅ Age normalized (z-score)")
print()

# Step 4: Encoding
print("6.4 ENCODING:")

# Binary encoding
df_ml['Sex_Binary'] = (df_ml['Sex'] == 'female').astype(int)
print("   ✅ Sex encoded (female=1, male=0)")

# One-hot encoding
embarked_dummies = pd.get_dummies(df_ml['Embarked'], prefix='Embarked')
title_dummies = pd.get_dummies(df_ml['Title_grouped'] if 'Title_grouped' in df_ml.columns else df_ml['Title'], prefix='Title')
pclass_dummies = pd.get_dummies(df_ml['Pclass'], prefix='Pclass')

df_ml = pd.concat([df_ml, embarked_dummies, title_dummies, pclass_dummies], axis=1)
print("   ✅ Embarked one-hot encoded")
print("   ✅ Title one-hot encoded")
print("   ✅ Pclass one-hot encoded")
print()

# Step 5: Select final features
print("6.5 FINAL FEATURE SELECTION: ")

features = [
    'Survived',
    'Sex_Binary', 'Age_Normalized', 'Fare_Log',
    'Family_Size','Is_Alone',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
    'Pclass_1', 'Pclass_2', 'Pclass_3'
]

# Add title dummies
title_cols = [c for c in df_ml.columns if c.startswith('Title_') ]
features.extend(title_cols)

# Add any availbale encoded columns
features = [f for f in features if f in df_ml.columns]

df_final = df_ml[features].copy()
df_final = df_final.dropna()

print(f"  ✅ Final feature set: {len(features)} features")
print(f"   ✅ Final dataset: {df_final.shape}")
print(f"\n   Features selected: ")
for f in features:
    print(f"     - {f}")

# Save ML-ready dataset
df_final.to_csv('titanic_ml_read.csv', index=False)
print(f"\n   ✅ Saved ML-ready dataset: 'titanic_ml_ready.csv'")

print()
print("="*70)
print("🎉 DATASET IS NOW ML-READY! ")
print("="*70)
print("""
Summary:
   Raw data: 12 columns, mixed types, missing values
   ML data:  15+ features, clean, encoded, normalized
      
Next steps in our roadmap:
   - Week 2: Linear Regression on this dataset
   - Week 3: More complex algorithms
   - Eventually: Full ML pipeline with this data!
""")