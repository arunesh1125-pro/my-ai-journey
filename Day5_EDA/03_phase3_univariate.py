import pandas as pd
import numpy as np

print("="*70)
print("PHASE 3: UNIVARIATE ANALYSIS")
print("="*70)

"""
UNIVARIATE = Analyzing ONE variable at a time
Goal: Understand the distribution of each feature individually
"""

df = pd.read_csv('titanic.csv')

 # 3.1 NUMERIC FEATURES

print("\n3.1 NUMERIC FEATURE ANALYSIS")
print("="*70)

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numeric_cols:
    data = df[col].dropna()

    print(f"\n{'─'*50}")
    print(f"📊 {col.upper()}")
    print(f"{'─'*50}")
    print(f"Count:    {len(data):>10.0f}")
    print(f"Missing:  {df[col].isnull().sum():>10d} ({df[col].isnull().mean()*100:.1f}%)")
    print(f"Mean:       {data.mean():>10.0f}")
    print(f"Median:     {data.median():>10.0f}")
    print(f"Std Dev:    {data.std():>10.0f}")
    print(f"Min:        {data.min():>10.0f}")
    print(f"Max:        {data.max():>10.0f}")
    print(f"Skewness:   {data.skew():>10.3f}", end=" ")

    skew = data.skew()
    if abs(skew) < 0.5:
        print("(Roughly symmetric)")
    elif skew > 0:
        print("(Right-skewed -> consider log transform)")
    else:
        print("(Left-skewed)")

    print(f"Kurtosis: {data.kurtosis():>10.3f}", end=" ")
    kurt = data.kurtosis()
    if kurt > 3:
        print("Heavy tails = many outliers")
    elif kurt < -1:
        print("(Light tails = few extreme)")
    else:
        print("(Normal-like tails)")

    # Percentils
    percentils = [0, 10, 25, 50, 75, 90, 100]
    pct_values = [np.percentile(data, p) for p in percentils]
    print(f"\nPercentiles: ")
    for p, v in zip(percentils, pct_values):
        bar_len = int((p/100)*20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  P{p:3d}: {v:8.2f}  |{bar}")

    # Outliers (IQR method)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    print(f"\nOutliers (IQR): {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")

    # Quick text histogram
    print(f"\nDistribution (ASCII): ")
    hist, edges = np.histogram(data, bins=10)
    max_count = max(hist)
    for i, count in enumerate(hist):
        bar_len = int((count / max_count) * 30)
        bar = "█" * bar_len
        print(f" [{edges[i]:6.1f}, {edges[i+1]:6.1f}]: "
              f"{bar} {count}")
    print()

# 3.2 CATEGORICAL FEATURES

print("\n3.2 CATEGORICAL FEATURE ANALYSIS")
print("="*70)

categorical_cols = ['Survived', 'Pclass', 'Sex', 'Embarked']

for col in categorical_cols:
    print(f"\n{'─'*50}")
    print(f"📊 {col.upper()}")
    print(f"{'─'*50}")

    value_counts = df[col].value_counts()
    value_pcts = df[col].value_counts(normalize=True) * 100
    missing = df[col].isnull().sum()

    print(f"Total unique values: {df[col].nunique()}")
    print(f"Missing: {missing} ({missing/len(df)*100:.1f}%)")
    print()
    print(f"{'Value':15s} {'Count':>8s} {'Percent':>10s}  Bar")
    print("-" * 55)

    for val, count in value_counts.items():
        pct = value_pcts[val]
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  {str(val):13s} {count:>8d} {pct:>8.1f}%  {bar}")
    print()

 #3.3 KEY UNIVARIATE INSIGHTS

print("="*70)
print("3.3 KEY UNIVARIATE INSIGHTS")
print("="*70)

survival_rate = df['Survived'].mean() * 100
median_age = df['Age'].median()
median_fare = df['Fare'].median()
pct_class3 = (df['Pclass'] == 3).mean() * 100
pct_male = (df['Sex'] == 'male').mean() * 100

print(f"""
Key Findings from Univariate Analysis:
{'─'*50}
📊 Survival:
   - Overall survival rate: {survival_rate:.1f}%
   - Majority ({100-survival_rate:.1f}%) did NOT survive
   - Class imbalance present

👥 Demographics:
   - Median passenger age: {median_age:.0f} years
   - Age range: 0.5 to 80 years
   - {pct_male:.1f}% male passengers
   - Many children aboard (infants visible in data)

💰 Economics:
   - Median fare: £{median_fare:.2f}
   - Fare is heavily right-skewed
   - Large variance suggests very different wealth levels

🚢 Class Distribution:
   - {pct_class3:.1f}% traveled 3rd class (majority!)
   - Clear economic stratification

⚠️  Data Quality:
   - Age missing in ~20% of records (needs imputation)
   - Cabin missing in ~77% (likely drop this column)
   - Fare missing in ~2% (easy to impute)

🔧 ML Preprocessing Needs:
   - Log-transform Fare (heavily skewed)
   - Impute Age (median or model-based)
   - Drop Cabin (too many missing)
   - Encode Sex and Embarked (categorical → numeric)
""")
