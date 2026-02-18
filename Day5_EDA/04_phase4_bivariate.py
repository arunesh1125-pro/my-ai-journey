import pandas as pd
import numpy as np

print("="*70)
print("PHASE 4: BIVARIATE ANALYSIS")
print("="*70)

"""
BIVARIATE = Analyzing TWO variables together
Goal: Find relationships between features and target (Survived)
These relationships guide feature selection in ML!
"""

df = pd.read_csv('titanic.csv')

# 4.1 NUMERIC vs TARGET (Survived)

print("\n4.1 NUMERIC FEATURES vs SURVIVAL")
print("="*70)

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

for col in numerical_cols:
    survived = df[df['Survived'] == 1][col].dropna()
    not_survived = df[df['Survived'] == 0][col].dropna()

    print(f"\n{'-'*55}")
    print(f"📊 {col} vs SURVIVAL")
    print(f"\n{'-'*55}")
    print(f"{'Metric':20s} {'Survived':>12s} {'Not Survived':>12s} {'Diff':>8s}")
    print(f"\n{'-'*55}")

    metrics = {
        'Count': (len(survived), len(not_survived)),
        'Mean': (survived.mean(), not_survived.mean()),
        'Median': (survived.median(), not_survived.median()),
        'Std Dev': (survived.std(), not_survived.std()),
        'Min': (survived.min(), not_survived.min()),
        'Max': (survived.max(), not_survived.max())
    }
    for metric, (s_val, ns_val) in metrics.items():
        diff = s_val - ns_val
        diff_str = f"{diff:+.2f}"
        print(f"  {metric:18s} {s_val:>12.2f} {ns_val:>12.2f} {diff_str:>8s}")

    # T-test equivalent (manual)
    from scipy import stats

    # Simple effect size
    pooled_std = np.sqrt((survived.var() + not_survived.var()) / 2)
    if pooled_std > 0:
        cohens_d = (survived.mean() - not_survived.mean()) / pooled_std
        print(f"\n  Effect size (Cohen's d): {cohens_d:.3f}", end=" ")
        if abs(cohens_d) < 0.2:
            print("(Small effect)")
        elif abs(cohens_d) < 0.5:
            print("Medium effect")
        else:
            print("Large effect - IMPORTANT feature")
        print()

 # 4.2 CATEGORICAL vs TARGET

print("\n4.2 CATEGORICAL FEATURES vs SURVIVAL")
print("="*70)

cat_cols = ['Pclass', 'Sex', 'Embarked']

for col in cat_cols:
    print(f"\n{'─'*60}")
    print(f"📊 {col.upper()} vs SURVIVAL")
    print(f"{'─'*60}")

    # Survival rate by category
    survival_by_cat = df.groupby(col)['Survived'].agg([
        'count', 'sum', 'mean'
    ]).round(4)
    survival_by_cat.columns = ['Total', 'Survived', 'Survival_Rate']
    survival_by_cat['Not_Survived'] = (
        survival_by_cat['Total'] - survival_by_cat['Survived']
    )
    survival_by_cat['Death_Rate'] = (
        survival_by_cat['Survival_Rate']
    ).round(4)

    print(f"\n{'Category':12s} {'Total':>7s} {'Survived':>9s} "
          f"{'Not Surv':>9s} {'Surv Rate':>10s}  Visual")
    print("-"*70)

    overall_rate = df['Survived'].mean()

    for idx, row in survival_by_cat.iterrows():
        rate = row['Survival_Rate']
        bar_len = int(rate * 40)
        bar = "█" * bar_len

        # Marker: above/below average
        marker = "▲" if rate > overall_rate else "▼"

        print(f"  {str(idx):10s} {row['Total']:>7.0f} {row['Survived']:>9.0f} "
              f"{row['Not_Survived']:>9.0f} {rate:>9.1%} {bar} {marker}")
    print(f"\n  Overall survival rate: {overall_rate:.1%} (baseline)")

    # Key insights
    if col == 'Sex':
        female_rate = survival_by_cat.loc['female', 'Survival_Rate']
        male_rate = survival_by_cat.loc['male', 'Survival_Rate']
        print(f"\n   💡 INSIGHT: Women survived at {female_rate:.1%} vs men at {male_rate:.1%}")
        print(f"  💡 'Women and children first' policy clearly visible!")
        print(f"  💡 Sex will be MOST IMPORTANT feature in ML model!")
    elif col == 'Pclass':
        p1 = survival_by_cat.loc[1, 'Survival_Rate']
        p3 = survival_by_cat.loc[3, 'Survival_Rate']
        print(f"\n  💡 INSIGHT: 1st class survived at {p1:.1%} vs 3rd class {p3:.1%}")
        print(f"  💡 Strong relationship: wealth = better survival odds!")
        print(f"  💡 Pclass will be IMPORTANT feature in ML model!")
    print()

 # NUMERIC vs NUMERIC (Correlation)

print("\n4.3 NUMERIC-NUMERIC CORRELATIONS")
print("="*70)

numeric_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
corr_matrix = numeric_df.corr().round(3)

print("\nCorrelation Matrix: ")
print(corr_matrix.to_string())
print()

# Correlation with target
print("\nCorrelation with Survived (Target): ")
print("-"*40)
target_corr = corr_matrix['Survived'].drop('Survived').sort_values(
    key=abs, ascending=False
)
for feature, corr in target_corr.items():
    bar_len = int(abs(corr) * 30)
    bar = "█" * bar_len
    direction = "+" if corr > 0 else "-"

    if abs(corr) > 0.3:
        strength = "STRONG"
    elif abs(corr) > 0.1:
        strength = "moderate"
    else:
        strength = "weak"

    print(f"   {feature:8s}: {corr:+.3f}   |{direction}{bar:<30s}|  {strength}")
print()

# 4.4 CROSS-TABULATION

print("\n4.4 CROSS-TABULATION ANALYSIS")
print("="*70)

# Sex vs Pclass Survival
print("\nSurvival Rate by Sex AND Class: ")
crosstab = pd.crosstab(
    df['Pclass'],
    df['Sex'],
    values = df['Survived'],
    aggfunc = 'mean'
).round(3)

print(crosstab)
print("💡 Pattern: Females in all classes had HIGHER survival!")
print("💡 1st class females: near 100% survival")
print("💡 3rd class males: very low survival")
print()

# Age goups vs survival
df['Age_Group'] = pd.cut(    #pd.cut() - used to segement and sort data values into bins
    df['Age'], # Input
    bins = [0, 12, 18, 35, 60, 100],  # Bins - 0to12, 12to18, etc
    labels=['Child', 'Teen', 'Yound Adult', 'Adult', 'Senior']  #HUman readable bin name
)

print("\nSurvival Rate by Age Group: ")
age_survival = df.groupby('Age_Group', observed=True)['Survived'].agg(
    ['count', 'mean']
).round(3)
age_survival.columns = ['Count', 'Survival_Rate']
print(age_survival)
print("💡 Children had highest survival rates!")
print("💡 'Women and children first' policy confirmed!")
print()

# 4.5 BIVARIATE INSIGHTS SUMMARY

print("="*70)
print("4.5 BIVARIATE ANALYSIS INSIGHTS")
print("="*70)

print("""
Top Feature Insights for ML Model:
{'─'*50}
🥇 SEX (Strongest predictor):
   - Females: ~74% survival rate
   - Males:   ~19% survival rate
   - Will be #1 or #2 most important feature

🥈 PCLASS (Second strongest):
   - 1st class: ~63% survival
   - 2nd class: ~47% survival
   - 3rd class: ~24% survival

🥉 FARE (Correlated with Pclass):
   - Higher fare = better survival
   - But partially redundant with Pclass
   - Log transform recommended

4️⃣ AGE:
   - Children had best odds
   - Moderate negative correlation overall
   - Missing values need careful imputation

5️⃣ FAMILY SIZE (SibSp + Parch):
   - Small families (1-3) better than alone
   - Very large families had poor outcomes
   - Engineering family_size feature could help

⚙️ ML FEATURE ENGINEERING RECOMMENDATIONS:
   1. Create family_size = SibSp + Parch + 1
   2. Extract title from Name (Mr, Mrs, Miss, etc.)
   3. Create is_alone = (family_size == 1)
   4. Log-transform Fare
   5. Fill Age with group median by title
""")