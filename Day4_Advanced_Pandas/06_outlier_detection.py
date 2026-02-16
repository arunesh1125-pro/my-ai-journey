import pandas as pd
import numpy as np

print("="*70)
print("OUTLIER DETECTION AND HANDLING")
print("="*70)

"""
Outliers are extreme values that deviate significantly from other data.
In ML: Outliers can severely affect model performance!
"""

# Create dataset with outliers
np.random.seed(42)
normal_data = np.random.normal(50, 10, 100)
# Add deliberate outliers
outliers_data = np.append(normal_data, [150, 200, -50, -100, 180])

df = pd.DataFrame({
    'Values':outliers_data
})
print("Dataset with outliers: ")
print(f"Shape: {df.shape}")
print(f"Basic stats:\n{df['Values'].describe()}")
print()

 # METHOD 1: Z-SCORE METHOD

print("="*70)
print("METHOD 1: Z-SCORE")
print("="*70)

"""
Z-score = (value - mean) / std
Outlier threshold: |z-score| > 3
"""

mean = df['Values'].mean()
std = df["Values"].std()

df['Z_Score'] = (df['Values']-mean)/std
df['Is_Outliers_Z'] = df['Z_Score'].abs() > 3

print(f"Z-score outliers (|z| > 3):")
print(df[df['Is_Outliers_Z']][['Values', 'Z_Score']])
print(f"Count: {df['Is_Outliers_Z'].sum()}")
print()

# Remove z-score outliers
df_no_outliers_z = df[~df['Is_Outliers_Z']]['Values']
print("After removing z-score outliers: ")
print(f"Original: {len(df)} rows -> Cleaned: {len(df_no_outliers_z)} rows")
print()

# METHOD 2: IQR METHOD (More Robust!)

print("="*70)
print("METHOD 2: IQR (Interquartile Range)")
print("="*70)

"""
IQR = Q3 - Q1
Lower fence = Q1 - 1.5 * IQR
Upper fence = Q3 + 1.5 * IQR
Anything outside fences = outlier
"""

Q1 = df['Values'].quantile(0.25)
Q3 = df['Values'].quantile(0.75)
IQR = Q3-Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"Q1: {Q1:.2f}")
print(f"Q3: {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print()

df['Is_Outlier_IQR'] = (df['Values'] < lower_fence) | (df['Values'] > upper_fence)

print("IQR outliers: ")
print(df[df['Is_Outlier_IQR']][['Values']])
print(f"Count: {df['Is_Outlier_IQR'].sum()}")
print()

# HANDLING OUTLIERS

print("="*70)
print("HANDLING STRATEGIES")
print("="*70)

Values = df['Values'].copy()

# Strategy 1: Remove outliers
values_removed = Values[~df['Is_Outlier_IQR']]
print(f"Remove: {len(Values)} -> {len(values_removed)} rows")

# Strategy 2: Cap/Clip (Winsorization)
values_capped = Values.clip(lower=lower_fence, upper=upper_fence)
print(f"\nBefore capping - Max: {Values.max():.2f}, Min: {Values.min():.2f}")
print(f"After capping - Max: {values_capped.max():.2f}, Min: {values_capped.min():.2f}")

# Strategy 3: Replace with median
median_value = Values.median()
values_replaced = Values.copy()
values_replaced[df['Is_Outlier_IQR']] == median_value
print(f"\nAfter replacing outliers with median ({median_value:.2f})")
print(f"Max: {values_replaced.max():.2f}, Min: {values_replaced.min():.2f}")
print()

 # WHEN To REMOVE vs KEEP OUTLIERS

guide = """
OUTLIER HANDLING DECISION GUIDE:

REMOVE outliers when:
  ✅ Clearly data entry errors (age = 999)
  ✅ Measurement errors
  ✅ Outlier is not representative of target population

KEEP outliers when:
  ✅ They represent real (rare) events
  ✅ Fraud detection (outliers ARE what you want to find!)
  ✅ Anomaly detection problems
  ✅ The outlier could be a signal, not noise

CAP/CLIP when:
  ✅ You want to preserve all data
  ✅ Outliers are extreme but valid
  ✅ Neural networks (sensitive to large values)

TRANSFORM (log, sqrt) when:
  ✅ Data is heavily right-skewed
  ✅ Income/price/count data
  ✅ Preserves relative differences
"""
print(guide)
