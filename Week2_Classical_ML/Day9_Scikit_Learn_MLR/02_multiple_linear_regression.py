"""
MULTIPLE LINEAR REGRESSION: HANDLING MANY FEATURES
===================================================
Real-world ML with 3-5+ features simultaneously
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("="*70)
print("MULTIPLE LINEAR REGRESSION")
print("="*70)

# THEORY: MULTIPLE LINEAR REGRESSION

print("""
╔══════════════════════════════════════════════════════════════╗
║           MULTIPLE LINEAR REGRESSION EQUATION                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Simple (1 feature):                                         ║
║  y = b₀ + b₁x₁                                               ║
║                                                              ║
║  Multiple (n features):                                      ║
║  y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ                          ║
║                                                              ║
║  Example: Salary Prediction                                  ║
║  Salary = b₀ + b₁(Experience) + b₂(Education) +             ║
║            b₃(City_Cost_Index) + b₄(Skill_Score)            ║
║                                                              ║
║  Where:                                                      ║
║  b₀ = Intercept (base value)                                 ║
║  b₁, b₂, ... = Coefficients (impact of each feature)         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# PROJECT 1: MARKETING MIX MODELING

print("\n" + "="*70)
print("PROJECT 1: MARKETING MIX MODEL (MMM)")
print("Predict Sales from Multiple Marketing Channels")
print("="*70)

# Generate realistic marketing mix data
np.random.seed(42)
n_months = 100

# Multiple marketing channels
tv_ads = np.random.uniform(20, 200, n_months)       # ₹ lakhs
radio_ads = np.random.uniform(10, 100, n_months)    # ₹ lakhs
social_media = np.random.uniform(5, 80, n_months)   # ₹ lakhs
influencer = np.random.uniform(10, 120, n_months)   # ₹ lakhs
season = np.random.choice([1, 2, 3, 4], n_months)   # Quarter

# True relationship (with different ROI per channel)
sales = (100 +                          # Base sales
         2.5 * tv_ads +                 # TV has high ROI
         1.8 * radio_ads +              # Radio moderate
         3.2 * social_media +           # Social media high ROI
         1.5 * influencer +             # Influencer moderate
         15 * season +                  # Seasonal effect
         np.random.normal(0, 30, n_months))  # Noise

# Create DataFrame
df_marketing = pd.DataFrame({
    'TV_Ads': tv_ads,
    'Radio_Ads': radio_ads,
    'Social_Media': social_media,
    'Influencer': influencer,
    'Season': season,
    'Sales': sales
})

print("\n📊 Dataset Overview:")
print(df_marketing.head(10))
print(f"\nDataset shape: {df_marketing.shape}")
print("\nDescriptive Statistics")
print(df_marketing.describe().round(2))

# Correlation analysis
print("\n📈 Feature Correlations with Sales:")
correlations = df_marketing.corr()['Sales'].sort_values(ascending=False)
print(correlations)

# Prepare features and target
X = df_marketing[['TV_Ads', 'Radio_Ads', 'Social_Media', 'Influencer', 'Season']]
y = df_marketing['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} months")
print(f"Test set: {len(X_test)} months")

# Train model
model_mmm = LinearRegression()
model_mmm.fit(X_train, y_train)

# Make predictions
y_pred = model_mmm.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"R² Score:      {r2:.4f}")
print(f"RMSE:          ₹{rmse:.2f} lakhs")
print(f"MAE:           ₹{mae:.2f} lakhs")

# Feature importance (coefficients)
print("\n" + "="*70)
print("FEATURE IMPORTANCE (ROI per Channel)")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_mmm.coef_,
    'Abs_Coefficient': np.abs(model_mmm.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nIntercept (Base Sales): ₹{model_mmm.intercept_:.2f} lakhs")
print(f"\n{'Channel':>15} {'ROI Multiplier':>15} {'Impact':>10}")
print("-"*45)
for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']:>15} {row['Coefficient']:>15.2f}x {'High' if abs(row['Coefficient']) > 2 else 'Moderate':>10}")

# Business insights
print("\n" + "="*70)
print("💡 BUSINESS INSIGHTS")
print("="*70)

print(f"""
Marketing Mix Model Results:
{'─'*50}
1. CHANNEL PERFORMANCE
   → Social Media: Best ROI (₹{feature_importance[feature_importance['Feature']=='Social_Media']['Coefficient'].values[0]:.2f}x return)
   → TV Ads: Strong ROI (₹{feature_importance[feature_importance['Feature']=='TV_Ads']['Coefficient'].values[0]:.2f}x return)
   → Radio: Moderate ROI (₹{feature_importance[feature_importance['Feature']=='Radio_Ads']['Coefficient'].values[0]:.2f}x return)
   → Influencer: Lower ROI (₹{feature_importance[feature_importance['Feature']=='Influencer']['Coefficient'].values[0]:.2f}x return)

2. RECOMMENDATIONS
   → Increase social media budget (highest ROI)
   → Maintain TV advertising (proven channel)
   → Consider reducing influencer spend (lower efficiency)

3. SEASONAL IMPACT
   → Sales vary by ₹{feature_importance[feature_importance['Feature']=='Season']['Coefficient'].values[0]:.2f}L per quarter
   → Plan inventory and campaigns accordingly

4. MODEL RELIABILITY
   → R² = {r2:.2%} of sales variance explained
   → Typical prediction error: ±₹{mae:.2f} lakhs
""")

# Budget optimization example
print("\n" + "="*70)
print("BUDGET OPTIMIZATION SCENARIO")
print("="*70)

budget_scenerio = pd.DataFrame({
    'TV_Ads': [100, 150, 80],
    'Radio_Ads': [50, 30, 60],
    'Social_Media': [40, 60, 80],
    'Influencer': [30, 20, 40],
    'Season': [3, 3, 3]   # Q3
})

predicted_sales = model_mmm.predict(budget_scenerio)
total_spend = budget_scenerio.drop('Season', axis=1).sum(axis=1)
roi = ((predicted_sales - total_spend) / total_spend * 100)

print(f"\n{'Scenario':>10} {'TV':>10} {'Radio':>10} {'Social':>10} {'Influencer':>12} {'Total':>10} {'Sales':>10} {'ROI %':>10}")
print("-"*90)
for i in range(len(budget_scenerio)):
    print(f"{'#'+str(i+1):>10} "
          f"{budget_scenerio.iloc[i]['TV_Ads']:>10.0f} "
          f"{budget_scenerio.iloc[i]['Radio_Ads']:>10.0f} "
          f"{budget_scenerio.iloc[i]['Social_Media']:>10.0f} "
          f"{budget_scenerio.iloc[i]['Influencer']:>12.0f} "
          f"{total_spend.iloc[i]:>10.0f} "
          f"{predicted_sales[i]:>10.2f} "
          f"{roi.iloc[i]:>10.1f}")
    
# Visualizations
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Actual vs Predicted (main)
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test, y_pred, alpha=0.6, s=80, color='#3498db', edgecolors='black')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Predictions')
ax1.set_xlabel('Actual Sales (₹ lakhs)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Predicted Sales (₹ lakhs)', fontweight='bold', fontsize=12)
ax1.set_title(f'Marketing Mix Model Performance (R²={r2:.3f})', 
              fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Feature Importance
ax2 = fig.add_subplot(gs[0, 2])
colors_imp = ['#2ecc71' if c > 2.5 else '#3498db' if c > 2.0 else '#f39c12' 
              for c in feature_importance['Coefficient']]
ax2.barh(feature_importance['Feature'], feature_importance['Coefficient'],
         color=colors_imp, edgecolor='black')
ax2.set_xlabel('ROI Multiplier', fontweight='bold', fontsize=11)
ax2.set_title('Channel Performance', fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

# Plots 3-7: Individual feature relationships
features_to_plot = ['TV_Ads', 'Social_Media', 'Radio_Ads', 'Influencer', 'Season']
positions = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

for (feature, pos) in zip(features_to_plot, positions):
    ax = fig.add_subplot(gs[pos])
    ax.scatter(df_marketing[feature], df_marketing['Sales'], 
               alpha=0.5, s=40, color='#9b59b6', edgecolors='black', linewidths=0.5)
    # Add trend line
    z = np.polyfit(df_marketing[feature], df_marketing['Sales'], 1)
    p = np.poly1d(z)
    ax.plot(df_marketing[feature].sort_values(), 
            p(df_marketing[feature].sort_values()), 
            "r-", linewidth=2, alpha=0.7)
    ax.set_xlabel(feature.replace('_', ' '), fontweight='bold', fontsize=10)
    ax.set_ylabel('Sales', fontweight='bold', fontsize=10)
    ax.set_title(f'{feature} Impact', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)

# Plot 8: Residuals
ax8 = fig.add_subplot(gs[2, 2])
residuals = y_test - y_pred
ax8.scatter(y_pred, residuals, alpha=0.6, s=60, color='#e74c3c', edgecolors='black')
ax8.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax8.set_xlabel('Predicted Sales', fontweight='bold', fontsize=11)
ax8.set_ylabel('Residuals', fontweight='bold', fontsize=11)
ax8.set_title('Residual Plot', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3)

plt.suptitle('MARKETING MIX MODEL - COMPLETE ANALYSIS', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('02_marketing_mix_model.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 02_marketing_mix_model.png")

print("\n" + "="*70)
print("PROJECT 1 COMPLETE: Marketing Mix Model")
print("="*70)