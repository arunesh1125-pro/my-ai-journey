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
    z = np.polyfit(df_marketing[feature], df_marketing['Sales'], 1) # Calculate the line of best fit (slope and intercept)
    p = np.poly1d(z) # Turns those calculations into function
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

# PROJECT 2: ADVANCED SALARY PREDICTOR

print("\n" + "="*70)
print("PROJECT 2: ADVANCED EMPLOYEE SALARY PREDICTOR")
print("Features: Experience, Education, City, Skills")
print("="*70)

# Generate realistic data
np.random.seed(123)
n_employees = 500

# Features
experience = np.random.uniform(0, 20, n_employees)
education = np.random.choice([12, 15, 16, 18, 21], n_employees) # Years of eduction
city_tier = np.random.choice([1, 2, 3], n_employees, p=[0.3, 0.4, 0.3])
skill_score = np.random.uniform(40, 100, n_employees)  # Technical assessment score

# Salary formula (complex real-world relationship)
base_salary = 400000  # ₹4L base

# Experience factor (diminishing returns)
exp_factor = 350000 * experience * (1 - 0.02 * experience)

# Education premium
edu_premium = {12:0, 15: 50000, 16: 100000, 18: 200000, 21: 400000}
edu_factor = np.array([edu_premium[e] for e in education])

# City cost of living adjustment
city_factor = {1: 200000, 2: 100000, 3: 0}  # Tier 1/2/3 cities
city_adj = np.array([city_factor[c] for c in city_tier])

# Skills premium
skill_factor = skill_score * 5000

# Final salary with noise
salary = (base_salary + exp_factor + edu_factor + city_adj + skill_factor +
          np.random.normal(0, 80000, n_employees))

# Convert to lakhs
salary_lakhs = salary / 100000

# Create DataFrame
df_salary = pd.DataFrame({
    'Experience_Years': experience,
    'Education_Years': education,
    'City_Tier': city_tier,
    'Skill_Score': skill_score,
    'Salary_Lakhs': salary_lakhs
})

print("\n📊 Employee Dataset Overview:")
print(df_salary.head(10))
print(f"\nDataset shape: {df_salary.shape}")
print("\nSalary Statistics:")
print(df_salary['Salary_Lakhs'].describe().round(2))

# Correlations
print("\n📈 Feature Correlations:")
print(df_salary.corr()['Salary_Lakhs'].sort_values(ascending=False).round(3))

# Prepare data
X_Salary = df_salary[['Experience_Years', 'Education_Years', 'City_Tier', 'Skill_Score']]
y_Salary = df_salary['Salary_Lakhs']

# Split
X_train_sal, X_test_sal, y_train_sal, y_test_sal = train_test_split(
    X_Salary, y_Salary, test_size=0.2, random_state=42
)

# Train model
model_salary = LinearRegression()
model_salary.fit(X_train_sal, y_train_sal)

# Predict
y_pred_sal = model_salary.predict(X_test_sal)

# Evaluate
r2_sal = r2_score(y_test_sal, y_pred_sal)
rmse_sal = np.sqrt(mean_squared_error(y_test_sal, y_pred_sal))
mae_sal = mean_absolute_error(y_test_sal, y_pred_sal)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"R² Score:      {r2_sal:.4f}")
print(f"RMSE:          ₹{rmse_sal:.2f} lakhs")
print(f"MAE:           ₹{mae_sal:.2f} lakhs")

# Feature importance
print("\n" + "="*70)
print("SALARY FACTORS ANALYSIS")
print("="*70)

salary_factors = pd.DataFrame({
    'Factor': X_Salary.columns,
    'Impact_Per_Unit': model_salary.coef_
}).sort_values('Impact_Per_Unit', ascending=False)

print(f"\nBase Salary: ₹{model_salary.intercept_:.2f} lakhs")
print(f"\n{'Factor':>20} {'Impact per Unit'}:>20")
print("-"*45)
for idx, row in salary_factors.iterrows():
    print(f"{row['Factor']:>20} ₹{row['Impact_Per_Unit']:>19.2f}L")

print("\n💡 Interpretation:")
print(f"  → Each year of experience: +₹{salary_factors[salary_factors['Factor']=='Experience_Years']['Impact_Per_Unit'].values[0]:.2f}L")
print(f"  → Each year of education: +₹{salary_factors[salary_factors['Factor']=='Education_Years']['Impact_Per_Unit'].values[0]:.2f}L")
print(f"  → Each skill point: +₹{salary_factors[salary_factors['Factor']=='Skill_Score']['Impact_Per_Unit'].values[0]:.3f}L")
print(f"  → City tier impact: +₹{salary_factors[salary_factors['Factor']=='City_Tier']['Impact_Per_Unit'].values[0]:.2f}L per tier")

# HR Use Cases
print("\n" + "="*70)
print("HR USE CASES: SALARY PREDICTIONS")
print("="*70)

candidates = pd.DataFrame({
    'Experience_Years': [2, 5, 8, 12, 15],
    'Education_Years': [16, 18, 16, 21, 18],
    'City_Tier': [1, 1, 2, 1, 2],
    'Skill_Score': [65, 75, 82, 90, 88]
})

predicted_salaries = model_salary.predict(candidates)

print(f"\n{'Profile':^80}")
print("="*80)
print(f"{'Exp':>5} {'Edu':>5} {'City':>6} {'Skills':>8} | {'Predicted Salary':>20} {'Annual CTC':>15}")
print("-"*80)
for i, pred in enumerate(predicted_salaries):
    annual_ctc = pred * 100000
    print(f"{candidates.iloc[i]['Experience_Years']:>5.0f}"
          f"{candidates.iloc[i]['Education_Years']:>5.0f}"
          f"{'T'+str(int(candidates.iloc[i]['City_Tier'])):>6}"
          f"{candidates.iloc[i]['Skill_Score']:>8.0f} | "
          f"₹{pred:>19.2f}L "
          f"₹{annual_ctc:>14,.0f}")
    
# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('EMPLOYEE SALARY PREDICTION MODEL', fontsize=16, fontweight='bold')

# Actual vs Predicted
axes[0, 0].scatter(y_test_sal, y_pred_sal, alpha=0.6, s=60,
                   color='#3498db', edgecolors='black')
axes[0, 0].plot([y_test_sal.min(), y_test_sal.max()],
                [y_test_sal.min(), y_test_sal.max()],
                'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual Salary (₹L)', fontweight='bold')
axes[0, 0].set_ylabel('Predicted Salary (₹L)', fontweight='bold')
axes[0, 0].set_title(f'Model Accuracy (R²={r2_sal:.3f})', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Experience impact
axes[0, 1].scatter(df_salary['Experience_Years'], df_salary['Experience_Years'],
                   alpha=0.5, s=40, color='#2ecc71', edgecolors='black', linewidths=0.5)
axes[0, 1].set_xlabel('Experience (Years)', fontweight='bold')
axes[0, 1].set_ylabel('Salary (₹L)', fontweight='bold')
axes[0, 1].set_title('Experience Impact', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Education Impact
edu_salary = df_salary.groupby('Education_Years')['Salary_Lakhs'].mean().sort_index()
axes[0, 2].bar(edu_salary.index, edu_salary.values, color='#e74c3c', 
               edgecolor='black', linewidth=1.5)
axes[0, 2].set_xlabel('Education (Years)', fontweight='bold')
axes[0, 2].set_ylabel('Avg Salary (₹L)', fontweight='bold')
axes[0, 2].set_title('Education Premium', fontweight='bold')
axes[0, 2].grid(axis='y', alpha=0.3)

#City tier comparison
city_salary = df_salary.groupby('City_Tier')['Salary_Lakhs'].mean()
axes[1, 0].bar(['Tier 1', 'Tier 2', 'Tier 3'], city_salary.values,
               color=['#FFD700', '#C0C0C0', '#CD7F32'], edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Avg Salary (₹L)', fontweight='bold')
axes[1, 0].set_title('City Tier Impact', fontweight='bold')

# Skills impact
axes[1, 1].scatter(df_salary['Skill_Score'], df_salary['Salary_Lakhs'],
                   alpha=0.5, s=40, color='#9b59b6', edgecolors='black', linewidths=0.5)
axes[1, 1].set_xlabel('Skill Score', fontweight='bold')
axes[1, 1].set_ylabel('Salary (₹L)', fontweight='bold')
axes[1, 1].set_title('Skills Premium', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Feature importance
axes[1, 2].barh(salary_factors['Factor'], np.abs(salary_factors['Impact_Per_Unit']),
                color='#3498db', edgecolor='black', linewidth=1.5)
axes[1, 2].set_xlabel('Impact (₹L per unit)', fontweight='bold')
axes[1, 2].set_title('Feature Importance', fontweight='bold')
axes[1, 2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('03_salary_prediction_advanced.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 03_salary_prediction_advanced.png")

print("\n" + "="*70)
print("PROJECT 2 COMPLETE: Advanced Salary Predictor")
print("="*70)
print("\nMULTIPLE LINEAR REGRESSION MASTERED!")
print("="*70)