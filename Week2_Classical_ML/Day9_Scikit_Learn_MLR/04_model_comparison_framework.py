"""
PROJECT 4: SYSTEMATIC MODEL COMPARISON FRAMEWORK
=================================================
Professional approach to comparing multiple ML models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time

print("="*70)
print("MODEL COMPARISON FRAMEWORK")
print("="*70)

# WHY SYSTEMATIC COMPARISON?

print("""
╔══════════════════════════════════════════════════════════════╗
║            WHY SYSTEMATIC MODEL COMPARISON?                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Amateur Approach:                                           ║
║  • Try one model                                             ║
║  • "Good enough" → Ship it                                   ║
║  • No comparison                                             ║
║                                                              ║
║  Professional Approach:                                      ║
║  • Try multiple models                                       ║
║  • Compare systematically                                    ║
║  • Use cross-validation                                      ║
║  • Measure multiple metrics                                  ║
║  • Consider trade-offs (accuracy vs speed vs interpretability)║
║  • Document decision rationale                               ║
║                                                              ║
║  This is what separates junior from senior ML engineers!     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# LOAD COMPREHENSIVE DATASET

print("\n" + "="*70)
print("DATASET: COMPREHENSIVE MARKETING ANALYTICS")
print("="*70)

# Generate comprehensive marketing dataset
np.random.seed(42)
n = 500

# Marketing channels
tv = np.random.uniform(20, 250, n)
radio = np.random.uniform(10, 120, n)
social = np.random.uniform(5, 100, n)
email = np.random.uniform(3, 80, n)
influencer = np.random.uniform(10, 150, n)

# Context features
season = np.random.choice([1, 2, 3, 4], n)
competitor_spend = np.random.uniform(50, 300, n)
economic_index = np.random.uniform(80, 120, n)

# Target (complex relationship)
sales = (150 +
         2.3 * tv +
         1.8 * radio +
         3.1 * social +
         2.2 * email +
         1.4 * influencer +
         20 * season - 
         0.3 * competitor_spend +
         1.5 * economic_index +
         np.random.normal(0, 25, n))

# Create DataFrame
df = pd.DataFrame({
    'TV': tv,
    'Radio': radio,
    'Social_Media': social,
    'Email': email,
    'Influencer': influencer,
    'Season': season,
    'Competitor_Spend': competitor_spend,
    'Economic_Index': economic_index,
    'Sales': sales
})

print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]-1} features")
print("\nFirst 5 rows:")
print(df.head())

print("\nCorrelation with Sales: ")
correlations = df.corr()['Sales'].sort_values(ascending=False)
print(correlations.round(3))

# Prepare data
X = df.drop('Sales', axis=1)
y = df['Sales']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# DEFINE MODELS TO COMPARE

print("\n" + "="*70)
print("MODELS TO COMPARE")
print("="*70)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10)': Ridge(alpha=10.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1, max_iter=10000),
    'Lasso (α=1.0)': Lasso(alpha=1.0, max_iter=10000),
    'Lasso (α=10)': Lasso(alpha=10.0, max_iter=10000),
    'ElasticNet (α=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
    'ElasticNet (α=1.0)': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
}

print(f"\nComparing {len(models)} different models/configurations: ")
for name in models.keys():
    print(f"  • {name}")

# EVALUATION METRICS

print("\n" + "="*70)
print("EVALUATION METRICS EXPLAINED")
print("="*70)

print("""
We'll evaluate each model on:
      
1. R² Score (Coefficient of Determination)
   → Range: 0 to 1 (can be negative)
   → Higher is better
   → "% of variance explained"
      
2. RMSE (Root Mean Squared Error)
   → Same units as target
   → Lower is better
   → Penalizes large errors heavily
      
3. MAE (Mean Absolute Error)
   → Same units as target
   → Lower is better
   → More robust to outliers
      
4. Cross-Validation Score
   → Average R² across 5 folds
   → More reliable than single train/test split
   → Reduces variance in evaluation
      
5. Training Time
   → Computational efficiency
   → Matters for large datasets
      
6. Number of Features Used
   → Simplicity (for Lasso)
   → Fewer features = easier to deploy
""")

# TRAIN AND EVALUATE ALL MODELS

print("\n" + "="*70)
print("TRAINING ALL MODELS...")
print("="*70)

results = []

for name, model in models.items():
    print(f"\nTraining: {name}")

    # Measure training time
    start_time = time.time()
    
    # Train
    model.fit(X_train, y_train)

    # Training time
    training_time = time.time() - start_time
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Feature count (for Lasso)
    if hasattr(model, 'coef_'):
        n_features = np.sum(np.abs(model.coef_) > 1e-5)
    else:
        n_features = X.shape[1]

    results.append({
            'Model': name,
            'R²_Train': r2_train,
            'R²_Test': r2_test,
            'RMSE_Train': rmse_train,
            'RMSE_Test': rmse_test,
            'MAE_Train': mae_train,
            'MAE_Test': mae_test,
            'CV_Mean': cv_mean,
            'CV_Std': cv_std,
            'Overfit_Gap': r2_train - r2_test,
            'Train_Time': training_time,
            'N_Features': n_features
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*70)
print("COMPLETE RESULTS TABLE")
print("="*70)
print(results_df.round(4).to_string(index=False))

# RANK MODELS

print("\n" + "="*70)
print("MODEL RANKINGS")
print("="*70)

# Rank by test R²
ranked_r2 = results_df.sort_values('R²_Test', ascending=False)
print("\n📊 Ranked by Test R² (Higher is Better):")
print("-"*60)
for i, (_, row) in enumerate(ranked_r2.iterrows(), 1):
    print(f"{i:>5} {row['Model']:>25} {row['R²_Test']:>12.4f} {row['CV_Mean']:>12.4f}")

# Rank by RMSE
ranked_rmse = results_df.sort_values('RMSE_Test', ascending=False)
print("\n📊 Ranked by Test RMSE (Lower is Better):")
print(f"{'Rank':>5} {'Model':>25} {'RMSE':>12}")
print("-"*45)
for i, (_, row) in enumerate(ranked_rmse.iterrows(), 1):
    print(f"{i:>5} {row['Model']:>25} {row['RMSE_Test']:>12.2f}")

# Rank by simplicity (fewer features)
ranked_simple = results_df.sort_values(['N_Features', 'R²_Test'],
                                       ascending=[True, False])
print("\n📊 Ranked by Simplicity (Fewest Features):")
print(f"{'Rank':>5} {'Model':>25} {'Features':>12} {'Test R²':>12}")
print("-"*60)
for i, (_, row) in enumerate(ranked_simple.iterrows(), 1):
    print(f"{i:>5} {row['Model']:>25} {row['N_Features']:>12.0f} {row['RMSE_Test']:>12.4f}")

# Best Overall (test R²)
best_overall = results_df.loc[results_df['R²_Test'].idxmax()]
print(f"\n✅ BEST OVERALL PERFORMANCE:")
print(f"  Model: {best_overall['Model']}")
print(f"  Test R²: {best_overall['R²_Test']:.4f}")
print(f"  RMSE: {best_overall['RMSE_Test']:.2f}")
print(f"  CV Score: {best_overall['CV_Mean']:.4f} ± {best_overall['CV_Std']:.4f}")

# Most stable (lowest CV Std)
most_stable = results_df.loc[results_df['CV_Std'].idxmax()]
print(f"\n✅ MOST STABLE MODEL:")
print(f"   Model: {most_stable['Model']}")
print(f"   CV Std: {most_stable['CV_Std']:.4f}")
print(f"   Test R²: {most_stable['R²_Test']:.4f}")

# Fastest
fastest = results_df.loc[results_df['Train_Time'].idxmax()]
print(f"\n✅ FASTEST TO TRAIN:")
print(f"   Model: {fastest['Model']}")
print(f"   Time: {fastest['Train_Time']*1000:.2f} ms")

# Simplest with good performance (Lasso with fewest features)
lasso_results = results_df[results_df['Model'].str.contains('Lasso')]
if len(lasso_results) > 0:
    simplest_good = lasso_results.sort_values(['N_Features', 'R²_Test'],
                                              ascending=[True, False]).iloc[0]
    print(f"\n✅ SIMPLEST MODEL (Feature Selection):")
    print(f"   Model: {simplest_good['Model']}")
    print(f"   Features: {simplest_good['N_Features']:.0f} / {X.shape[1]}")
    print(f"   Test R²: {simplest_good['R²_Test']:.4f}")

# VISUALIZATIONS

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: R² Comparison
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(results_df))
ax1.barh(x_pos, results_df['R²_Test'], color='#3498db', 
         alpha=0.7, label='Test R²', edgecolor='black')
ax1.barh(x_pos, results_df['R²_Train'], color='#e74c3c', 
         alpha=0.4, label='Train R²', edgecolor='black')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(results_df['Model'], fontsize=9)
ax1.set_xlabel('R² Score', fontweight='bold', fontsize=11)
ax1.set_title('Model Performance Comparison (R²)', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(x=0.95, color='green', linestyle='--', linewidth=2, 
            alpha=0.5, label='Excellent (0.95)')

# Plot2: RMSE Comparison
ax2 = fig.add_subplot(gs[1, 0])
sorted_rmse = results_df.sort_values('RMSE_Test')
ax2.bar(range(len(sorted_rmse)), sorted_rmse['RMSE_Test'], 
        color='#2ecc71', edgecolor='black')
ax2.set_xticks(range(len(sorted_rmse)))
ax2.set_xticklabels(sorted_rmse['Model'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('RMSE', fontweight='bold')
ax2.set_title('RMSE Comparison (Lower=Better)', fontweight='bold', fontsize=12)
ax2.grid(axis='y', alpha=0.3)

# Plot3: Cross-Validation Scores with Error Bars
ax3 = fig.add_subplot(gs[1, 1])
ax3.errorbar(range(len(results_df)), results_df['CV_Mean'],
                  yerr=results_df['CV_Std'], fmt='o', markersize=8,
                  capsize=5, capthick=2, color='#9b59b6', 
             ecolor='#9b59b6', elinewidth=2)
ax3.set_xticks(range(len(results_df)))
ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('CV Scores', fontweight='bold')
ax3.set_title('Cross-Validation Scores', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)

# Plot4: Overfitting Analysis
ax4 = fig.add_subplot(gs[1, 2])
colors_overfit = ['#2ecc71' if gap < 0.05 else '#f39c12' if gap < 0.1 else '#e74c3c' 
                  for gap in results_df['Overfit_Gap']]
ax4.bar(range(len(results_df)), results_df['Overfit_Gap'],
        color=colors_overfit, edgecolor='black')
ax4.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, 
            alpha=0.7, label='Warning (0.05)')
ax4.axhline(y=0.1, color='red', linestyle='--', linewidth=2, 
            alpha=0.7, label='Overfitting (0.1)')
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('Train R² - Test R²', fontweight='bold')
ax4.set_title('Overfitting Analysis', fontweight='bold', fontsize=12)
ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3)

# Plot5: Feature Count (Lasso models)
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(range(len(results_df)), results_df['N_Features'],
        color='#e74c3c', edgecolor='black', alpha=0.7)
ax5.axhline(y=X.shape[1], color='blue', linestyle='--', linewidth=2,
            label=f'All Features ({X.shape[1]})')
ax5.set_xticks(range(len(results_df)))
ax5.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax5.set_ylabel('Number of Features', fontweight='bold')
ax5.set_title('Model Complexity (Features Used)', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Plot6: Training Time
ax6 = fig.add_subplot(gs[2, 1])
train_times_ms = results_df['Train_Time'] * 1000
ax6.bar(range(len(results_df)), train_times_ms, 
        color='#f39c12', edgecolor='black', alpha=0.7)
ax6.set_xticks(range(len(results_df)))
ax6.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax6.set_ylabel('Time (milliseconds)', fontweight='bold')
ax6.set_title('Training Time', fontweight='bold', fontsize=12)
ax6.grid(axis='y', alpha=0.3)

# Plot7: Accuracy vs Simplicity Scatter
ax7 = fig.add_subplot(gs[2, 2])
scatter = ax7.scatter(results_df['N_Features'], results_df['R²_Test'],
                      s=200, c=results_df['RMSE_Test'], cmap='RdYlGn_r',
                      edgecolors='black', linewidth=2, alpha=0.7)
for i, row in results_df.iterrows():
    ax7.annotate(row['Model'].split()[0],
                 (row['N_Features'], row['RMSE_Test']),
                 fontsize=7, ha='center')
ax7.set_xlabel('Number of Features', fontweight='bold')
ax7.set_ylabel('Test R²', fontweight='bold')
ax7.set_title('Accuracy vs Simplicity Trade-off', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3)    
plt.colorbar(scatter, ax=ax7, label='RMSE')

plt.suptitle('COMPREHENSIVE MODEL COMPARISON DASHBOARD', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('08_model_comparison_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 08_model_comparison_dashboard.png")

# FINAL RECOMMENDATION

print("\n" + "="*70)
print("📋 FINAL RECOMMENDATION")
print("="*70)

recommendation = f"""
RECOMMENDED MODEL: {best_overall['Model']}
{'═'*50}

JUSTIFICATION:
  ✅ Highest test R² score: {best_overall['R²_Test']:.4f}
  ✅ Low RMSE: {best_overall['RMSE_Test']:.2f}
  ✅ Cross-validation: {best_overall['CV_Mean']:.4f} ± {best_overall['CV_Std']:.4f}
  ✅ Overfit gap: {best_overall['Overfit_Gap']:.4f} (acceptable)

DEPLOYMENT SPECIFICATIONS:
  • Training data: {len(X_train)} samples
  • Features: {best_overall['N_Features']:.0f}
  • Expected error: ±{best_overall['MAE_Test']:.2f} (MAE)
  • Training time: {best_overall['Train_Time']*1000:.2f} ms

ALTERNATIVE CONSIDERATION:
If interpretability is priority → Use {simplest_good['Model']}
  • Only {simplest_good['N_Features']:.0f} features (vs {X.shape[1]})
  • R² = {simplest_good['R²_Test']:.4f} (slight drop)
  • Easier to explain to stakeholders

MONITORING PLAN:
  1. Track prediction errors monthly
  2. Retrain if test R² drops below {best_overall['R²_Test'] * 0.95:.3f}
  3. Add new features if market changes
  4. Compare against baseline (R² = {results_df[results_df['Model']=='Linear Regression']['R²_Test'].values[0]:.3f})
"""
print(recommendation)

# Save recommendation to file
with open('model_recommendation.txt', 'w', encoding='utf-8') as f:
    f.write(recommendation)
print("\n✅ Saved: model_recommendation.txt")

print("\n" + "="*70)
print("PROJECT 4 COMPLETE: MODEL COMPARISON FRAMEWORK")
print("="*70)