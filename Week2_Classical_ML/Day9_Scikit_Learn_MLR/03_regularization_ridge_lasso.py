"""
PROJECT 3: REGULARIZATION - PREVENTING OVERFITTING
===================================================
Ridge (L2) and Lasso (L1) Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("="*70)
print("REGULARIZATION: RIDGE & LASSO REGRESSION")
print("="*70)

# WHAT IS REGULARIZATION?

print("""
╔══════════════════════════════════════════════════════════════╗
║                   WHAT IS REGULARIZATION?                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Problem: Overfitting                                        ║
║  Model learns training data TOO well (memorizes noise)       ║
║  → Performs great on training data                           ║
║  → Performs POORLY on new data                               ║
║                                                              ║
║  Solution: Regularization                                    ║
║  Add penalty for complex models                              ║
║  → Forces simpler, more generalizable models                 ║
║                                                              ║
║  Three Types:                                                ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ 1. Ridge (L2): Shrink all coefficients                │  ║
║  │    Cost = MSE + α × Σ(coefficient²)                   │  ║
║  │    → Good when all features are somewhat useful        │  ║
║  │                                                        │  ║
║  │ 2. Lasso (L1): Force some coefficients to ZERO        │  ║
║  │    Cost = MSE + α × Σ|coefficient|                    │  ║
║  │    → Feature selection (auto removes useless features) │  ║
║  │                                                        │  ║
║  │ 3. ElasticNet: Combination of Ridge + Lasso           │  ║
║  │    Cost = MSE + α₁×Σ(coef²) + α₂×Σ|coef|             │  ║
║  │    → Best of both worlds                               │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  α (alpha): Regularization strength                          ║
║  • α = 0     → No regularization (standard regression)       ║
║  • α small   → Light penalty                                 ║
║  • α large   → Strong penalty (simpler model)                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# DEMONSTRATE OVERFITTING PROBLEM

print("\n" + "="*70)
print("PART 1: DEMONSTRATING THE OVERFITTING PROBLEM")
print("="*70)

# Generate data with noise

# High model complexity: model memorizes, Small dataset: not enough generalization, High noise: model learns noise - Leads to Overfitting problem

np.random.seed(42)
n_samples = 20
X_demo = np.linspace(0, 10, n_samples)
y_true = 2 * X_demo + 3 # True realtionship
y_demo = y_true + np.random.normal(0, 8, n_samples) # Add noise

# Create polynomial features (1, x, x², x³, ..., x¹⁰)
poly = PolynomialFeatures(degree=10, include_bias=False) # include_bias=False means it doesn't manually add a column of 1s (intercept), as most regression models handle that automatically.
X_poly = poly.fit_transform(X_demo.reshape(-1, 1))

#scaler = StandardScaler()
#X_poly = scaler.fit_transform(X_poly)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_demo, test_size=0.3, random_state=42
)

# Train models
model_overfit = LinearRegression()
model_overfit.fit(X_train, y_train)

# Evaluate
train_score = model_overfit.score(X_train, y_train)
test_score = model_overfit.score(X_test, y_test)

print("\nPolynomial Model (degree 10) - NO REGULARIZATION: ")
print(f"   Training R²: {train_score:.4f}")
print(f"   Test R²:     {test_score:.4f}")
print(f"   Gap:         {train_score - test_score:.4f}")

if train_score - test_score > 0.1:
    print(f"  ⚠️  OVERFITTING DETECTED!")
    print(f"     Model memorized training data but fails on new data")
else:
    print(f"  ✅ No overfitting")

# APPLY REGULARIZATION

print("\n" + "="*70)
print("PART 2: APPLYING REGULARIZATION TO FIX OVERFITTING")
print("="*70)

# Test different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = []
lasso_scores = []

print(f"\n{'Alpha':>10} {'Ridge Train':>15} {'Ridge Test':>15} {'Lasso Train':>15} {'Lasso Test':>15}")
print("-"*75)

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_train = ridge.score(X_train, y_train)
    ridge_test = ridge.score(X_test, y_test)
    ridge_scores.append((ridge_train, ridge_test))

    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_train = lasso.score(X_train, y_train)
    lasso_test = lasso.score(X_test, y_test)
    lasso_scores.append((lasso_train, lasso_test))

    print(f"{alpha:>10.3f} {ridge_train:>15.4f} {ridge_test:>15.4f} "
          f"{lasso_train:>15.4f} {lasso_test:>15.4f}")
    
# Visualize regularization effect
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('REGULARIZATION: PREVENTING OVERFITTING', fontsize=16, fontweight='bold')

# Plot 1: Overfitting demonstration
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_pred_overfit = model_overfit.predict(X_plot_poly)

axes[0, 0].scatter(X_demo, y_demo, alpha=0.6, s=60, label='Training Data',
                   color='blue', edgecolors='black')
axes[0, 0].plot(X_plot, y_pred_overfit, 'r-', linewidth=2,
                label='Overfit Model (wiggly)')
axes[0, 0].plot(X_plot, 2*X_plot +3, 'g--', linewidth=2,
                label='True Pattern (straight)')
axes[0, 0].set_xlabel('X', fontweight='bold')
axes[0, 0].set_ylabel('y', fontweight='bold')
axes[0, 0].set_title('Overfitting Problem', fontweight='bold', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(-5, 35)

# Plot 2: Ridge regularization effect
ridge_best = Ridge(alpha=1.0)
ridge_best.fit(X_train, y_train)
y_pred_ridge = ridge_best.predict(X_plot_poly)

axes[0, 1].scatter(X_demo, y_demo, alpha=0.6, s=60, label='Training Data',
                   color='blue', edgecolors='black')
axes[0, 1].plot(X_plot, y_pred_ridge, 'purple', linewidth=3,
                label='Ridge Model (smooth)')
axes[0, 1].plot(X_plot, 2*X_plot +3, 'g--', linewidth=2,
                label='True Pattern')
axes[0, 1].set_xlabel('X', fontweight='bold')
axes[0, 1].set_ylabel('y', fontweight='bold')
axes[0, 1].set_title('Ridge Regression (α=1.0)', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-5, 35)

# Plot 3: Alpha Tunning for Ridge
train_scores = [s[0] for s in ridge_scores]
test_scores = [s[1] for s in ridge_scores]

axes[1, 0].plot(alphas, train_scores, 'o--', linewidth=2, markersize=8,
                label='Training Score', color='blue')
axes[1, 0].plot(alphas, test_scores, 's-', linewidth=2, markersize=8,
                label='Test Score', color='red')
axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('Alpha (Regularization Strength)', fontweight='bold')
axes[1, 0].set_ylabel('R² Score', fontweight='bold')
axes[1, 0].set_title('Ridge: Finding Optimal Alpha', fontweight='bold', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=1.0, color='green', linestyle='--', linewidth=2, 
                    label='Best α', alpha=0.5)

# Plot 4: Ridge vs Lasso comparison
ridge_test_scores = [s[1] for s in ridge_scores]
lasso_test_scores = [s[1] for s in lasso_scores]

axes[1, 1].plot(alphas, ridge_test_scores, 'o-', linewidth=2, markersize=8,
                label='Ridge (L2)', color='purple')
axes[1, 1].plot(alphas, lasso_test_scores, 's-', linewidth=2, markersize=8,
                label='Lasso (L1)', color='orange')
axes[1, 1].set_xscale('log')
axes[1, 1].set_xlabel('Alpha', fontweight='bold')
axes[1, 1].set_ylabel('Test R² Score', fontweight='bold')
axes[1, 1].set_title('Ridge vs Lasso Performance', fontweight='bold', fontsize=14)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_regularization_demo.png', dpi=300, bbox_inches='tight')
plt.close()

# REAL BUSINESS EXAMPLE: MARKETING WITH REGULARIZATION

print("\n" + "="*70)
print("REAL EXAMPLE: MARKETING MIX WITH MANY FEATURES")
print("="*70)

# Generating marketing data with MANY features (some irrelevant)
np.random.seed(99)
n = 200

# Relevant features
tv = np.random.uniform(20, 200, n)
social = np.random.uniform(10, 100, n)
email = np.random.uniform(5, 50, n)

# IRRELEVANT features (noise)
billboard = np.random.uniform(10, 80, n)  # Doesn't affect sales
print_media = np.random.uniform(5, 60, n) # Doesn't affect sales
events = np.random.uniform(10, 100, n)    # Doesn't affect sales
podcast = np.random.uniform(5, 40, n)     # Doesn't affect sales

# True relationship (only TV, Social, Email matter)
sales = (100 +
         2.5 * tv +
         3.0 * social +
         1.5 * email +
         np.random.normal(0, 20, n))

# Create DataFrame with ALL features
df_marketing_full = pd.DataFrame({
    'TV': tv,
    'Social': social,
    'Email': email,
    'Billboard': billboard,         # NOISE
    'Print_Media': print_media,     # NOISE
    'Events': events,               # NOISE
    'Podcast': podcast,             # NOISE
    'Sales': sales
})

print("\nDatset with 7 features (3 relevant, 4 noise): ")
print(df_marketing_full.head())

# Prepare data
X_full = df_marketing_full.drop('Sales', axis=1)
y_full = df_marketing_full['Sales']

# Standardize features (important for regularization!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_full.columns)

# Split
X_train_Sal, X_test_Sal, y_train_Sal, y_test_Sal = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=42
)

# COMPARE: LINEAR vs RIDGE vs LASSO

print("\n" + "="*70)
print("MODEL COMPARISON: LINEAR vs RIDGE vs LASSO")
print("="*70)

# Standard Linear Regression
lr = LinearRegression()
lr.fit(X_train_Sal, y_train_Sal)
lr_train = lr.score(X_train_Sal, y_train_Sal)
lr_test = lr.score(X_test_Sal, y_test_Sal)

# Ridge Regression
ridger = Ridge(alpha=1.0)
ridger.fit(X_train_Sal, y_train_Sal)
ridger_train = ridger.score(X_train_Sal, y_train_Sal)
ridger_test = ridger.score(X_test_Sal, y_test_Sal)

# Lasso Regression
Lassor = Lasso(alpha=0.5, max_iter=10000)
Lassor.fit(X_train_Sal, y_train_Sal)
Lassor_train = Lassor.score(X_train_Sal, y_train_Sal)
Lassor_test = Lassor.score(X_test_Sal, y_test_Sal)

# Result
print(f"\n{'Model':>20} {'Train R²':>12} {'Test R²':>12} {'Overfit Gap':>15}")
print("-"*65)
print(f"{'Linear Regression':>20} {lr_train:>12.4f} {lr_test:>12.4f} {lr_train - lr_test:>15.4f}")
print(f"{'Ridge (α=1.0)':>20} {ridger_train:>12.4f} {ridger_test:12.4f} {ridger_train - ridger_test:>15.4f}")
print(f"{'Lasso (α=0.5)':>20} {Lassor_train:12.4f} {Lassor_test:12.4f} {Lassor_train - Lassor_test:15.4f}")

print("\n💡 Interpretation:")
if ridger_test > lr_test:
    print("  → Ridge performs BETTER than Linear Regression on test data")
    print("  → Regularization successfully reduced overfitting!")

# FEATURE SELECTION WITH LASSO

print("\n" + "="*70)
print("LASSO: AUTOMATIC FEATURE SELECTION")
print("="*70)

# Compare Coefficient
coef_comparison = pd.DataFrame({
    'Feature': X_full.columns,
    'Linear_Coef': lr.coef_,
    'Ridge_Coef': ridger.coef_,
    'Lasso_Coef': Lassor.coef_
})

print("\nCoefficient Comparison: ")
print(coef_comparison.round(4))

# Count non-zero coefficients in Lasso
non_zero_lasso = np.sum(np.abs(Lassor.coef_) > 0.01)
print(f"\nLasso selected {non_zero_lasso} out of {len(X_full.columns)} features")

# Identify eliminated feature
eliminated = coef_comparison[np.abs(coef_comparison['Lasso_Coef']) < 0.01]['Feature'].tolist()
if eliminated:
    print(f"Features eliminated by Lasso: {eliminated}")
    print("→ These features had little predictive power (likely noise!)")

# VISUALIZE COEFFICIENTS

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('COEFFICIENT COMPARISON: LINEAR vs RIDGE vs LASSO',
             fontsize=16, fontweight='bold')

# Linear coefficients
axes[0].barh(X_full.columns, lr.coef_, color='#3498db', edgecolor='black')
axes[0].set_xlabel('Coefficient Value', fontweight='bold')
axes[0].set_title('Linear Regression\n(No Regularization)', fontweight='bold', fontsize=13)
axes[0].grid(axis='x', alpha=0.3)
axes[0].axvline(x=0, color='black', linewidth=1)

# Ridge Coefficients
axes[1].barh(X_full.columns, ridger.coef_, color='purple', edgecolor='black')
axes[1].set_xlabel('Coefficient Value', fontweight='bold')
axes[1].set_title('Ridge Regression\n(L2: Shrinks all coefficients)', fontweight='bold', fontsize=13)
axes[1].grid(axis='x', alpha=0.3)
axes[1].axvline(x=0, color='black', linewidth=1)

# Lasso Coefficients
colors = ['#2ecc71' if abs(c) > 0.01 else '#e74c3c' for c in lasso.coef_]
axes[2].barh(X_full.columns, Lassor.coef_, color=colors, edgecolor='black')
axes[2].set_xlabel('Coefficient Value', fontweight='bold')
axes[2].set_title('Lasso Regression\n(L1: Sets some to ZERO)', fontweight='bold', fontsize=13)
axes[2].grid(axis='x', alpha=0.3)
axes[2].axvline(x=0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig('06_coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 06_coefficient_comparison.png")

# HYPERPARAMETER TUNNING

print("\n" + "="*70)
print("HYPERPARAMETER TUNING: FINDING OPTIMAL ALPHA")
print("="*70)

alphas_grid = np.logspace(-3, 3, 50)  # 0.001 to 1000
ridge_cv_scores = []
lasso_cv_scores = []

print("Testing 50 different alpha values")

for alpha in alphas_grid:
    # Ridge with cross-validation
    ridge_model = Ridge(alpha=alpha)
    ridge_scores = cross_val_score(ridge_model, X_train_Sal, y_train_Sal,
                                   cv=5, scoring='r2')
    ridge_cv_scores.append(ridge_scores.mean())

    # Lasso with cross-validation
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_scores = cross_val_score(lasso_model, X_train_Sal, y_train_Sal,
                                   cv=5, scoring='r2')
    lasso_cv_scores.append(lasso_scores.mean())

# Find best alphas
best_ridge_idx = np.argmax(ridge_cv_scores)
best_lasso_idx = np.argmax(lasso_cv_scores)

best_ridge_alpha = alphas_grid[best_ridge_idx]
best_lasso_alpha = alphas_grid[best_lasso_idx]

print(f"\n✅ OPTIMAL HYPERPARAMETERS:")
print(f"   Ridge best alpha: {best_ridge_alpha:.4f} (CV  R² = {ridge_cv_scores[best_ridge_idx]:.4f})")
print(f"   Lasso best alpha: {best_lasso_alpha:.4f} (CV R² = {lasso_cv_scores[best_lasso_idx]:.4f})")

# Train final models with best alphas
final_ridge = Ridge(alpha=best_ridge_alpha)
final_ridge.fit(X_train_Sal, y_train_Sal)

final_lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
final_lasso.fit(X_train_Sal, y_train_Sal)

# Evaluate on test set
print(f"\n📊 FINAL TEST PERFORMANCE:")
print(f" Ridge: R² = {final_ridge.score(X_test_Sal, y_test_Sal):.4f}")
print(f" Lasso: R² = {final_lasso.score(X_test_Sal, y_test_Sal):.4f}")

# Visualize hyperparameter tunning
plt.figure(figsize=(12, 6))
plt.plot(alphas_grid, ridge_cv_scores, 'o-', linewidth=2, markersize=5,
         label='Ridge', color='purple')
plt.plot(alphas_grid, lasso_cv_scores, 's-', linewidth=2, markersize=5,
         label='Lasso', color='orange')
plt.axvline(x=best_ridge_alpha, color='purple', linestyle='--',
            linewidth=2, alpha=0.5, label=f'Best Ridge α={best_ridge_alpha:.3f}')
plt.axvline(x=best_lasso_alpha, color='orange', linestyle='--',
            linewidth=2, alpha=0.5, label=f'Best Lasso α={best_lasso_alpha:.3f}')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)', fontweight='bold', fontsize=12)
plt.ylabel('Cross-Validation R² Score', fontweight='bold', fontsize=12)
plt.title('Hyperparameter Tuning: Finding Optimal Alpha', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('07_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 07_hyperparameter_tuning.png")

# BUSINESS RECOMMENDATIONS

print("\n" + "="*70)
print("💡 BUSINESS INSIGHTS FROM REGULARIZATION")
print("="*70)

# Get important features from Lasso
important_features = coef_comparison[
    np.abs(coef_comparison['Lasso_Coef']) > 0.01
].sort_values('Lasso_Coef', ascending=False)

print(f"""
MARKETING CHANNEL OPTIMIZATION:
{'─'*50}

Channels Worth Investing In:
{chr(10).join([f'  ✅ {row["Feature"]}: ROI coefficient = {row["Lasso_Coef"]:.3f}' 
               for _, row in important_features.iterrows()])}

Channels to REDUCE/ELIMINATE:
{chr(10).join([f'  ❌ {feat}: No measurable impact on sales' 
               for feat in eliminated])}

RECOMMENDATION:
→ Reallocate budget from eliminated channels to high-ROI channels
→ Expected impact: {((Lassor_test - 0.5) * 100):.0f}% improvement in prediction accuracy
→ Model confidence: R² = {Lassor_test:.2%}

""")


print("="*70)
print("PROJECT 3 COMPLETE: REGULARIZATION MASTERED!")
print("="*70)