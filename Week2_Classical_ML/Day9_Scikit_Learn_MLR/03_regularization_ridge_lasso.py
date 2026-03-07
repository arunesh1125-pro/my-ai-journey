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