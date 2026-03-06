"""
SCIKIT-LEARN: THE INDUSTRY STANDARD ML LIBRARY
===============================================
Learn the toolkit that 90% of ML engineers use daily
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("="*70)
print("SCIKIT-LEARN: PROFESSIONAL ML WORKFLOW")
print("="*70)

# WHY SCIKIT-LEARN ?
print("""
╔══════════════════════════════════════════════════════════════╗
║                    WHY SCIKIT-LEARN?                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ✅ Industry Standard: Used by 90% of ML practitioners       ║
║  ✅ Consistent API: All algorithms work the same way         ║
║  ✅ Well-Tested: Battle-tested in production for 10+ years   ║
║  ✅ Comprehensive: 100+ ML algorithms built-in               ║
║  ✅ Fast: Optimized C/Cython underneath                      ║
║  ✅ Well-Documented: Excellent tutorials and examples        ║
║                                                              ║
║  Used By:                                                    ║
║  • Spotify (recommendation systems)                          ║
║  • Booking.com (price prediction)                            ║
║  • Evernote (content classification)                         ║
║  • Every data science team globally                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# THE SCI-KIT LEARN API PATTERN

print("\n" + "="*70)
print("THE SCIKIT-LEARN API: CONSISTENT ACROSS ALL ALGORITHMS")
print("="*70)

print("""
Every scikit-learn model follows this pattern:
      
1. IMPORT
   from slearn.xxx import ModelName
      
2. INSTANTIATE (create model object)
   model = ModelName(hyperparameter1=value1, ...)
      
3. FIT (train on data)
   model.fit(X_train, y_train)
      
4. PREDICT (make predictions)
   predictions = model.predict(X_test)

5. EVALUATE (measure performance)
   score = model.score(X_test, y_test)
      
The SAME pattern works for:
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Neural Networks
- Everything!

Master this pattern → Master scikit-learn!
""")

# EXAMPLE: SIMPLE LINEAR REGRESSION

print("\n" + "="*70)
print("STEP-BY-STEP EXAMPLE: MARKETING ROI")
print("="*70)

# Generate sample data
np.random.seed(42)
ad_spend = np.array([50, 65, 80, 85, 110, 125, 140, 155, 170, 185, 200, 215])
sales = 2.3 * ad_spend + 50 + np.random.normal(0, 15, len(ad_spend))

print("\nStep 1: Prepare Data")
print("-"*50)
# Reshape to 2D array (sklearn requirement)
X = ad_spend.reshape(-1, 1)   # Features must be 2D
y = sales                     # Target can be 1D

print(f"X shape: {X.shape} (n_samples, n_features)")
print(f"y shape: {y.shape} (n_samples,)")
print(f"\nFirst 3 rows of data:")
print(f"{'Ad Spend':>12} {'Sales':>12}")
for i in range(3):
    print(f"{X[i, 0]:>12.0f} {y[i]:>12.2f}")

print("\nStep 2: Split into Train/Test")
print("-"*50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")
print(f"Split ratio:  80/20 (standard practice)")

print("\nStep 3: Create Model")
print("-"*50)
model = LinearRegression()
print(f"Model: {model}")
print("Parameters: None specified (using defaults)")

print("\nStep 4: Train Model")
print("-"*50)
model.fit(X_train, y_train)
print("✅ Model trained!")
print(f"\nLearned Parameters:")
print(f"  Coefficient (m): {model.coef_[0]:.3f}")
print(f"  Intercept (b):   {model.intercept_:.2f}")
print(f"\nEquation: Sales = {model.coef_[0]:.3f} × Ad_Spend + {model.intercept_:.2f}")

print("\nStep 5: Make Predictions")
print("-"*50)
y_pred = model.predict(X_test)
print(f"{'Actual':>12} {'Predicted':>12} {'Error':>12}")
print("-"*40)
for actual, pred in zip(y_test, y_pred):
    error = abs(actual - pred)
    print(f"{actual:>12.2f} {pred:>12.2f} {error:>12.2f}")

print("\nStep 6: Evaluate Model")
print("-"*50)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score:  {r2:.4f} (1.0 = perfect, 0.0 = useless)")
print(f"RMSE:      ₹{rmse:.2f} lakhs")
print(f"MAE:       ₹{mae:.2f} lakhs")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training and Test Data
axes[0].scatter(X_train, y_train, color='blue', s=100, 
                label='Training Data', alpha=0.6, edgecolors='black')
axes[0].scatter(X_test, y_test, color='red', s=100, 
                label='Test Data', alpha=0.6, edgecolors='black')
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
axes[0].plot(X_line, y_line, 'g-', linewidth=3, label='Fitted Line')
axes[0].set_xlabel('Ad Spend (₹ lakhs)', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Sales (₹ lakhs)', fontweight='bold', fontsize=12)
axes[0].set_title('Train/Test Split & Model Fit', fontweight='bold', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual
axes[1].scatter(y_test, y_pred, color='purple', s=100,
                alpha=0.6, edgecolors='black')
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Predictions')
axes[1].set_xlabel('Actual Sales', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Predicted Sales', fontweight='bold', fontsize=12)
axes[1].set_title(f'Prediction Quality (R²={r2:.3f})', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_sklearn_workflow.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_sklearn_workflow.png")

# KEY CONCEPTS

print("\n" + "="*70)
print("KEY CONCEPTS EXPLAINED")
print("="*70)

concepts = """
1. TRAIN/TEST SPLIT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Why? Can't evaluate on same data we trained on!
   That's like giving students the same exam they studied from.

   Standard split: 80/20 or 70/30
   Training set: Used to learn parameters
   Test set: Used to evaluate real-world performance

2. R² SCORE (Coefficient of Determination)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Range: 0 to 1 (can be negative if model is terrible)

   R² = 1.0  → Perfect predictions
   R² = 0.7  → Good model
   R² = 0.5  → Mediocre
   R² = 0.0  → Model is useless (predicting mean is better!)
   R² < 0    → Model is worse than predicting mean 

   Interpretation: "Model explains X% of variance in target"

3. RMSE (Root Mean Squared Error)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   same units as target variable
   Average prediction error

   If RMSE = ₹20 lakhs, predictions are off by ±₹20L on average
   Lower is better

4. MAE (Mean Absolute Error)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   More robust to outliers than RMSE
   Easier to interpret: "Average error is X units"

5. MODEL PERSISTENCE
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Once trained, you can save and load models:

   import joblib
   joblib.dump(model, 'model.pkl')   # Save
   model = joblib.load('model.pkl')   # Load

   This is how you deploy models to production!
"""

print(concepts)

print("\n" + "="*70)
print("SCIKIT-LEARN BASICS: COMPLETE!")
print("="*70)