"""
LINEAR REGRESSION: COMPLETE IMPLEMENTATION FROM SCRATCH
========================================================
Build the algorithm yourself to truly understand it
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("="*70)
print("LINEAR REGRESSION: BUILDING FROM SCRATCH")
print("="*70)

class LinearRegressionScratch:
    """
    Linear Regression implemented from first principles
    No libraries except NumPy for math operations
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize model

        Parameters:
        -----------
        learning_rate: float
            How big of steps to take during learning
        iteration: int
            How many times to upgrade parameters
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = None  # Slope
        self.b = None  # Intercepts
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model on data

        Parameters:
        -----------
        X: array-like, shape (n_samples,)
           Training features
        y: array-like, shape (n_samples,)
           Target values
        """
        # Convent to numpy arrays
        X = np.array(X)
        y = np.array(y)

        n = len(y)

        # Initialize parameters randomly
        self.m = np.random.randn()
        self.b = np.random.randn()

        # Gradient Descent
        for i in range(self.iterations):
            # Make predictions
            y_pred = self.m * X + self.b

            # Calculate cost (MSE)
            cost = np.sum((y_pred - y) ** 2) / n
            self.cost_history.append(cost)

            # Calculate gradients
            dm = (2/n) * np.sum(X * (y_pred - y))
            db = (2/n) * np.sum(y_pred - y)

            # Update parameters
            self.m = self.m - self.learning_rate * dm
            self.b = self.b - self.learning_rate * db

        return self
    
    def predict(self, X):
        """
        Make predictions on new data

        Parameters:
        -----------
        X: array-like
           Features to predict on

        Returns:
        --------
        predictions: array
            Predicted values
        """
        X = np.array(X)
        return self.m * X + self.b
        
    
    def score(self, X, y):
        """
        Calculate R² score (Coefficient of determination)

        R² = 1 - (SS_res / SS_tot)

        where:
        SS_res = Σ(y - ŷ)²  (residual sum of squares)
        SS_tot = Σ(y - ȳ)²  (total sum of squares)
        """
        X = np.array(X)
        y = np.array(y)

        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - (ss_res / ss_tot)

        return r2
    
# Example 1: MARKETING ROI

print("\n" + "="*70)
print("EXAMPLE 1: MARKETING SPEND → SALES REVENUE")
print("="*70)

# Generate realistic marketing data
np.random.seed(42)
months = 50
ad_spend = np.linspace(30, 250, months) # ₹30L to ₹250L
# True realtionship: Sales = 2.5 × AdSpend + 60 + noise
sales = 2.5 * ad_spend + 60 + np.random.normal(0, 20, months)

print(f"\nDataset: {months} months of marketing data")
print(f"Ad Spend range: ₹{ad_spend.min():.0f}L - ₹{ad_spend.max():.0f}L")
print(f"Sales range: ₹{sales.min():.0f}L - ₹{sales.max():.0f}L")

#RuntimeWarning: invalid value encountered in scalar subtract: self.m = self.m - self.learning_rate * dm = It becomes infinity
# Train Model: becomes Sales = nan × Ad_Spend + nan, Business Insights: → ROI: nan% return on every rupee spent
# So Before Train the model, Normalize the Feature X and target y and after convert back to the real scale as m_final and b_final or another alternative solution was learning_rate = 0.001 -> 0.0000001

# Normalize features and Train model
#Normalize X, y
X_norm = (ad_spend - ad_spend.mean()) / ad_spend.std()
y_norm = (sales - sales.mean()) / sales.std()

# Train model
model = LinearRegressionScratch(learning_rate=0.001, iterations=1000)
model.fit(X_norm, y_norm)

# Train model
# model = LinearRegressionScratch(learning_rate=0.001, iterations=1000)
# model.fit(ad_spend, sales)

# Then convert back to real scale
m_final = model.m * (sales.std() / ad_spend.std()) # m_final = model.m * (y.std() / X.std())
b_final = sales.mean() - m_final * ad_spend.mean() # b_final = y.mean() - m_final * X.mean()


print(f"\nTrain Model:")
print(f" Sales = {m_final:.3f} × Ad_Spend + {b_final:.2f}")
print(f"Business Insights:")
print(f"  → ROI: {(m_final - 1) * 100:.1f}% return on every rupee spent")
print(f"  → Break-even: ₹{abs(b_final / m_final):.2f} lakhs ad spend")
print(f"  → Model accuracy (R²): {model.score(X_norm, y_norm):.4f}")

# Make business predictions
new_budgets = np.array([100, 150, 200, 300])

# Normalize inputs
new_budgets_norm = (new_budgets - ad_spend.mean()) / ad_spend.std()

#Predict in normalized space
predicted_sales_norm = model.predict(new_budgets_norm)

# Convert back to real scale
predicted_sales = predicted_sales_norm * sales.std() + sales.mean()

print(f"\n{'Budget Scenerios:':^50}")
print(f"{'─'*50}")
print(f"{'Ad Budget (₹L)':>15} {'Predicted Sales (₹L)':>20} {'Expected ROI':>15}")
print(f"{'─'*50}")
for budget, pred_sale in zip(new_budgets, predicted_sales):
    roi = ((pred_sale - budget) / budget) * 100
    print(f"{budget:>15.0f} {pred_sale:>20.2f} {roi:>14.1f}%")

# Visulaization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Marketing ROI Analysis - Linear Regression',
             fontsize=16, fontweight='bold')

# Plot 1: Scatter + Regression Line
axes[0, 0].scatter(X_norm, y_norm, alpha=0.6, s=50,
                   label=f'Actual Data', color='#3498db', edgecolors='black')
axes[0, 0].plot(X_norm, model.predict(X_norm), 'r-',
                linewidth=3, label=f'Model: y={model.m:.2f}x+{model.b:.2f}')
axes[0, 0].set_xlabel('Ad Spend (₹ lakhs)', fontweight='bold')
axes[0, 0].set_ylabel('Sales (₹ lakhs)', fontweight='bold')
axes[0, 0].set_title('Ad Spend vs Sales Revenue', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals (errors)
residuals = y_norm - model.predict(X_norm)
axes[0, 1].scatter(model.predict(X_norm), residuals,
                   alpha=0.6, s=50, color='#e74c3c', edgecolors='black')
axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Sales', fontweight='bold')
axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontweight='bold')
axes[0, 1].set_title('Residual Plot (Error Analysis)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

#Plot 3: Cost History
axes[1, 0].plot(model.cost_history, linewidth=2, color='#2ecc71')
axes[1, 0].set_xlabel('Iteration', fontweight='bold')
axes[1, 0].set_ylabel('Cost (MSE)', fontweight='bold')
axes[1, 0].set_title('Learning Curve (Cost Reduction)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

#Plot 4: Prediction vs Actual
axes[1, 1].scatter(y_norm, model.predict(X_norm),
                   alpha=0.6, s=50, color='#9b59b6', edgecolors='black')
axes[1, 1].plot([y_norm.min(), y_norm.max()], [y_norm.min(), y_norm.max()],
                'r--', linewidth=2, label='Perfect Predictions')
axes[1, 1].set_xlabel('Actual Sales', fontweight='bold')
axes[1, 1].set_ylabel('Predicted Sales', fontweight='bold')
axes[1, 1].set_title('Predicted vs Actual', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_marketing_roi_model.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 03_marketing_roi_model.png")