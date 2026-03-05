"""
LINEAR REGRESSION: MATHEMATICAL FOUNDATION
==========================================
Understanding the math behind the magic
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("LINEAR REGRESSION: THEORY & MATHEMATICS")
print("="*70)

# THE LINEAR REGRESSION EQUATION

print("""
╔══════════════════════════════════════════════════════════════╗
║              LINEAR REGRESSION EQUATION                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Simple Linear Regression (1 feature):                       ║
║  ŷ = m × X + b                                               ║
║                                                              ║
║  Where:                                                      ║
║  ŷ (y-hat) = Predicted value                                 ║
║  X          = Feature (input variable)                       ║
║  m          = Slope (how much y changes per unit X)          ║
║  b          = Intercept (y value when X = 0)                 ║
║                                                              ║
║  Multiple Linear Regression (many features):                 ║
║  ŷ = b₀ + b₁X₁ + b₂X₂ + ... + bₙXₙ                          ║
║                                                              ║
║  Matrix Form (what computers actually use):                  ║
║  ŷ = X · θ                                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# BUSINESS EXAMPLE: MARKETING ROI

print("\n" + "="*70)
print("REAL BUSINESS EXAMPLE: MARKETING ROI PREDICTION")
print("="*70)

# Simulated marketing data (based on real patterns)
np.random.seed(42)
months = 12
ad_spend = np.array([50, 65, 80, 95, 110, 125, 140, 155, 170, 185, 200, 215]) # ₹ lakhs
sales = ad_spend * 2.5 + 45 + np.random.normal(0, 15, months)  # ₹ lakhs

print("""
Scenerio: E-commerce Startup Marketing Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Company: "QuickCart" (online grocery)
Data : 12 months of ad Spend vs sales revenue
Question: What's the ROI on marketing spend?
      
Historical Data:
""")

print(f"{'Month':>6} {'Ad Spend (₹L)':>15} {'Sales (₹L)':>15}")
print("-"*40)
for i, (spend, sale) in enumerate(zip(ad_spend, sales), 1):
    print(f"{i:>6} {spend:>15.0f} {sale:>15.2f}")

# COST FUNCTION (Mean Squared Error)

print("\n" + "="*70)
print("COST FUNCTION: MEASURING HOW WRONG WE ARE")
print("="*70)

print("""
Goal: Find m and b that minimizes prediction error

Cost Function (Mean Squared Error - MSE):
           n
          Σ (yᵢ - ŷᵢ)²  
          i=1
J(m,b) = ─────────────
              n
      
where :
- yᵢ = actual value
- ŷᵢ = predicted value (m×Xᵢ + b)
- n = number of samples
      
WHY SQUARED error?
1. Penalizes large errors more
2. Always positive
3. Mathematically convenient (differentiable)
4. Industry standard
""")

def compute_cost(X, y, m, b):
    n = len(y)
    predictions = m * X + b
    squared_errors = (predictions - y) ** 2
    mse = np.sum(squared_errors) / n
    return mse

# Test with deifferent parameters
print("\nTesting different models:")
print(f"{'m (slope)':>12} {'b (intercept)':>15} {'MSE (error)':>15}")
print("-"*45)

test_parms = [
    (1.0, 50),  # Bad guess
    (2.0, 45),  # Better
    (2.3, 45),  # Close to optimal
    (3.0, 30)  # Overfitting
]

for m, b in test_parms:
    mse = compute_cost(ad_spend, sales, m, b)
    print(f"{m:>12.1f} {b:>15.1f} {mse:>15.2f}")

print("\n→ Lower MSE = Better model!")

# GRADIENT DECENT: THE LEARNING ALGORITHM

print("\n" + "="*70)
print("GRADIENT DESCENT: HOW MACHINES LEARN")
print("="*70)

print("""
Gradient Descent Algorithm:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Imagine you're blinfolded on a hill and want to reach the valley.
Strategy: Feel the Slope, take small steps downhill.

In ML:
1. Start with random m and b
2. Calculate cost (how wrong we are)
3. Calculate gradient (which direction to adjust)
4. Update parameters: m = m - α × gradient
5. Repeat until cost stops decreasing
      
α (alpha) = Learning rate (step size)
- Too small: Takes forever to learn
- Too large: Might overshoot minimum
- Typical value: 0.01 to 0.001
      
Mathematical Update Rules:
∂J
m := m - α × ──
∂m

∂J
b := b - α × ──
∂b
      
Where ∂J/∂m and ∂J/∂b are partial derivatives
(how much cost changes when we change m or b)
""")

def gradient_descent(X, y, learing_rate=0.001, iterations=1000):
    """
    learn optimal m and b using gradient descent
    """
    n = len(y)
    m = 0.0 # Start with random parameter
    b = 0.0

    cost_history = []

    for i in range(iterations): # Loop for Training
        # Predictions
        y_pred = m * X + b

        # Calculate cost
        cost = np.sum((y_pred - y) ** 2) / n # MSE
        cost_history.append(cost)

        # Calculate gradients (how to adjust parameters)
        dm = (2/n) * np.sum(X * (y_pred - y)) # Derivative w.r.t.m
        db = (2/n) * np.sum(y_pred - y)       # Derivative w.r.t.b

        # Update parameters (take step downhill)
        m = m - learing_rate * dm
        b = b - learing_rate * db

        # Print progress every 100 iterations
        if (i + 1)%100 == 0:
            print(f"Iteration {i+1:>4}: Cost = {cost:>10.2f}, m = {m:>6.3f}, b = {b:>6.2f}")

    return m, b, cost_history

print("\nTraining Linear Regression Model:")
print("="*50)

# Normalize data for faster convergence
X_normalized = (ad_spend - ad_spend.mean()) / ad_spend.std()
y_normalized = (sales - sales.mean()) / sales.std()

m_norm, b_norm, cost_history = gradient_descent(
    X_normalized, y_normalized,
    learing_rate=0.001,
    iterations=1000
)

# Convert back to original scale
m_final = m_norm * (sales.std() / ad_spend.std())
b_final = sales.mean() - m_final * ad_spend.mean()

print(f"\n{'='*50}")
print("FINAL MODEL:")
print(f"{'='*50}")
print(f"Sales = {m_final:.3f} × Ad_Spend + {b_final:.2f}")
print()
print("Business Interpretation:")
print(f"  → For every ₹1 lakh spent on ads, sales increase by ₹{m_final:.2f} lakhs")
print(f"  → Base sales (no ads): ₹{b_final:.2f} lakhs")
print(f"  → ROI: {(m_final - 1) * 100:.1f}% return on ad spend")

# VISUALIZE LEARNING PROCESS

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Cost over time
axes[0].plot(cost_history, linewidth=2, color='#e74c3c')
axes[0].set_xlabel('Iteration', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Cost (MSE)', fontweight='bold', fontsize=12)
axes[0].set_title('Gradient Descent: Cost Reduction', fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=cost_history[-1], color='green', linestyle='--',
                label=f'Final Cost: {cost_history[-1]:.4f}')
axes[0].legend()

# Plot 2: Final fit
axes[1].scatter(ad_spend, sales, color='red', s=100,
                label='Actual Sales', zorder=3, edgecolor='black')
predictions = m_final * ad_spend + b_final
axes[1].plot(ad_spend, predictions, 'b-', linewidth=3,
             label=f'Learned Model: y = {m_final:.2f}x + {b_final:.2f}')
axes[1].set_xlabel('Ad Spend (₹ lakhs)', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Sales (₹ lakhs)', fontweight='bold', fontsize=12)
axes[1].set_title('Marketing ROI Model', fontweight='bold', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_gradient_descent.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 02_gradient_descent.png")

print("\n" + "="*70)
print("LINEAR REGRESSION THEORY COMPLETE!")
print("="*70)